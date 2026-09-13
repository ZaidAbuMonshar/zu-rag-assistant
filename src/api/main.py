"""
ZU RAG Assistant — Retrieval + Generation API (FastAPI)
Implements /search and /ask per api_contract_retrieval_v1.md

Run locally (from the project root):
    pip install fastapi uvicorn chromadb llama-cpp-python
    uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive Swagger UI.

Note: startup loads both BGE-M3 (embedding) and Qwen2.5-3B-Instruct (generation)
into memory once, so the first request after starting the server will be slow
to arrive at all -- that's model loading, not a hang.

Verified against src/ingest/build_chroma_db.py:
- This file lives at src/api/main.py (project root is two levels up).
- `BGEM3EmbeddingFunction` in `src/ingest/build_chroma_db.py` is the correct
  class name — matches as-is.
- The persisted collection is at `data/chroma_db/`, collection name `zu_docs_v1`
  — matches build_chroma_db.py.
- IMPORTANT: build_chroma_db.py stores `embedding_text` (title-prefixed) as
  the Chroma "document", and the clean `chunk_text` only in metadata. The
  original draft read chunk_text from raw["documents"], which would have
  returned the title-prefixed text to API callers instead of the clean
  display text — fixed below to read it from metadata instead.
"""

import logging
import os
import secrets
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — same self-locating pattern used in build_chroma_db.py /
# eval_recall_v3.py, so this works regardless of the working directory it's
# launched from.
#
# __file__ = .../zu-rag-assistant/src/api/main.py
# .parent            -> .../zu-rag-assistant/src/api
# .parent.parent     -> .../zu-rag-assistant/src
# .parent.parent.parent -> .../zu-rag-assistant   (project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src" / "rag"))

from retrieve import get_collection, retrieve_chunks  # noqa: E402
from generate import generate_answer, load_llm  # noqa: E402

DEFAULT_TOP_K = 3
MAX_TOP_K = 10

logger = logging.getLogger("zu_rag_api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# API key — a single shared secret required on /search and /ask. This is
# deliberately a *simple* check, not a full auth system: one static key
# checked via a header, sized to the actual scope of a solo/local-only
# project rather than over-engineered.
#
# If ZU_RAG_API_KEY isn't set in the environment, a random key is generated
# for this run and logged once at startup -- so a local demo still works
# with zero configuration (no request can succeed without *some* key, but
# nobody has to remember to set one just to run `python src/api/main.py`).
# A longer-lived deployment should set ZU_RAG_API_KEY explicitly so the key
# survives a server restart.
# ---------------------------------------------------------------------------
API_KEY_HEADER = "X-API-Key"
_env_key = os.environ.get("ZU_RAG_API_KEY")
API_KEY = _env_key or secrets.token_urlsafe(24)
if not _env_key:
    logger.info(
        "ZU_RAG_API_KEY not set -- generated a one-time key for this run: %s "
        "(the bundled UI picks this up automatically; set ZU_RAG_API_KEY "
        "explicitly for any deployment that needs a stable key across restarts)",
        API_KEY,
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


# ---------------------------------------------------------------------------
# Chroma collection + LLM — both loaded once at startup, reused across
# requests. Same reason for both: get_collection() loads BGE-M3, load_llm()
# loads the ~2GB Qwen2.5-3B weights -- recreating either per-request would
# make every /search or /ask call pay a multi-second (or worse) reload cost.
# ---------------------------------------------------------------------------
app = FastAPI(title="ZU RAG Assistant — Retrieval + Generation API", version="1.0")

_collection = get_collection()
_llm = load_llm()

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Thin UI — a single self-contained HTML page (no build step, no external
# dependencies) served at the app root, so opening http://127.0.0.1:8000/ in
# a browser gives a usable demo without needing a separate frontend project.
# Not behind the API key itself (a browser needs to load the page before it
# has any key to send) -- the key gets injected into the page at serve time
# instead, so its own fetch("/ask") call can include it automatically.
# ---------------------------------------------------------------------------
@app.get("/")
def ui():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("__ZU_RAG_API_KEY__", API_KEY)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Request / response models — mirror api_contract_retrieval_v1.md exactly
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question (Arabic or English).")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class SearchResult(BaseModel):
    chunk_id: str
    chunk_text: str
    source_doc: str
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question (Arabic or English).")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query field is required")

    try:
        chunks = retrieve_chunks(_collection, req.query, top_k=req.top_k)
    except Exception:
        # Log the real exception server-side; the client only ever sees a
        # generic message -- exception text (stack internals, file paths,
        # library errors) must not leave the machine. See
        # api_contract_retrieval_v1.md's Errors section.
        logger.exception("Unhandled error in /search")
        raise HTTPException(status_code=500, detail="an internal error occurred")

    results = [SearchResult(**chunk) for chunk in chunks]
    return SearchResponse(query=req.query, results=results)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query field is required")

    try:
        result = generate_answer(
            req.query,
            top_k=req.top_k,
            llm=_llm,
            collection=_collection,
        )
    except Exception:
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=500, detail="an internal error occurred")

    return AskResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[SearchResult(**src) for src in result["sources"]],
    )


# ---------------------------------------------------------------------------
# Custom error shape per contract §5 (FastAPI's default 422/500 bodies won't
# match this automatically — this handler normalizes HTTPException output)
# ---------------------------------------------------------------------------
_ERROR_CODES = {400: "invalid_request", 401: "unauthorized"}


@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException):
    code = _ERROR_CODES.get(exc.status_code, "internal_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": exc.detail}},
    )


# ---------------------------------------------------------------------------
# Lets `python src/api/main.py` (e.g. PyCharm's Run button) start the server
# directly, instead of only working via `uvicorn src.api.main:app --reload`.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
