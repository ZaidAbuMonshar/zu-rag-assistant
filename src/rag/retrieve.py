"""
Shared retrieval helper: query Chroma, return clean top-k chunks with metadata.

Extracted from src/api/main.py's /search handler once generate.py needed the
exact same "top-k chunks with clean chunk_text + metadata" logic -- two call
sites doing the same query/unpack was the point to share it.

Note this is deliberately NOT the same as eval_recall_v3.py's
top_k_distinct_docs(): the eval script dedupes to distinct *documents* for the
recall@3 metric. This module returns raw top-k *chunks* (a document can appear
more than once), which is what both /search and generation actually want.
"""

import sys
from pathlib import Path
from typing import TypedDict

import chromadb

# retrieve.py lives in src/rag/, build_chroma_db.py lives in src/ingest/ --
# same sys.path pattern as eval_recall_v3.py in this same folder.
_INGEST_DIR = Path(__file__).resolve().parent.parent / "ingest"
sys.path.insert(0, str(_INGEST_DIR))

from build_chroma_db import (  # noqa: E402
    BGEM3EmbeddingFunction,
    CHROMA_PATH,
    COLLECTION_NAME,
)
from query_expansion import expand_query  # noqa: E402

DEFAULT_TOP_K = 3


class RetrievedChunk(TypedDict):
    chunk_id: str
    chunk_text: str
    source_doc: str
    chunk_index: int
    score: float


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = BGEM3EmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def retrieve_chunks(collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    # expand_query only appends terms for the documented informal/formal
    # terminology gap; it's a no-op for any other query.
    raw = collection.query(query_texts=[expand_query(query)], n_results=top_k)

    ids = raw.get("ids", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[RetrievedChunk] = []
    for chunk_id, meta, distance in zip(ids, metadatas, distances):
        score = max(0.0, 1.0 - distance)
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                # metadata["chunk_text"] is the clean text -- Chroma's
                # "documents" field holds the title-prefixed embedding_text,
                # not what should be shown/fed to the LLM. See src/api/main.py.
                chunk_text=meta.get("chunk_text", ""),
                source_doc=meta.get("source_doc", ""),
                chunk_index=meta.get("chunk_index", -1),
                score=round(score, 4),
            )
        )
    return results
