# ZU RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions
in Arabic over a university department's internal policy documents — built solo
(retrieval, generation, API, and deployment) on fully self-hosted, CPU-only
infrastructure, with no third-party model APIs.

## Why self-hosted

The source documents are internal institutional policies, and there was no confirmed
clearance to send that data to a third-party API (e.g. OpenAI embeddings). So the
entire pipeline — embeddings, vector store, and the LLM itself — runs locally: no GPU,
no paid API, no data leaving the machine it runs on.

## What it does

- **Dense retrieval** — BGE-M3 multilingual embeddings over Chroma (cosine similarity),
  36 Arabic PDF policy documents chunked into title-aware passages.
- **Query expansion** — a small, hand-diagnosed phrase table closing specific
  informal-language retrieval gaps found through eval failures, not brute-force
  reranking. Took Recall@3 from 82.35% to **100% (17/17)** on a hand-built eval set.
- **Hybrid search (BM25 + dense)** — Okapi BM25 implemented from scratch (custom
  inverted index, hand-picked Arabic stopword list, a tokenizer that separates
  digit-runs from letter-runs to handle PDF table extraction gluing course codes to
  adjacent words), fused with dense retrieval via weighted Reciprocal Rank Fusion.
  Matches the same 100% recall independently.
- **Grounded generation** — Qwen2.5-3B-Instruct (GGUF, quantized) via
  `llama-cpp-python`, fully local. The prompt explicitly instructs the model to answer
  only from retrieved context and decline rather than guess when it doesn't know.
- **LangChain integration** — the retriever, hybrid retriever, and local LLM are each
  wrapped behind LangChain's `BaseRetriever` / `BaseChatModel` interfaces, verified
  output-for-output against the hand-built originals before being adopted.
- **API** — FastAPI with two endpoints: `/search` (retrieval only) and `/ask`
  (retrieval + grounded generation), API-key auth, sanitized error responses.
- **UI** — a single self-contained Arabic RTL HTML page, no build step, no external
  dependencies.
- **Deployment** — Dockerized (CPU-only PyTorch pinned explicitly so pip's resolver
  never pulls in CUDA dependencies transitively), deployed and verified end-to-end on
  a cloud VM.

## Tech stack

Python · FastAPI · ChromaDB · BGE-M3 · `llama-cpp-python` + Qwen2.5-3B-Instruct (GGUF)
· LangChain (core) · Docker / Docker Compose

## Results

- **Recall@3: 100% (17/17)** on a hand-built Arabic eval set, achieved through targeted
  query-expansion fixes rather than a larger/heavier retrieval model.
- Hybrid (BM25 + dense) retrieval independently matches that same 100% recall.
- Fully grounded generation: verified to correctly decline out-of-corpus questions
  rather than hallucinate an answer.
- Runs end-to-end on commodity CPU hardware — no GPU required.

## Running it locally

Requires Docker and Docker Compose.

```bash
git clone https://github.com/ZaidAbuMonshar/zu-rag-assistant.git
cd zu-rag-assistant
cp .env.example .env   # set ZU_RAG_API_KEY, or leave unset to auto-generate one
docker compose up -d --build
```

The API will be available at `http://localhost:8000` (`/search`, `/ask`), with a
bundled web UI served at `/`.

Note: this repo does not include the source documents, the vector database, or model
weights (`data/` is intentionally excluded) — you'll need your own document corpus and
a build/ingestion pass to populate a working `data/` directory before this runs
end-to-end.

## License

MIT — see [LICENSE](LICENSE).
