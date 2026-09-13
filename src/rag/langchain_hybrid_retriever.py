"""
LangChain retriever wrapper around hybrid_search() (hybrid_search.py).

Same pattern as langchain_retriever.py's ChromaBGEM3Retriever, applied to
the dense+BM25 RRF fusion instead of dense alone. Nothing about the
underlying retrieval (BGE-M3 + Chroma + query expansion, fused with the
hand-built BM25 index via weighted RRF) changes -- this only adapts the
RetrievedChunk output to LangChain's Document shape.

Run:
    python langchain_hybrid_retriever.py "ما هو المتطلب السابق للمادة 1503473"
"""

import sys
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from bm25_search import build_index
from hybrid_search import CANDIDATE_POOL_SIZE, DEFAULT_TOP_K, hybrid_search
from retrieve import get_collection


class HybridRRFRetriever(BaseRetriever):
    """Wraps hybrid_search(): weighted RRF fusion of dense (BGE-M3) + BM25."""

    # Same reasoning as ChromaBGEM3Retriever's `collection` field: pydantic
    # needs a typed field to hold state, and `Any` skips validating these
    # plain Python objects (a Chroma collection, a BM25Index) as models.
    collection: Any
    bm25_index: Any
    top_k: int = DEFAULT_TOP_K
    weights: tuple[float, float] = (0.5, 0.5)
    candidate_pool_size: int = CANDIDATE_POOL_SIZE

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = hybrid_search(
            self.collection,
            self.bm25_index,
            query,
            top_k=self.top_k,
            weights=self.weights,
            candidate_pool_size=self.candidate_pool_size,
        )
        return [
            Document(
                page_content=chunk["chunk_text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "source_doc": chunk["source_doc"],
                    "chunk_index": chunk["chunk_index"],
                    "score": chunk["score"],
                },
            )
            for chunk in chunks
        ]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    retriever = HybridRRFRetriever(
        collection=get_collection(),
        bm25_index=build_index(),
        top_k=3,
    )
    query = sys.argv[1] if len(sys.argv) > 1 else "ما هو المتطلب السابق للمادة 1503473"
    docs = retriever.invoke(query)

    print(f"Query: {query}\n")
    for doc in docs:
        meta = doc.metadata
        print(f"  {meta['source_doc']} (chunk {meta['chunk_index']}) -- RRF score {meta['score']}")
        print(f"    {doc.page_content[:80]}...")
