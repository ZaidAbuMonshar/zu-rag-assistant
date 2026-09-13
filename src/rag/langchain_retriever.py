"""
LangChain retriever wrapper around retrieve_chunks() (retrieve.py).

First step of the LangChain migration. This adapts the existing,
already-verified dense retrieval path (BGE-M3 +
Chroma + query expansion) to LangChain's BaseRetriever interface -- it
does not change how retrieval works, only how its results are shaped
(RetrievedChunk dict -> LangChain Document) so the rest of a LangChain
pipeline (prompt templates, chains) can consume it like any other
retriever.

Run:
    python langchain_retriever.py "ما هي خدمات مركز الحاسوب؟"
"""

import sys
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retrieve import DEFAULT_TOP_K, get_collection, retrieve_chunks


class ChromaBGEM3Retriever(BaseRetriever):
    """Wraps retrieve_chunks(): dense BGE-M3/Chroma retrieval + query expansion."""

    # BaseRetriever is a pydantic model, so state has to be declared as
    # typed fields (not set in __init__) -- `Any` here skips pydantic
    # trying to validate/serialize the Chroma collection object itself.
    collection: Any
    top_k: int = DEFAULT_TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = retrieve_chunks(self.collection, query, top_k=self.top_k)
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

    retriever = ChromaBGEM3Retriever(collection=get_collection(), top_k=3)
    query = sys.argv[1] if len(sys.argv) > 1 else "ما هي خدمات مركز الحاسوب؟"
    docs = retriever.invoke(query)

    print(f"Query: {query}\n")
    for doc in docs:
        meta = doc.metadata
        print(f"  {meta['source_doc']} (chunk {meta['chunk_index']}) -- score {meta['score']}")
        print(f"    {doc.page_content[:80]}...")
