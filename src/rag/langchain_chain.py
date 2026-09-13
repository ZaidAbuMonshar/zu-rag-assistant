"""
LCEL chain wiring the wrapped pieces together: a retriever (langchain_retriever.py
or langchain_hybrid_retriever.py), a prompt built from generate.py's SYSTEM_PROMPT,
and the Qwen ChatModel (langchain_chat_model.py).

Deliberately NOT one single `retriever | prompt | model` chain -- see the
module-level explanation in the conversation this was built in. Short
version: generate_answer()'s context-overflow retry (drop the lowest-
scoring chunk, rebuild, retry) needs the ranked list of retrieved
Documents across a retry, which a single linear chain has no place to
hold once retrieval hands off to formatting. So retrieval happens once,
outside the chain; the chain itself (format -> prompt -> model -> parse)
is what gets retried.

Run:
    python langchain_chain.py "ما هي خدمات مركز الحاسوب؟"
"""

import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from generate import MAX_CHUNK_PROMPT_TOKENS, SYSTEM_PROMPT, _truncate_for_prompt, load_llm
from langchain_chat_model import ChatQwenLlamaCpp
from langchain_retriever import ChromaBGEM3Retriever
from retrieve import get_collection

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "المقاطع المرجعية:\n\n{context}\n\nسؤال المستخدم: {question}"),
    ]
)


def _format_docs(llm, docs: list[Document]) -> str:
    # Same truncation generate.py's _build_context_block applies to
    # RetrievedChunk dicts -- reused as-is here (imported, not
    # reimplemented) against LangChain Document objects instead.
    parts = []
    for i, doc in enumerate(docs, start=1):
        text = _truncate_for_prompt(llm, doc.page_content, MAX_CHUNK_PROMPT_TOKENS)
        parts.append(f"[مقطع {i} — المصدر: {doc.metadata['source_doc']}]\n{text}")
    return "\n\n".join(parts)


def build_chain(chat_model: ChatQwenLlamaCpp):
    """
    The LCEL chain proper: takes {"docs": [...], "question": str}, returns
    a plain answer string. Does not retrieve -- that happens once, outside,
    in generate_answer_lcel() below, so the ranked doc list survives a retry.
    """
    return (
        RunnableLambda(
            lambda x: {
                "context": _format_docs(chat_model.llm, x["docs"]),
                "question": x["question"],
            }
        )
        | PROMPT
        | chat_model
        | StrOutputParser()
    )


def generate_answer_lcel(
    query: str,
    retriever: BaseRetriever,
    chat_model: ChatQwenLlamaCpp,
) -> dict:
    """
    LCEL equivalent of generate.py's generate_answer(): retrieve once,
    then invoke the chain, dropping the lowest-scoring doc and retrying
    on context overflow -- same safety net, same trigger condition.
    Returns {"query", "answer", "sources"} in the same shape as the
    original, with `sources` as LangChain Documents instead of
    RetrievedChunk dicts.
    """
    docs = retriever.invoke(query)

    if not docs:
        return {
            "query": query,
            "answer": "لا تتوفر لديّ معلومات كافية في الوثائق المتاحة للإجابة على هذا السؤال.",
            "sources": [],
        }

    chain = build_chain(chat_model)

    remaining = list(docs)
    answer = None
    while answer is None:
        try:
            answer = chain.invoke({"docs": remaining, "question": query})
        except ValueError as e:
            if "context window" not in str(e) or len(remaining) == 1:
                raise
            remaining.pop()  # drop the lowest-scoring doc, retry with less context

    return {"query": query, "answer": answer, "sources": remaining}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    retriever = ChromaBGEM3Retriever(collection=get_collection(), top_k=3)
    chat_model = ChatQwenLlamaCpp(llm=load_llm())

    query = sys.argv[1] if len(sys.argv) > 1 else "ما هي خدمات مركز الحاسوب؟"
    result = generate_answer_lcel(query, retriever, chat_model)

    print(f"Query: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for doc in result["sources"]:
        meta = doc.metadata
        print(f"  - {meta['source_doc']} (chunk {meta['chunk_index']}, score {meta['score']})")
