"""
Generation layer: retrieve top-k chunks (via retrieve.py), ground a local LLM's
answer in them, return the answer plus the sources it was built from.

Model: Qwen2.5-3B-Instruct, GGUF Q4_K_M quantization, run via llama-cpp-python.
Chosen against the same constraints as BGE-M3: CPU-only laptop, $0 budget, no
data leaving the machine. A 3B model won't reason deeply, but this task is
narrow: paraphrase/summarize retrieved Arabic institutional text, not
open-ended generation, so it doesn't need to.

Run:
    python generate.py "ما هي خدمات مركز الحاسوب؟"
"""

import sys
from pathlib import Path

from llama_cpp import Llama

from retrieve import RetrievedChunk, get_collection, retrieve_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "أنت مساعد افتراضي لمركز الحاسوب في جامعة الزرقاء. مهمتك الإجابة على أسئلة "
    "المستخدمين بالاعتماد حصريًا على المقاطع المرجعية المزودة أدناه، والمأخوذة "
    "من وثائق الجامعة الرسمية.\n"
    "قواعد صارمة:\n"
    "- إذا كانت المقاطع لا تحتوي على إجابة واضحة للسؤال، قل بوضوح أنك لا تملك "
    "معلومات كافية في الوثائق المتاحة للإجابة، ولا تخمّن.\n"
    "- لا تضف أي معلومة غير موجودة حرفيًا أو ضمنيًا في المقاطع المرجعية.\n"
    "- أجب باللغة العربية بأسلوب واضح ومختصر."
)


def load_llm() -> Llama:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Download it first, e.g.:\n"
            f"  huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF "
            f"qwen2.5-3b-instruct-q4_k_m.gguf --local-dir data/models"
        )
    return Llama(
        model_path=str(MODEL_PATH),
        # 16384, not 4096: measured directly against Qwen's real tokenizer,
        # chunks.csv's word-based token_count column undercounts by ~1.6x on
        # average -- the densest chunk (تعليمات منح درجة البكالوريوس.pdf)
        # tokenizes to 2262 tokens alone, so 3 retrieved chunks from it plus
        # the system prompt can exceed 4096 outright (this crashed a live
        # request). Qwen2.5-3B natively supports up to 32768; 16384 gives
        # comfortable headroom over the measured worst case (~7500) without
        # the extra KV-cache memory cost of going all the way to 32768.
        n_ctx=16384,
        n_threads=None,  # let llama.cpp pick based on available cores
        verbose=False,
    )


# Cap per Qwen's real tokenizer, not chunks.csv's word-based token_count
# column (measured ~1.6x undercount -- see load_llm()'s n_ctx comment). Chosen
# against the actual distribution across all 72 chunks: 900 sits just above
# the median (~910 real tokens), so most chunks pass through untouched, while
# the long tail (75th percentile 1303, max 2262 -- mostly one dense document,
# تعليمات منح درجة البكالوريوس.pdf) gets meaningfully cut. This is what
# actually drove both the earlier context-overflow crash and the 142s/request
# latency: a few oversized chunks, not the typical case.
MAX_CHUNK_PROMPT_TOKENS = 900


def _truncate_for_prompt(llm: Llama, text: str, max_tokens: int) -> str:
    tokens = llm.tokenize(text.encode("utf-8"))
    if len(tokens) <= max_tokens:
        return text
    truncated = llm.detokenize(tokens[:max_tokens]).decode("utf-8", errors="ignore")
    return truncated + " ..."


def _build_context_block(llm: Llama, chunks: list[RetrievedChunk]) -> str:
    # NOTE: truncation here only shortens what the LLM *sees* in the prompt --
    # `chunks` (returned to the caller as `sources`) keeps the original,
    # untruncated chunk_text, so the API/UI still shows the real retrieved
    # text, not a cut-off version of it.
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = _truncate_for_prompt(llm, chunk["chunk_text"], MAX_CHUNK_PROMPT_TOKENS)
        parts.append(f"[مقطع {i} — المصدر: {chunk['source_doc']}]\n{text}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    top_k: int = 3,
    llm: Llama | None = None,
    collection=None,
) -> dict:
    """
    Retrieve top_k chunks for `query`, ground an LLM answer in them.
    Returns {"query", "answer", "sources"} -- sources is the same shape
    retrieve_chunks() returns, so callers can show what the answer was based on.

    Pass `llm` and `collection` when calling this repeatedly (e.g. from an API
    server) -- both `load_llm()` and `get_collection()` load multi-GB model
    weights, so recreating them per-call (the standalone-script default below)
    is only acceptable for a one-off run, not for serving requests.
    """
    collection = collection or get_collection()
    chunks = retrieve_chunks(collection, query, top_k=top_k)

    if not chunks:
        return {
            "query": query,
            "answer": "لا تتوفر لديّ معلومات كافية في الوثائق المتاحة للإجابة على هذا السؤال.",
            "sources": [],
        }

    llm = llm or load_llm()

    # Safety net on top of the n_ctx increase above: chunks are ranked by
    # score, so if the context still doesn't fit (an even denser document, or
    # a future larger top_k), drop the *lowest*-scoring chunk and retry rather
    # than crashing the request. Degrades answer quality a little in a rare
    # edge case instead of failing outright -- important for a live demo.
    remaining = list(chunks)
    response = None
    while response is None:
        context_block = _build_context_block(llm, remaining)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"المقاطع المرجعية:\n\n{context_block}\n\nسؤال المستخدم: {query}",
            },
        ]
        try:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.2,  # low temperature: grounded/factual, not creative
                max_tokens=512,
            )
        except ValueError as e:
            if "context window" not in str(e) or len(remaining) == 1:
                raise
            remaining.pop()  # drop the lowest-scoring chunk, retry with less context

    answer = response["choices"][0]["message"]["content"].strip()

    return {"query": query, "answer": answer, "sources": remaining}


if __name__ == "__main__":
    # Windows console default codepage (cp1252) can't encode Arabic; without
    # this, print() crashes on the first Arabic character rather than just
    # displaying it oddly (which is what happens in terminals that can).
    sys.stdout.reconfigure(encoding="utf-8")

    query = sys.argv[1] if len(sys.argv) > 1 else "ما هي خدمات مركز الحاسوب؟"
    result = generate_answer(query)

    print(f"Query: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['source_doc']} (chunk {src['chunk_index']}, score {src['score']})")
