"""
Re-run recall@3 against Chroma instead of the hand-rolled cosine similarity
from eval_recall_v2.py.

This is the regression check. Same eval set (arabic_qa_eval_v1.csv), same
"dedupe to top-3 distinct source documents" logic from v2 -- the only thing
that changed is the search backend (Chroma's HNSW approximate NN instead of
exact cosine over all vectors).

Expect ~82.35% (14/17), matching the earlier cosine-similarity baseline. A
drop of a couple points could just be approximate-NN noise at this tiny
scale (72 vectors) --
worth a manual look at which question(s) flipped, not necessarily a bug.
A large drop (multiple questions) means something is actually wrong
(e.g. embedding config mismatch between ingestion and query time).
"""

import sys
from pathlib import Path

import chromadb
import pandas as pd

# eval_recall_v3.py lives in src/rag/, build_chroma_db.py lives in src/ingest/
# -- siblings, not the same folder, so Python won't find it automatically.
# Explicitly add src/ingest/ to the import search path before importing from
# it. This is a pragmatic fix for a 2-file project; if this repo grows into
# many cross-importing modules, that's the point to properly package `src/`
# with __init__.py files and absolute imports instead of patching sys.path
# per-script.
_INGEST_DIR = Path(__file__).resolve().parent.parent / "ingest"
sys.path.insert(0, str(_INGEST_DIR))

from build_chroma_db import (
    BGEM3EmbeddingFunction,
    CHROMA_PATH,
    COLLECTION_NAME,
    PROJECT_ROOT,
)
from query_expansion import expand_query

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "eval_set.csv"
N_RESULTS_RAW = 10  # over-fetch raw chunks, then dedupe down to top-3 *docs*
TOP_K_DOCS = 3


def load_eval_set(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"id", "question_ar", "target_doc", "question_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"eval CSV missing columns: {missing}. Found: {list(df.columns)}"
        )
    return df


def top_k_distinct_docs(collection, query: str, k: int = TOP_K_DOCS) -> list[str]:
    """
    Query returns raw chunks ranked by similarity. Since a document can now
    have multiple chunks, dedupe by source_doc while preserving rank order,
    so one document's chunks can't crowd out a different correct document
    from the top-k -- same rule established in the earlier v2 eval.
    """
    result = collection.query(query_texts=[expand_query(query)], n_results=N_RESULTS_RAW)
    metadatas = result["metadatas"][0]  # list of dicts, rank-ordered

    seen_docs = []
    for meta in metadatas:
        doc = meta["source_doc"]
        if doc not in seen_docs:
            seen_docs.append(doc)
        if len(seen_docs) == k:
            break
    return seen_docs


def run_eval():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = BGEM3EmbeddingFunction()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    eval_df = load_eval_set(EVAL_CSV)

    results = []
    for row in eval_df.itertuples():
        retrieved = top_k_distinct_docs(collection, row.question_ar)
        hit = row.target_doc in retrieved
        results.append(
            {
                "id": row.id,
                "question": row.question_ar,
                "question_type": row.question_type,
                "target_doc": row.target_doc,
                "retrieved_top3": retrieved,
                "hit": hit,
            }
        )

    results_df = pd.DataFrame(results)

    overall_recall = results_df["hit"].mean()
    print(f"Overall recall@3: {overall_recall:.2%} "
          f"({results_df['hit'].sum()}/{len(results_df)})\n")

    print("By question_type:")
    print(results_df.groupby("question_type")["hit"].mean().apply(lambda x: f"{x:.1%}"))

    print("\nMisses:")
    misses = results_df[~results_df["hit"]]
    if misses.empty:
        print("  (none)")
    else:
        for row in misses.itertuples():
            print(f"  [{row.id} / {row.question_type}] {row.question}")
            print(f"    expected: {row.target_doc}")
            print(f"    got:      {row.retrieved_top3}")

    out_path = Path("data/eval/recall_results_chroma_v3.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")

    return results_df


if __name__ == "__main__":
    run_eval()