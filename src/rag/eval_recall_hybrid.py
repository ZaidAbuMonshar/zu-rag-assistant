"""
Recall@3 v4 -- same eval set and "dedupe raw chunks to top-3 distinct source
documents" logic as eval_recall_v3.py, but querying hybrid_search() (dense +
hand-built BM25, fused via RRF) instead of Chroma dense-only.

This is the real verdict on whether hybrid search helps: the diagnostics
used while building hybrid_search.py all centered on one deliberately hard,
adversarial course-code query -- useful for understanding the mechanism,
not representative of the eval set as a whole. Comparing this script's
recall@3 against eval_recall_v3.py's 100% (17/17) is the actual before/after
that matters.
"""

import sys
from pathlib import Path

import pandas as pd

from hybrid_search import hybrid_search  # noqa: E402
from retrieve import get_collection  # noqa: E402
from bm25_search import build_index  # noqa: E402

# Same sys.path pattern as eval_recall_v3.py / bm25_search.py -- needed for
# PROJECT_ROOT, since build_chroma_db.py lives in the sibling src/ingest/
# folder Python won't see automatically.
_INGEST_DIR = Path(__file__).resolve().parent.parent / "ingest"
sys.path.insert(0, str(_INGEST_DIR))

from build_chroma_db import PROJECT_ROOT  # noqa: E402

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "eval_set.csv"
N_RESULTS_RAW = 15  # fused candidates to fetch before deduping to top-3 docs
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


def top_k_distinct_docs(
    collection, bm25_index, query: str, weights: tuple[float, float] = (0.5, 0.5), k: int = TOP_K_DOCS
) -> list[str]:
    """Same dedup-to-distinct-documents rule as eval_recall_v3.py's function
    of the same name -- recall@3 is a document-level metric, so one
    document's chunks can't crowd out a different correct document."""
    results = hybrid_search(collection, bm25_index, query, top_k=N_RESULTS_RAW, weights=weights)

    seen_docs = []
    for chunk in results:
        doc = chunk["source_doc"]
        if doc not in seen_docs:
            seen_docs.append(doc)
        if len(seen_docs) == k:
            break
    return seen_docs


def run_eval(weights: tuple[float, float] = (0.5, 0.5)):
    collection = get_collection()
    bm25_index = build_index()

    eval_df = load_eval_set(EVAL_CSV)

    results = []
    for row in eval_df.itertuples():
        retrieved = top_k_distinct_docs(collection, bm25_index, row.question_ar, weights=weights)
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

    out_path = Path("data/eval/recall_results_hybrid_v4.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")

    return results_df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # e.g. `python eval_recall_hybrid.py 0.9 0.1` to test a dense-weighted RRF
    if len(sys.argv) >= 3:
        w = (float(sys.argv[1]), float(sys.argv[2]))
    else:
        w = (0.5, 0.5)
    print(f"weights (dense, bm25) = {w}\n")
    run_eval(weights=w)
