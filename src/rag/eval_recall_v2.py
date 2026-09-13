"""
Recall@3 evaluation for BGE-M3 embeddings, using real chunks (data/processed/chunks.csv)
instead of whole-document embeddings.

Key difference from the v1 script: since a document can now have multiple chunks,
"top-3" means top-3 DISTINCT source documents, ranked by their best-scoring chunk —
not top-3 raw chunks (which could all come from the same document and crowd out
a correct different one).

Run from the project root, AFTER running chunk_documents.py:
    python src/rag/eval_recall_v2.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks.csv"
EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "eval_set.csv"
K = 3


def top_k_distinct_docs(sims_row: np.ndarray, source_docs: list[str], k: int) -> list[str]:
    """Rank all chunks by similarity, walk down the ranked list, and keep the
    first k DISTINCT source documents encountered (skip repeat docs)."""
    ranked_idx = np.argsort(-sims_row)
    seen_docs = []
    for idx in ranked_idx:
        doc = source_docs[idx]
        if doc not in seen_docs:
            seen_docs.append(doc)
        if len(seen_docs) == k:
            break
    return seen_docs


def main():
    print(f"Loading chunks from {CHUNKS_CSV} ...")
    chunks_df = pd.read_csv(CHUNKS_CSV)
    print(f"Loaded {len(chunks_df)} chunks from {chunks_df['source_doc'].nunique()} documents.\n")

    print("Loading BGE-M3 ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Embedding chunks ...")
    # Embed embedding_text (title + chunk), not raw chunk_text — the title
    # prefix is a deliberate fix for sibling-document confusion, see recall@3
    # v2 results. chunk_text stays available for display purposes elsewhere.
    chunk_texts = chunks_df["embedding_text"].tolist()
    source_docs = chunks_df["source_doc"].tolist()
    chunk_embeddings = model.encode(chunk_texts, return_dense=True)["dense_vecs"]

    print(f"Loading eval set from {EVAL_CSV} ...")
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"Loaded {len(eval_df)} eval questions.\n")

    questions = eval_df["question_ar"].tolist()
    print("Embedding questions ...")
    question_embeddings = model.encode(questions, return_dense=True)["dense_vecs"]

    sims = question_embeddings @ chunk_embeddings.T  # (n_questions, n_chunks)

    all_source_docs = set(source_docs)
    results = []
    for i, row in eval_df.iterrows():
        target = row["target_doc"]
        if target not in all_source_docs:
            results.append({
                "id": row["id"], "question_type": row["question_type"],
                "target_doc": target, "hit": None, "top3_docs": [],
            })
            continue

        top3_docs = top_k_distinct_docs(sims[i], source_docs, K)
        hit = target in top3_docs
        results.append({
            "id": row["id"], "question_type": row["question_type"],
            "target_doc": target, "hit": hit, "top3_docs": top3_docs,
        })

    results_df = pd.DataFrame(results)

    missing = results_df[results_df["hit"].isna()]
    if not missing.empty:
        print("WARNING: these target_doc values were not found among chunked documents:")
        for _, r in missing.iterrows():
            print(f"  id={r['id']}: '{r['target_doc']}'")
        print()

    scored = results_df[results_df["hit"].notna()]
    overall_recall = scored["hit"].mean()
    print(f"Overall recall@3: {overall_recall:.2%} ({scored['hit'].sum()}/{len(scored)})\n")

    print("Recall@3 by question type:")
    print(scored.groupby("question_type")["hit"].agg(["mean", "count"]).rename(
        columns={"mean": "recall@3", "count": "n"}
    ).to_string())
    print()

    print("Per-question detail (failures first):")
    for _, r in scored.sort_values("hit").iterrows():
        status = "HIT " if r["hit"] else "MISS"
        print(f"[{status}] id={r['id']} ({r['question_type']}) target={r['target_doc']}")
        if not r["hit"]:
            print(f"         retrieved: {r['top3_docs']}")

    out_path = PROJECT_ROOT / "data" / "eval" / "recall_results_bge_m3_chunked.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()