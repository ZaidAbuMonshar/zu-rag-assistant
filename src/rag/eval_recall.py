"""
Recall@3 evaluation for BGE-M3 embeddings on ZU RAG assistant docs.

What this does:
1. Loads all PDFs from data/raw/
2. Extracts text, treats each PDF as a single chunk (matches the
   "1-2 chunks per short doc" plan)
3. Embeds all doc-chunks with BGE-M3
4. Loads the eval set (data/eval/arabic_qa_eval_v1.csv)
5. Embeds each question, retrieves top-3 nearest doc-chunks
6. Checks if the correct target_doc is in the top-3
7. Reports recall@3 overall, and broken down by question_type

Run from the project root:
    python src/rag/eval_recall.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "eval_set.csv"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_documents() -> dict[str, str]:
    """Returns {filename: extracted_text} for every PDF in data/raw/."""
    docs = {}
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR} — check the path.")
    for pdf_path in pdf_files:
        text = extract_text(pdf_path)
        if not text:
            print(f"  WARNING: no extractable text in {pdf_path.name} — check if it's a scanned image PDF")
        docs[pdf_path.name] = text
    return docs


def main():
    print(f"Loading PDFs from {RAW_DIR} ...")
    docs = load_documents()
    filenames = list(docs.keys())
    texts = list(docs.values())
    print(f"Loaded {len(filenames)} documents.\n")

    print("Loading BGE-M3 ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Embedding documents ...")
    doc_embeddings = model.encode(texts, return_dense=True)["dense_vecs"]

    print(f"Loading eval set from {EVAL_CSV} ...")
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"Loaded {len(eval_df)} eval questions.\n")

    questions = eval_df["question_ar"].tolist()
    print("Embedding questions ...")
    question_embeddings = model.encode(questions, return_dense=True)["dense_vecs"]

    # Cosine similarity (embeddings are already normalized by BGE-M3)
    sims = question_embeddings @ doc_embeddings.T  # shape: (n_questions, n_docs)

    results = []
    for i, row in eval_df.iterrows():
        target = row["target_doc"]
        if target not in filenames:
            results.append({
                "id": row["id"],
                "question_type": row["question_type"],
                "target_doc": target,
                "hit": None,  # target doc missing from data/raw/ entirely
                "top3": [],
            })
            continue

        top3_idx = np.argsort(-sims[i])[:3]
        top3_docs = [filenames[j] for j in top3_idx]
        hit = target in top3_docs

        results.append({
            "id": row["id"],
            "question_type": row["question_type"],
            "target_doc": target,
            "hit": hit,
            "top3": top3_docs,
        })

    results_df = pd.DataFrame(results)

    # Flag any target_doc that doesn't exist in data/raw/ — a data problem, not a model problem
    missing = results_df[results_df["hit"].isna()]
    if not missing.empty:
        print("WARNING: these target_doc values were not found in data/raw/ — fix the CSV, not the model:")
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
    detail = scored.sort_values("hit")  # False (failures) sort before True
    for _, r in detail.iterrows():
        status = "HIT " if r["hit"] else "MISS"
        print(f"[{status}] id={r['id']} ({r['question_type']}) target={r['target_doc']}")
        if not r["hit"]:
            print(f"         retrieved: {r['top3']}")

    results_df.to_csv(PROJECT_ROOT / "data" / "eval" / "recall_results_bge_m3.csv", index=False)
    print(f"\nFull results saved to data/eval/recall_results_bge_m3.csv")


if __name__ == "__main__":
    main()