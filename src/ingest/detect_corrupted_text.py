"""
Scans every chunk for signs of broken Arabic PDF text extraction:
1. Arabic PRESENTATION FORM characters (U+FB50-FDFF, U+FE70-FEFF) — these are
   glyph-rendering variants, not standard Arabic letters. Their presence means
   the extraction produced visual/presentation encoding instead of logical text.
2. Suspicious space density — justified-text extraction sometimes injects
   spaces mid-word, inflating the ratio of spaces to total characters.
3. Leaked non-content artifacts (browser chrome, URLs) — sign of a
   print-to-PDF export gone wrong.

Run from the project root, AFTER chunk_documents.py:
    python src/ingest/detect_corrupted_text.py
"""

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks.csv"

PRESENTATION_FORM_PATTERN = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
ARTIFACT_PATTERNS = [
    re.compile(r"about:blank"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(AM|PM)"),  # timestamps
    re.compile(r"DIV Contents"),
]


def analyze_chunk(text: str) -> dict:
    presentation_form_chars = len(PRESENTATION_FORM_PATTERN.findall(text))
    total_chars = max(len(text), 1)
    presentation_form_ratio = presentation_form_chars / total_chars

    space_count = text.count(" ")
    space_ratio = space_count / total_chars

    artifacts_found = [p.pattern for p in ARTIFACT_PATTERNS if p.search(text)]

    return {
        "presentation_form_ratio": round(presentation_form_ratio, 4),
        "space_ratio": round(space_ratio, 4),
        "has_artifacts": bool(artifacts_found),
        "artifact_types": artifacts_found,
    }


def main():
    df = pd.read_csv(CHUNKS_CSV)
    analysis = df["chunk_text"].apply(analyze_chunk).apply(pd.Series)
    df = pd.concat([df, analysis], axis=1)

    # Flag thresholds — tune if these miss/over-trigger on your actual data
    df["likely_corrupted"] = (
        (df["presentation_form_ratio"] > 0.05) |  # >5% presentation-form chars = suspicious
        (df["space_ratio"] > 0.25) |               # unusually space-heavy
        (df["has_artifacts"])
    )

    flagged = df[df["likely_corrupted"]]
    flagged_docs = flagged["source_doc"].unique()

    print(f"Scanned {len(df)} chunks from {df['source_doc'].nunique()} documents.\n")
    print(f"Flagged {len(flagged)} chunk(s) as likely corrupted, "
          f"spanning {len(flagged_docs)} document(s):\n")

    for doc in sorted(flagged_docs):
        doc_chunks = flagged[flagged["source_doc"] == doc]
        reasons = []
        if (doc_chunks["presentation_form_ratio"] > 0.05).any():
            reasons.append("presentation-form Arabic")
        if (doc_chunks["space_ratio"] > 0.25).any():
            reasons.append("excessive spacing")
        if doc_chunks["has_artifacts"].any():
            reasons.append("leaked artifacts")
        print(f"  - {doc}  [{', '.join(reasons)}]")

    out_path = PROJECT_ROOT / "data" / "processed" / "corruption_audit.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nFull audit saved to {out_path}")
    print("\nReview the flagged documents manually before deciding on a fix —")
    print("thresholds above are a starting point, not ground truth.")


if __name__ == "__main__":
    main()