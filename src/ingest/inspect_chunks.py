"""
Quick diagnostic: shows the smallest and largest chunks from chunks.csv,
so you can see exactly what's in them before trusting the eval results.

Run from the project root:
    python src/ingest/inspect_chunks.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks.csv"

df = pd.read_csv(CHUNKS_CSV)

print("=" * 60)
print("SMALLEST CHUNKS (likely junk fragments)")
print("=" * 60)
smallest = df.nsmallest(5, "token_count")
for _, row in smallest.iterrows():
    print(f"\n[{row['chunk_id']}] {row['token_count']} tokens, from {row['source_doc']}")
    print(f"  text: {row['chunk_text']!r}")

print("\n" + "=" * 60)
print("LARGEST CHUNKS (paragraph-splitting likely failed)")
print("=" * 60)
largest = df.nlargest(3, "token_count")
for _, row in largest.iterrows():
    print(f"\n[{row['chunk_id']}] {row['token_count']} tokens, from {row['source_doc']}")
    print(f"  first 300 chars: {row['chunk_text'][:300]!r}")
    print(f"  ...")
    print(f"  last 300 chars: {row['chunk_text'][-300:]!r}")