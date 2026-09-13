"""
Chunks all PDFs in data/raw/ into paragraph-respecting chunks sized for BGE-M3,
and saves the result to data/processed/chunks.csv.

Design choices:
- Target chunk size: 500-800 tokens (larger than typical RAG defaults, since
  source docs are short 1-2 pagers and we want each chunk to carry full context)
- Chunks never split mid-paragraph — paragraph boundaries are the atomic unit
- Token counts use BGE-M3's own tokenizer, not a word-count approximation,
  since Arabic subword tokenization doesn't map 1:1 to word count
- Each chunk keeps source metadata (filename) attached

Run from the project root:
    python src/ingest/chunk_documents.py
"""

from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF — pypdf and pdfplumber both corrupt this corpus's Arabic
# encoding (presentation-form glyphs, reversed word order); PyMuPDF extracts
# cleanly. See src/ingest/test_extraction_methods.py for the comparison.
from transformers import AutoTokenizer
import re

# Some source PDFs were browser print-to-PDF exports and leak page chrome
# into the text layer (timestamps, "about:blank", "DIV Contents"). Strip
# these before they pollute chunk content.
ARTIFACT_PATTERNS = [
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(AM|PM)"),
    re.compile(r"about:blank"),
    re.compile(r"DIV Contents"),
]


def clean_artifacts(text: str) -> str:
    for pattern in ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIN_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 800

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_paragraphs(pdf_path: Path) -> list[str]:
    """Extracts paragraph-level text blocks directly from PDF layout structure,
    instead of guessing paragraph boundaries from newline characters — PyMuPDF's
    clean text output doesn't reliably preserve blank-line paragraph separators,
    so block-level extraction (using the PDF's actual text-box layout) is a more
    reliable signal than whitespace heuristics."""
    doc = fitz.open(str(pdf_path))
    paragraphs = []
    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        # block_no preserves PyMuPDF's detected reading order
        for block in sorted(blocks, key=lambda b: b[5]):
            text = clean_artifacts(block[4].strip())
            if text:
                paragraphs.append(text)
    doc.close()
    return paragraphs


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedily merge paragraphs into chunks between MIN and MAX tokens,
    never splitting a paragraph in half."""
    chunks = []
    current_chunk_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Single paragraph already exceeds max — keep it as its own chunk
        # rather than force-splitting mid-sentence (acceptable rare case for
        # short institutional docs; flag if this fires often)
        if para_tokens > MAX_CHUNK_TOKENS:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts, current_tokens = [], 0
            chunks.append(para)
            continue

        if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current_tokens >= MIN_CHUNK_TOKENS:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts, current_tokens = [], 0

        current_chunk_parts.append(para)
        current_tokens += para_tokens

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}")

    print(f"Found {len(pdf_files)} PDFs. Chunking...\n")

    rows = []
    single_chunk_count = 0
    multi_chunk_count = 0
    oversized_paragraph_count = 0

    for pdf_path in pdf_files:
        paragraphs = extract_paragraphs(pdf_path)
        if not paragraphs:
            print(f"  WARNING: no extractable text in {pdf_path.name} — check if it's a scanned image PDF")
            continue

        chunks = chunk_paragraphs(paragraphs)

        if len(chunks) == 1:
            single_chunk_count += 1
        else:
            multi_chunk_count += 1

        for i, chunk_text in enumerate(chunks):
            tok_count = count_tokens(chunk_text)
            if tok_count > MAX_CHUNK_TOKENS:
                oversized_paragraph_count += 1
            # Document title (filename without extension) prepended for embedding
            # only — not for display. Short institutional docs (vision/mission,
            # chairman's word, dept overview, etc.) share heavy boilerplate
            # phrasing; the title is the clearest disambiguating signal we have,
            # and it currently sits unused in metadata. See recall@3 v2 results:
            # sibling department docs were being confused for one another.
            title = pdf_path.stem
            embedding_text = f"{title}\n\n{chunk_text}"
            rows.append({
                "chunk_id": f"{pdf_path.stem}__{i}",
                "source_doc": pdf_path.name,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "embedding_text": embedding_text,
                "token_count": tok_count,
            })

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "chunks.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Done. {len(df)} chunks from {len(pdf_files)} documents.")
    print(f"  Docs that stayed as 1 chunk: {single_chunk_count}")
    print(f"  Docs split into 2+ chunks:   {multi_chunk_count}")
    if oversized_paragraph_count:
        print(f"  WARNING: {oversized_paragraph_count} chunk(s) exceed {MAX_CHUNK_TOKENS} tokens "
              f"(single paragraph too long to split) — review these manually")
    print(f"\nToken count stats:\n{df['token_count'].describe().to_string()}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()