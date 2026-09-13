"""
Tests three different PDF text extraction libraries on a sample of flagged
documents, to see if switching libraries fixes the presentation-form/reversed
Arabic issue systemically — before committing to any per-document manual fix.

Requires: pip install pdfplumber pymupdf

Run from the project root:
    python src/ingest/test_extraction_methods.py
"""

from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PRESENTATION_FORM_PATTERN = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")

# Test on a few known-flagged documents — short ones, easy to eyeball
SAMPLE_FILES = [
    "رؤية جامعة الزرقاء ورسالتها.pdf",
    "الهيئة التدريسية لقسم علم الحاسوب.pdf",
    "التخصصات و الرسوم الجامعية.pdf",
]


def presentation_form_ratio(text: str) -> float:
    if not text:
        return 1.0  # empty extraction counts as fully broken
    return len(PRESENTATION_FORM_PATTERN.findall(text)) / len(text)


def extract_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdfplumber(path: Path) -> str:
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_pymupdf(path: Path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def main():
    methods = {
        "pypdf": extract_pypdf,
        "pdfplumber": extract_pdfplumber,
        "pymupdf": extract_pymupdf,
    }

    for filename in SAMPLE_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            print(f"SKIP (not found): {filename}")
            continue

        print("=" * 70)
        print(f"FILE: {filename}")
        print("=" * 70)

        for method_name, extract_fn in methods.items():
            try:
                text = extract_fn(path)
                ratio = presentation_form_ratio(text)
                preview = text[:150].replace("\n", " ")
                status = "CLEAN" if ratio < 0.05 else "CORRUPTED"
                print(f"\n  [{method_name}] presentation-form ratio: {ratio:.4f}  -> {status}")
                print(f"  preview: {preview!r}")
            except Exception as e:
                print(f"\n  [{method_name}] ERROR: {e}")
        print()


if __name__ == "__main__":
    main()