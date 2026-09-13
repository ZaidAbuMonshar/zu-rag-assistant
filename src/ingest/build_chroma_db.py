"""
Custom BGE-M3 embedding function, chunk_id sanitization, and bulk ingestion
of chunks.csv into a persistent Chroma collection.

ASSUMPTIONS (adjust if your chunks.csv columns differ):
  - data/processed/chunks.csv has columns:
      source_doc      -> original filename (may contain Arabic, spaces, etc.)
      chunk_index      -> integer, 0-based index of this chunk within its doc
      chunk_text        -> the clean text shown to the end user
      embedding_text     -> title-prefixed text actually sent to the embedder
  - FlagEmbedding + BGE-M3 are installed and working.

Run:
    python build_chroma_db.py --test        # sanity check on 3 rows only
    python build_chroma_db.py                # full ingest
"""

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import chromadb
import pandas as pd
from chromadb import Documents, Embeddings, EmbeddingFunction
from FlagEmbedding import BGEM3FlagModel

# ---------------------------------------------------------------------------
# Config -- paths anchored to the project root, not to whatever directory
# the process happens to be launched from.
#
# __file__ = this script's own location:
#   .../zu-rag-assistant/src/ingest/build_chroma_db.py
# .parent            -> .../zu-rag-assistant/src/ingest
# .parent.parent     -> .../zu-rag-assistant/src
# .parent.parent.parent -> .../zu-rag-assistant   (project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks.csv"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"  # persistent, on-disk
COLLECTION_NAME = "zu_docs_v1"


# ---------------------------------------------------------------------------
# Step: chunk_id sanitization
# ---------------------------------------------------------------------------
def slugify_filename(filename: str, max_len: int = 40) -> str:
    """
    Turn a (possibly Arabic, possibly messy) filename into a short, stable,
    ASCII-safe slug suitable for use in a Chroma/vector-DB ID or a URL.

    Arabic (and any non-Latin) text is NOT transliterated -- it's dropped,
    and a short content hash is appended instead. This keeps the ID:
      - ASCII-only (safe in URLs, logs, Flutter debug prints)
      - stable across re-runs (same filename -> same slug, always)
      - collision-resistant even when many Arabic filenames strip down to
        nothing after ASCII filtering (the hash is what actually guarantees
        uniqueness -- the ASCII remainder is just for human skimmability)

    The original, human-readable filename should still be kept in metadata
    (e.g. `source_doc`) for anything user-facing. This slug is for the ID
    only.
    """
    name = Path(filename).stem  # drop extension
    name = unicodedata.normalize("NFKD", name)

    # keep ASCII letters/digits only for the readable part
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    ascii_part = ascii_part[:max_len] if ascii_part else "doc"

    # short hash of the FULL original filename guarantees uniqueness even
    # when two Arabic filenames both strip down to "doc"
    h = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]

    return f"{ascii_part}_{h}"


def make_chunk_id(source_doc: str, chunk_index: int) -> str:
    return f"{slugify_filename(source_doc)}__chunk_{chunk_index}"


# ---------------------------------------------------------------------------
# Step: custom BGE-M3 embedding function for Chroma
# ---------------------------------------------------------------------------
class BGEM3EmbeddingFunction(EmbeddingFunction):
    """
    Wraps FlagEmbedding's BGEM3FlagModel to match Chroma's EmbeddingFunction
    interface: a callable that takes a list of strings and returns a list
    of embedding vectors.

    Subclasses chromadb.EmbeddingFunction (not just a plain callable) because
    newer Chroma versions store *which* embedding function a collection was
    built with, and check it against whatever function you pass in on later
    calls (e.g. get_or_create_collection, or a future query). That check is
    why `name()` is required here -- it's how Chroma identifies "is this the
    same embedding function as before, or did something change under me."
    This matters in practice: it's the guardrail that would catch you
    accidentally querying a BGE-M3-built collection with a different model
    later and silently getting garbage results.

    IMPORTANT: use_fp16 and any normalization choice here must match what
    was used during the original recall@3 validation, or Chroma's recall
    numbers won't be comparable to that baseline -- they'd be measuring a
    different embedding config, not just a different search backend.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def __call__(self, input: Documents) -> Embeddings:
        # dense vecs only -- BGE-M3 also supports sparse/colbert vectors,
        # but the original recall@3 work was dense-only, so stay consistent
        output = self.model.encode(
            input,
            batch_size=12,
            max_length=8192,
        )
        return output["dense_vecs"].tolist()

    @staticmethod
    def name() -> str:
        # Chroma persists this string alongside the collection so it can
        # later warn you (or error, depending on version) if you try to
        # query with a differently-named embedding function.
        return "bge-m3-dense"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def load_chunks(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"source_doc", "chunk_index", "chunk_text", "embedding_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"chunks.csv is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Adjust load_chunks() to match your actual column names."
        )
    return df


def build_collection(df: pd.DataFrame, test_mode: bool = False):
    if test_mode:
        df = df.head(3)
        print(f"[TEST MODE] Ingesting only {len(df)} rows for a sanity check.\n")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = BGEM3EmbeddingFunction()

    # get_or_create so re-running this script doesn't error on a second run;
    # delete first if you want a clean rebuild after a pipeline change
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # match your baseline's cosine sim
    )

    ids = [
        make_chunk_id(row.source_doc, row.chunk_index) for row in df.itertuples()
    ]
    embedding_texts = df["embedding_text"].tolist()
    metadatas = [
        {
            "source_doc": row.source_doc,
            "chunk_index": int(row.chunk_index),
            "chunk_text": row.chunk_text,
        }
        for row in df.itertuples()
    ]

    # sanity: ids must be unique -- catch a slug collision before Chroma
    # silently overwrites one chunk with another
    dupes = [i for i in set(ids) if ids.count(i) > 1]
    if dupes:
        raise ValueError(
            f"Duplicate chunk_ids detected, would silently overwrite chunks: "
            f"{dupes}. Check slugify_filename() collisions."
        )

    collection.add(
        ids=ids,
        documents=embedding_texts,  # what gets embedded
        metadatas=metadatas,        # what you can filter/display on later
    )

    print(f"Ingested {len(ids)} chunks into collection '{COLLECTION_NAME}'.")
    print(f"Persisted at: {CHROMA_PATH.resolve()}")

    if test_mode:
        print("\nSample IDs generated:")
        for i in ids:
            print(f"  {i}")
        print(
            "\nRun a test query to confirm retrieval works before bulk ingest:"
        )
        result = collection.query(query_texts=["مركز الحاسوب"], n_results=2)
        print(result)

    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true", help="Ingest only 3 rows as a sanity check"
    )
    args = parser.parse_args()

    df = load_chunks(CHUNKS_CSV)
    build_collection(df, test_mode=args.test)