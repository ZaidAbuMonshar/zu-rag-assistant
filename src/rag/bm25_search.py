"""
Hand-built BM25 (Okapi) keyword search over the same 72 chunks Chroma
indexes — a learning exercise, not a production need: the corpus is tiny,
so a library (`rank_bm25`) would work fine, but implementing the formula
by hand is the point.

Design decisions, made explicit rather than left implicit:

- **Reads `data/processed/chunks.csv` directly**, the same source of truth
  `build_chroma_db.py` ingests from — this is a second, independent,
  in-memory index, not something layered on top of Chroma (Chroma has no
  sparse/BM25 index built in).
- **Indexes `embedding_text`, not `chunk_text`.** The dense side (BGE-M3)
  embeds `embedding_text` (title-prefixed) — indexing the same field here
  means a later hybrid-search comparison tests *retrieval mechanism*
  (semantic vs. lexical) on identical input text, not "which method got
  the better text."
- **`chunk_id` uses the same `make_chunk_id()` as `build_chroma_db.py`.**
  A BM25 result and a dense (Chroma) result for the same physical chunk
  must share one ID, or a future RRF fusion step can't tell they're the
  same document.
- **Tokenization is deliberately simple** (whitespace/punctuation split,
  ASCII-lowercased) — no Arabic-specific normalization (definite article
  "ال", diacritics, hamza forms) yet. Get the baseline working, measure
  whether that actually hurts real queries, then decide if it's worth
  adding — not before.

Run:
    python bm25_search.py "ما هو المتطلب السابق للمادة 1503473"
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

import pandas as pd

# bm25_search.py lives in src/rag/, build_chroma_db.py lives in src/ingest/
# -- same sys.path pattern as retrieve.py / eval_recall_v3.py in this folder.
_INGEST_DIR = Path(__file__).resolve().parent.parent / "ingest"
sys.path.insert(0, str(_INGEST_DIR))

from build_chroma_db import CHUNKS_CSV, make_chunk_id  # noqa: E402
from query_expansion import expand_query  # noqa: E402

# Classic Okapi BM25 defaults (same values the `rank_bm25` library ships
# with) -- tuning knobs shaped by IR research and practice, not something
# derived from this corpus. k1 controls term-frequency saturation (how
# fast repeated mentions of a term stop adding score); b controls how much
# document length is penalized relative to the corpus average.
K1 = 1.5
B = 0.75

# Two alternatives, not one \w+: a plain \w+ treats a digit run and an
# adjacent letter run as ONE token whenever they're not separated by
# whitespace -- which happens for real in this corpus's PDF-extracted
# course tables (e.g. "1503473فحص", the course code glued directly to the
# course name with no space in the original PDF). That fused token can
# never match a query's bare "1503473" token, silently breaking exact-code
# lookups -- found by testing against a real query, not spotted by
# inspection. Matching \d+ first splits the digits off as their own token
# before the letter-run alternative can swallow them.
_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+", re.UNICODE)

# A small, hand-picked set of high-frequency Arabic function words --
# pronouns, prepositions, particles, forms of "to be" -- that carry almost
# no discriminating signal for retrieval. Their IDF isn't literally zero
# (measured: "قسم" scores 1.13, not ~0), so left in -- several of them
# summed across a query can rival one genuinely rare term's
# contribution. This is a FIXED list, not learned from this corpus -- at
# 72 chunks, trying to auto-derive "too common" from this corpus's own
# document frequencies would be tuning on far too little data (same
# reasoning as query_expansion.py's hand-curated phrase table).
ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "هو", "هي", "هم", "أنت", "أنا", "نحن",
    "كيف", "ما", "ماذا", "هل", "لا", "لم", "لن", "إن", "أن", "كان",
    "يكون", "و", "أو", "ثم", "قد", "لقد", "كل", "بعض", "غير", "بين",
    "عند", "حتى", "إذا", "لكن", "أيضا", "فقط", "جدا", "أي", "لماذا",
    "متى", "أين", "هناك", "هنا",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Split into digit runs and letter runs separately (see _TOKEN_RE);
    lowercase (affects ASCII/course-code digits and any Latin text, a no-op
    on Arabic script, which has no case). `remove_stopwords` defaults True
    -- kept as a toggle (not baked in unconditionally) so it's easy to
    compare scoring with it on vs. off, same corpus, same query."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(str(text))]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ARABIC_STOPWORDS]
    return tokens


class BM25Chunk(TypedDict):
    chunk_id: str
    chunk_text: str
    source_doc: str
    chunk_index: int


class BM25Index:
    """
    Two-pass design: build() computes corpus-wide statistics once (average
    document length, per-term document frequency, and an inverted index --
    term -> {doc_idx: count_in_that_doc}); score() then only ever touches
    documents that share at least one term with the query, via the inverted
    index's postings, instead of rescanning every document per query. This
    is the same architectural shape real search engines use (Lucene,
    Elasticsearch) -- at 72 chunks it's not a performance necessity, but
    it's the honest way to implement BM25, not a shortcut specific to this
    corpus's tiny size.
    """

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.chunks: list[BM25Chunk] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.doc_freq: Counter = Counter()  # term -> number of docs containing it
        self.inverted_index: dict[str, dict[int, int]] = {}  # term -> {doc_idx: tf}
        self.n_docs: int = 0

    def build(self, df: pd.DataFrame) -> None:
        inverted_index: dict[str, dict[int, int]] = {}

        for doc_idx, row in enumerate(df.itertuples()):
            tokens = tokenize(row.embedding_text)
            self.chunks.append(
                BM25Chunk(
                    chunk_id=make_chunk_id(row.source_doc, row.chunk_index),
                    chunk_text=row.chunk_text,
                    source_doc=row.source_doc,
                    chunk_index=int(row.chunk_index),
                )
            )
            self.doc_lengths.append(len(tokens))

            counts = Counter(tokens)
            for term, count in counts.items():
                inverted_index.setdefault(term, {})[doc_idx] = count
                self.doc_freq[term] += 1  # presence per doc, not raw count

        self.inverted_index = inverted_index
        self.n_docs = len(self.chunks)
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        # Smoothed variant (the "+ 1" inside the log): the classic
        # Robertson/Sparck-Jones IDF can go *negative* for a term that
        # appears in more than half the corpus. Adding 1 guarantees IDF
        # >= 0 always -- a term that's in every document contributes
        # nothing to the score, but never actively subtracts from it.
        n_qi = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1)

    def score(self, query: str, top_k: int = 3) -> list[dict]:
        # expand_query() was only wired into the dense side (retrieve.py),
        # not here -- so a query expansion fix for the dense retriever could
        # still get outvoted in RRF fusion by BM25 seeing the raw,
        # unexpanded query. Applying it here too keeps both retrievers
        # looking at the same query-side signal.
        query_terms = tokenize(expand_query(query))
        scores: Counter = Counter()  # doc_idx -> accumulated score

        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue  # term never seen in the corpus -- contributes 0
            idf = self._idf(term)
            for doc_idx, f in postings.items():
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avgdl
                )
                scores[doc_idx] += idf * (f * (self.k1 + 1)) / denom

        results = []
        for doc_idx, s in scores.most_common(top_k):
            chunk = self.chunks[doc_idx]
            results.append({**chunk, "score": round(s, 4)})
        return results


def build_index() -> BM25Index:
    df = pd.read_csv(CHUNKS_CSV)
    index = BM25Index()
    index.build(df)
    return index


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    index = build_index()
    query = sys.argv[1] if len(sys.argv) > 1 else "ما هو المتطلب السابق للمادة 1503473"
    results = index.score(query, top_k=5)

    print(f"Query: {query}\n")
    if not results:
        print("  (no matches -- none of the query's terms appear in the corpus)")
    for r in results:
        print(f"  {r['source_doc']} (chunk {r['chunk_index']}) -- score {r['score']}")
