"""
Hybrid search: merge dense (BGE-M3/Chroma, via retrieve.py) and sparse
(hand-built BM25, via bm25_search.py) retrieval using weighted Reciprocal
Rank Fusion -- the same fusion mechanism LangChain's EnsembleRetriever uses
internally. Built by hand first, deliberately, before introducing
LangChain's retriever abstraction over these same two pieces.

Why RRF instead of blending raw scores directly: dense cosine similarity
and BM25's score live on completely different, incomparable numeric scales
-- blending them with a weight like `0.5 * dense + 0.5 * bm25` would require
normalizing
both onto a comparable range first, and getting that normalization subtly
wrong is a common real bug. RRF sidesteps the problem by fusing on *rank
position* only (1st, 2nd, 3rd...), which is already on the same scale no
matter how differently each retriever computes its own underlying score.

Both retrievers' chunk_ids come from the same make_chunk_id() (Chroma's ID
scheme, reused by bm25_search.py) -- required for RRF to recognize a dense
result and a BM25 result as the same physical chunk.

Note: retrieve_chunks() (the dense side) already applies query_expansion.py
internally -- this hybrid search builds on
top of that improved dense retrieval, not a stripped-down baseline. Worth
knowing, not hiding: it means the dense side here already has a hand-tuned
advantage BM25 doesn't get, on the three previously-diagnosed queries.
"""

import sys
from pathlib import Path
from typing import Optional

from bm25_search import BM25Index, build_index  # noqa: E402
from retrieve import RetrievedChunk, get_collection, retrieve_chunks  # noqa: E402

DEFAULT_TOP_K = 3

# How many candidates to pull from EACH retriever before fusing -- wider
# than the final top_k on purpose. A chunk ranked #8 by dense search but
# #1 by BM25 deserves a shot at the final top-k; fetching only each side's
# top-k would never let the fusion step see that chunk at all.
CANDIDATE_POOL_SIZE = 15

# LangChain's EnsembleRetriever default -- the constant from the original
# RRF paper (Cormack, Clarke & Buettcher 2009). Controls how much rank
# position matters: a small k makes rank #1 dominate heavily, a large k
# flattens the gap between neighboring ranks. Not derived from this corpus.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]],
    weights: Optional[list[float]] = None,
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Generic RRF over any number of ranked lists of chunk_ids (best first) --
    not hardcoded to exactly two retrievers, the same way LangChain's
    EnsembleRetriever accepts an arbitrary list of retrievers. Returns
    chunk_id -> fused score. A chunk_id absent from one list simply gets no
    contribution from that list; it isn't penalized beyond that.
    """
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    if len(weights) != len(ranked_id_lists):
        raise ValueError("weights must have one entry per ranked list")

    scores: dict[str, float] = {}
    for ranked_ids, weight in zip(ranked_id_lists, weights):
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


def hybrid_search(
    collection,
    bm25_index: BM25Index,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    weights: tuple[float, float] = (0.5, 0.5),
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[RetrievedChunk]:
    """
    weights = (dense_weight, bm25_weight) -- this is the "beta" from our
    earlier discussion, just applied inside RRF's rank-based formula
    instead of to raw scores directly (see reciprocal_rank_fusion's
    docstring for why raw-score blending was avoided).
    """
    dense_results = retrieve_chunks(collection, query, top_k=candidate_pool_size)
    bm25_results = bm25_index.score(query, top_k=candidate_pool_size)

    # chunk_id -> full chunk dict, from whichever side saw it first. Both
    # sides describe the same physical chunk identically (same chunk_id
    # scheme), so it doesn't matter which one "wins" here.
    chunk_by_id = {c["chunk_id"]: c for c in dense_results}
    for c in bm25_results:
        chunk_by_id.setdefault(c["chunk_id"], c)

    ranked_lists = [
        [c["chunk_id"] for c in dense_results],
        [c["chunk_id"] for c in bm25_results],
    ]
    fused_scores = reciprocal_rank_fusion(ranked_lists, weights=list(weights))

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    results: list[RetrievedChunk] = []
    for chunk_id in ranked_ids[:top_k]:
        chunk = chunk_by_id[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=chunk["chunk_text"],
                source_doc=chunk["source_doc"],
                chunk_index=chunk["chunk_index"],
                score=round(fused_scores[chunk_id], 5),
            )
        )
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    collection = get_collection()
    bm25_index = build_index()

    query = sys.argv[1] if len(sys.argv) > 1 else "ما هو المتطلب السابق للمادة 1503473"
    results = hybrid_search(collection, bm25_index, query, top_k=5)

    print(f"Query: {query}\n")
    for r in results:
        print(f"  {r['source_doc']} (chunk {r['chunk_index']}) -- RRF score {r['score']}")
