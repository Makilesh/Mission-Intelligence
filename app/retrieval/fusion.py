"""Reciprocal Rank Fusion.

RRF(d) = sum_i 1 / (k + rank_i(d))

Raw similarity scores are deliberately NOT fused: BM25 scores and cosine similarities live
on incomparable scales, and normalising them makes fusion sensitive to the score
distribution of whichever corpus happens to be loaded. Ranks are scale-free.
"""
from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def rank_positions(ranked: Sequence[str]) -> dict[str, int]:
    return {doc_id: i + 1 for i, doc_id in enumerate(ranked)}
