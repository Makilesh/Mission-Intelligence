"""BM25 sparse retrieval. Tokenisation keeps hyphenated identifiers (T-88, V-17) intact."""
from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self, texts: Sequence[str], ids: Sequence[str]) -> None:
        self.ids = list(ids)
        self._tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
