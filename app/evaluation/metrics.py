"""Metric implementations. All deterministic, all computed in Python."""
from __future__ import annotations

import math
import re
from typing import Sequence

#: Phrases that assert absence. Used for fabrication detection - if the answer state is
#: UNKNOWN, none of these may appear unhedged.
_ABSENCE_ASSERTIONS = [
    r"no contacts were observed",
    r"no contacts were detected",
    r"no contacts detected",
    r"nothing was present",
    r"nothing was detected",
    r"the area was clear",
    r"no vessels were present",
    r"there were no contacts",
]
_HEDGES = [
    "cannot determine",
    "unknown",
    "not sufficiently observed",
    "insufficient",
    "did not look",
    "no claim is made",
    "unobserved",
]


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    top = list(retrieved[:k])
    hits = sum(1 for d in top if d in set(relevant))
    return hits / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def evidence_coverage(cited: Sequence[str], relevant: Sequence[str]) -> float:
    """Fraction of the ground-truth evidence that actually reached the answer."""
    if not relevant:
        return float("nan")
    return len(set(cited) & set(relevant)) / len(set(relevant))


def trap_rate(retrieved: Sequence[str], forbidden: Sequence[str], k: int) -> float:
    if not forbidden:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(forbidden)) / len(set(forbidden))


def is_fabricated_absence(answer_text: str, state: str) -> bool:
    """An UNKNOWN answer that nonetheless asserts absence is a fabrication."""
    if state != "UNKNOWN":
        return False
    text = answer_text.lower()
    asserts_absence = any(re.search(p, text) for p in _ABSENCE_ASSERTIONS)
    if not asserts_absence:
        return False
    return not any(h in text for h in _HEDGES)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = mean_rank
            i = j + 1
        return ranks

    return pearson(rank(xs), rank(ys))


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")
