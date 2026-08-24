"""Dense vector index. FAISS when available, exact numpy dot-product otherwise."""
from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, vectors: np.ndarray, ids: list[str]) -> None:
        assert vectors.shape[0] == len(ids)
        self.ids = ids
        self.vectors = np.ascontiguousarray(vectors.astype("float32"))
        self.backend = "numpy"
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss_index = index
            self.backend = "faiss.IndexFlatIP"
        except Exception:  # pragma: no cover - environment dependent
            self._faiss_index = None

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        query_vectors = np.ascontiguousarray(query_vectors.astype("float32"))
        k = min(k, len(self.ids))
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(query_vectors, k)
        else:
            sims = query_vectors @ self.vectors.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        out: list[list[tuple[str, float]]] = []
        for row_ids, row_scores in zip(idx, scores):
            out.append(
                [
                    (self.ids[int(i)], float(s))
                    for i, s in zip(row_ids, row_scores)
                    if int(i) >= 0
                ]
            )
        return out
