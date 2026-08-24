"""Pluggable dense encoders.

`sentence_transformers` is used when the model is available locally; otherwise the system
falls back to a deterministic TF-IDF + SVD encoder so that the prototype stays fully
runnable (and reproducible) offline. Whichever backend is used is recorded in every
benchmark report - provider variability must never contaminate a retrieval measurement.
"""
from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np

from app.config import SETTINGS


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from transformers.utils import logging as hf_logging  # local import: heavy

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        from sentence_transformers import SentenceTransformer  # local import: heavy

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


class TfidfSvdEmbedder:
    """Deterministic offline fallback. Fitted once on the corpus, then frozen."""

    def __init__(self, corpus_texts: Sequence[str], dim: int = 384) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w\-]+\b",
            min_df=1,
        )
        matrix = self._vectorizer.fit_transform(list(corpus_texts))
        n_components = int(min(dim, max(2, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._svd.fit(matrix)
        self.name = f"tfidf-svd-{n_components}"
        self.dim = n_components

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(list(texts))
        dense = self._svd.transform(matrix).astype("float32")
        return _normalise(dense)


def get_embedder(corpus_texts: Sequence[str]) -> Embedder:
    cfg = SETTINGS.retrieval
    backend = cfg.dense_backend
    if backend in ("auto", "sentence_transformers"):
        try:
            return SentenceTransformerEmbedder(cfg.dense_model)
        except Exception as exc:  # pragma: no cover - environment dependent
            if backend == "sentence_transformers":
                raise
            print(f"[retrieval] sentence-transformers unavailable ({exc}); using TF-IDF+SVD")
    return TfidfSvdEmbedder(corpus_texts, dim=cfg.embedding_dim_fallback)
