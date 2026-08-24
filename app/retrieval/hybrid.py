"""Hybrid retriever: dense + sparse, fused with RRF, then metadata-aware reranking.

Every decomposed sub-question is retrieved for **independently** and the sub-query results
are merged afterwards, so a narrow sub-question cannot starve a broad one.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from app.config import INDEX_DIR, SETTINGS
from app.models.schemas import Modality, RetrievedDoc, SubQuery, SubQueryType
from app.retrieval.corpus import Corpus, get_corpus
from app.retrieval.dense import DenseIndex
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.fusion import rank_positions, reciprocal_rank_fusion
from app.retrieval.rerank import rerank
from app.retrieval.sparse import SparseIndex


class HybridRetriever:
    def __init__(self, corpus: Corpus, embedder: Embedder | None = None) -> None:
        self.corpus = corpus
        self.embedder = embedder or get_embedder(corpus.texts)
        vectors = self.embedder.encode(corpus.texts)
        self.dense = DenseIndex(vectors, corpus.ids)
        self.sparse = SparseIndex(corpus.texts, corpus.ids)
        self.build_info = {
            "embedding_model": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "dense_backend": self.dense.backend,
            "sparse_backend": "rank_bm25.BM25Okapi",
            "documents": len(corpus),
            "rrf_k": SETTINGS.retrieval.rrf_k,
            "top_k_dense": SETTINGS.retrieval.top_k_dense,
            "top_k_sparse": SETTINGS.retrieval.top_k_sparse,
            "rerank": SETTINGS.retrieval.rerank,
        }

    # ------------------------------------------------------------------ primitives ----
    def dense_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        k = k or SETTINGS.retrieval.top_k_dense
        vec = self.embedder.encode([query])
        return self.dense.search(vec, k)[0]

    def sparse_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        k = k or SETTINGS.retrieval.top_k_sparse
        return self.sparse.search(query, k)

    # ---------------------------------------------------------------- sub-question ----
    def search_subquery(self, subquery: SubQuery, k: int | None = None) -> list[RetrievedDoc]:
        cfg = SETTINGS.retrieval
        k = k or cfg.final_k
        query = subquery.text
        dense_hits = self.dense_search(query)
        sparse_hits = self.sparse_search(query)

        dense_ids = [d for d, _ in dense_hits]
        sparse_ids = [d for d, _ in sparse_hits]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=cfg.rrf_k)
        dense_pos = rank_positions(dense_ids)
        sparse_pos = rank_positions(sparse_ids)

        candidates = [
            RetrievedDoc(
                record_id=doc_id,
                score=score,
                fusion_score=score,
                dense_rank=dense_pos.get(doc_id),
                sparse_rank=sparse_pos.get(doc_id),
                subquery_id=subquery.subquery_id,
                why=[
                    w
                    for w in [
                        f"dense rank {dense_pos[doc_id]}" if doc_id in dense_pos else "",
                        f"bm25 rank {sparse_pos[doc_id]}" if doc_id in sparse_pos else "",
                    ]
                    if w
                ],
            )
            for doc_id, score in sorted(fused.items(), key=lambda kv: -kv[1])
        ]
        if cfg.rerank:
            candidates = rerank(candidates, self.corpus.by_id, subquery)
        return candidates[:k]

    async def asearch_subquery(self, subquery: SubQuery, k: int | None = None) -> list[RetrievedDoc]:
        return await asyncio.to_thread(self.search_subquery, subquery, k)

    async def search_plan(self, subqueries: list[SubQuery], k: int | None = None) -> dict[str, list[RetrievedDoc]]:
        """Retrieve for every sub-question in parallel."""
        results = await asyncio.gather(*(self.asearch_subquery(sq, k) for sq in subqueries))
        return {sq.subquery_id: docs for sq, docs in zip(subqueries, results)}

    # --------------------------------------------------------------------- merging ----
    @staticmethod
    def merge(results: dict[str, list[RetrievedDoc]], limit: int) -> list[RetrievedDoc]:
        """Round-robin merge so every sub-question contributes to the final evidence set."""
        merged: list[RetrievedDoc] = []
        seen: set[str] = set()
        pools = [list(v) for v in results.values()]
        depth = max((len(p) for p in pools), default=0)
        for i in range(depth):
            for pool in pools:
                if i >= len(pool):
                    continue
                doc = pool[i]
                if doc.record_id in seen:
                    continue
                seen.add(doc.record_id)
                merged.append(doc)
                if len(merged) >= limit:
                    return merged
        return merged

    # ------------------------------------------------------------------ simple API ----
    def search(self, question: str, k: int | None = None, region: str | None = None) -> list[RetrievedDoc]:
        """Retrieval-only entry point (no decomposition, no LLM). Used by the benchmark."""
        sq = SubQuery(
            subquery_id="q0",
            type=SubQueryType.RETRIEVE_CONTEXT,
            text=question,
            region=region,
        )
        return self.search_subquery(sq, k)

    def save_manifest(self, path: Path | None = None) -> Path:
        path = path or (INDEX_DIR / "index_manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.build_info)
        payload["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["dataset_version"] = SETTINGS.dataset_version
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


_RETRIEVER: HybridRetriever | None = None


def get_retriever(reload: bool = False) -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or reload:
        _RETRIEVER = HybridRetriever(get_corpus(reload=reload))
    return _RETRIEVER


def set_retriever(retriever: HybridRetriever | None) -> None:
    global _RETRIEVER
    _RETRIEVER = retriever
