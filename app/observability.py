"""Structured tracing. Every query produces a stage-by-stage trace with latencies."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.models.schemas import QueryTrace, TraceStage

logger = logging.getLogger("mission_intel")
if not logger.handlers:  # pragma: no cover - logging setup
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Tracer:
    """Collects per-stage latency and detail for one operator query."""

    def __init__(self, question: str) -> None:
        self.trace = QueryTrace(trace_id=uuid.uuid4().hex[:12], question=question)
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        payload: dict[str, Any] = dict(detail)
        try:
            yield payload
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.trace.stages.append(
                TraceStage(name=name, latency_ms=round(elapsed, 2), detail=payload)
            )

    def finish(self) -> QueryTrace:
        self.trace.total_latency_ms = round((time.perf_counter() - self._t0) * 1000.0, 2)
        retrieval_stages = {"dense_sparse_retrieval", "fusion_rerank", "coverage_check"}
        reasoning_stages = {"llm_synthesis", "evidence_classification", "contradiction_detection"}
        self.trace.retrieval_latency_ms = round(
            sum(s.latency_ms for s in self.trace.stages if s.name in retrieval_stages), 2
        )
        self.trace.reasoning_latency_ms = round(
            sum(s.latency_ms for s in self.trace.stages if s.name in reasoning_stages), 2
        )
        logger.info(
            json.dumps(
                {
                    "trace_id": self.trace.trace_id,
                    "question": self.trace.question,
                    "total_ms": self.trace.total_latency_ms,
                    "retrieval_ms": self.trace.retrieval_latency_ms,
                    "reasoning_ms": self.trace.reasoning_latency_ms,
                    "stages": {s.name: s.latency_ms for s in self.trace.stages},
                }
            )
        )
        return self.trace
