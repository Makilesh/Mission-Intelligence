"""FastAPI service.

The API exposes the same structured answer the pipeline produces internally - answer,
state, confidence, coverage, evidence, gaps, contradictions - plus the full stage trace.
Nothing is summarised away at the boundary: an operator client and a debugging client see
the same object.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import SETTINGS
from app.coverage.ledger import get_ledger, set_ledger
from app.dataset import world
from app.evaluation import harness
from app.models.schemas import (
    CoverageEntry,
    CoverageReport,
    MissionAnswer,
    Modality,
    SourceRecord,
    TimeRange,
)
from app.observability import logger
from app.reasoning import llm
from app.reasoning.pipeline import answer_question
from app.retrieval.corpus import Corpus, get_corpus, set_corpus
from app.retrieval.hybrid import HybridRetriever, get_retriever, set_retriever

STARTED_AT = time.time()

_METRICS: dict[str, Any] = {
    "queries_total": 0,
    "latencies_ms": [],
    "state_counts": {},
    "errors_total": 0,
    "ingests_total": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Warm the index once at start-up so the first query is not paying model-load latency.
    await asyncio.to_thread(get_retriever)
    get_ledger()
    logger.info("mission-intelligence API ready")
    yield


app = FastAPI(
    title="Mission Intelligence — Coverage-Aware Retrieval",
    version="0.1.0",
    description=(
        "An empty result is not evidence of absence. Observation coverage is represented "
        "explicitly and evaluated independently of retrieval."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    include_trace: bool = True


class IngestRequest(BaseModel):
    records: list[SourceRecord] = Field(default_factory=list)
    coverage_entries: list[CoverageEntry] = Field(default_factory=list)
    rebuild_index: bool = True


class EvaluationRequest(BaseModel):
    suite: Literal["all", "retrieval", "end_to_end", "calibration"] = "all"


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.post("/query", response_model=MissionAnswer)
async def query(request: QueryRequest) -> MissionAnswer:
    started = time.perf_counter()
    try:
        answer = await answer_question(request.question, include_trace=request.include_trace)
    except Exception as exc:  # pragma: no cover - defensive
        _METRICS["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    _METRICS["queries_total"] += 1
    _METRICS["latencies_ms"].append(round(elapsed, 2))
    _METRICS["state_counts"][answer.state.value] = (
        _METRICS["state_counts"].get(answer.state.value, 0) + 1
    )
    return answer


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, Any]:
    """Add new observations and/or coverage assertions.

    Records and coverage entries are ingested through *separate* fields on purpose: the
    caller cannot create coverage by adding documents.
    """
    corpus = get_corpus()
    ledger = get_ledger()

    if request.coverage_entries:
        for entry in request.coverage_entries:
            ledger.add(entry)
        set_ledger(ledger)

    rebuilt = False
    if request.records:
        merged = Corpus(list(corpus.records) + list(request.records))
        set_corpus(merged)
        if request.rebuild_index:
            retriever = get_retriever()
            set_retriever(
                await asyncio.to_thread(HybridRetriever, merged, retriever.embedder)
            )
            rebuilt = True

    _METRICS["ingests_total"] += 1
    return {
        "records_added": len(request.records),
        "coverage_entries_added": len(request.coverage_entries),
        "index_rebuilt": rebuilt,
        "documents": len(get_corpus()),
        "coverage_entries": len(get_ledger().entries),
    }


@app.get("/coverage", response_model=CoverageReport)
async def coverage(
    region: str = Query(..., description="region id, e.g. sector_alpha or grid_b7"),
    start: str = Query(..., description="ISO timestamp or HH:MM on the mission day"),
    end: str = Query(..., description="ISO timestamp or HH:MM on the mission day"),
    modalities: list[str] | None = Query(None),
) -> CoverageReport:
    """Query the coverage ledger directly, with no retrieval involved at all."""

    def _parse(value: str) -> datetime:
        if ":" in value and len(value) <= 5:
            hh, mm = (int(x) for x in value.split(":"))
            return world.t(hh, mm)
        return datetime.fromisoformat(value)

    try:
        window = TimeRange(start=_parse(start), end=_parse(end))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad time range: {exc}") from exc

    mods = None
    if modalities:
        try:
            mods = [Modality(m) for m in modalities]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_ledger().check(region, window, mods)


@app.get("/evidence/{evidence_id}")
async def evidence(evidence_id: str) -> dict[str, Any]:
    record = get_corpus().get(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no such record: {evidence_id}")
    ledger = get_ledger()
    window = TimeRange(
        start=record.timestamp - world.minutes(2), end=record.timestamp + world.minutes(2)
    )
    report = ledger.check(record.region, window, [record.modality])
    return {
        "record": record.model_dump(mode="json"),
        "coverage_at_record": {
            "status": report.status.value,
            "covered_fraction": report.covered_fraction,
            "absence_claim_supported": report.absence_claim_supported,
            "ledger_entries": report.ledger_entries,
        },
        "sensor": {
            "id": record.sensor,
            "reliability": record.reliability,
            "description": world.SENSORS[record.sensor].description
            if record.sensor in world.SENSORS
            else "",
        },
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    corpus = get_corpus()
    ledger = get_ledger()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "documents": len(corpus),
        "coverage_entries": len(ledger.entries),
        "dataset_version": SETTINGS.dataset_version,
        "query_set_version": SETTINGS.query_set_version,
        "llm": llm.provider_info(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    latencies = _METRICS["latencies_ms"]
    retriever = get_retriever()
    percentiles: dict[str, float] = {}
    if latencies:
        ordered = sorted(latencies)
        percentiles = {
            "p50_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "mean_ms": round(statistics.fmean(ordered), 2),
            "max_ms": ordered[-1],
        }
    return {
        "queries_total": _METRICS["queries_total"],
        "errors_total": _METRICS["errors_total"],
        "ingests_total": _METRICS["ingests_total"],
        "state_counts": _METRICS["state_counts"],
        "latency": percentiles,
        "index": retriever.build_info,
        "coverage_ledger": get_ledger().summary(),
    }


@app.post("/evaluation/run")
async def run_evaluation(request: EvaluationRequest = Body(default=EvaluationRequest())) -> dict[str, Any]:
    """Run the evaluation suite. Blocking work is pushed to a worker thread."""
    retriever = get_retriever()
    if request.suite == "retrieval":
        return {
            "environment": harness.environment(retriever),
            "retrieval_only": (
                await asyncio.to_thread(harness.run_retrieval_benchmark, None, retriever)
            )["summary"],
        }
    if request.suite == "end_to_end":
        return {
            "environment": harness.environment(retriever),
            "end_to_end": (await asyncio.to_thread(harness.run_end_to_end))["summary"],
        }
    if request.suite == "calibration":
        return {
            "environment": harness.environment(retriever),
            "calibration": (await asyncio.to_thread(harness.run_calibration))["summary"],
        }
    report = await asyncio.to_thread(harness.run_all)
    return {
        "environment": report["environment"],
        "retrieval_only": report["retrieval_only"]["summary"],
        "end_to_end": report["end_to_end"]["summary"],
        "calibration": report["calibration"]["summary"],
        "report_path": report["report_path"],
    }
