"""Evaluation harness.

Two independent benchmarks:

1. **Retrieval-only** - query -> retrieval -> evidence, with no generator involved. This is
   what retrieval quality is judged on, so that swapping or upgrading the LLM cannot move
   the retrieval numbers.
2. **End-to-end** - the full pipeline, judged on answer state, coverage classification,
   blind-window detection, fabrication and contradiction handling.

Plus a **confidence calibration sweep**: coverage is artificially reduced to
100/80/60/40/20 % and the correlation between coverage and confidence is measured.

Every report records the embedding model, LLM provider/model, dataset version, retrieval
configuration, document count and query-set version.
"""
from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import EVAL_DIR, QUERY_SET_VERSION, SETTINGS
from app.coverage.ledger import CoverageLedger, get_ledger
from app.dataset.world import t
from app.evaluation import metrics as M
from app.evaluation.golden import GoldenQuestion, build
from app.models.schemas import AnswerState, CoverageStatus, SubQuery, SubQueryType, TimeRange
from app.reasoning import llm
from app.reasoning.decomposition import decompose
from app.reasoning.pipeline import answer_question
from app.retrieval.corpus import Corpus, get_corpus
from app.retrieval.hybrid import HybridRetriever, get_retriever

K_VALUES = (5, 10, 20)


# --------------------------------------------------------------------------------------
# 1. Retrieval-only benchmark
# --------------------------------------------------------------------------------------
def run_retrieval_benchmark(
    questions: list[GoldenQuestion] | None = None,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    questions = questions or build()
    retriever = retriever or get_retriever()

    per_question: list[dict[str, Any]] = []
    started = time.perf_counter()
    for q in questions:
        plan = decompose(q.question)
        # Retrieve for each sub-question independently, then merge - exactly as the
        # pipeline does, but with no evidence classification and no generator.
        results = asyncio.run(retriever.search_plan(plan.subqueries))
        merged = [d.record_id for d in HybridRetriever.merge(results, limit=max(K_VALUES))]

        single = [
            d.record_id
            for d in retriever.search_subquery(
                SubQuery(
                    subquery_id="flat",
                    type=SubQueryType.RETRIEVE_CONTEXT,
                    text=q.question,
                    region=q.region,
                    time_range=q.time_range(),
                ),
                k=max(K_VALUES),
            )
        ]

        row: dict[str, Any] = {
            "qid": q.qid,
            "category": q.category,
            "retrieved": merged,
            "relevant_count": len(q.relevant_ids),
            "mrr": M.reciprocal_rank(merged, q.relevant_ids),
            "mrr_flat_baseline": M.reciprocal_rank(single, q.relevant_ids),
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = M.recall_at_k(merged, q.relevant_ids, k)
            row[f"precision@{k}"] = M.precision_at_k(merged, q.relevant_ids, k)
            row[f"flat_recall@{k}"] = M.recall_at_k(single, q.relevant_ids, k)
        row["trap_rate@10"] = M.trap_rate(merged, q.forbidden_ids, 10)
        per_question.append(row)

    summary: dict[str, Any] = {"questions": len(questions)}
    for k in K_VALUES:
        summary[f"recall@{k}"] = round(M.mean([r[f"recall@{k}"] for r in per_question]), 4)
        summary[f"precision@{k}"] = round(M.mean([r[f"precision@{k}"] for r in per_question]), 4)
        summary[f"decomposed_vs_flat_recall@{k}"] = round(
            M.mean([r[f"recall@{k}"] for r in per_question])
            - M.mean([r[f"flat_recall@{k}"] for r in per_question]),
            4,
        )
    summary["mrr"] = round(M.mean([r["mrr"] for r in per_question]), 4)
    summary["mrr_flat_baseline"] = round(M.mean([r["mrr_flat_baseline"] for r in per_question]), 4)
    summary["trap_rate@10"] = round(M.mean([r["trap_rate@10"] for r in per_question]), 4)
    summary["latency_seconds"] = round(time.perf_counter() - started, 2)
    return {"summary": summary, "per_question": per_question}


# --------------------------------------------------------------------------------------
# 2. End-to-end benchmark
# --------------------------------------------------------------------------------------
async def _run_all(questions: list[GoldenQuestion], retriever, ledger, corpus):  # noqa: ANN001
    out = []
    for q in questions:
        answer = await answer_question(
            q.question, retriever=retriever, ledger=ledger, corpus=corpus
        )
        out.append((q, answer))
    return out


def run_end_to_end(
    questions: list[GoldenQuestion] | None = None,
    retriever: HybridRetriever | None = None,
    ledger: CoverageLedger | None = None,
    corpus: Corpus | None = None,
) -> dict[str, Any]:
    questions = questions or build()
    retriever = retriever or get_retriever()
    ledger = ledger or get_ledger()
    corpus = corpus or get_corpus()

    started = time.perf_counter()
    pairs = asyncio.run(_run_all(questions, retriever, ledger, corpus))

    rows: list[dict[str, Any]] = []
    for q, answer in pairs:
        cited = [e["id"] for e in answer.evidence]
        # A trap only counts as "cited" if it actually supported the claim; showing it
        # in the evidence list marked STALE/UNOBSERVED is transparency, not a failure.
        claim_ids = answer.meta.get("claim_evidence_ids", cited)
        blind_detected = bool(answer.coverage.missing_intervals) or any(
            "no sensing coverage" in g.lower() or "coverage missing" in g.lower()
            for g in answer.gaps
        )
        contradiction_reported = any(
            c["severity_label"] in ("moderate", "high") for c in answer.contradictions
        )
        lo, hi = q.expected_coverage_band
        rows.append(
            {
                "qid": q.qid,
                "category": q.category,
                "question": q.question,
                "expected_state": q.expected_state.value,
                "actual_state": answer.state.value,
                "state_correct": answer.state.value == q.expected_state.value,
                "confidence": answer.confidence,
                "coverage_fraction": answer.coverage.coverage_fraction,
                "coverage_status": answer.coverage.status.value,
                "coverage_in_band": lo - 1e-9 <= answer.coverage.coverage_fraction <= hi + 1e-9,
                "expects_blind_window": q.expects_blind_window,
                "blind_window_detected": blind_detected,
                "expects_contradiction": q.expects_contradiction,
                "contradiction_reported": contradiction_reported,
                "fabricated_absence": M.is_fabricated_absence(answer.answer, answer.state.value),
                "cited_forbidden": sorted(set(claim_ids) & set(q.forbidden_ids)),
                "listed_forbidden": sorted(set(cited) & set(q.forbidden_ids)),
                "evidence_coverage": M.evidence_coverage(cited, q.relevant_ids),
                "gap_mentioned": bool(answer.gaps),
                "gap_required": q.must_mention_gap,
                "ungrounded_citations": [
                    v for v in (answer.trace.stage("llm_synthesis").detail.get("violations") or [])
                ]
                if answer.trace and answer.trace.stage("llm_synthesis")
                else [],
                "latency_ms": answer.meta.get("total_latency_ms"),
            }
        )

    unknown_rows = [r for r in rows if r["expected_state"] == "UNKNOWN"]
    absence_rows = [r for r in rows if r["expected_state"] == "OBSERVED_ABSENCE"]
    blind_rows = [r for r in rows if r["expects_blind_window"]]
    contradiction_rows = [r for r in rows if r["expects_contradiction"]]
    clean_rows = [r for r in rows if not r["expects_contradiction"]]
    gap_rows = [r for r in rows if r["gap_required"]]

    def frac(items: list[dict], key: str) -> float:
        return round(sum(1 for r in items if r[key]) / len(items), 4) if items else float("nan")

    summary = {
        "questions": len(rows),
        "state_accuracy": frac(rows, "state_correct"),
        "unknown_recall": frac(unknown_rows, "state_correct"),
        "observed_absence_accuracy": frac(absence_rows, "state_correct"),
        "coverage_classification_accuracy": frac(rows, "coverage_in_band"),
        "blind_window_detection": frac(blind_rows, "blind_window_detected"),
        "fabrication_rate_on_planted_absences": round(
            sum(1 for r in unknown_rows if r["fabricated_absence"]) / len(unknown_rows), 4
        )
        if unknown_rows
        else 0.0,
        "fabrication_rate_overall": round(
            sum(1 for r in rows if r["fabricated_absence"]) / len(rows), 4
        ),
        "ungrounded_citation_rate": round(
            sum(1 for r in rows if r["ungrounded_citations"]) / len(rows), 4
        ),
        "contradiction_recall": frac(contradiction_rows, "contradiction_reported"),
        "contradiction_false_positive_rate": round(
            sum(1 for r in clean_rows if r["contradiction_reported"]) / len(clean_rows), 4
        )
        if clean_rows
        else float("nan"),
        "trap_citation_rate": round(
            sum(1 for r in rows if r["cited_forbidden"]) / len(rows), 4
        ),
        "gap_reporting_rate": frac(gap_rows, "gap_mentioned"),
        "mean_evidence_coverage": round(M.mean([r["evidence_coverage"] for r in rows]), 4),
        "mean_latency_ms": round(M.mean([r["latency_ms"] or 0.0 for r in rows]), 1),
        "p95_latency_ms": round(
            sorted(r["latency_ms"] or 0.0 for r in rows)[int(0.95 * (len(rows) - 1))], 1
        ),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    by_category: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(row["state_correct"])
    summary["state_accuracy_by_category"] = {
        cat: round(v["correct"] / v["n"], 4) for cat, v in sorted(by_category.items())
    }
    return {"summary": summary, "per_question": rows}


# --------------------------------------------------------------------------------------
# 3. Confidence calibration sweep
# --------------------------------------------------------------------------------------
CALIBRATION_QUESTIONS = [
    ("sector_alpha", ((4, 0), (4, 20)), "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?"),
    ("grid_a1", ((4, 0), (4, 20)), "Were there any contacts in Grid A1 between 04:00 and 04:20?"),
    ("sector_bravo", ((4, 0), (4, 30)), "Were there any contacts in Sector Bravo between 04:00 and 04:30?"),
]


def run_calibration(
    retriever: HybridRetriever | None = None,
    ledger: CoverageLedger | None = None,
    corpus: Corpus | None = None,
    levels: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> dict[str, Any]:
    retriever = retriever or get_retriever()
    base_ledger = ledger or get_ledger()
    corpus = corpus or get_corpus()

    points: list[dict[str, Any]] = []
    for region, window, question in CALIBRATION_QUESTIONS:
        tr = TimeRange(start=t(*window[0]), end=t(*window[1]))
        for level in levels:
            injected = base_ledger.with_coverage_loss(level, region=region, window=tr)
            answer = asyncio.run(
                answer_question(question, retriever=retriever, ledger=injected, corpus=corpus)
            )
            points.append(
                {
                    "region": region,
                    "injected_level": level,
                    "coverage_fraction": answer.coverage.coverage_fraction,
                    "confidence": answer.confidence,
                    "state": answer.state.value,
                }
            )

    coverages = [p["coverage_fraction"] for p in points]
    confidences = [p["confidence"] for p in points]
    monotone_violations = 0
    for region, _w, _q in CALIBRATION_QUESTIONS:
        series = [p for p in points if p["region"] == region]
        series.sort(key=lambda p: -p["injected_level"])
        for a, b in zip(series, series[1:]):
            if b["confidence"] > a["confidence"] + 1e-6:
                monotone_violations += 1

    return {
        "summary": {
            "pearson_confidence_vs_coverage": round(M.pearson(coverages, confidences), 4),
            "spearman_confidence_vs_coverage": round(M.spearman(coverages, confidences), 4),
            "monotonicity_violations": monotone_violations,
            "levels": list(levels),
        },
        "points": points,
    }


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------
def environment(retriever: HybridRetriever) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": SETTINGS.dataset_version,
        "query_set_version": QUERY_SET_VERSION,
        "documents": retriever.build_info["documents"],
        "embedding_model": retriever.build_info["embedding_model"],
        "embedding_dim": retriever.build_info["embedding_dim"],
        "dense_backend": retriever.build_info["dense_backend"],
        "sparse_backend": retriever.build_info["sparse_backend"],
        "retrieval_config": SETTINGS.retrieval.model_dump(),
        "coverage_config": SETTINGS.coverage.model_dump(),
        "confidence_config": SETTINGS.confidence.model_dump(),
        "llm": llm.provider_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def run_all(write_path: Path | None = None) -> dict[str, Any]:
    retriever = get_retriever()
    report = {
        "environment": environment(retriever),
        "retrieval_only": run_retrieval_benchmark(retriever=retriever),
        "end_to_end": run_end_to_end(retriever=retriever),
        "calibration": run_calibration(retriever=retriever),
    }
    path = write_path or (EVAL_DIR / "report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(path)
    return report
