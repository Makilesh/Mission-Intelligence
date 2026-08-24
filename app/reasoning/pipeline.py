"""The end-to-end query pipeline.

    question
      -> decomposition
      -> [ coverage check | dense retrieval | sparse retrieval | metadata ]  (parallel)
      -> fusion + rerank
      -> evidence classification
      -> contradiction detection
      -> deterministic state + confidence
      -> LLM synthesis
      -> operator response

Deterministic work happens outside the model. The only LLM call in the whole pipeline is
the final synthesis, and even that is optional.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from app.confidence.model import (
    calculate_confidence,
    decide_state,
    in_scope,
    supporting_evidence,
)
from app.contradiction.engine import aggregate_severity, detect, mark_evidence
from app.coverage.ledger import CoverageLedger, get_ledger
from app.dataset import world
from app.evidence.aggregator import EvidenceBundle, aggregate
from app.evidence.classifier import classify
from app.models.schemas import (
    AnswerState,
    CoverageReport,
    CoverageSummary,
    Evidence,
    MissionAnswer,
    Modality,
    QueryIntent,
    QueryPlan,
    SubQuery,
    SubQueryType,
    TimeRange,
)
from app.observability import Tracer
from app.reasoning import llm
from app.reasoning.association import assess
from app.reasoning.decomposition import decompose
from app.retrieval.corpus import Corpus, get_corpus
from app.retrieval.hybrid import HybridRetriever, get_retriever

DEFAULT_REGION = "mission_area"


def _default_window() -> TimeRange:
    return TimeRange(start=world.MISSION_START, end=world.MISSION_NOW)


def _gap_strings(coverage: CoverageReport) -> list[str]:
    gaps: list[str] = []
    for start, end in coverage.missing_intervals:
        where = ", ".join(coverage.blind_subregions) or coverage.region
        gaps.append(
            f"No sensing coverage of {where} between {start.strftime('%H:%M')} and "
            f"{end.strftime('%H:%M')}Z"
        )
    for modality in coverage.missing_modalities:
        gaps.append(
            f"{modality.value.replace('_', ' ').upper()} coverage missing for "
            f"{coverage.region} during {coverage.time_range.label()}Z"
        )
    for modality in coverage.degraded_modalities:
        gaps.append(
            f"{modality.value.replace('_', ' ').upper()} coverage degraded during "
            f"{coverage.time_range.label()}Z"
        )
    if coverage.no_information_fraction > 0:
        gaps.append(
            f"{coverage.no_information_fraction:.0%} of the queried "
            "(sub-region x modality x time) volume has no ledger entry at all"
        )
    return gaps


def _uncertainty(
    state: AnswerState,
    coverage: CoverageReport,
    bundle: EvidenceBundle,
    reasons: list[str],
) -> list[str]:
    out = list(reasons)
    if bundle.stale:
        out.append(
            f"{len(bundle.stale)} retrieved record(s) were excluded as stale: "
            + ", ".join(e.evidence_id for e in bundle.stale[:5])
        )
    if bundle.low_confidence:
        out.append(
            f"{len(bundle.low_confidence)} record(s) carry low source confidence: "
            + ", ".join(e.evidence_id for e in bundle.low_confidence[:5])
        )
    off_region = [e for e in bundle.evidence if not e.attributes.get("region_relevant", True)]
    if off_region:
        out.append(
            f"{len(off_region)} retrieved record(s) were outside the queried region and did "
            "not contribute to the claim: " + ", ".join(e.evidence_id for e in off_region[:5])
        )
    if state is AnswerState.OBSERVED_ABSENCE and coverage.covered_fraction < 1.0:
        out.append(
            f"absence is asserted only for the observed {coverage.covered_fraction:.0%} of the "
            "queried volume"
        )
    return out


def _operator_view(answer: MissionAnswer, coverage: CoverageReport) -> str:
    filled = int(round(coverage.covered_fraction * 24))
    bar = "#" * filled + "." * (24 - filled)
    lines = [
        f"ANSWER   [{answer.state.value}]",
        answer.answer,
        "",
        f"COVERAGE {bar} {coverage.covered_fraction:.0%} of {coverage.region} "
        f"({coverage.time_range.label()}Z, {coverage.status.value})",
        f"CONFIDENCE {answer.confidence:.2f}",
    ]
    if answer.gaps:
        lines.append("")
        lines.append("GAPS")
        lines.extend(f"  - {g}" for g in answer.gaps)
    if answer.contradictions:
        lines.append("")
        lines.append("CONTRADICTIONS")
        for c in answer.contradictions:
            lines.append(f"  - [{c['severity_label']}] {c['dimension']}: {c['reason']}")
    if answer.evidence:
        lines.append("")
        lines.append("EVIDENCE")
        for e in answer.evidence[:8]:
            lines.append(
                f"  - [{e['id']}] {e['source']}/{e['sensor']} {e['time_range']}Z "
                f"({e['state']}) {e['claim'][:110]}"
            )
    return "\n".join(lines)


async def answer_question(
    question: str,
    retriever: HybridRetriever | None = None,
    ledger: CoverageLedger | None = None,
    corpus: Corpus | None = None,
    include_trace: bool = True,
) -> MissionAnswer:
    retriever = retriever or get_retriever()
    ledger = ledger or get_ledger()
    corpus = corpus or retriever.corpus

    tracer = Tracer(question)

    # ---- 1. decomposition -------------------------------------------------------------
    with tracer.stage("decomposition") as detail:
        plan: QueryPlan = decompose(question)
        detail.update(
            {
                "intent": plan.intent.value,
                "region": plan.region,
                "time_range": plan.time_range.label() if plan.time_range else None,
                "entities": plan.entities,
                "subqueries": [
                    {"id": s.subquery_id, "type": s.type.value, "text": s.text}
                    for s in plan.subqueries
                ],
            }
        )

    region = plan.region or DEFAULT_REGION
    window = plan.time_range or _default_window()

    # ---- 2. coverage + retrieval, in parallel ----------------------------------------
    coverage_holder: dict[str, Any] = {}

    async def coverage_task() -> CoverageReport:
        return await asyncio.to_thread(
            ledger.check, region, window, plan.preferred_modalities or None
        )

    async def retrieval_task():  # noqa: ANN202
        return await retriever.search_plan(plan.subqueries)

    with tracer.stage("dense_sparse_retrieval") as detail:
        coverage, per_subquery = await asyncio.gather(coverage_task(), retrieval_task())
        coverage_holder["report"] = coverage
        detail.update(
            {
                "subquery_hits": {k: len(v) for k, v in per_subquery.items()},
                "embedding_model": retriever.build_info["embedding_model"],
                "dense_backend": retriever.build_info["dense_backend"],
            }
        )

    with tracer.stage("coverage_check") as detail:
        detail.update(
            {
                "region": coverage.region,
                "status": coverage.status.value,
                "covered_fraction": coverage.covered_fraction,
                "missing_intervals": [
                    f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in coverage.missing_intervals
                ],
                "missing_modalities": [m.value for m in coverage.missing_modalities],
                "absence_claim_supported": coverage.absence_claim_supported,
                "ledger_entries": len(coverage.ledger_entries),
            }
        )

    # ---- 3. fusion / merge ------------------------------------------------------------
    with tracer.stage("fusion_rerank") as detail:
        # The evidence pool fed to the deterministic layers is deliberately wider than what
        # the operator is shown: contradiction detection and association reasoning need the
        # long tail, while the operator view stays focused on the top evidence.
        merged = HybridRetriever.merge(per_subquery, limit=max(20, len(plan.subqueries) * 8))
        detail.update({"merged_documents": len(merged), "rrf_k": retriever.build_info["rrf_k"]})
    tracer.trace.retrieved = merged

    # ---- 4. evidence classification ---------------------------------------------------
    with tracer.stage("evidence_classification") as detail:
        bundle = aggregate(plan, merged, corpus, ledger, coverage)
        detail.update(
            {
                "evidence": len(bundle.evidence),
                "presence": len(bundle.presence),
                "observed_absence": len(bundle.absence),
                "unobserved": len(bundle.unobserved),
                "partial_coverage": len(bundle.partial),
                "stale": len(bundle.stale),
                "low_confidence": len(bundle.low_confidence),
            }
        )

    # ---- 4b. multi-hop expansion -------------------------------------------------------
    # A genuine multi-hop question cannot be answered by one round of retrieval: hop 4
    # (identity) depends on the *result* of hop 2 (which contact we are talking about).
    # So for association questions we pick the later contact first, then go back to the
    # index for identity claims co-located with it.
    if plan.intent is QueryIntent.ASSOCIATION:
        with tracer.stage("multihop_expansion") as detail:
            pre = assess(bundle, ledger, plan.comparison_targets)
            added: list[str] = []
            if pre.later is not None:
                anchor = pre.later
                expand = SubQuery(
                    subquery_id="sq-expand",
                    type=SubQueryType.RETRIEVE_IDENTITY,
                    text=(
                        f"identity vessel name MMSI hull identification of the contact in "
                        f"{anchor.region} at {anchor.time_range.start.strftime('%H:%M')}"
                    ),
                    region=anchor.region,
                    time_range=TimeRange(
                        start=anchor.time_range.start - timedelta(minutes=10),
                        end=anchor.time_range.end + timedelta(minutes=10),
                    ),
                    preferred_modalities=[
                        Modality.AIS,
                        Modality.MISSION_REPORT,
                        Modality.EO_IR,
                        Modality.IMAGERY,
                    ],
                )
                known = bundle.ids()
                for doc in await retriever.asearch_subquery(expand, k=8):
                    if doc.record_id in known:
                        continue
                    record = corpus.get(doc.record_id)
                    if record is None:
                        continue
                    bundle.evidence.append(
                        classify(
                            record=record,
                            ledger=ledger,
                            window=plan.time_range,
                            query_region=plan.region,
                            retrieval_score=doc.score,
                        )
                    )
                    added.append(doc.record_id)
            detail.update(
                {
                    "anchor": pre.later.evidence_id if pre.later else None,
                    "added_evidence": added,
                }
            )

    # ---- 5. contradiction detection ---------------------------------------------------
    with tracer.stage("contradiction_detection") as detail:
        all_contradictions = detect(bundle.evidence)
        mark_evidence(bundle.evidence, all_contradictions)
        # A real disagreement elsewhere in the mission is not an answer to *this* question.
        # It is still surfaced, but as "related", and it does not drive the state or the
        # confidence penalty.
        contradictions = [
            c for c in all_contradictions if in_scope(c, coverage, plan.region, plan.entities)
        ]
        related = [c for c in all_contradictions if c not in contradictions]
        severity = aggregate_severity(contradictions)
        detail.update(
            {
                "contradictions": len(contradictions),
                "related_out_of_scope": len(related),
                "aggregate_severity": severity,
                "dimensions": [c.dimension.value for c in contradictions],
            }
        )

    # ---- 6. deterministic state + confidence -----------------------------------------
    with tracer.stage("confidence_calculation") as detail:
        state, reasons = decide_state(
            bundle=bundle,
            coverage=coverage,
            contradictions=contradictions,
            contradiction_severity=severity,
            intent_is_absence=plan.intent is QueryIntent.ABSENCE_CHECK,
            region=plan.region,
            entities=plan.entities,
        )
        breakdown = calculate_confidence(
            state=state,
            coverage=coverage,
            bundle=bundle,
            contradictions=contradictions,
            contradiction_severity=severity,
        )
        detail.update({"state": state.value, "confidence": breakdown.confidence, "reasons": reasons})

    # ---- 6b. multi-hop association (only for association questions) --------------------
    association_text: str | None = None
    association_payload: dict[str, Any] | None = None
    if plan.intent is QueryIntent.ASSOCIATION:
        with tracer.stage("association_analysis") as detail:
            assessment = assess(bundle, ledger, plan.comparison_targets)
            association_text = assessment.narrative
            association_payload = assessment.to_dict()
            detail.update(association_payload)
            if assessment.verdict in ("CANNOT_ASSOCIATE", "INSUFFICIENT_EVIDENCE"):
                state = AnswerState.UNKNOWN
                reasons.append(
                    f"association verdict {assessment.verdict}: the two reports cannot be "
                    "linked with the available evidence"
                )
                breakdown = calculate_confidence(
                    state=state,
                    coverage=coverage,
                    bundle=bundle,
                    contradictions=contradictions,
                    contradiction_severity=severity,
                )
            else:
                adjusted = max(
                    0.02, min(0.99, breakdown.confidence + assessment.confidence_modifier)
                )
                breakdown = breakdown.model_copy(update={"confidence": round(adjusted, 4)})
                reasons.append(f"association verdict {assessment.verdict}")

    # ---- 7. synthesis ------------------------------------------------------------------
    gaps = _gap_strings(coverage)
    claim_support = supporting_evidence(bundle, state)
    pinned = set(association_payload.get("supporting_evidence") or []) if association_payload else set()
    shown: list[Evidence] = _select_for_display(bundle, state, pinned=pinned)
    with tracer.stage("llm_synthesis") as detail:
        result = llm.synthesise(
            plan=plan,
            state=state,
            coverage=coverage,
            evidence=shown,
            contradictions=contradictions,
            confidence=breakdown.confidence,
            gaps=gaps,
            presence=bundle.presence,
            absence=bundle.absence,
            unobserved=bundle.unobserved,
            association=association_text,
        )
        detail.update(
            {
                "provider": result.provider,
                "model": result.model,
                "grounded": result.grounded,
                "violations": result.violations,
                "fallback_used": result.fallback_used,
            }
        )

    trace = tracer.finish()

    answer = MissionAnswer(
        answer=result.text,
        state=state,
        confidence=breakdown.confidence,
        coverage=CoverageSummary(
            region=coverage.region,
            time_range=coverage.time_range.label() + "Z",
            coverage_fraction=coverage.covered_fraction,
            status=coverage.status,
            modalities=[m.value for m in coverage.requested_modalities],
            missing_modalities=[m.value for m in coverage.missing_modalities],
            missing_intervals=[
                f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}Z"
                for s, e in coverage.missing_intervals
            ],
        ),
        evidence=[e.to_public() for e in shown],
        gaps=gaps,
        contradictions=[
            {
                "id": c.contradiction_id,
                "dimension": c.dimension.value,
                "severity": c.severity,
                "severity_label": c.severity_label,
                "reason": c.reason,
                "entity": c.entity,
                "claims": [
                    {
                        "evidence_id": cl.evidence_id,
                        "source": cl.source.value,
                        "sensor": cl.sensor,
                        "value": cl.value,
                        "reliability": cl.reliability,
                    }
                    for cl in c.claims
                ],
                "resolved": False,
            }
            for c in contradictions
        ],
        uncertainty=_uncertainty(state, coverage, bundle, reasons),
        confidence_breakdown=breakdown,
        plan=plan,
        trace=trace if include_trace else None,
        meta={
            "related_contradictions": [
                {
                    "id": c.contradiction_id,
                    "dimension": c.dimension.value,
                    "severity": c.severity,
                    "severity_label": c.severity_label,
                    "reason": c.reason,
                    "region": c.region,
                    "time_range": c.time_range.label(),
                    "note": "outside the queried region/window; not used to answer",
                }
                for c in related
            ],
            "dataset_version": retriever.build_info.get("dataset_version", ""),
            "documents": retriever.build_info["documents"],
            "embedding_model": retriever.build_info["embedding_model"],
            "llm": llm.provider_info(),
            "evidence_count": len(bundle.evidence),
            "coverage_percent": round(coverage.covered_fraction * 100, 1),
            "total_latency_ms": trace.total_latency_ms,
            "retrieval_latency_ms": trace.retrieval_latency_ms,
            "reasoning_latency_ms": trace.reasoning_latency_ms,
            "confidence_features": breakdown.explain(),
            "association": association_payload,
            "claim_evidence_ids": [e.evidence_id for e in claim_support],
        },
    )
    answer.operator_view = _operator_view(answer, coverage)
    return answer


def _select_for_display(
    bundle: EvidenceBundle,
    state: AnswerState,
    limit: int = 12,
    pinned: set[str] | None = None,
) -> list[Evidence]:
    """Evidence shown to the operator and to the LLM: claim-supporting first, then context."""
    ordered: list[Evidence] = []
    seen: set[str] = set()

    def push(items: list[Evidence]) -> None:
        for e in items:
            if e.evidence_id in seen:
                continue
            seen.add(e.evidence_id)
            ordered.append(e)

    if pinned:
        # Evidence the deterministic analysis explicitly cites is always shown.
        push([e for e in bundle.evidence if e.evidence_id in pinned])
    if state is AnswerState.PRESENCE:
        push(bundle.presence)
    elif state is AnswerState.OBSERVED_ABSENCE:
        push(bundle.absence)
    elif state is AnswerState.CONTRADICTION:
        push([e for e in bundle.evidence if e.attributes.get("contradiction_ids")])
        push(bundle.presence)
    else:
        push(bundle.unobserved)
        push(bundle.partial)
    push([e for e in bundle.operational if e.state.value != "STALE"])
    push(bundle.evidence)
    return ordered[:limit]


def answer_question_sync(question: str, **kwargs: Any) -> MissionAnswer:
    return asyncio.run(answer_question(question, **kwargs))
