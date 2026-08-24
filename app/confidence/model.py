"""Deterministic confidence model.

Confidence is computed in Python from measurable features. The LLM never reports it, never
sees it before it is fixed, and cannot change it.

Features: coverage, source reliability, recency, retrieval agreement, evidence count,
contradiction severity, staleness.

The property that matters most, and that is unit-tested:

    **confidence is monotonically non-decreasing in coverage.**

For an absence claim the coverage term is applied twice - once in the weighted sum and once
as a multiplier - because "we saw nothing" is worth exactly as much as the observation that
backs it.
"""
from __future__ import annotations

from app.config import SETTINGS
from app.evidence.aggregator import EvidenceBundle, base_state
from app.models.schemas import (
    AnswerState,
    ConfidenceBreakdown,
    Contradiction,
    CoverageReport,
    Evidence,
    EvidenceState,
    Modality,
)


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _quality(e: Evidence) -> float:
    return float(e.reliability) * float(e.attributes.get("confidence", 1.0) or 1.0)


def supporting_evidence(bundle: EvidenceBundle, state: AnswerState) -> list[Evidence]:
    """The evidence that actually backs the answer we are about to give."""
    if state is AnswerState.PRESENCE:
        return bundle.presence
    if state is AnswerState.OBSERVED_ABSENCE:
        return bundle.absence
    if state is AnswerState.CONTRADICTION:
        return [e for e in bundle.operational if e.attributes.get("contradiction_ids")] or bundle.presence
    # UNKNOWN: the gap evidence is the support for saying "we cannot tell".
    return bundle.unobserved + bundle.partial


def retrieval_agreement(bundle: EvidenceBundle, support: list[Evidence]) -> float:
    """How much of the available sensor picture points the same way."""
    sensing = [
        e
        for e in bundle.operational
        if e.source not in (Modality.STANDING_ORDER, Modality.TERRAIN)
        and e.attributes.get("kind") is None
    ]
    if not sensing:
        return 0.0
    available = {e.source for e in sensing}
    agreeing = {e.source for e in support if e.source in available}
    if not available:
        return 0.0
    modality_share = len(agreeing) / len(available)
    # A single corroborating record is weaker than several.
    count_factor = min(1.0, len(support) / 3.0)
    return round(0.65 * modality_share + 0.35 * count_factor, 4)


def calculate_confidence(
    state: AnswerState,
    coverage: CoverageReport | None,
    bundle: EvidenceBundle,
    contradictions: list[Contradiction],
    contradiction_severity: float = 0.0,
) -> ConfidenceBreakdown:
    cfg = SETTINGS.confidence

    covered = coverage.covered_fraction if coverage else 0.0
    quality = coverage.coverage_quality if coverage else 0.0
    # Sensor quality modulates coverage but can never invert its ordering.
    coverage_term = covered * (0.75 + 0.25 * quality)

    support = supporting_evidence(bundle, state)
    reliability = _mean([_quality(e) for e in support], default=0.30)
    recency = _mean([e.recency for e in support], default=0.50)
    agreement = retrieval_agreement(bundle, support)
    evidence_support = min(1.0, len(support) / cfg.max_evidence_saturation)

    relevant = [e for e in bundle.evidence if e.attributes.get("region_relevant", True)]
    stale_share = (len(bundle.stale) / len(relevant)) if relevant else 0.0
    stale_penalty = round(cfg.stale_penalty * stale_share, 4)

    contradiction_penalty = round(
        min(0.85, cfg.contradiction_weight * max(contradiction_severity, 0.0)), 4
    )

    raw = (
        cfg.w_coverage * coverage_term
        + cfg.w_reliability * reliability
        + cfg.w_recency * recency
        + cfg.w_agreement * agreement
        + cfg.w_evidence * evidence_support
    )

    absence_multiplier = 1.0
    if state is AnswerState.OBSERVED_ABSENCE:
        # An absence claim is only as strong as the observation behind it.
        absence_multiplier = round(covered, 4)

    confidence = raw * (1.0 - contradiction_penalty) * absence_multiplier - stale_penalty

    unknown_ceiling_applied = False
    if state is AnswerState.UNKNOWN:
        # A claim the system cannot substantiate is capped - but it must still *decline*
        # with coverage. A hard ceiling alone would flatten the curve and make "unknown at
        # 75% coverage" indistinguishable from "unknown at 5% coverage", which is exactly
        # the collapse this system exists to prevent.
        confidence = min(confidence, cfg.unknown_ceiling) * (0.5 + 0.5 * covered)
        unknown_ceiling_applied = True

    confidence = max(0.02, min(0.99, confidence))

    return ConfidenceBreakdown(
        coverage=round(coverage_term, 4),
        source_reliability=round(reliability, 4),
        recency=round(recency, 4),
        retrieval_agreement=round(agreement, 4),
        evidence_support=round(evidence_support, 4),
        contradiction_penalty=contradiction_penalty,
        stale_penalty=stale_penalty,
        absence_coverage_multiplier=absence_multiplier,
        unknown_ceiling_applied=unknown_ceiling_applied,
        raw_score=round(raw, 4),
        confidence=round(confidence, 4),
    )


def in_scope(
    contradiction: Contradiction,
    coverage: CoverageReport | None,
    region: str | None,
    entities: list[str] | None = None,
) -> bool:
    """A disagreement only defines the answer if it happened *where and when we asked*.

    Without this, the Grid B7 identity dispute at 05:20 would hijack every question about
    Grid B7 at any other time - including the 04:07-04:11 blind-window question.
    """
    from datetime import timedelta

    from app.dataset import world

    if entities:
        # A question about T-42 is not answered "contradiction" because two other vessels
        # disagreed nearby.
        haystack = " ".join(
            [str(contradiction.entity or "")]
            + [str(c.value) for c in contradiction.claims]
        ).lower()
        if not any(e.lower() in haystack for e in entities):
            return False

    target_region = region or (coverage.region if coverage else None)
    if target_region and not (
        world.region_matches(contradiction.region, target_region)
        or world.region_matches(target_region, contradiction.region)
    ):
        return False
    if coverage is None:
        return True
    slack = timedelta(minutes=10)
    return (
        contradiction.time_range.start <= coverage.time_range.end + slack
        and contradiction.time_range.end >= coverage.time_range.start - slack
    )


def decide_state(
    bundle: EvidenceBundle,
    coverage: CoverageReport | None,
    contradictions: list[Contradiction],
    contradiction_severity: float,
    intent_is_absence: bool,
    region: str | None = None,
    entities: list[str] | None = None,
) -> tuple[AnswerState, list[str]]:
    """Deterministic answer-state decision. The LLM does not get a vote.

    Order of precedence:
      1. A high-severity contradiction dominates - the operator must see it.
      2. Credible in-window presence evidence -> PRESENCE.
      3. No presence evidence: absence is only allowed if the ledger supports it.
      4. Otherwise UNKNOWN.
    """
    reasons: list[str] = []
    presence = bundle.presence
    absence_supported = bool(coverage and coverage.absence_claim_supported)
    in_scope_contradictions = [
        c for c in contradictions if in_scope(c, coverage, region, entities)
    ]

    if in_scope_contradictions and contradiction_severity >= 0.55:
        reasons.append(
            f"contradiction severity {contradiction_severity:.2f} exceeds the reporting "
            "threshold and the disagreement falls inside the queried region and window"
        )
        return AnswerState.CONTRADICTION, reasons
    if contradictions and not in_scope_contradictions:
        reasons.append(
            f"{len(contradictions)} contradiction(s) were detected outside the queried "
            "region/window; reported as context but not allowed to define the answer"
        )

    if presence:
        reasons.append(
            f"{len(presence)} in-window presence record(s) from "
            f"{len({e.source for e in presence})} modality/modalities"
        )
        return AnswerState.PRESENCE, reasons

    if absence_supported:
        if bundle.absence:
            reasons.append(
                "no presence evidence, and the ledger confirms the area/interval was observed"
            )
            return AnswerState.OBSERVED_ABSENCE, reasons
        reasons.append(
            "coverage is sufficient, but no negative report corroborates the empty picture"
        )
        return AnswerState.OBSERVED_ABSENCE, reasons

    reasons.append(
        "insufficient observation coverage to distinguish absence from non-observation"
        + (f": {coverage.absence_block_reason}" if coverage and coverage.absence_block_reason else "")
    )
    return AnswerState.UNKNOWN, reasons
