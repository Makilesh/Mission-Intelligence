"""Phase 6 and 8 gates: evidence states, and confidence that tracks coverage."""
from __future__ import annotations

import pytest

from app.confidence.model import calculate_confidence
from app.coverage.ledger import CoverageLedger
from app.dataset.world import t
from app.evidence.aggregator import EvidenceBundle
from app.evidence.classifier import classify, is_detection, recency_score
from app.models.schemas import (
    AnswerState,
    Evidence,
    EvidenceState,
    Modality,
    TimeRange,
)


def _record(records, record_id):
    return next(r for r in records if r.record_id == record_id)


# ------------------------------------------------------------------ evidence states ----
def test_negative_report_with_coverage_is_observed_absence(records, ledger):
    ev = classify(
        _record(records, "RADAR-102"),
        ledger,
        TimeRange(start=t(4, 0), end=t(4, 20)),
        "sector_alpha",
    )
    assert ev.state is EvidenceState.OBSERVED_ABSENCE


def test_negative_report_without_coverage_is_not_absence(records, ledger):
    """The same wording over a blind window must NOT become an absence claim."""
    ev = classify(
        _record(records, "RADAR-104"),
        ledger,
        TimeRange(start=t(4, 7), end=t(4, 11)),
        "sector_alpha",
    )
    assert ev.state in (EvidenceState.PARTIAL_COVERAGE, EvidenceState.UNOBSERVED)
    assert ev.state is not EvidenceState.OBSERVED_ABSENCE


def test_trap_region_negative_report_is_unobserved(records, ledger):
    """The training-annex distractor has no ledger coverage at all."""
    ev = classify(
        _record(records, "TRAP-802"),
        ledger,
        TimeRange(start=t(4, 7), end=t(4, 11)),
        "sector_alpha",
    )
    assert ev.state is EvidenceState.UNOBSERVED
    assert ev.attributes["region_relevant"] is False


def test_previous_mission_record_is_stale(records, ledger):
    ev = classify(
        _record(records, "MR-099"),
        ledger,
        TimeRange(start=t(5, 10), end=t(5, 30)),
        "grid_b7",
    )
    assert ev.state is EvidenceState.STALE


def test_unverified_analyst_note_is_low_confidence(records, ledger):
    ev = classify(
        _record(records, "NOTE-701"),
        ledger,
        TimeRange(start=t(5, 0), end=t(5, 10)),
        "grid_a2",
    )
    assert ev.state is EvidenceState.LOW_CONFIDENCE


def test_documents_are_not_detections(records):
    assert is_detection(_record(records, "MR-030")) is False  # "no radar tasked" note
    assert is_detection(_record(records, "SO-001")) is False
    assert is_detection(_record(records, "RADAR-221")) is True
    assert is_detection(_record(records, "RADAR-102")) is False  # negative report


def test_recency_decays_outside_window():
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    assert recency_score(t(4, 10), window) == 1.0
    assert recency_score(t(3, 35), window) < 1.0
    assert recency_score(t(2, 10), window) < recency_score(t(3, 35), window)


# ---------------------------------------------------------------------- confidence ----
def _bundle_with(state: EvidenceState, n: int = 3) -> EvidenceBundle:
    bundle = EvidenceBundle()
    for i in range(n):
        bundle.evidence.append(
            Evidence(
                evidence_id=f"E-{i}",
                source_id=f"E-{i}",
                source=Modality.SURFACE_RADAR,
                sensor="radar_01",
                claim="test",
                state=state,
                region="sector_alpha",
                time_range=TimeRange(start=t(4, 0), end=t(4, 20)),
                reliability=0.9,
                recency=1.0,
                attributes={
                    "base_state": state.value,
                    "confidence": 0.9,
                    "in_window": True,
                    "region_relevant": True,
                    "detection": state is EvidenceState.PRESENCE,
                },
            )
        )
    return bundle


@pytest.mark.parametrize("state", [AnswerState.OBSERVED_ABSENCE, AnswerState.PRESENCE])
def test_confidence_is_monotone_in_coverage(ledger: CoverageLedger, state):
    """THE property: less coverage can never mean more confidence."""
    evidence_state = (
        EvidenceState.OBSERVED_ABSENCE
        if state is AnswerState.OBSERVED_ABSENCE
        else EvidenceState.PRESENCE
    )
    bundle = _bundle_with(evidence_state)
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    confidences = []
    for kept in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        report = ledger.with_coverage_loss(
            kept, region="sector_alpha", window=window
        ).check("sector_alpha", window)
        breakdown = calculate_confidence(state, report, bundle, [], 0.0)
        confidences.append(breakdown.confidence)
    assert confidences == sorted(confidences, reverse=True)
    assert confidences[0] > confidences[-1]


def test_absence_confidence_decays_faster_than_presence(ledger):
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    report = ledger.with_coverage_loss(0.4, region="sector_alpha", window=window).check(
        "sector_alpha", window
    )
    absence = calculate_confidence(
        AnswerState.OBSERVED_ABSENCE, report, _bundle_with(EvidenceState.OBSERVED_ABSENCE), [], 0.0
    )
    presence = calculate_confidence(
        AnswerState.PRESENCE, report, _bundle_with(EvidenceState.PRESENCE), [], 0.0
    )
    assert absence.confidence < presence.confidence


def test_unknown_state_is_capped(ledger):
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    report = ledger.check("sector_alpha", window)
    breakdown = calculate_confidence(
        AnswerState.UNKNOWN, report, _bundle_with(EvidenceState.UNOBSERVED), [], 0.0
    )
    assert breakdown.unknown_ceiling_applied
    assert breakdown.confidence <= 0.45


def test_confidence_strictly_decreases_with_fraction_only(ledger):
    """Isolate the coverage term: hold everything else fixed, vary only the fraction."""
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    base = ledger.check("sector_alpha", window)
    bundle = _bundle_with(EvidenceState.OBSERVED_ABSENCE)
    scores = []
    for fraction in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
        report = base.model_copy(update={"covered_fraction": fraction})
        scores.append(
            calculate_confidence(
                AnswerState.OBSERVED_ABSENCE, report, bundle, [], 0.0
            ).confidence
        )
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_contradiction_reduces_confidence(ledger):
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    report = ledger.check("sector_alpha", window)
    bundle = _bundle_with(EvidenceState.PRESENCE)
    clean = calculate_confidence(AnswerState.PRESENCE, report, bundle, [], 0.0)
    conflicted = calculate_confidence(AnswerState.PRESENCE, report, bundle, [], 0.8)
    assert conflicted.confidence < clean.confidence
    assert conflicted.contradiction_penalty > 0
