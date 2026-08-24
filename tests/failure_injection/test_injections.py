"""Phase 11 gate: the system must break in the right direction, not just keep answering."""
from __future__ import annotations

import asyncio

import pytest

from app.contradiction.engine import detect
from app.coverage.ledger import CoverageLedger
from app.dataset.world import t
from app.evaluation import failure_injection as FI
from app.evidence.classifier import classify
from app.models.schemas import AnswerState, ContradictionDimension, EvidenceState, TimeRange
from app.reasoning.pipeline import answer_question
from app.retrieval.corpus import Corpus
from app.retrieval.hybrid import HybridRetriever, get_retriever

DEMO1 = "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?"


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return get_retriever()


def _answer(question: str, retriever, ledger=None, corpus=None):  # noqa: ANN001
    return asyncio.run(
        answer_question(question, retriever=retriever, ledger=ledger, corpus=corpus)
    )


# ---------------------------------------------------------------- sensor dropout ----
def test_sensor_dropout_degrades_coverage_confidence_and_state(retriever, ledger):
    before = _answer(DEMO1, retriever, ledger=ledger)
    injection = FI.sensor_dropout(ledger, "radar_01", TimeRange(start=t(4, 0), end=t(4, 20)))
    after = _answer(DEMO1, retriever, ledger=injection.ledger)

    assert before.state is AnswerState.OBSERVED_ABSENCE
    assert after.coverage.coverage_fraction < before.coverage.coverage_fraction
    assert after.confidence < before.confidence
    assert after.state is AnswerState.UNKNOWN, "an absence claim must be withdrawn"
    assert after.gaps


def test_sensor_dropout_is_asserted_not_forgotten(ledger):
    """The dropout must produce NOT_OBSERVED entries, never a silent absence of entries."""
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    injected = ledger.with_sensor_dropout("radar_01", window)
    report = injected.check("sector_alpha", window)
    assert report.no_information_fraction == 0.0
    assert any(
        e.coverage_status.value == "NOT_OBSERVED"
        for e in injected.entries_for("grid_a1", window)
    )


# -------------------------------------------------------------------- stale data ----
def test_stale_injection_is_not_promoted(retriever, ledger):
    injection = FI.stale_data()
    poisoned = FI.apply_records(retriever.corpus, injection.records)
    evidence = classify(
        injection.records[0],
        ledger,
        TimeRange(start=t(4, 0), end=t(4, 20)),
        "sector_alpha",
    )
    assert evidence.state is EvidenceState.STALE
    assert evidence.recency < 0.05

    rebuilt = HybridRetriever(poisoned, embedder=retriever.embedder)
    answer = _answer(DEMO1, rebuilt, ledger=ledger, corpus=poisoned)
    assert "INJ-STALE-01" not in answer.meta["claim_evidence_ids"]


# --------------------------------------------------------------- contradictions ----
def _severity(records, ledger, window, region, dimensions):  # noqa: ANN001
    evidence = [classify(r, ledger, window, region) for r in records]
    found = [c for c in detect(evidence) if c.dimension in dimensions]
    return found


def test_false_contradiction_is_low_severity(ledger):
    injection = FI.false_contradiction()
    found = _severity(
        injection.records,
        ledger,
        TimeRange(start=t(5, 20), end=t(5, 30)),
        "grid_a1",
        {ContradictionDimension.IDENTITY, ContradictionDimension.HEADING},
    )
    assert found, "a disagreement between weak sources must still be flagged"
    assert max(c.severity for c in found) < 0.55
    assert all(c.severity_label in ("low", "moderate") for c in found)


def test_true_contradiction_is_high_severity(ledger):
    injection = FI.true_contradiction()
    found = _severity(
        injection.records,
        ledger,
        TimeRange(start=t(5, 20), end=t(5, 30)),
        "grid_a2",
        {ContradictionDimension.HEADING, ContradictionDimension.SPEED},
    )
    assert found
    assert max(c.severity for c in found) >= 0.55
    assert any(c.severity_label == "high" for c in found)


def test_true_contradiction_outranks_false_one(ledger):
    window_a = TimeRange(start=t(5, 20), end=t(5, 30))
    weak = _severity(
        FI.false_contradiction().records, ledger, window_a, "grid_a1",
        {ContradictionDimension.HEADING},
    )
    strong = _severity(
        FI.true_contradiction().records, ledger, window_a, "grid_a2",
        {ContradictionDimension.HEADING},
    )
    assert max(c.severity for c in strong) > max(c.severity for c in weak)


def test_contradictions_are_never_resolved(ledger):
    found = _severity(
        FI.true_contradiction().records,
        ledger,
        TimeRange(start=t(5, 20), end=t(5, 30)),
        "grid_a2",
        {ContradictionDimension.HEADING, ContradictionDimension.SPEED},
    )
    for c in found:
        assert c.resolvable is False
        assert len(c.claims) >= 2, "every conflicting claim must be preserved"


# --------------------------------------------------------- retrieval poisoning ----
def test_retrieval_poisoning_does_not_produce_a_false_absence(retriever, ledger):
    question = "Were there any contacts in Sector Alpha between 04:07 and 04:11?"
    injection = FI.retrieval_poisoning()
    poisoned = FI.apply_records(retriever.corpus, injection.records)
    rebuilt = HybridRetriever(poisoned, embedder=retriever.embedder)

    answer = _answer(question, rebuilt, ledger=ledger, corpus=poisoned)
    retrieved = {d.record_id for d in answer.trace.retrieved}

    assert any(r.startswith("INJ-POISON") for r in retrieved), "the poison should be retrieved"
    assert answer.state is AnswerState.UNKNOWN, "but it must not create an absence claim"
    assert answer.coverage.coverage_fraction < 0.85

    # The poisoned records claim full coverage of Grid B7. The ledger says otherwise, so
    # every one of them must be downgraded out of OBSERVED_ABSENCE. They may still be shown
    # (and even cited as evidence *that coverage was partial*) - what they must never do is
    # support an absence claim.
    poison_states = {
        e["state"] for e in answer.evidence if e["id"].startswith("INJ-POISON")
    }
    assert poison_states, "the poison should reach the evidence listing"
    assert poison_states <= {"PARTIAL_COVERAGE", "UNOBSERVED", "LOW_CONFIDENCE"}
    assert "OBSERVED_ABSENCE" not in poison_states
