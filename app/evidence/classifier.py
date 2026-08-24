"""Evidence state classification.

Every retrieved record is turned into a typed `Evidence` object with an explicit epistemic
state *before* any LLM sees it. The critical rule lives here:

    a record that says "no contacts detected" only becomes OBSERVED_ABSENCE if the
    **coverage ledger independently confirms** that the area and interval were observed.

Otherwise the same record becomes PARTIAL_COVERAGE or UNOBSERVED. A sensor cannot vouch
for its own coverage.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from app.coverage.ledger import CoverageLedger
from app.dataset import world
from app.models.schemas import (
    Evidence,
    EvidenceState,
    Modality,
    SourceRecord,
    TimeRange,
)

#: How far after a window a record may still be describing that window.
WINDOW_TOLERANCE = {
    Modality.MISSION_REPORT: timedelta(minutes=15),
    Modality.STANDING_ORDER: timedelta(days=3650),  # always applicable
    Modality.TERRAIN: timedelta(days=3650),
    Modality.IMAGERY: timedelta(minutes=5),
}
DEFAULT_TOLERANCE = timedelta(minutes=2)

STALE_RECENCY_THRESHOLD = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.45


def recency_score(timestamp: datetime, window: TimeRange | None, half_life_minutes: float = 45.0) -> float:
    """1.0 inside the window, exponential decay outside it, ~0 for a previous mission."""
    reference_end = window.end if window else world.MISSION_NOW
    reference_start = window.start if window else world.MISSION_START
    if reference_start <= timestamp <= reference_end:
        return 1.0
    if timestamp > reference_end:
        delta_min = (timestamp - reference_end).total_seconds() / 60.0
    else:
        delta_min = (reference_start - timestamp).total_seconds() / 60.0
    return float(math.pow(0.5, delta_min / half_life_minutes))


def in_window(record: SourceRecord, window: TimeRange | None) -> bool:
    if window is None:
        return True
    tolerance = WINDOW_TOLERANCE.get(record.modality, DEFAULT_TOLERANCE)
    return (window.start - tolerance) <= record.timestamp <= (window.end + tolerance)


def region_relevant(record: SourceRecord, region: str | None) -> bool:
    if region is None:
        return True
    return world.region_matches(record.region, region) or world.region_matches(region, record.region)


def is_detection(record: SourceRecord) -> bool:
    """True only when the record asserts that *something was there*.

    A watch summary saying "no radar was tasked" is a document about the sensor picture,
    not a contact. Counting such documents as presence evidence would let narrative text
    manufacture contacts.
    """
    if record.is_absence_report:
        return False
    if record.modality in (Modality.STANDING_ORDER, Modality.TERRAIN):
        return False
    return any(
        [
            record.track_id,
            record.mmsi,
            record.vessel_name,
            record.object_type,
            record.frequency_mhz is not None,
            record.classification not in (None, "", "pending", "unclassified"),
        ]
    )


def _claim(record: SourceRecord) -> str:
    text = record.text.strip()
    return text if len(text) <= 240 else text[:237] + "..."


def classify(
    record: SourceRecord,
    ledger: CoverageLedger,
    window: TimeRange | None,
    query_region: str | None,
    retrieval_score: float = 0.0,
) -> Evidence:
    notes: list[str] = []
    record_window = TimeRange(
        start=record.timestamp - timedelta(minutes=2), end=record.timestamp + timedelta(minutes=2)
    )
    effective_window = window or record_window
    recency = recency_score(record.timestamp, window)

    # --- base epistemic state ---------------------------------------------------------
    if record.is_absence_report:
        # Ask the ledger, not the record, whether the area was actually observed.
        claim_window = window or record_window
        report = ledger.check(record.region, claim_window, [record.modality])
        if report.absence_claim_supported:
            base = EvidenceState.OBSERVED_ABSENCE
            notes.append(
                f"ledger confirms {report.covered_fraction:.0%} coverage of {record.region} "
                f"for {claim_window.label()}"
            )
        elif report.covered_fraction > 0:
            base = EvidenceState.PARTIAL_COVERAGE
            notes.append(
                f"negative report downgraded: ledger reports only "
                f"{report.covered_fraction:.0%} coverage ({report.absence_block_reason})"
            )
        else:
            base = EvidenceState.UNOBSERVED
            notes.append(
                "negative report rejected as absence evidence: the ledger has no observation "
                f"of {record.region} for {claim_window.label()}"
            )
    elif record.modality in (Modality.STANDING_ORDER, Modality.TERRAIN):
        base = EvidenceState.PRESENCE  # contextual fact, not a sensor observation
        notes.append("contextual document (not a sensor observation)")
    else:
        base = EvidenceState.PRESENCE

    # --- overriding qualifiers ---------------------------------------------------------
    state = base
    quality = record.reliability * record.confidence
    if recency < STALE_RECENCY_THRESHOLD and record.modality not in (
        Modality.STANDING_ORDER,
        Modality.TERRAIN,
    ):
        state = EvidenceState.STALE
        notes.append(
            f"stale: timestamped {record.timestamp.strftime('%Y-%m-%d %H:%M')}Z, outside the "
            "requested window"
        )
    elif quality < LOW_CONFIDENCE_THRESHOLD:
        state = EvidenceState.LOW_CONFIDENCE
        notes.append(
            f"low confidence: reliability {record.reliability:.2f} x confidence "
            f"{record.confidence:.2f} = {quality:.2f}"
        )

    if not region_relevant(record, query_region):
        notes.append(
            f"region {record.region} is outside the queried region {query_region} "
            "(retained for transparency, excluded from the operational claim)"
        )

    return Evidence(
        evidence_id=record.record_id,
        source_id=record.record_id,
        source=record.modality,
        sensor=record.sensor,
        claim=_claim(record),
        state=state,
        region=record.region,
        time_range=record_window,
        reliability=record.reliability,
        recency=round(recency, 4),
        retrieval_score=round(retrieval_score, 6),
        entities=record.entities,
        attributes={
            "base_state": base.value,
            "confidence": record.confidence,
            "heading": record.heading,
            "speed": record.speed,
            "position": list(record.position) if record.position else None,
            "track_id": record.track_id,
            "mmsi": record.mmsi,
            "vessel_name": record.vessel_name,
            "object_type": record.object_type,
            "classification": record.classification,
            "frequency_mhz": record.frequency_mhz,
            "timestamp": record.timestamp.isoformat(),
            "in_window": in_window(record, window),
            "region_relevant": region_relevant(record, query_region),
            "quality": round(quality, 4),
            "effective_window": effective_window.label(),
            "detection": is_detection(record),
        },
        notes=notes,
    )
