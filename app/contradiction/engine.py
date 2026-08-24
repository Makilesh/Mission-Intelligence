"""Deterministic contradiction detection.

Rules, not vibes. Evidence that plausibly describes the same contact (same area, times
chained within a few minutes) is compared across identity, position, heading, speed,
classification and timestamp.

Two principles:

* **Cross-source only.** Two AIS reports naming two different vessels is two vessels, not a
  contradiction. A contradiction requires at least two *different* sources to disagree.
* **Never resolved.** The engine returns every conflicting claim with its source and
  reliability, and refuses to pick a winner. No majority vote, no highest-score wins.

Severity scales with the *magnitude* of the disagreement and with the *quality of the
weakest* participating source, so two unreliable sources disagreeing is a low-severity
flag while two trusted sources disagreeing is high severity.
"""
from __future__ import annotations

import math
from datetime import timedelta

from app.models.schemas import (
    Contradiction,
    ContradictionClaim,
    ContradictionDimension,
    Evidence,
    EvidenceState,
    Modality,
    TimeRange,
)

MAX_CLUSTER_GAP = timedelta(minutes=8)
HEADING_TOLERANCE_DEG = 45.0
SPEED_RELATIVE_TOLERANCE = 0.40
SPEED_ABSOLUTE_TOLERANCE_KN = 3.0
POSITION_TOLERANCE_DEG = 0.05

NON_COMMITTAL = {"pending", "unclassified", "unidentified", "unknown", "none", ""}

DIMENSION_WEIGHT = {
    ContradictionDimension.IDENTITY: 1.00,
    ContradictionDimension.HEADING: 0.95,
    ContradictionDimension.CLASSIFICATION: 0.80,
    ContradictionDimension.SPEED: 0.70,
    ContradictionDimension.POSITION: 0.85,
    ContradictionDimension.TIMESTAMP: 0.50,
}

SEVERITY_HIGH = 0.55
SEVERITY_MODERATE = 0.28


def _quality(e: Evidence) -> float:
    return float(e.reliability) * float(e.attributes.get("confidence", 1.0) or 1.0)


def _ts(e: Evidence):  # noqa: ANN202
    return e.time_range.start


def _angular_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _eligible(evidence: list[Evidence]) -> list[Evidence]:
    """Stale evidence and coverage-gap markers never participate in contradictions."""
    return [
        e
        for e in evidence
        if e.state is not EvidenceState.STALE
        and e.attributes.get("kind") != "coverage_gap"
        and e.attributes.get("kind") != "missing_modality"
        and e.source not in (Modality.STANDING_ORDER, Modality.TERRAIN)
        and e.attributes.get("region_relevant", True)
    ]


def cluster(evidence: list[Evidence]) -> list[list[Evidence]]:
    """Chain evidence into candidate same-contact clusters by region and time proximity."""
    buckets: dict[str, list[Evidence]] = {}
    for e in _eligible(evidence):
        buckets.setdefault(e.region, []).append(e)

    clusters: list[list[Evidence]] = []
    for _region, items in buckets.items():
        items.sort(key=_ts)
        current: list[Evidence] = []
        for e in items:
            if current and (_ts(e) - _ts(current[-1])) > MAX_CLUSTER_GAP:
                clusters.append(current)
                current = []
            current.append(e)
        if current:
            clusters.append(current)
    return [c for c in clusters if len(c) > 1]


def _claims(items: list[Evidence], value_fn) -> list[tuple[Evidence, object]]:  # noqa: ANN001
    out = []
    for e in items:
        value = value_fn(e)
        if value is None:
            continue
        out.append((e, value))
    return out


def _make(
    dimension: ContradictionDimension,
    items: list[tuple[Evidence, object]],
    magnitude: float,
    reason: str,
    index: int,
) -> Contradiction:
    qualities = [_quality(e) for e, _ in items]
    weakest = min(qualities)
    severity = max(0.0, min(1.0, magnitude * weakest * DIMENSION_WEIGHT[dimension]))
    label = "high" if severity >= SEVERITY_HIGH else "moderate" if severity >= SEVERITY_MODERATE else "low"
    start = min(_ts(e) for e, _ in items)
    end = max(_ts(e) for e, _ in items)
    entity = next(
        (
            str(e.attributes.get("track_id") or e.attributes.get("vessel_name"))
            for e, _ in items
            if e.attributes.get("track_id") or e.attributes.get("vessel_name")
        ),
        None,
    )
    return Contradiction(
        contradiction_id=f"CON-{index:02d}",
        dimension=dimension,
        entity=entity,
        region=items[0][0].region,
        time_range=TimeRange(start=start, end=end),
        claims=[
            ContradictionClaim(
                evidence_id=e.evidence_id,
                source=e.source,
                sensor=e.sensor,
                value=value,
                reliability=round(_quality(e), 4),
            )
            for e, value in items
        ],
        reason=reason,
        severity=round(severity, 4),
        severity_label=label,  # type: ignore[arg-type]
        resolvable=False,
    )


def _cross_source(items: list[tuple[Evidence, object]]) -> bool:
    return len({e.source for e, _ in items}) >= 2


def detect(evidence: list[Evidence]) -> list[Contradiction]:
    contradictions: list[Contradiction] = []
    index = 0

    for group in cluster(evidence):
        # ---------------- identity ----------------------------------------------------
        identity = _claims(
            group,
            lambda e: (e.attributes.get("vessel_name") or e.attributes.get("mmsi")) or None,
        )
        identity = [(e, v) for e, v in identity if str(v).strip().lower() not in NON_COMMITTAL]
        distinct = {str(v).upper() for _e, v in identity}
        if len(distinct) > 1 and _cross_source(identity):
            index += 1
            contradictions.append(
                _make(
                    ContradictionDimension.IDENTITY,
                    identity,
                    magnitude=1.0,
                    reason=(
                        "Sources disagree on the identity of the contact: "
                        + " vs ".join(sorted(distinct))
                    ),
                    index=index,
                )
            )

        # ---------------- heading -----------------------------------------------------
        headings = _claims(group, lambda e: e.attributes.get("heading"))
        if len(headings) > 1 and _cross_source(headings):
            worst = 0.0
            pair = None
            for i in range(len(headings)):
                for j in range(i + 1, len(headings)):
                    d = _angular_difference(float(headings[i][1]), float(headings[j][1]))  # type: ignore[arg-type]
                    if d > worst:
                        worst, pair = d, (headings[i], headings[j])
            if worst > HEADING_TOLERANCE_DEG and pair:
                index += 1
                contradictions.append(
                    _make(
                        ContradictionDimension.HEADING,
                        list(pair),
                        magnitude=min(1.0, worst / 180.0),
                        reason=(
                            f"Reported headings differ by {worst:.0f} degrees "
                            f"({pair[0][1]} vs {pair[1][1]})"
                        ),
                        index=index,
                    )
                )

        # ---------------- speed -------------------------------------------------------
        speeds = _claims(group, lambda e: e.attributes.get("speed"))
        if len(speeds) > 1 and _cross_source(speeds):
            lo = min(speeds, key=lambda kv: float(kv[1]))  # type: ignore[arg-type]
            hi = max(speeds, key=lambda kv: float(kv[1]))  # type: ignore[arg-type]
            diff = float(hi[1]) - float(lo[1])  # type: ignore[arg-type]
            rel = diff / max(float(hi[1]), 1e-6)  # type: ignore[arg-type]
            if diff >= SPEED_ABSOLUTE_TOLERANCE_KN and rel >= SPEED_RELATIVE_TOLERANCE:
                index += 1
                contradictions.append(
                    _make(
                        ContradictionDimension.SPEED,
                        [lo, hi],
                        magnitude=min(1.0, rel),
                        reason=f"Reported speeds differ by {diff:.1f} knots ({lo[1]} vs {hi[1]})",
                        index=index,
                    )
                )

        # ---------------- classification ----------------------------------------------
        classes = _claims(group, lambda e: e.attributes.get("classification"))
        classes = [(e, v) for e, v in classes if str(v).strip().lower() not in NON_COMMITTAL]
        distinct_classes = {str(v).lower() for _e, v in classes}
        if len(distinct_classes) > 1 and _cross_source(classes):
            index += 1
            contradictions.append(
                _make(
                    ContradictionDimension.CLASSIFICATION,
                    classes,
                    magnitude=0.9,
                    reason=(
                        "Sources disagree on classification: " + " vs ".join(sorted(distinct_classes))
                    ),
                    index=index,
                )
            )

        # ---------------- position ----------------------------------------------------
        positions = _claims(group, lambda e: e.attributes.get("position"))
        if len(positions) > 1 and _cross_source(positions):
            worst = 0.0
            pair = None
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    a, b = positions[i][1], positions[j][1]  # type: ignore[assignment]
                    d = math.dist(tuple(a), tuple(b))  # type: ignore[arg-type]
                    if d > worst:
                        worst, pair = d, (positions[i], positions[j])
            if worst > POSITION_TOLERANCE_DEG and pair:
                index += 1
                contradictions.append(
                    _make(
                        ContradictionDimension.POSITION,
                        list(pair),
                        magnitude=min(1.0, worst / (POSITION_TOLERANCE_DEG * 4)),
                        reason=f"Reported positions differ by {worst:.3f} degrees",
                        index=index,
                    )
                )

    contradictions.sort(key=lambda c: -c.severity)
    return contradictions


def mark_evidence(evidence: list[Evidence], contradictions: list[Contradiction]) -> None:
    """Annotate participating evidence in place (base_state is preserved in attributes)."""
    involved: dict[str, list[str]] = {}
    for c in contradictions:
        for claim in c.claims:
            involved.setdefault(claim.evidence_id, []).append(c.contradiction_id)
    for e in evidence:
        ids = involved.get(e.evidence_id)
        if not ids:
            continue
        e.attributes["contradiction_ids"] = ids
        e.notes.append(f"participates in contradiction(s): {', '.join(ids)}")
        if e.state in (EvidenceState.PRESENCE, EvidenceState.OBSERVED_ABSENCE):
            e.state = EvidenceState.CONTRADICTION


def aggregate_severity(contradictions: list[Contradiction]) -> float:
    """Overall contradiction pressure on the answer: dominated by the worst conflict."""
    if not contradictions:
        return 0.0
    top = max(c.severity for c in contradictions)
    extra = sum(c.severity for c in contradictions) - top
    return round(min(1.0, top + 0.15 * extra), 4)
