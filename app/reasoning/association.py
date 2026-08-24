"""Multi-hop track association (Case E).

"Is the vessel at 05:20 the same one tracked at 04:00?" is not a retrieval question. It
needs an explicit, deterministic comparison of the two hops:

  hop 1  earlier track          -> heading, speed, position, time
  hop 2  later detection        -> heading, speed, position, time
  hop 3  movement plausibility  -> could a vessel at that speed have covered the distance?
  hop 4  identity               -> what do AIS and the mission reports call each contact?
  hop 5  custody                -> did the coverage ledger have continuous observation?

The verdict is computed in Python. The LLM only narrates it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.coverage.ledger import CoverageLedger
from app.dataset import world
from app.evidence.aggregator import EvidenceBundle
from app.models.schemas import CoverageStatus, Evidence, Modality, TimeRange

#: 1 degree of latitude is ~60 nautical miles. Good enough for a synthetic world.
NM_PER_DEGREE = 60.0
HEADING_CONSISTENT_DEG = 25.0
SPEED_CONSISTENT_RELATIVE = 0.25
CUSTODY_GAP_LIMIT = timedelta(minutes=10)  # standing order SO-005


@dataclass
class AssociationAssessment:
    verdict: str = "INSUFFICIENT_EVIDENCE"
    confidence_modifier: float = 0.0
    earlier: Evidence | None = None
    later: Evidence | None = None
    heading_delta: float | None = None
    speed_delta: float | None = None
    required_speed_kn: float | None = None
    kinematically_plausible: bool | None = None
    custody_gaps: list[str] = field(default_factory=list)
    identity_claims: dict[str, str] = field(default_factory=dict)
    identity_conflict: bool = False
    supporting_ids: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "earlier": self.earlier.evidence_id if self.earlier else None,
            "later": self.later.evidence_id if self.later else None,
            "heading_delta_deg": self.heading_delta,
            "speed_delta_kn": self.speed_delta,
            "required_speed_kn": self.required_speed_kn,
            "kinematically_plausible": self.kinematically_plausible,
            "custody_gaps": self.custody_gaps,
            "identity_claims": self.identity_claims,
            "identity_conflict": self.identity_conflict,
            "supporting_evidence": self.supporting_ids,
            "narrative": self.narrative,
        }


def _angular_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _position(e: Evidence) -> tuple[float, float] | None:
    pos = e.attributes.get("position")
    if pos:
        return (float(pos[0]), float(pos[1]))
    region = world.REGIONS.get(e.region)
    return region.centroid if region and region.centroid else None


def _kinematic(evidence: list[Evidence]) -> list[Evidence]:
    return [
        e
        for e in evidence
        if e.attributes.get("heading") is not None
        and e.attributes.get("speed") is not None
        and e.source in (Modality.SURFACE_RADAR, Modality.AIS, Modality.EO_IR)
    ]


def _nearest(evidence: list[Evidence], anchor: datetime) -> Evidence | None:
    if not evidence:
        return None
    return min(evidence, key=lambda e: abs(e.time_range.start - anchor))


def pair_score(earlier: Evidence, later: Evidence) -> float:
    """How well does this candidate pair hang together as one contact?

    Association is a *search over candidate pairs*, not "whichever record happens to sit
    closest to the clock". Picking by timestamp alone would associate the first vessel that
    happened to report at 05:20, regardless of whether it could possibly be the same hull.
    """
    h1, h2 = float(earlier.attributes["heading"]), float(later.attributes["heading"])
    s1, s2 = float(earlier.attributes["speed"]), float(later.attributes["speed"])
    heading_score = 1.0 - _angular_difference(h1, h2) / 180.0
    speed_score = 1.0 - min(1.0, abs(s1 - s2) / max(s1, s2, 1e-6))

    transit_score = 0.5
    p1, p2 = _position(earlier), _position(later)
    elapsed_h = (later.time_range.start - earlier.time_range.start).total_seconds() / 3600.0
    if p1 and p2 and elapsed_h > 0:
        required = (math.dist(p1, p2) * NM_PER_DEGREE) / elapsed_h
        mean_reported = (s1 + s2) / 2.0
        transit_score = 1.0 if required <= mean_reported * 1.35 else max(
            0.0, 1.0 - (required - mean_reported * 1.35) / max(mean_reported, 1e-6)
        )

    identity_score = 0.5
    for key in ("track_id", "vessel_name", "mmsi"):
        va, vb = earlier.attributes.get(key), later.attributes.get(key)
        if va and vb:
            identity_score = 1.0 if str(va).upper() == str(vb).upper() else 0.2
            break

    relevance = (earlier.retrieval_score + later.retrieval_score) / 2.0
    return (
        0.34 * heading_score
        + 0.26 * speed_score
        + 0.25 * transit_score
        + 0.10 * identity_score
        + 0.05 * min(1.0, relevance / 2.0)
    )


def custody_gaps(
    ledger: CoverageLedger, regions: list[str], window: TimeRange
) -> list[str]:
    """Any asserted NOT_OBSERVED interval along the transit invalidates continuous custody."""
    gaps: list[str] = []
    for region in regions:
        for entry in ledger.entries_for(region, window, [Modality.SURFACE_RADAR]):
            if entry.coverage_status is not CoverageStatus.NOT_OBSERVED:
                continue
            overlap = entry.time_range.intersection(window)
            if overlap is None or overlap.duration_seconds <= 0:
                continue
            gaps.append(
                f"{region} radar custody lost {overlap.start.strftime('%H:%M')}-"
                f"{overlap.end.strftime('%H:%M')}Z ({entry.reason or 'no reason recorded'})"
            )
    return sorted(set(gaps))


def assess(
    bundle: EvidenceBundle,
    ledger: CoverageLedger,
    anchors: list[str],
) -> AssociationAssessment:
    result = AssociationAssessment()
    if len(anchors) < 2:
        result.narrative = "The question did not name two comparable times."
        return result

    def parse(label: str) -> datetime:
        hh, mm = (int(x) for x in label.split(":"))
        return world.t(hh, mm)

    t_early, t_late = parse(anchors[0]), parse(anchors[-1])
    candidates = _kinematic([e for e in bundle.evidence if e.attributes.get("detection")])
    early_pool = [e for e in candidates if abs(e.time_range.start - t_early) <= timedelta(minutes=12)]
    late_pool = [e for e in candidates if abs(e.time_range.start - t_late) <= timedelta(minutes=12)]

    if not early_pool or not late_pool:
        result.narrative = (
            "Could not retrieve a kinematic report near "
            f"{'the earlier time' if not early_pool else 'the later time'}."
        )
        return result

    earlier, later, best = None, None, -1.0
    for a in early_pool:
        for b in late_pool:
            if a.evidence_id == b.evidence_id:
                continue
            score = pair_score(a, b)
            if score > best:
                earlier, later, best = a, b, score
    if earlier is None or later is None:
        result.narrative = "No candidate pair could be formed from the retrieved evidence."
        return result

    result.earlier, result.later = earlier, later
    result.supporting_ids = [earlier.evidence_id, later.evidence_id]

    h1 = float(earlier.attributes["heading"])
    h2 = float(later.attributes["heading"])
    s1 = float(earlier.attributes["speed"])
    s2 = float(later.attributes["speed"])
    result.heading_delta = round(_angular_difference(h1, h2), 1)
    result.speed_delta = round(abs(s1 - s2), 1)

    # --- hop 3: is the transit physically possible at the reported speed? --------------
    p1, p2 = _position(earlier), _position(later)
    elapsed_h = max((later.time_range.start - earlier.time_range.start).total_seconds() / 3600.0, 1e-6)
    if p1 and p2:
        distance_nm = math.dist(p1, p2) * NM_PER_DEGREE
        required = distance_nm / elapsed_h
        result.required_speed_kn = round(required, 1)
        mean_reported = (s1 + s2) / 2.0
        result.kinematically_plausible = required <= mean_reported * 1.35

    heading_ok = result.heading_delta <= HEADING_CONSISTENT_DEG
    speed_ok = result.speed_delta <= max(s1, s2) * SPEED_CONSISTENT_RELATIVE

    # --- hop 4: identity ----------------------------------------------------------------
    for e in bundle.evidence:
        name = e.attributes.get("vessel_name") or e.attributes.get("track_id")
        if name and e.attributes.get("region_relevant", True):
            result.identity_claims.setdefault(str(name), e.evidence_id)
    # Identity is only "contested" among reports co-located with the later detection.
    named = {
        str(e.attributes.get("vessel_name"))
        for e in bundle.evidence
        if e.attributes.get("vessel_name")
        and abs(e.time_range.start - later.time_range.start) <= timedelta(minutes=10)
        and (e.region == later.region or world.region_matches(e.region, later.region))
    }
    result.identity_conflict = len(named) > 1

    # --- hop 5: custody ------------------------------------------------------------------
    corridor = sorted({earlier.region, later.region})
    result.custody_gaps = custody_gaps(
        ledger, corridor, TimeRange(start=earlier.time_range.start, end=later.time_range.start)
    )

    # --- verdict --------------------------------------------------------------------------
    if heading_ok and speed_ok and result.kinematically_plausible is not False:
        if result.custody_gaps or result.identity_conflict:
            result.verdict = "PROBABLE_SAME_UNCONFIRMED"
            result.confidence_modifier = -0.10
        else:
            result.verdict = "LIKELY_SAME"
            result.confidence_modifier = 0.05
    elif not heading_ok and not speed_ok:
        result.verdict = "LIKELY_DIFFERENT"
        result.confidence_modifier = -0.05
    else:
        result.verdict = "CANNOT_ASSOCIATE"
        result.confidence_modifier = -0.15

    bits = [
        f"Earlier report [{earlier.evidence_id}] at {earlier.time_range.start.strftime('%H:%M')}Z "
        f"in {earlier.region}: heading {h1:.0f} deg, speed {s1:.1f} kn.",
        f"Later report [{later.evidence_id}] at {later.time_range.start.strftime('%H:%M')}Z "
        f"in {later.region}: heading {h2:.0f} deg, speed {s2:.1f} kn.",
        f"Heading differs by {result.heading_delta:.0f} deg and speed by "
        f"{result.speed_delta:.1f} kn.",
    ]
    if result.required_speed_kn is not None:
        bits.append(
            f"The transit would require {result.required_speed_kn:.1f} kn against a reported "
            f"{(s1 + s2) / 2:.1f} kn, which is "
            f"{'plausible' if result.kinematically_plausible else 'not plausible'}."
        )
    if result.custody_gaps:
        bits.append(
            "Continuous custody could not be demonstrated: " + "; ".join(result.custody_gaps) + "."
        )
    if result.identity_conflict:
        bits.append(
            "Identity is contested near the later time ("
            + ", ".join(sorted(named))
            + "), so the association cannot be confirmed by identity."
        )
    result.narrative = " ".join(bits)
    return result
