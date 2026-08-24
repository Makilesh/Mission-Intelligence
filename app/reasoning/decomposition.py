"""Query decomposition: deterministic extraction first, optional LLM refinement second.

Everything that can be computed with a rule is computed with a rule (times, regions,
identifiers, intent). The LLM is only allowed to *add* information to the plan, never to
remove a region, a time bound or a modality that the deterministic layer found - an LLM
that silently narrows the plan is exactly the parent-filter failure mode from spec 8.
"""
from __future__ import annotations

import re
from datetime import timedelta

from app.dataset import world
from app.models.schemas import (
    Modality,
    QueryIntent,
    QueryPlan,
    SubQuery,
    SubQueryType,
    TimeRange,
)

# --------------------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------------------
REGION_ALIASES: dict[str, str] = {
    "sector alpha training annex": "sector_alpha_training_annex",
    "training annex": "sector_alpha_training_annex",
    "sector alpha": "sector_alpha",
    "sector a": "sector_alpha",
    "sector bravo": "sector_bravo",
    "sector b": "sector_bravo",
    "sector charlie": "sector_charlie",
    "sector c": "sector_charlie",
}
for _rid, _region in world.REGIONS.items():
    if _rid.startswith("grid_"):
        short = _rid.split("_", 1)[1]
        REGION_ALIASES[f"grid {short}"] = _rid
        REGION_ALIASES[_rid.replace("_", " ")] = _rid

MODALITY_KEYWORDS: dict[Modality, tuple[str, ...]] = {
    Modality.SURFACE_RADAR: ("radar", "surface search", "track", "sweep"),
    Modality.AIS: ("ais", "transponder", "mmsi", "cooperative"),
    Modality.EO_IR: ("eo", "eo/ir", "ir", "infrared", "electro-optical", "thermal", "visual"),
    Modality.RF: ("rf", "emission", "emissions", "signal", "frequency", "mhz"),
    Modality.IMAGERY: ("imagery", "satellite", "image", "photo"),
    Modality.MISSION_REPORT: ("report", "mission report", "analyst", "watch"),
    Modality.STANDING_ORDER: ("standing order", "orders", "policy", "procedure"),
}

_ABSENCE_PATTERNS = (
    "no contacts",
    "nothing was",
    "can we conclude",
    "conclude that no",
    "were there any",
    "was there any",
    "any contacts",
    "absence",
    "clear of",
    "empty",
)
_ASSOCIATION_PATTERNS = ("same vessel", "same contact", "same as", "is this the", "identical")
_DISAGREEMENT_PATTERNS = ("disagree", "different vessel identities", "conflict", "why are these two", "contradiction")
_IDENTITY_PATTERNS = ("what vessel", "which vessel", "who is", "identify", "identity of", "was vessel")

# --------------------------------------------------------------------------------------
# Extraction helpers
# --------------------------------------------------------------------------------------
_TIME_HHMM = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_MIL = re.compile(r"\b(\d{2})(\d{2})\s*z?\b", re.IGNORECASE)
_RELATIVE = re.compile(r"last\s+(\d+)\s*(minute|minutes|min|hour|hours|hr)\b", re.IGNORECASE)
_ENTITY_TRACK = re.compile(r"\b([VT]-\d{1,3})\b", re.IGNORECASE)
_ENTITY_MMSI = re.compile(r"\b(\d{9})\b")
_ENTITY_DETECTION = re.compile(r"\b((?:EO|RADAR|AIS|RF|IMG|MR)-\d{2,4})\b", re.IGNORECASE)
_ENTITY_VESSEL = re.compile(r"\b((?:MV|FV|MT)\s+[A-Z][A-Z\s]{2,})\b")


def _parse_clock(text: str) -> list[tuple[int, int]]:
    """Collect explicit clock times in either 04:00 or 0400Z form, preserving order."""
    found: list[tuple[int, int, int]] = []  # (position, hh, mm)
    for m in _TIME_HHMM.finditer(text):
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            found.append((m.start(), hh, mm))
    for m in _TIME_MIL.finditer(text):
        # Skip anything already captured as HH:MM and obvious non-times.
        if any(abs(m.start() - pos) < 6 for pos, _, _ in found):
            continue
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59 and m.group(0).lower().endswith("z"):
            found.append((m.start(), hh, mm))
    found.sort()
    return [(hh, mm) for _pos, hh, mm in found]


def extract_time_range(question: str) -> tuple[TimeRange | None, str]:
    """Return (time_range, rationale). Times resolve against the mission clock."""
    q = question.lower()

    rel = _RELATIVE.search(q)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2)
        delta = timedelta(hours=amount) if unit.startswith(("hour", "hr")) else timedelta(minutes=amount)
        return (
            TimeRange(start=world.MISSION_NOW - delta, end=world.MISSION_NOW),
            f"relative window: last {amount} {unit}",
        )

    clocks = _parse_clock(question)
    if len(clocks) >= 2:
        start = world.t(*clocks[0])
        end = world.t(*clocks[1])
        if end < start:
            start, end = end, start
        return TimeRange(start=start, end=end), "explicit start and end times"
    if len(clocks) == 1:
        anchor = world.t(*clocks[0])
        if "after" in q or "since" in q:
            return TimeRange(start=anchor, end=world.MISSION_NOW), "open-ended window after anchor"
        if "before" in q or "prior to" in q or "until" in q:
            return TimeRange(start=world.MISSION_START, end=anchor), "window before anchor"
        return (
            TimeRange(start=anchor - timedelta(minutes=5), end=anchor + timedelta(minutes=5)),
            "point-in-time query widened to +/-5 min",
        )
    return None, "no explicit time reference"


def extract_region(question: str) -> str | None:
    q = question.lower()
    # Longest alias first so "sector alpha training annex" beats "sector alpha".
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return REGION_ALIASES[alias]
    return None


def extract_entities(question: str) -> list[str]:
    out: list[str] = []
    for pattern in (_ENTITY_TRACK, _ENTITY_MMSI, _ENTITY_DETECTION):
        out.extend(m.group(1).upper() for m in pattern.finditer(question))
    out.extend(m.group(1).strip() for m in _ENTITY_VESSEL.finditer(question))
    seen: set[str] = set()
    unique = []
    for e in out:
        if e.upper() in seen:
            continue
        seen.add(e.upper())
        unique.append(e)
    return unique


def extract_modalities(question: str) -> tuple[list[Modality], list[Modality]]:
    """Return (preferred, hard). `hard` is only populated on an explicit operator restriction."""
    q = question.lower()
    preferred: list[Modality] = []
    for modality, keywords in MODALITY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in keywords):
            preferred.append(modality)
    hard: list[Modality] = []
    if re.search(r"\b(only|exclusively|restrict(ed)? to|just)\b", q):
        hard = [m for m in preferred if m in world.MODALITY_ADEQUACY]
    if not preferred:
        preferred = list(world.DEFAULT_QUERY_MODALITIES)
    return preferred, hard


def detect_intent(question: str) -> QueryIntent:
    q = question.lower()
    if any(p in q for p in _DISAGREEMENT_PATTERNS):
        return QueryIntent.EXPLAIN_DISAGREEMENT
    if any(p in q for p in _ASSOCIATION_PATTERNS):
        return QueryIntent.ASSOCIATION
    if any(p in q for p in _IDENTITY_PATTERNS):
        return QueryIntent.IDENTITY_RESOLUTION
    if any(p in q for p in _ABSENCE_PATTERNS):
        return QueryIntent.ABSENCE_CHECK
    if re.search(r"\b(observed|detected|contacts|activity|present)\b", q):
        return QueryIntent.PRESENCE_CHECK
    return QueryIntent.SUMMARY


# --------------------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------------------
def _sq(
    idx: int,
    kind: SubQueryType,
    text: str,
    plan_region: str | None,
    time_range: TimeRange | None,
    entities: list[str],
    preferred: list[Modality],
    hard: list[Modality],
    rationale: str,
) -> SubQuery:
    return SubQuery(
        subquery_id=f"sq{idx}",
        type=kind,
        text=text,
        entities=entities,
        time_range=time_range,
        region=plan_region,
        preferred_modalities=preferred,
        hard_modalities=hard,
        rationale=rationale,
    )


def build_subqueries(plan: QueryPlan) -> list[SubQuery]:
    """One retrieval per evidence channel. Channels are never collapsed into one query."""
    region_name = (
        world.REGIONS[plan.region].name if plan.region in world.REGIONS else (plan.region or "the area of interest")
    )
    window = plan.time_range.label() if plan.time_range else "the mission window"
    entities = plan.entities
    hard = plan.hard_modalities
    subqueries: list[SubQuery] = []
    idx = 0

    def add(kind, text, mods, rationale, time_range=..., region=...):  # type: ignore[no-untyped-def]
        nonlocal idx
        idx += 1
        subqueries.append(
            _sq(
                idx,
                kind,
                text,
                plan.region if region is ... else region,
                plan.time_range if time_range is ... else time_range,
                entities,
                mods,
                hard,
                rationale,
            )
        )

    if plan.intent in (QueryIntent.PRESENCE_CHECK, QueryIntent.ABSENCE_CHECK, QueryIntent.SUMMARY):
        # Independent retrieval per sensing modality: a single blended query lets one
        # dominant modality mask the others.
        for modality, phrase in (
            (Modality.SURFACE_RADAR, "radar surface track sweep contacts"),
            (Modality.AIS, "AIS transponder vessel report"),
            (Modality.EO_IR, "EO/IR thermal visual detection"),
            (Modality.RF, "RF emission signal detection"),
        ):
            add(
                SubQueryType.RETRIEVE_PRESENCE,
                f"{phrase} in {region_name} during {window}",
                [modality],
                f"independent {modality.value} channel",
            )
        add(
            SubQueryType.RETRIEVE_CONTEXT,
            f"mission report analyst assessment for {region_name} during {window}",
            [Modality.MISSION_REPORT, Modality.IMAGERY],
            "narrative context and imagery",
        )
    elif plan.intent is QueryIntent.ASSOCIATION:
        anchor = plan.comparison_targets
        early = anchor[0] if anchor else "earlier"
        late = anchor[1] if len(anchor) > 1 else "later"
        early_tr = (
            TimeRange(start=world.t(*_hhmm(early)) - timedelta(minutes=10), end=world.t(*_hhmm(early)) + timedelta(minutes=10))
            if _hhmm(early)
            else plan.time_range
        )
        late_tr = (
            TimeRange(start=world.t(*_hhmm(late)) - timedelta(minutes=10), end=world.t(*_hhmm(late)) + timedelta(minutes=10))
            if _hhmm(late)
            else plan.time_range
        )
        add(
            SubQueryType.RETRIEVE_TRACK,
            f"surface radar track held at {early} heading speed",
            [Modality.SURFACE_RADAR],
            "the earlier track (hop 1)",
            time_range=early_tr,
            region=None,  # the earlier track may sit in a different sector
        )
        add(
            SubQueryType.RETRIEVE_CURRENT_DETECTION,
            f"detection at {late} in {region_name} heading speed",
            [Modality.SURFACE_RADAR, Modality.EO_IR, Modality.IMAGERY],
            "the later detection (hop 2)",
            time_range=late_tr,
        )
        add(
            SubQueryType.RETRIEVE_TRAJECTORY,
            "track trajectory movement between sectors custody handover gap",
            [Modality.SURFACE_RADAR, Modality.MISSION_REPORT],
            "movement continuity (hop 3)",
            time_range=None,
            region=None,
        )
        add(
            SubQueryType.RETRIEVE_IDENTITY,
            "AIS identity MMSI vessel name mission report identification",
            [Modality.AIS, Modality.MISSION_REPORT],
            "identity resolution (hop 4)",
            time_range=None,
            region=None,
        )
        add(
            SubQueryType.RETRIEVE_ORDERS,
            "standing order track association custody gap identity confirmation",
            [Modality.STANDING_ORDER],
            "applicable standing orders (hop 5)",
            time_range=None,
            region=None,
        )
    else:  # IDENTITY_RESOLUTION / EXPLAIN_DISAGREEMENT
        add(
            SubQueryType.RETRIEVE_CURRENT_DETECTION,
            f"contact detected in {region_name} during {window} heading speed classification",
            [Modality.SURFACE_RADAR, Modality.EO_IR],
            "kinematic picture",
        )
        add(
            SubQueryType.RETRIEVE_IDENTITY,
            f"vessel identity AIS MMSI name in {region_name} during {window}",
            [Modality.AIS],
            "cooperative identity",
        )
        add(
            SubQueryType.RETRIEVE_CONTEXT,
            f"mission report identification assessment {region_name} {window}",
            [Modality.MISSION_REPORT, Modality.IMAGERY],
            "analyst identification",
        )
        add(
            SubQueryType.RETRIEVE_ORDERS,
            "standing order conflicting identity reports escalation",
            [Modality.STANDING_ORDER],
            "applicable standing orders",
        )
    return subqueries


def _hhmm(token: str) -> tuple[int, int] | None:
    m = _TIME_HHMM.fullmatch(token.strip()) or _TIME_MIL.fullmatch(token.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def decompose(question: str) -> QueryPlan:
    intent = detect_intent(question)
    time_range, time_note = extract_time_range(question)
    region = extract_region(question)
    entities = extract_entities(question)
    preferred, hard = extract_modalities(question)

    clocks = _parse_clock(question)
    comparison_targets = [f"{hh:02d}:{mm:02d}" for hh, mm in clocks]

    notes = [f"intent={intent.value}", f"time: {time_note}"]
    if region is None:
        notes.append("no region named: coverage will be evaluated over the whole mission area")
    if hard:
        notes.append(f"operator restricted sources to {[m.value for m in hard]}")

    plan = QueryPlan(
        raw_question=question,
        intent=intent,
        entities=entities,
        region=region,
        time_range=time_range,
        preferred_modalities=preferred,
        hard_modalities=hard,
        comparison_targets=comparison_targets,
        requested_relationship="same_vessel" if intent is QueryIntent.ASSOCIATION else None,
        notes=notes,
    )
    plan.subqueries = build_subqueries(plan)
    return plan
