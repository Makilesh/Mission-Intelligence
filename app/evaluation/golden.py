"""Golden evaluation set.

Ground truth is derived from the *world*, not from the pipeline:

* `expected_state` is either pinned by hand (for the planted cases, so an oracle bug can
  never silently redefine what the right answer is) or computed by :func:`oracle_state`,
  which reads the full record set and the coverage ledger directly.
* `relevant_ids` - the documents a perfect retriever would surface - are computed by an
  oracle filter over the corpus (region, window, modality, kind), again without touching
  the retriever.
* `forbidden_ids` are the planted traps that must not drive the answer.

That separation is what makes the retrieval numbers meaningful: the retriever is scored
against a ground truth it played no part in defining.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from app.config import EVAL_DIR, QUERY_SET_VERSION
from app.coverage.ledger import CoverageLedger, get_ledger
from app.dataset import world
from app.dataset.world import t
from app.models.schemas import AnswerState, Modality, SourceRecord, TimeRange
from app.retrieval.corpus import Corpus, get_corpus

CONTRADICTION_RECORDS = {"RADAR-221", "AIS-1312", "MR-014", "EO-1042"}
TRAP_IDS = {"RADAR-090", "TRAP-802", "TRAP-805", "MR-099", "RF-610"}


@dataclass
class GoldenQuestion:
    qid: str
    question: str
    category: str
    region: str | None
    window: tuple[tuple[int, int], tuple[int, int]] | None
    expected_state: AnswerState
    expects_contradiction: bool = False
    expects_blind_window: bool = False
    expected_coverage_band: tuple[float, float] = (0.0, 1.0)
    relevant_ids: list[str] = field(default_factory=list)
    forbidden_ids: list[str] = field(default_factory=list)
    must_mention_gap: bool = False
    notes: str = ""

    def time_range(self) -> TimeRange | None:
        if self.window is None:
            return None
        return TimeRange(start=t(*self.window[0]), end=t(*self.window[1]))

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "category": self.category,
            "region": self.region,
            "window": [list(self.window[0]), list(self.window[1])] if self.window else None,
            "expected_state": self.expected_state.value,
            "expects_contradiction": self.expects_contradiction,
            "expects_blind_window": self.expects_blind_window,
            "expected_coverage_band": list(self.expected_coverage_band),
            "relevant_ids": self.relevant_ids,
            "forbidden_ids": self.forbidden_ids,
            "must_mention_gap": self.must_mention_gap,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Oracles (read the world directly; never the retriever)
# --------------------------------------------------------------------------------------
def _in_region(record: SourceRecord, region: str | None) -> bool:
    if region is None:
        return True
    return world.region_matches(record.region, region) or world.region_matches(
        region, record.region
    )


def _in_window(record: SourceRecord, window: TimeRange | None, slack_minutes: int = 3) -> bool:
    if window is None:
        return True
    from datetime import timedelta

    slack = timedelta(minutes=slack_minutes)
    if record.modality is Modality.MISSION_REPORT:
        slack = timedelta(minutes=15)
    return (window.start - slack) <= record.timestamp <= (window.end + slack)


def true_contacts(
    corpus: Corpus, region: str | None, window: TimeRange | None
) -> list[SourceRecord]:
    """Records that assert a real contact inside the region/window (the world's truth)."""
    from app.evidence.classifier import is_detection

    return [
        r
        for r in corpus.records
        if is_detection(r)
        and _in_region(r, region)
        and _in_window(r, window, slack_minutes=0)
        and "trap" not in r.tags
        and r.timestamp >= world.MISSION_START
    ]


def relevant_ids(
    corpus: Corpus,
    region: str | None,
    window: TimeRange | None,
    modalities: Sequence[Modality] | None = None,
    include_absence_reports: bool = True,
    extra: Iterable[str] = (),
) -> list[str]:
    out: list[str] = []
    for r in corpus.records:
        if r.record_id in TRAP_IDS:
            continue
        if modalities and r.modality not in modalities:
            continue
        if not _in_region(r, region) or not _in_window(r, window):
            continue
        if r.is_absence_report and not include_absence_reports:
            continue
        if r.modality in (Modality.STANDING_ORDER, Modality.TERRAIN):
            continue
        out.append(r.record_id)
    out.extend(extra)
    return sorted(set(out))


def oracle_state(
    corpus: Corpus,
    ledger: CoverageLedger,
    region: str | None,
    window: TimeRange | None,
) -> AnswerState:
    """The state a perfect system would return, computed from the world."""
    effective_region = region or "mission_area"
    effective_window = window or TimeRange(start=world.MISSION_START, end=world.MISSION_NOW)
    contacts = true_contacts(corpus, region, window)
    if contacts:
        return AnswerState.PRESENCE
    report = ledger.check(effective_region, effective_window)
    return AnswerState.OBSERVED_ABSENCE if report.absence_claim_supported else AnswerState.UNKNOWN


# --------------------------------------------------------------------------------------
# The question set
# --------------------------------------------------------------------------------------
_SPEC: list[dict] = [
    # ---------------- planted true absence (Case A) ---------------------------------
    dict(qid="G01", question="Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
         category="planted_absence", region="sector_alpha", window=((4, 0), (4, 20)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.9, 1.0),
         must_mention_gap=True, forbidden=["RADAR-090", "TRAP-802"],
         notes="Demo 1. Coverage 0.95; the 4-minute Grid B7 gap must still be reported."),
    dict(qid="G02", question="Were there any contacts in Grid A1 between 04:00 and 04:20?",
         category="planted_absence", region="grid_a1", window=((4, 0), (4, 20)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.95, 1.0)),
    dict(qid="G03", question="Was Grid A2 clear of surface contacts between 04:00 and 04:20?",
         category="planted_absence", region="grid_a2", window=((4, 0), (4, 20)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.95, 1.0)),
    dict(qid="G04", question="Were any vessels detected in Grid A3 between 04:00 and 04:20?",
         category="planted_absence", region="grid_a3", window=((4, 0), (4, 20)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.95, 1.0)),
    dict(qid="G05", question="Were there any surface contacts in Sector Alpha between 04:00 and 04:07?",
         category="planted_absence", region="sector_alpha", window=((4, 0), (4, 7)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.95, 1.0),
         notes="Fully covered sub-window: coverage must be higher than G01."),
    dict(qid="G06", question="Were there any contacts in Grid B7 between 04:00 and 04:07?",
         category="planted_absence", region="grid_b7", window=((4, 0), (4, 7)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.9, 1.0)),

    # ---------------- planted blind window (Case B) ---------------------------------
    dict(qid="G07", question="Were there any contacts in Sector Alpha between 04:07 and 04:11?",
         category="blind_window", region="sector_alpha", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.6, 0.85), must_mention_gap=True,
         forbidden=["TRAP-802", "TRAP-805"],
         notes="Demo 2. The training-annex distractors must not carry the answer."),
    dict(qid="G08", question="Were there any contacts in Grid B7 between 04:07 and 04:11?",
         category="blind_window", region="grid_b7", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.0, 0.05), must_mention_gap=True,
         forbidden=["TRAP-802", "TRAP-805"]),
    dict(qid="G09", question="Can we conclude that nothing was present in Grid B7 at 04:09?",
         category="blind_window", region="grid_b7", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.0, 0.4), must_mention_gap=True),
    dict(qid="G10", question="Was Grid B7 observed between 04:07 and 04:11?",
         category="blind_window", region="grid_b7", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.0, 0.05)),
    dict(qid="G11", question="Were there any contacts in Sector Alpha between 04:05 and 04:13?",
         category="blind_window", region="sector_alpha", window=((4, 5), (4, 13)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.7, 0.95), must_mention_gap=True,
         notes="Window straddles the blackout: a whole sub-region is dark for half of it."),

    # ---------------- coverage-limited / partial ------------------------------------
    dict(qid="G12", question="Can we conclude that no contacts were present in Sector C?",
         category="partial_coverage", region="sector_charlie", window=None,
         expected_state=AnswerState.PRESENCE, expected_coverage_band=(0.0, 0.5),
         notes="AIS sees cooperative traffic only; an absence claim must be refused."),
    dict(qid="G13", question="Were there any non-cooperative contacts in Sector Charlie between 04:00 and 04:30?",
         category="partial_coverage", region="sector_charlie", window=((4, 0), (4, 30)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.5)),
    dict(qid="G14", question="Were there any contacts in Grid C1 between 05:00 and 05:20?",
         category="partial_coverage", region="grid_c1", window=((5, 0), (5, 20)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.5)),
    dict(qid="G15", question="Was Sector Alpha observed between 05:40 and 05:55?",
         category="partial_coverage", region="sector_alpha", window=((5, 40), (5, 55)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.2, 0.7),
         notes="Radar degraded by sea clutter after 05:35; EO/IR off station."),
    dict(qid="G16", question="Were there any AIS contacts in Sector Alpha between 04:41 and 04:51?",
         category="partial_coverage", region="sector_alpha", window=((4, 41), (4, 51)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.5, 0.95),
         notes="AIS receiver reset: the AIS channel is blind even though radar is not."),
    dict(qid="G17", question="Were there any RF emissions in Sector Alpha between 04:41 and 04:55?",
         category="partial_coverage", region="sector_alpha", window=((4, 41), (4, 55)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.6),
         notes="RF receiver dropout 04:40-05:00."),

    # ---------------- contradiction (Case C) -----------------------------------------
    dict(qid="G18", question="What vessel was detected near Grid B7?",
         category="contradiction", region="grid_b7", window=((5, 10), (5, 35)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=sorted(CONTRADICTION_RECORDS), forbidden=["MR-099"],
         notes="Demo 3. AIS says V-17, the mission report says V-21."),
    dict(qid="G19", question="What is the identity of the contact in Grid B7 at 05:20?",
         category="contradiction", region="grid_b7", window=((5, 15), (5, 30)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=sorted(CONTRADICTION_RECORDS), forbidden=["MR-099"]),
    dict(qid="G20", question="Why are these two sources reporting different vessel identities in Grid B7 at 05:20?",
         category="contradiction", region="grid_b7", window=((5, 15), (5, 30)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=sorted(CONTRADICTION_RECORDS)),
    dict(qid="G21", question="What heading was the contact in Grid B7 making at 05:20?",
         category="contradiction", region="grid_b7", window=((5, 15), (5, 30)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["RADAR-221", "AIS-1312"]),
    dict(qid="G22", question="How fast was the contact in Grid B7 travelling at 05:20?",
         category="contradiction", region="grid_b7", window=((5, 15), (5, 30)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["RADAR-221", "AIS-1312"]),

    # ---------------- false contradiction (low severity) ------------------------------
    dict(qid="G23", question="What was the RF emission detected in Grid A2 at 05:04?",
         category="false_contradiction", region="grid_a2", window=((5, 0), (5, 10)),
         expected_state=AnswerState.PRESENCE, expects_contradiction=False,
         relevant=["RF-601", "NOTE-701"],
         notes="Two low-reliability sources disagree: flag it, but do not let it dominate."),

    # ---------------- multi-hop (Case E) ----------------------------------------------
    dict(qid="G24", question="Is the vessel detected at 05:20 the same vessel tracked at 04:00?",
         category="multi_hop", region=None, window=((4, 0), (5, 20)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["RADAR-210", "RADAR-221", "MR-021", "AIS-1310", "AIS-1312", "MR-014"],
         notes="Demo 4. Kinematics match, identity is contested, custody was lost."),
    dict(qid="G25", question="Where was track T-42 last held?",
         category="multi_hop", region="sector_bravo", window=((4, 30), (4, 50)),
         expected_state=AnswerState.PRESENCE, relevant=["RADAR-212", "AIS-1311"]),
    dict(qid="G26", question="Was track T-88 in Grid B7 after 05:15?",
         category="multi_hop", region="grid_b7", window=((5, 15), (5, 40)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["RADAR-221", "RADAR-222"]),
    dict(qid="G27", question="Was radar custody continuous between 04:40 and 05:20?",
         category="multi_hop", region="sector_bravo", window=((4, 40), (5, 20)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.3, 0.9), must_mention_gap=True,
         notes="The Bravo/Alpha handover gap must be surfaced."),

    # ---------------- presence ---------------------------------------------------------
    dict(qid="G28", question="What contacts were observed in Grid B1 between 05:00 and 05:10?",
         category="presence", region="grid_b1", window=((5, 0), (5, 10)),
         expected_state=AnswerState.PRESENCE, relevant=["RADAR-230", "AIS-1320"]),
    dict(qid="G29", question="Was vessel MV KESTREL observed in Sector Bravo?",
         category="presence", region="sector_bravo", window=((3, 55), (4, 45)),
         expected_state=AnswerState.PRESENCE, relevant=["AIS-1310", "AIS-1311"]),
    dict(qid="G30", question="What contacts were observed in Grid A1 between 05:30 and 05:40?",
         category="presence", region="grid_a1", window=((5, 30), (5, 40)),
         expected_state=AnswerState.PRESENCE, relevant=["RADAR-240", "AIS-1330"]),
    dict(qid="G31", question="Was vessel V-17 observed after 04:00?",
         category="presence", region=None, window=((4, 0), (5, 40)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["AIS-1312", "MR-014"], forbidden=["MR-099", "RF-610"],
         notes="V-17 appears in a previous-mission report and in a stale RF association."),
    dict(qid="G32", question="What did the imagery pass over Grid B7 at 05:17 show?",
         category="presence", region="grid_b7", window=((5, 12), (5, 22)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         relevant=["IMG-401"]),
    dict(qid="G33", question="What contacts were observed in Sector Alpha during the last 20 minutes?",
         category="presence", region="sector_alpha", window=((5, 20), (5, 40)),
         expected_state=AnswerState.PRESENCE, relevant=["RADAR-240", "AIS-1330"]),
    dict(qid="G34", question="Was there any AIS traffic in Grid C2 between 04:00 and 05:30?",
         category="presence", region="grid_c2", window=((4, 0), (5, 30)),
         expected_state=AnswerState.PRESENCE),
    dict(qid="G35", question="Were there any suspicious RF emissions in Sector Alpha during the last hour?",
         category="presence", region="sector_alpha", window=((4, 40), (5, 40)),
         expected_state=AnswerState.PRESENCE, relevant=["RF-601", "RF-603"]),

    # ---------------- stale evidence ----------------------------------------------------
    dict(qid="G36", question="Was vessel V-17 positively identified in Grid B7 during this mission?",
         category="stale", region="grid_b7", window=((5, 10), (5, 35)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         forbidden=["MR-099"],
         notes="MR-099 is a previous-day identification and must not be treated as current."),
    dict(qid="G37", question="Were there any surface contacts in Sector Alpha at 02:10?",
         category="stale", region="sector_alpha", window=((2, 5), (2, 15)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.05),
         forbidden=[],
         notes="Before the mission window: the ledger has no entries at all."),
    dict(qid="G38", question="What was the last confirmed identity of the Grid B7 contact?",
         category="stale", region="grid_b7", window=((5, 10), (5, 35)),
         expected_state=AnswerState.CONTRADICTION, expects_contradiction=True,
         forbidden=["MR-099"]),
    dict(qid="G39", question="Are there emissions associated with V-17 in Sector Bravo right now?",
         category="stale", region="sector_bravo", window=((5, 20), (5, 40)),
         expected_state=AnswerState.UNKNOWN, forbidden=["RF-610"],
         notes="RF-610 is a historical association from 04:15, not a current detection."),

    # ---------------- retrieval traps -----------------------------------------------------
    dict(qid="G40", question="Was the Sector Alpha sweep at 04:09 negative?",
         category="trap", region="sector_alpha", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         forbidden=["TRAP-802", "TRAP-805"], must_mention_gap=True,
         notes="Near-identical annex records exist for exactly this window."),
    dict(qid="G41", question="Were there contacts in the Sector Alpha Training Annex between 04:07 and 04:11?",
         category="trap", region="sector_alpha_training_annex", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.05),
         notes="The annex is real but has no coverage ledger entries at all -> UNKNOWN."),
    dict(qid="G42", question="Has Grid B7 ever been reported clear of contacts?",
         category="trap", region="grid_b7", window=((3, 30), (5, 40)),
         expected_state=AnswerState.PRESENCE, forbidden=["TRAP-802", "TRAP-805"]),
    dict(qid="G43", question="Was Sector Alpha swept and found empty before 03:00?",
         category="trap", region="sector_alpha", window=((2, 0), (3, 0)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.0, 0.05),
         forbidden=["RADAR-090"],
         notes="RADAR-090 says exactly this, but it predates the mission's coverage."),

    # ---------------- coverage / modality questions ------------------------------------
    dict(qid="G44", question="Which modalities covered Sector Alpha between 04:00 and 04:20?",
         category="coverage", region="sector_alpha", window=((4, 0), (4, 20)),
         expected_state=AnswerState.OBSERVED_ABSENCE, expected_coverage_band=(0.9, 1.0)),
    dict(qid="G45", question="Was EO/IR available over Sector Alpha at 03:35?",
         category="coverage", region="sector_alpha", window=((3, 30), (3, 40)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.5, 1.0),
         notes="The UAV comes on station at 03:45; before that EO/IR has no entries."),
    dict(qid="G46", question="Was Sector Charlie covered by radar at any point in the mission?",
         category="coverage", region="sector_charlie", window=None,
         expected_state=AnswerState.PRESENCE, expected_coverage_band=(0.0, 0.5),
         notes="Radar is asserted NOT_OBSERVED for Charlie - an assertion, not silence."),
    dict(qid="G47", question="Were there any contacts in Grid A3 between 05:35 and 05:40?",
         category="coverage", region="grid_a3", window=((5, 35), (5, 40)),
         expected_state=AnswerState.UNKNOWN, expected_coverage_band=(0.2, 0.8),
         notes="Radar degraded, EO/IR off station."),
    dict(qid="G48", question="What was observed in Grid B2 between 04:15 and 04:25?",
         category="presence", region="grid_b2", window=((4, 15), (4, 25)),
         expected_state=AnswerState.PRESENCE, relevant=["RADAR-211"]),
    dict(qid="G49", question="Were there any contacts in Sector Bravo between 04:52 and 05:12?",
         category="blind_window", region="sector_bravo", window=((4, 52), (5, 12)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.0, 0.6), must_mention_gap=True,
         notes="Radar_02 is down for the custody handover."),
    dict(qid="G50", question="Can we conclude that no contacts were present in Sector Alpha between 04:07 and 04:11?",
         category="blind_window", region="sector_alpha", window=((4, 7), (4, 11)),
         expected_state=AnswerState.UNKNOWN, expects_blind_window=True,
         expected_coverage_band=(0.6, 0.85), must_mention_gap=True,
         forbidden=["TRAP-802", "TRAP-805"]),
]


def build(corpus: Corpus | None = None, ledger: CoverageLedger | None = None) -> list[GoldenQuestion]:
    corpus = corpus or get_corpus()
    ledger = ledger or get_ledger()
    questions: list[GoldenQuestion] = []
    for spec in _SPEC:
        window = spec.get("window")
        tr = (
            TimeRange(start=t(*window[0]), end=t(*window[1]))
            if window
            else None
        )
        explicit = spec.get("relevant")
        ids = (
            sorted(set(explicit))
            if explicit
            else relevant_ids(corpus, spec.get("region"), tr)
        )
        questions.append(
            GoldenQuestion(
                qid=spec["qid"],
                question=spec["question"],
                category=spec["category"],
                region=spec.get("region"),
                window=window,
                expected_state=spec["expected_state"],
                expects_contradiction=spec.get("expects_contradiction", False),
                expects_blind_window=spec.get("expects_blind_window", False),
                expected_coverage_band=tuple(spec.get("expected_coverage_band", (0.0, 1.0))),
                relevant_ids=ids,
                forbidden_ids=sorted(set(spec.get("forbidden", []))),
                must_mention_gap=spec.get("must_mention_gap", False),
                notes=spec.get("notes", ""),
            )
        )
    return questions


def write(path: Path | None = None) -> Path:
    path = path or (EVAL_DIR / "golden.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    questions = build()
    payload = {
        "query_set_version": QUERY_SET_VERSION,
        "count": len(questions),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "questions": [q.to_dict() for q in questions],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    p = write()
    print(f"golden set -> {p} ({len(build())} questions)")
