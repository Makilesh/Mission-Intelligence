"""Synthetic mission dataset generator.

Produces two independent artefacts:

* the **coverage ledger** (what each sensor did or did not observe, where, when), and
* the **source records** (heterogeneous observations that get indexed for retrieval).

They are deliberately generated separately: the ledger must never be derivable from the
records, because the whole point of the system is that "no record" is not "nothing there".

The generator plants the failure cases required by the spec:

* Case A - true absence  : Sector Alpha 04:00-04:20, fully observed, genuinely empty.
* Case B - blind window  : Grid B7 04:07-04:11, every sensing modality NOT_OBSERVED.
* Case C - contradiction : Grid B7 ~05:20, radar / AIS / EO-IR / mission report disagree.
* Case D - retrieval trap: near-duplicate records from other regions / other days.
* Case E - multi-hop     : T-42 (04:00, Grid B2) -> T-88 (05:20, Grid B7) with a custody gap.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from app.config import (
    COVERAGE_DIR,
    DATASET_VERSION,
    RANDOM_SEED,
    SYNTHETIC_DIR,
    ensure_dirs,
)
from app.dataset import world
from app.dataset.world import SENSORS, atomic_regions, minutes, t
from app.models.schemas import (
    CoverageEntry,
    CoverageStatus,
    Modality,
    SourceRecord,
)

Exception_ = tuple[datetime, datetime, CoverageStatus, float, str]


@dataclass
class SensorPlan:
    sensor: str
    modality: Modality
    grids: list[str]
    base: tuple[datetime, datetime] | None
    base_conf: float
    exceptions: list[Exception_]
    windows: list[tuple[datetime, datetime]] | None = None  # for pass-based sensors


def _split(
    base_start: datetime,
    base_end: datetime,
    base_status: CoverageStatus,
    base_conf: float,
    exceptions: list[Exception_],
) -> list[tuple[datetime, datetime, CoverageStatus, float, str]]:
    """Carve a base interval into segments, applying exception windows on top."""
    cuts = {base_start, base_end}
    for s, e, _st, _c, _r in exceptions:
        if e > base_start and s < base_end:
            cuts.add(max(s, base_start))
            cuts.add(min(e, base_end))
    ordered = sorted(cuts)
    out: list[tuple[datetime, datetime, CoverageStatus, float, str]] = []
    for a, b in zip(ordered, ordered[1:]):
        status, conf, reason = base_status, base_conf, ""
        for s, e, st, c, r in exceptions:
            if s <= a and b <= e:
                status, conf, reason = st, c, r
        if out and out[-1][2] == status and out[-1][3] == conf and out[-1][4] == reason:
            prev = out.pop()
            out.append((prev[0], b, status, conf, reason))
        else:
            out.append((a, b, status, conf, reason))
    return out


def _blackout_exception(grid: str) -> list[Exception_]:
    """Case B: total sensor blackout over Grid B7, 04:07-04:11."""
    if grid != world.BLACKOUT_REGION:
        return []
    s, e = world.BLACKOUT_WINDOW
    return [
        (
            s,
            e,
            CoverageStatus.NOT_OBSERVED,
            0.99,
            "All sensors re-tasked to a SAR event; Grid B7 unobserved for this window.",
        )
    ]


def build_sensor_plans() -> list[SensorPlan]:
    ms, me = world.MISSION_START, world.MISSION_END
    alpha = atomic_regions("sector_alpha")
    bravo = atomic_regions("sector_bravo")
    charlie = atomic_regions("sector_charlie")

    plans: list[SensorPlan] = [
        SensorPlan(
            sensor="radar_01",
            modality=Modality.SURFACE_RADAR,
            grids=alpha,
            base=(ms, me),
            base_conf=0.97,
            exceptions=[
                (t(5, 35), me, CoverageStatus.DEGRADED, 0.55,
                 "Heavy sea clutter; detection threshold raised."),
            ],
        ),
        SensorPlan(
            sensor="radar_02",
            modality=Modality.SURFACE_RADAR,
            grids=bravo,
            base=(ms, me),
            base_conf=0.93,
            exceptions=[
                (*world.HANDOVER_GAP, CoverageStatus.NOT_OBSERVED, 0.95,
                 "Own-ship maneuver; radar down during Bravo/Alpha custody handover."),
            ],
        ),
        SensorPlan(
            sensor="eo_ir_01",
            modality=Modality.EO_IR,
            grids=alpha + ["grid_b1"],
            base=(t(3, 45), t(5, 35)),  # UAV on-station window only
            base_conf=0.90,
            exceptions=[
                (t(4, 25), t(4, 40), CoverageStatus.DEGRADED, 0.50,
                 "Sun glare on the sensor axis."),
            ],
        ),
        SensorPlan(
            sensor="ais_rx_01",
            modality=Modality.AIS,
            grids=alpha + bravo + charlie,
            base=(ms, me),
            base_conf=0.85,
            exceptions=[
                (t(4, 40), t(4, 52), CoverageStatus.NOT_OBSERVED, 0.9,
                 "AIS receiver reset."),
            ],
        ),
        SensorPlan(
            sensor="rf_01",
            modality=Modality.RF,
            grids=alpha + bravo,
            base=(ms, me),
            base_conf=0.70,
            exceptions=[
                (t(4, 40), t(5, 0), CoverageStatus.NOT_OBSERVED, 0.92,
                 "RF receiver dropout."),
            ],
        ),
        SensorPlan(
            sensor="sat_img_01",
            modality=Modality.IMAGERY,
            grids=alpha + bravo + charlie,
            base=None,
            base_conf=0.80,
            exceptions=[],
            windows=[(t(3, 45), t(3, 50)), (t(4, 30), t(4, 35)), (t(5, 15), t(5, 20))],
        ),
        # Sector Charlie has NO radar asset tasked. This is an *asserted* blind area
        # (NOT_OBSERVED), which is different from having no ledger entry at all (UNKNOWN).
        SensorPlan(
            sensor="none_tasked",
            modality=Modality.SURFACE_RADAR,
            grids=charlie,
            base=(ms, me),
            base_conf=0.98,
            exceptions=[
                (ms, me, CoverageStatus.NOT_OBSERVED, 0.98,
                 "No surface radar asset tasked for Sector Charlie this mission."),
            ],
        ),
    ]
    return plans


def build_coverage_entries() -> list[CoverageEntry]:
    entries: list[CoverageEntry] = []
    counter = 0

    def emit(
        grid: str,
        start: datetime,
        end: datetime,
        modality: Modality,
        sensor: str,
        status: CoverageStatus,
        conf: float,
        reason: str,
    ) -> None:
        nonlocal counter
        counter += 1
        entries.append(
            CoverageEntry(
                entry_id=f"COV-{counter:04d}",
                region=grid,
                time_start=start,
                time_end=end,
                modality=modality,
                sensor=sensor,
                coverage_status=status,
                coverage_confidence=conf,
                reason=reason,
            )
        )

    for plan in build_sensor_plans():
        for grid in plan.grids:
            exceptions = list(plan.exceptions) + _blackout_exception(grid)
            if plan.windows:
                for ws, we in plan.windows:
                    for s, e, st, c, r in _split(
                        ws, we, CoverageStatus.OBSERVED, plan.base_conf, exceptions
                    ):
                        emit(grid, s, e, plan.modality, plan.sensor, st, c, r)
                continue
            assert plan.base is not None
            bs, be = plan.base
            for s, e, st, c, r in _split(
                bs, be, CoverageStatus.OBSERVED, plan.base_conf, exceptions
            ):
                emit(grid, s, e, plan.modality, plan.sensor, st, c, r)

    return entries


# --------------------------------------------------------------------------------------
# Source records
# --------------------------------------------------------------------------------------
def _rec(**kwargs) -> SourceRecord:  # type: ignore[no-untyped-def]
    sensor = kwargs.get("sensor")
    if "reliability" not in kwargs and sensor in SENSORS:
        kwargs["reliability"] = SENSORS[sensor].reliability
    return SourceRecord(**kwargs)


def _handcrafted() -> list[SourceRecord]:
    R: list[SourceRecord] = []

    # ---------------- Case A: true absence, Sector Alpha 04:00-04:20 -------------------
    for i, (mm, note) in enumerate(
        [
            (55, "0355Z"),
            (60, "0400Z"),
            (65, "0405Z"),
            (70, "0410Z"),
            (75, "0415Z"),
            (80, "0420Z"),
        ]
    ):
        ts = world.MISSION_START.replace(hour=3, minute=0) + minutes(mm)
        blanked = t(4, 7) <= ts <= t(4, 11)
        text = (
            f"Surface search sweep of Sector Alpha completed at {note}. "
            "No surface contacts held. Sea state 2, no clutter returns above threshold."
        )
        if blanked:
            text += (
                " Grid B7 not scanned this sweep (sector blanking, asset re-tasked); "
                "no assessment possible for Grid B7."
            )
        R.append(
            _rec(
                record_id=f"RADAR-10{i+1}",
                modality=Modality.SURFACE_RADAR,
                sensor="radar_01",
                timestamp=ts,
                region="sector_alpha",
                text=text,
                confidence=0.95,
                is_absence_report=True,
                tags=["case_a", "planted_absence"],
                payload={"sweep": note, "contacts": 0},
            )
        )

    R.append(
        _rec(
            record_id="EO-1201",
            modality=Modality.EO_IR,
            sensor="eo_ir_01",
            timestamp=t(4, 2),
            region="sector_alpha",
            text=(
                "EO/IR sweep of Sector Alpha: no thermal signatures above sea clutter "
                "threshold, no visible surface vessels."
            ),
            confidence=0.88,
            is_absence_report=True,
            tags=["case_a", "planted_absence"],
        )
    )
    R.append(
        _rec(
            record_id="EO-1202",
            modality=Modality.EO_IR,
            sensor="eo_ir_01",
            timestamp=t(4, 13),
            region="sector_alpha",
            text=(
                "EO/IR re-sweep of Sector Alpha at 0413Z: still no thermal signatures or "
                "surface vessels observed."
            ),
            confidence=0.87,
            is_absence_report=True,
            tags=["case_a", "planted_absence"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1301",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(4, 5),
            region="sector_alpha",
            text=(
                "No AIS transponder returns received from within Sector Alpha during the "
                "0400-0420Z window."
            ),
            confidence=0.8,
            is_absence_report=True,
            tags=["case_a", "planted_absence"],
        )
    )
    R.append(
        _rec(
            record_id="MR-001",
            modality=Modality.MISSION_REPORT,
            sensor="watch_officer",
            timestamp=t(4, 22),
            region="sector_alpha",
            text=(
                "Sector Alpha assessed quiet for 0400-0420Z. Radar sweeps negative, EO/IR "
                "negative, no AIS traffic. Caveat: Grid B7 was unobserved 0407-0411Z due to "
                "asset re-tasking; no determination is possible for that grid and interval."
            ),
            confidence=0.9,
            is_absence_report=True,
            tags=["case_a", "case_b", "planted_absence"],
        )
    )

    # ---------------- Case E: multi-hop transit T-42 -> T-88 --------------------------
    transit = [
        ("RADAR-210", t(4, 0), "grid_b2", 148, 17.6, 0.89),
        ("RADAR-211", t(4, 20), "grid_b2", 146, 17.9, 0.9),
        ("RADAR-212", t(4, 40), "grid_b3", 145, 18.0, 0.88),
    ]
    for rid, ts, grid, hdg, spd, conf in transit:
        R.append(
            _rec(
                record_id=rid,
                modality=Modality.SURFACE_RADAR,
                sensor="radar_02",
                timestamp=ts,
                region=grid,
                text=(
                    f"Surface track T-42 held in {grid.replace('_', ' ').title()}: "
                    f"heading {hdg} degrees, speed {spd} knots, medium radar cross-section."
                ),
                confidence=conf,
                entities=["T-42"],
                track_id="T-42",
                heading=float(hdg),
                speed=spd,
                object_type="vessel",
                position=world.REGIONS[grid].centroid,
                tags=["case_e", "trajectory"],
            )
        )

    R.append(
        _rec(
            record_id="AIS-1310",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(4, 0),
            region="grid_b2",
            text=(
                "AIS report: MMSI 412700111, vessel MV KESTREL, position Grid B2, "
                "heading 148 degrees, speed 17.5 knots, navigation status under way."
            ),
            confidence=0.86,
            entities=["MV KESTREL", "412700111"],
            mmsi="412700111",
            vessel_name="MV KESTREL",
            heading=148.0,
            speed=17.5,
            object_type="vessel",
            position=world.REGIONS["grid_b2"].centroid,
            tags=["case_e", "identity"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1311",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(4, 35),
            region="grid_b3",
            text=(
                "AIS report: MMSI 412700111, vessel MV KESTREL, position Grid B3, "
                "heading 145 degrees, speed 17.8 knots."
            ),
            confidence=0.86,
            entities=["MV KESTREL", "412700111"],
            mmsi="412700111",
            vessel_name="MV KESTREL",
            heading=145.0,
            speed=17.8,
            object_type="vessel",
            position=world.REGIONS["grid_b3"].centroid,
            tags=["case_e", "identity"],
        )
    )

    # ---------------- Case C: contradiction cluster, Grid B7 ~05:20 -------------------
    R.append(
        _rec(
            record_id="RADAR-221",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_01",
            timestamp=t(5, 20),
            region="grid_b7",
            text=(
                "Surface track T-88 held in Grid B7: heading 145 degrees, speed 18.2 knots, "
                "small-to-medium radar cross-section, no IFF response."
            ),
            confidence=0.88,
            entities=["T-88"],
            track_id="T-88",
            heading=145.0,
            speed=18.2,
            object_type="vessel",
            position=world.REGIONS["grid_b7"].centroid,
            tags=["case_c", "case_e"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1312",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(5, 20),
            region="grid_b7",
            text=(
                "AIS report: MMSI 412345678, vessel V-17, position Grid B7, heading 310 "
                "degrees, speed 11.4 knots, navigation status under way using engine."
            ),
            confidence=0.83,
            entities=["V-17", "412345678"],
            mmsi="412345678",
            vessel_name="V-17",
            heading=310.0,
            speed=11.4,
            object_type="vessel",
            position=world.REGIONS["grid_b7"].centroid,
            tags=["case_c", "identity"],
        )
    )
    R.append(
        _rec(
            record_id="EO-1042",
            modality=Modality.EO_IR,
            sensor="eo_ir_01",
            timestamp=t(5, 21),
            region="grid_b7",
            text=(
                "EO/IR detection EO-1042: small fast-moving surface vessel in Grid B7, "
                "no visible markings or hull number, unidentified."
            ),
            confidence=0.79,
            entities=["EO-1042"],
            object_type="vessel",
            classification="unidentified",
            position=(12.281, 45.712),
            tags=["case_c"],
        )
    )
    R.append(
        _rec(
            record_id="MR-014",
            modality=Modality.MISSION_REPORT,
            sensor="watch_officer",
            timestamp=t(5, 26),
            region="grid_b7",
            text=(
                "Contact near Grid B7 assessed as vessel V-21 based on hull profile and mast "
                "configuration observed on the 0517Z imagery pass. AIS in the area reports "
                "V-17; correlation between the AIS track and the radar track is not established."
            ),
            confidence=0.72,
            entities=["V-21", "V-17"],
            vessel_name="V-21",
            object_type="vessel",
            tags=["case_c", "identity"],
        )
    )
    R.append(
        _rec(
            record_id="RADAR-222",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_01",
            timestamp=t(5, 32),
            region="grid_b7",
            text=(
                "Surface track T-88 continues in Grid B7: heading 147 degrees, speed 18.0 "
                "knots, steady course."
            ),
            confidence=0.87,
            entities=["T-88"],
            track_id="T-88",
            heading=147.0,
            speed=18.0,
            object_type="vessel",
            position=world.REGIONS["grid_b7"].centroid,
            tags=["case_c", "case_e"],
        )
    )
    R.append(
        _rec(
            record_id="MR-021",
            modality=Modality.MISSION_REPORT,
            sensor="watch_officer",
            timestamp=t(5, 34),
            region="sector_alpha",
            text=(
                "Association note: track T-88 kinematics (145 degrees / 18.2 knots) are "
                "consistent with track T-42 last held in Grid B3 at 0440Z on a 145 degree "
                "course. Radar custody was lost 0452-0512Z during the Bravo/Alpha handover, "
                "so continuous custody cannot be demonstrated. Identity not confirmed."
            ),
            confidence=0.75,
            entities=["T-88", "T-42"],
            tags=["case_e", "association"],
        )
    )
    R.append(
        _rec(
            record_id="IMG-401",
            modality=Modality.IMAGERY,
            sensor="sat_img_01",
            timestamp=t(5, 17),
            region="grid_b7",
            text=(
                "Imagery pass over Grid B7 at 0517Z: one small surface vessel with visible "
                "wake, estimated length 22 metres, heading south-east. No hull markings "
                "resolvable at this GSD."
            ),
            confidence=0.74,
            object_type="vessel",
            position=world.REGIONS["grid_b7"].centroid,
            tags=["case_c", "case_e"],
        )
    )

    # ---------------- False contradiction: two low-reliability sources ----------------
    R.append(
        _rec(
            record_id="RF-601",
            modality=Modality.RF,
            sensor="rf_01",
            timestamp=t(5, 4),
            region="grid_a2",
            text=(
                "RF emission detected in Grid A2 at 9410 MHz, pulse characteristics "
                "consistent with a commercial navigation radar."
            ),
            confidence=0.62,
            frequency_mhz=9410.0,
            classification="commercial_navigation_radar",
            tags=["false_contradiction"],
        )
    )
    R.append(
        _rec(
            record_id="NOTE-701",
            modality=Modality.MISSION_REPORT,
            sensor="analyst_unverified",
            timestamp=t(5, 6),
            region="grid_a2",
            text=(
                "Unverified analyst note: emission in Grid A2 may be a military fire-control "
                "radar. Single-source, not corroborated, low confidence."
            ),
            confidence=0.35,
            classification="military_fire_control",
            tags=["false_contradiction", "low_confidence"],
        )
    )

    # ---------------- Other RF activity ------------------------------------------------
    R.append(
        _rec(
            record_id="RF-602",
            modality=Modality.RF,
            sensor="rf_01",
            timestamp=t(4, 35),
            region="grid_b1",
            text=(
                "Anomalous short-duration emission at 243.0 MHz in Grid B1 (international "
                "distress band). Source not localised beyond grid resolution."
            ),
            confidence=0.55,
            frequency_mhz=243.0,
            classification="distress_band_anomaly",
            tags=["rf"],
        )
    )
    R.append(
        _rec(
            record_id="RF-603",
            modality=Modality.RF,
            sensor="rf_01",
            timestamp=t(5, 32),
            region="grid_a3",
            text=(
                "Unclassified pulsed emission at 1240 MHz in Grid A3. Classification "
                "uncertain; insufficient dwell time for fingerprinting."
            ),
            confidence=0.41,
            frequency_mhz=1240.0,
            classification="unclassified",
            tags=["rf", "low_confidence"],
        )
    )
    R.append(
        _rec(
            record_id="RF-604",
            modality=Modality.RF,
            sensor="rf_01",
            timestamp=t(5, 20),
            region="grid_b7",
            text=(
                "Surface search radar emission at 8900 MHz in Grid B7, consistent with a "
                "small vessel navigation set."
            ),
            confidence=0.66,
            frequency_mhz=8900.0,
            classification="navigation_radar",
            tags=["case_c", "rf"],
        )
    )

    # ---------------- Corroborating (non-contradictory) contacts ----------------------
    R.append(
        _rec(
            record_id="RADAR-230",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_02",
            timestamp=t(5, 5),
            region="grid_b1",
            text=(
                "Surface track T-51 held in Grid B1: heading 090 degrees, speed 8.2 knots, "
                "large radar cross-section."
            ),
            confidence=0.9,
            entities=["T-51"],
            track_id="T-51",
            heading=90.0,
            speed=8.2,
            object_type="vessel",
            tags=["presence"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1320",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(5, 5),
            region="grid_b1",
            text=(
                "AIS report: MMSI 412500222, vessel FV NORTHERN STAR, position Grid B1, "
                "heading 088 degrees, speed 8.0 knots, fishing vessel."
            ),
            confidence=0.85,
            entities=["FV NORTHERN STAR", "412500222"],
            mmsi="412500222",
            vessel_name="FV NORTHERN STAR",
            heading=88.0,
            speed=8.0,
            object_type="vessel",
            tags=["presence", "agreement"],
        )
    )
    R.append(
        _rec(
            record_id="RADAR-240",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_01",
            timestamp=t(5, 36),
            region="grid_a1",
            text=(
                "Slow-moving surface contact T-93 held in Grid A1: heading 220 degrees, "
                "speed 6.1 knots. Detection quality reduced by sea clutter."
            ),
            confidence=0.66,
            entities=["T-93"],
            track_id="T-93",
            heading=220.0,
            speed=6.1,
            object_type="vessel",
            tags=["presence", "degraded"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1330",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(5, 36),
            region="grid_a1",
            text=(
                "AIS report: MMSI 412600333, vessel MV ALDER, position Grid A1, heading 221 "
                "degrees, speed 6.0 knots."
            ),
            confidence=0.84,
            entities=["MV ALDER", "412600333"],
            mmsi="412600333",
            vessel_name="MV ALDER",
            heading=221.0,
            speed=6.0,
            object_type="vessel",
            tags=["presence", "agreement"],
        )
    )
    R.append(
        _rec(
            record_id="AIS-1340",
            modality=Modality.AIS,
            sensor="ais_rx_01",
            timestamp=t(4, 50),
            region="grid_c1",
            text=(
                "AIS report: MMSI 412800444, vessel MV CORAL, position Grid C1, heading 015 "
                "degrees, speed 9.3 knots."
            ),
            confidence=0.84,
            entities=["MV CORAL", "412800444"],
            mmsi="412800444",
            vessel_name="MV CORAL",
            heading=15.0,
            speed=9.3,
            object_type="vessel",
            tags=["presence", "charlie"],
        )
    )
    R.append(
        _rec(
            record_id="MR-030",
            modality=Modality.MISSION_REPORT,
            sensor="watch_officer",
            timestamp=t(5, 0),
            region="sector_charlie",
            text=(
                "Sector Charlie: no organic surface radar tasked for this mission. Only AIS "
                "is available, which sees cooperative traffic exclusively. Any statement about "
                "non-cooperative contacts in Sector Charlie is unsupported."
            ),
            confidence=0.9,
            tags=["charlie", "coverage_note"],
        )
    )

    # ---------------- Case D: retrieval traps -----------------------------------------
    R.append(
        _rec(
            record_id="RADAR-090",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_01",
            timestamp=t(2, 10),
            region="sector_alpha",
            text=(
                "Surface search sweep of Sector Alpha completed at 0210Z. No surface contacts "
                "held. Sea state 2."
            ),
            confidence=0.94,
            is_absence_report=True,
            tags=["trap", "stale"],
        )
    )
    R.append(
        _rec(
            record_id="TRAP-802",
            modality=Modality.SURFACE_RADAR,
            sensor="radar_01",
            timestamp=t(4, 9),
            region="sector_alpha_training_annex",
            text=(
                "Sector Alpha Training Annex: surface sweep negative for the 0407-0411Z window. "
                "No contacts observed during the exercise serial. Sea state 2, no clutter."
            ),
            confidence=0.9,
            is_absence_report=True,
            tags=["trap"],
        )
    )
    R.append(
        _rec(
            record_id="TRAP-805",
            modality=Modality.EO_IR,
            sensor="eo_ir_01",
            timestamp=t(4, 9),
            region="sector_alpha_training_annex",
            text=(
                "EO/IR: no contacts visible in the annex during the 0407-0411Z window. "
                "Clear optics, no thermal returns."
            ),
            confidence=0.86,
            is_absence_report=True,
            tags=["trap"],
        )
    )
    R.append(
        _rec(
            record_id="MR-099",
            modality=Modality.MISSION_REPORT,
            sensor="watch_officer",
            timestamp=t(5, 20, day=21),
            region="grid_b7",
            text=(
                "Vessel V-17 positively identified in Grid B7, heading 145 degrees, speed 18 "
                "knots. Identity confirmed by boarding team."
            ),
            confidence=0.95,
            entities=["V-17"],
            vessel_name="V-17",
            heading=145.0,
            speed=18.0,
            tags=["trap", "stale"],
        )
    )
    R.append(
        _rec(
            record_id="RF-610",
            modality=Modality.RF,
            sensor="rf_01",
            timestamp=t(4, 15),
            region="sector_bravo",
            text=(
                "Emissions previously associated with vessel V-17 detected in Sector Bravo at "
                "9375 MHz. Association is historical and not re-verified."
            ),
            confidence=0.5,
            entities=["V-17"],
            frequency_mhz=9375.0,
            tags=["trap"],
        )
    )

    # ---------------- Standing orders --------------------------------------------------
    orders = [
        (
            "SO-001",
            "Any unidentified surface contact within Sector Alpha must be cross-checked "
            "against AIS and prior mission reports before classification.",
        ),
        (
            "SO-002",
            "Absence of contacts may only be reported for intervals and areas with confirmed "
            "sensor coverage. Unobserved intervals must be reported as unknown.",
        ),
        (
            "SO-003",
            "AIS alone is insufficient to establish absence of non-cooperative vessels; "
            "corroboration by radar, EO/IR or imagery is required.",
        ),
        (
            "SO-004",
            "Conflicting identity reports must be escalated to the watch officer and reported "
            "as unresolved until corroborated by an independent source.",
        ),
        (
            "SO-005",
            "Track custody gaps exceeding 10 minutes invalidate automatic track association; "
            "association must be reported as inferred, not confirmed.",
        ),
    ]
    for oid, text in orders:
        R.append(
            _rec(
                record_id=oid,
                modality=Modality.STANDING_ORDER,
                sensor="j3_orders",
                timestamp=world.MISSION_START,
                region="sector_alpha",
                text=text,
                confidence=1.0,
                tags=["standing_order"],
            )
        )

    # ---------------- Terrain / imagery metadata --------------------------------------
    for rid, region in enumerate(
        ["sector_alpha", "sector_bravo", "sector_charlie", "grid_b7", "grid_c1"], start=1
    ):
        reg = world.REGIONS[region]
        R.append(
            _rec(
                record_id=f"TERR-{rid:03d}",
                modality=Modality.TERRAIN,
                sensor="geo_cell",
                timestamp=world.MISSION_START,
                region=region,
                text=(
                    f"{reg.name} terrain characteristics: {reg.terrain.replace('_', ' ')}. "
                    f"{reg.notes}"
                ),
                confidence=0.9,
                tags=["terrain"],
            )
        )

    return R


def _filler(rng: random.Random) -> list[SourceRecord]:
    """Background traffic. Never places a contact inside a planted-absence window."""
    out: list[SourceRecord] = []
    alpha = atomic_regions("sector_alpha")
    bravo = atomic_regions("sector_bravo")
    charlie = atomic_regions("sector_charlie")
    all_grids = alpha + bravo + charlie

    abs_start, abs_end = world.TRUE_ABSENCE_WINDOW
    bo_start, bo_end = world.BLACKOUT_WINDOW

    def forbidden(region: str, ts: datetime, absence_ok: bool) -> bool:
        # Protect Case A: no *contact* may exist in Sector Alpha 04:00-04:20.
        if not absence_ok and region in alpha and abs_start <= ts <= abs_end:
            return True
        # Protect Case B: nothing at all was observed in Grid B7 during the blackout.
        if region == world.BLACKOUT_REGION and bo_start <= ts < bo_end:
            return True
        return False

    def rand_ts() -> datetime:
        span = int((world.MISSION_END - world.MISSION_START).total_seconds() // 60)
        return world.MISSION_START + timedelta(minutes=rng.randrange(0, span))

    names = [
        "MV LARCH", "FV SEA SWALLOW", "MV BASALT", "MT ORION TRADER", "FV BLUE HERON",
        "MV SANDPIPER", "MV TIDEWATER", "FV GANNET", "MV QUARTZ", "MT CEDAR POINT",
    ]

    idx = 0
    # Routine radar sweeps and tracks
    for _ in range(38):
        idx += 1
        grid = rng.choice(all_grids)
        ts = rand_ts()
        if grid in charlie:
            continue  # no radar in Charlie, by design
        sensor = "radar_01" if grid in alpha else "radar_02"
        negative = rng.random() < 0.55
        if forbidden(grid, ts, absence_ok=negative):
            continue
        if negative:
            out.append(
                _rec(
                    record_id=f"RADAR-3{idx:03d}",
                    modality=Modality.SURFACE_RADAR,
                    sensor=sensor,
                    timestamp=ts,
                    region=grid,
                    text=(
                        f"Routine surface sweep of {grid.replace('_', ' ').title()} at "
                        f"{ts.strftime('%H%M')}Z: no new contacts held."
                    ),
                    confidence=round(rng.uniform(0.82, 0.96), 2),
                    is_absence_report=True,
                    tags=["filler"],
                )
            )
        else:
            track = f"T-{rng.randrange(10, 99)}"
            hdg = round(rng.uniform(0, 359), 0)
            spd = round(rng.uniform(3, 22), 1)
            out.append(
                _rec(
                    record_id=f"RADAR-3{idx:03d}",
                    modality=Modality.SURFACE_RADAR,
                    sensor=sensor,
                    timestamp=ts,
                    region=grid,
                    text=(
                        f"Surface track {track} held in {grid.replace('_', ' ').title()}: "
                        f"heading {int(hdg)} degrees, speed {spd} knots."
                    ),
                    confidence=round(rng.uniform(0.7, 0.95), 2),
                    entities=[track],
                    track_id=track,
                    heading=hdg,
                    speed=spd,
                    object_type="vessel",
                    tags=["filler"],
                )
            )

    # AIS traffic
    for _ in range(34):
        idx += 1
        grid = rng.choice(all_grids)
        ts = rand_ts()
        if forbidden(grid, ts, absence_ok=False):
            continue
        name = rng.choice(names)
        mmsi = str(412000000 + rng.randrange(100000, 999999))
        hdg = round(rng.uniform(0, 359), 0)
        spd = round(rng.uniform(2, 19), 1)
        out.append(
            _rec(
                record_id=f"AIS-3{idx:03d}",
                modality=Modality.AIS,
                sensor="ais_rx_01",
                timestamp=ts,
                region=grid,
                text=(
                    f"AIS report: MMSI {mmsi}, vessel {name}, position "
                    f"{grid.replace('_', ' ').title()}, heading {int(hdg)} degrees, "
                    f"speed {spd} knots."
                ),
                confidence=round(rng.uniform(0.78, 0.9), 2),
                entities=[name, mmsi],
                mmsi=mmsi,
                vessel_name=name,
                heading=hdg,
                speed=spd,
                object_type="vessel",
                tags=["filler"],
            )
        )

    # EO/IR detections
    for _ in range(16):
        idx += 1
        grid = rng.choice(alpha + ["grid_b1"])
        ts = rand_ts()
        if not (t(3, 45) <= ts <= t(5, 35)):
            continue
        if forbidden(grid, ts, absence_ok=False):
            continue
        out.append(
            _rec(
                record_id=f"EO-3{idx:03d}",
                modality=Modality.EO_IR,
                sensor="eo_ir_01",
                timestamp=ts,
                region=grid,
                text=(
                    f"EO/IR detection in {grid.replace('_', ' ').title()}: "
                    f"{rng.choice(['small surface vessel', 'fishing craft', 'wake signature', 'thermal contact'])}"
                    ", classification pending."
                ),
                confidence=round(rng.uniform(0.55, 0.9), 2),
                object_type="vessel",
                classification="pending",
                tags=["filler"],
            )
        )

    # RF background
    for _ in range(14):
        idx += 1
        grid = rng.choice(alpha + bravo)
        ts = rand_ts()
        if forbidden(grid, ts, absence_ok=False):
            continue
        freq = round(rng.choice([156.8, 243.0, 1240.0, 8900.0, 9410.0, 9375.0]), 1)
        out.append(
            _rec(
                record_id=f"RF-3{idx:03d}",
                modality=Modality.RF,
                sensor="rf_01",
                timestamp=ts,
                region=grid,
                text=(
                    f"RF emission at {freq} MHz detected in {grid.replace('_', ' ').title()}, "
                    f"classified as {rng.choice(['maritime VHF voice', 'navigation radar', 'unclassified'])}."
                ),
                confidence=round(rng.uniform(0.4, 0.75), 2),
                frequency_mhz=freq,
                tags=["filler"],
            )
        )

    # Mission reports
    for _ in range(14):
        idx += 1
        region = rng.choice(["sector_alpha", "sector_bravo", "sector_charlie"])
        ts = rand_ts()
        out.append(
            _rec(
                record_id=f"MR-3{idx:03d}",
                modality=Modality.MISSION_REPORT,
                sensor="watch_officer",
                timestamp=ts,
                region=region,
                text=(
                    f"Watch summary for {world.REGIONS[region].name} at {ts.strftime('%H%M')}Z: "
                    f"{rng.choice(['routine traffic', 'no significant activity', 'increased fishing activity', 'weather deteriorating'])}. "
                    "Analyst note: assessment based on available sensor picture only."
                ),
                confidence=round(rng.uniform(0.6, 0.9), 2),
                tags=["filler"],
            )
        )

    # Imagery passes
    for i, (ws, _we) in enumerate([(t(3, 45), None), (t(4, 30), None), (t(5, 15), None)]):
        for grid in ["grid_a1", "grid_b2", "grid_c2"]:
            idx += 1
            if forbidden(grid, ws, absence_ok=True):
                continue
            out.append(
                _rec(
                    record_id=f"IMG-3{idx:03d}",
                    modality=Modality.IMAGERY,
                    sensor="sat_img_01",
                    timestamp=ws,
                    region=grid,
                    text=(
                        f"Imagery pass {i+1} over {grid.replace('_', ' ').title()} at "
                        f"{ws.strftime('%H%M')}Z: no vessels of interest resolved."
                    ),
                    confidence=0.72,
                    is_absence_report=True,
                    tags=["filler"],
                )
            )

    return out


def build_records() -> list[SourceRecord]:
    rng = random.Random(RANDOM_SEED)
    records = _handcrafted() + _filler(rng)
    seen: set[str] = set()
    unique: list[SourceRecord] = []
    for r in records:
        if r.record_id in seen:
            continue
        seen.add(r.record_id)
        unique.append(r)
    unique.sort(key=lambda r: (r.timestamp, r.record_id))
    return unique


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def _dump(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def generate(write: bool = True) -> tuple[list[SourceRecord], list[CoverageEntry]]:
    ensure_dirs()
    records = build_records()
    coverage = build_coverage_entries()
    if write:
        _dump(SYNTHETIC_DIR / "records.json", [r.model_dump(mode="json") for r in records])
        _dump(COVERAGE_DIR / "ledger.json", [c.model_dump(mode="json") for c in coverage])
        by_modality: dict[str, list[dict]] = {}
        for r in records:
            by_modality.setdefault(r.modality.value, []).append(r.model_dump(mode="json"))
        for modality, rows in by_modality.items():
            _dump(SYNTHETIC_DIR / f"{modality}.json", rows)
        manifest = {
            "dataset_version": DATASET_VERSION,
            "records": len(records),
            "coverage_entries": len(coverage),
            "regions": len(world.REGIONS),
            "sensors": len(SENSORS),
            "mission_start": world.MISSION_START.isoformat(),
            "mission_end": world.MISSION_END.isoformat(),
            "mission_now": world.MISSION_NOW.isoformat(),
            "planted_cases": {
                "case_a_true_absence": {
                    "region": world.TRUE_ABSENCE_REGION,
                    "window": [w.isoformat() for w in world.TRUE_ABSENCE_WINDOW],
                },
                "case_b_blind_window": {
                    "region": world.BLACKOUT_REGION,
                    "window": [w.isoformat() for w in world.BLACKOUT_WINDOW],
                },
                "case_c_contradiction": {"region": "grid_b7", "time": "05:20"},
                "case_d_traps": ["RADAR-090", "TRAP-802", "TRAP-805", "MR-099", "RF-610"],
                "case_e_multihop": ["RADAR-210", "RADAR-212", "RADAR-221", "AIS-1310", "MR-021"],
            },
            "modality_counts": {k: len(v) for k, v in sorted(by_modality.items())},
        }
        (SYNTHETIC_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return records, coverage


if __name__ == "__main__":  # pragma: no cover
    recs, cov = generate()
    print(f"records={len(recs)} coverage_entries={len(cov)}")
