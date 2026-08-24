"""Failure injection framework.

Deliberately break the system and assert that it degrades in the *right direction*. Each
injector returns a modified world (ledger and/or corpus); the harness runs the same
question before and after and reports the deltas.

Injectors
---------
sensor_dropout      disable a sensor for a window  -> coverage down, confidence down,
                                                      absence claims withdrawn
stale_data          make old evidence look highly relevant -> recency penalty, no promotion
false_contradiction two low-quality sources disagree      -> flagged, low severity
true_contradiction  two reliable sources disagree         -> flagged, high severity
retrieval_poisoning near-duplicate but wrong records      -> retrieved but not believed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from app.coverage.ledger import CoverageLedger
from app.dataset.world import t
from app.models.schemas import Modality, SourceRecord, TimeRange
from app.retrieval.corpus import Corpus


@dataclass
class Injection:
    name: str
    description: str
    ledger: CoverageLedger | None = None
    records: list[SourceRecord] = field(default_factory=list)
    expectation: str = ""


def _record(**kwargs: Any) -> SourceRecord:
    return SourceRecord(**kwargs)


# --------------------------------------------------------------------------------------
def sensor_dropout(
    ledger: CoverageLedger,
    sensor: str = "radar_01",
    window: TimeRange | None = None,
    regions: list[str] | None = None,
) -> Injection:
    window = window or TimeRange(start=t(4, 0), end=t(4, 20))
    return Injection(
        name="sensor_dropout",
        description=f"{sensor} offline {window.label()}Z over {regions or 'its whole footprint'}",
        ledger=ledger.with_sensor_dropout(sensor, window, regions),
        expectation="coverage decreases, confidence decreases, an absence claim becomes UNKNOWN",
    )


def stale_data() -> Injection:
    """An old, highly relevant-looking record placed where a naive retriever will love it."""
    return Injection(
        name="stale_data",
        description="a previous-mission sweep report worded exactly like the current ones",
        records=[
            _record(
                record_id="INJ-STALE-01",
                modality=Modality.SURFACE_RADAR,
                sensor="radar_01",
                timestamp=t(4, 5, day=20),
                region="sector_alpha",
                text=(
                    "Surface search sweep of Sector Alpha completed at 0405Z. No surface "
                    "contacts held. Sea state 2, no clutter returns above threshold. "
                    "Sector Alpha assessed quiet."
                ),
                reliability=0.94,
                confidence=0.97,
                is_absence_report=True,
                tags=["injected", "stale"],
            )
        ],
        expectation="classified STALE, excluded from the claim, recency penalty applied",
    )


def false_contradiction() -> Injection:
    """Two low-reliability sources disagree. Should be flagged, but weakly."""
    return Injection(
        name="false_contradiction",
        description="two unreliable sources disagree about a contact in Grid A1",
        records=[
            _record(
                record_id="INJ-FALSE-A",
                modality=Modality.RF,
                sensor="rf_01",
                timestamp=t(5, 25),
                region="grid_a1",
                text=(
                    "Weak RF emission in Grid A1 tentatively associated with vessel FV DRIFTER, "
                    "heading 030 degrees. Single-look, poor SNR."
                ),
                reliability=0.35,
                confidence=0.30,
                vessel_name="FV DRIFTER",
                heading=30.0,
                speed=9.0,
                position=(12.40, 45.60),
                tags=["injected", "false_contradiction"],
            ),
            _record(
                record_id="INJ-FALSE-B",
                modality=Modality.MISSION_REPORT,
                sensor="analyst_unverified",
                timestamp=t(5, 26),
                region="grid_a1",
                text=(
                    "Unverified note: the Grid A1 contact may be FV SEA WREN on a 200 degree "
                    "course. Not corroborated."
                ),
                reliability=0.40,
                confidence=0.30,
                vessel_name="FV SEA WREN",
                heading=200.0,
                speed=9.5,
                position=(12.401, 45.601),
                tags=["injected", "false_contradiction"],
            ),
        ],
        expectation="contradiction detected with LOW severity (weakest source is unreliable)",
    )


def true_contradiction() -> Injection:
    """Two reliable sources disagree. Should be flagged loudly."""
    return Injection(
        name="true_contradiction",
        description="radar and AIS disagree about a contact in Grid A2, both reliable",
        records=[
            _record(
                record_id="INJ-TRUE-A",
                modality=Modality.SURFACE_RADAR,
                sensor="radar_01",
                timestamp=t(5, 25),
                region="grid_a2",
                text=(
                    "Surface track T-77 held in Grid A2: heading 020 degrees, speed 19.5 knots, "
                    "firm track, high signal-to-noise."
                ),
                reliability=0.94,
                confidence=0.95,
                track_id="T-77",
                heading=20.0,
                speed=19.5,
                position=(12.36, 45.66),
                object_type="vessel",
                tags=["injected", "true_contradiction"],
            ),
            _record(
                record_id="INJ-TRUE-B",
                modality=Modality.AIS,
                sensor="ais_rx_01",
                timestamp=t(5, 26),
                region="grid_a2",
                text=(
                    "AIS report: MMSI 412900555, vessel MV IRONWOOD, position Grid A2, heading "
                    "200 degrees, speed 6.0 knots, navigation status under way."
                ),
                reliability=0.90,
                confidence=0.94,
                mmsi="412900555",
                vessel_name="MV IRONWOOD",
                heading=200.0,
                speed=6.0,
                position=(12.361, 45.661),
                object_type="vessel",
                tags=["injected", "true_contradiction"],
            ),
        ],
        expectation="contradiction detected with HIGH severity on heading and speed",
    )


def retrieval_poisoning() -> Injection:
    """Highly similar but wrong records, designed to win a similarity contest."""
    return Injection(
        name="retrieval_poisoning",
        description="near-duplicate records asserting coverage the ledger does not have",
        records=[
            _record(
                record_id=f"INJ-POISON-{i:02d}",
                modality=Modality.SURFACE_RADAR,
                sensor="radar_01",
                timestamp=t(4, 9),
                # Deliberately the *right* region and the *right* window, so region and time
                # affinity cannot filter it out. The only thing standing between this record
                # and a false absence claim is the coverage ledger.
                region="sector_alpha",
                text=(
                    "Surface search sweep of Sector Alpha completed for the 0407-0411Z window. "
                    "No surface contacts held anywhere in the sector. Full sensor coverage "
                    "confirmed, Grid B7 included. Sea state 2, no clutter above threshold."
                ),
                reliability=0.94,
                confidence=0.98,
                is_absence_report=True,
                tags=["injected", "trap", "poison"],
            )
            for i in range(1, 6)
        ],
        expectation=(
            "records are retrieved but the ledger refuses to confirm coverage, so the answer "
            "stays UNKNOWN and no absence is asserted"
        ),
    )


ALL_INJECTIONS: dict[str, Callable[..., Injection]] = {
    "sensor_dropout": sensor_dropout,
    "stale_data": stale_data,
    "false_contradiction": false_contradiction,
    "true_contradiction": true_contradiction,
    "retrieval_poisoning": retrieval_poisoning,
}


def apply_records(corpus: Corpus, records: list[SourceRecord]) -> Corpus:
    """Return a new corpus with the injected records appended."""
    return Corpus(list(corpus.records) + list(records))
