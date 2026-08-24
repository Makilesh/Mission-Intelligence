"""Static definition of the synthetic mission world.

Everything here is fictional. Regions, sensors and reliabilities are invented for the
prototype; no real-world or operational data is used.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.schemas import Modality, Region

UTC = timezone.utc


def t(hh: int, mm: int, day: int = 22) -> datetime:
    """Mission-clock helper: 2026-08-<day> hh:mm UTC."""
    return datetime(2026, 8, day, hh, mm, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Regions: sectors decompose into atomic grids. Coverage is rasterised over atomic grids.
# --------------------------------------------------------------------------------------
REGIONS: dict[str, Region] = {}


def _add(region: Region) -> None:
    REGIONS[region.region_id] = region


_add(
    Region(
        region_id="sector_alpha",
        name="Sector Alpha",
        atomic=False,
        children=["grid_a1", "grid_a2", "grid_a3", "grid_b7"],
        centroid=(12.35, 45.65),
        terrain="coastal_approach",
        notes="Primary area of interest; shipping lane transits the southern edge.",
    )
)
_add(
    Region(
        region_id="mission_area",
        name="Mission Area",
        atomic=False,
        children=["sector_alpha", "sector_bravo", "sector_charlie"],
        centroid=(12.20, 45.63),
        terrain="mixed",
        notes="Union of all tasked sectors; used when the operator names no region.",
    )
)
_add(
    Region(
        region_id="sector_bravo",
        name="Sector Bravo",
        atomic=False,
        children=["grid_b1", "grid_b2", "grid_b3"],
        centroid=(12.20, 45.80),
        terrain="open_water",
        notes="Open water east of Sector Alpha; main transit corridor toward Grid B7.",
    )
)
_add(
    Region(
        region_id="sector_charlie",
        name="Sector Charlie",
        atomic=False,
        children=["grid_c1", "grid_c2"],
        centroid=(12.05, 45.45),
        terrain="island_archipelago",
        notes="Archipelago with heavy radar shadowing; no organic radar tasked this mission.",
    )
)
# Deliberate retrieval trap: a similarly named region that is NOT part of Sector Alpha.
_add(
    Region(
        region_id="sector_alpha_training_annex",
        name="Sector Alpha Training Annex",
        atomic=True,
        parent=None,
        centroid=(11.80, 46.20),
        terrain="open_water",
        notes="Exercise range 60 NM south. Lexically similar to Sector Alpha; not part of it.",
    )
)

_ATOMIC = {
    "grid_a1": ((12.40, 45.60), "open_water", "sector_alpha"),
    "grid_a2": ((12.36, 45.66), "open_water", "sector_alpha"),
    "grid_a3": ((12.32, 45.62), "coastal_shelf", "sector_alpha"),
    "grid_b7": ((12.28, 45.71), "coastal_shelf", "sector_alpha"),
    "grid_b1": ((12.22, 45.78), "open_water", "sector_bravo"),
    "grid_b2": ((12.18, 45.84), "open_water", "sector_bravo"),
    "grid_b3": ((12.24, 45.90), "open_water", "sector_bravo"),
    "grid_c1": ((12.06, 45.42), "island_archipelago", "sector_charlie"),
    "grid_c2": ((12.02, 45.48), "island_archipelago", "sector_charlie"),
}
for _rid, (_c, _terr, _parent) in _ATOMIC.items():
    _add(
        Region(
            region_id=_rid,
            name=_rid.replace("_", " ").title(),
            atomic=True,
            parent=_parent,
            centroid=_c,
            terrain=_terr,
        )
    )


def atomic_regions(region_id: str) -> list[str]:
    """Expand a region id into its atomic leaves. Unknown ids resolve to themselves."""
    region = REGIONS.get(region_id)
    if region is None:
        return [region_id]
    if region.atomic:
        return [region.region_id]
    leaves: list[str] = []
    for child in region.children:
        leaves.extend(atomic_regions(child))
    return leaves


def parent_of(region_id: str) -> str | None:
    region = REGIONS.get(region_id)
    return region.parent if region else None


def region_matches(record_region: str, query_region: str) -> bool:
    """True when record_region is inside query_region (or identical)."""
    if record_region == query_region:
        return True
    return record_region in set(atomic_regions(query_region))


# --------------------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------------------
class Sensor:
    def __init__(
        self,
        sensor_id: str,
        modality: Modality,
        reliability: float,
        regions: list[str],
        description: str,
    ) -> None:
        self.sensor_id = sensor_id
        self.modality = modality
        self.reliability = reliability
        self.regions = regions
        self.description = description


SENSORS: dict[str, Sensor] = {
    s.sensor_id: s
    for s in [
        Sensor("radar_01", Modality.SURFACE_RADAR, 0.94, ["sector_alpha"],
               "Shore-based surface search radar covering Sector Alpha."),
        Sensor("radar_02", Modality.SURFACE_RADAR, 0.90, ["sector_bravo"],
               "Shipborne surface search radar covering Sector Bravo."),
        Sensor("eo_ir_01", Modality.EO_IR, 0.88, ["sector_alpha", "grid_b1"],
               "UAV electro-optical / infrared turret."),
        Sensor("ais_rx_01", Modality.AIS, 0.82,
               ["sector_alpha", "sector_bravo", "sector_charlie"],
               "AIS receiver. Cooperative traffic only - cannot see non-transmitting vessels."),
        Sensor("rf_01", Modality.RF, 0.68, ["sector_alpha", "sector_bravo"],
               "Wideband RF direction-finding receiver."),
        Sensor("sat_img_01", Modality.IMAGERY, 0.75,
               ["sector_alpha", "sector_bravo", "sector_charlie"],
               "Commercial imagery satellite, periodic passes."),
        Sensor("analyst_unverified", Modality.MISSION_REPORT, 0.45, ["sector_alpha"],
               "Unverified single-analyst visual note. Low reliability by policy."),
        Sensor("watch_officer", Modality.MISSION_REPORT, 0.86,
               ["sector_alpha", "sector_bravo", "sector_charlie"],
               "Watch officer mission reporting."),
        Sensor("j3_orders", Modality.STANDING_ORDER, 0.99,
               ["sector_alpha", "sector_bravo", "sector_charlie"],
               "Standing orders desk."),
        Sensor("geo_cell", Modality.TERRAIN, 0.9,
               ["sector_alpha", "sector_bravo", "sector_charlie"],
               "Geospatial cell terrain and imagery metadata."),
    ]
}

# How capable each modality is of establishing presence/absence of a NON-cooperative
# surface contact. AIS only sees vessels that choose to transmit, so AIS-only coverage can
# never support a confident absence claim. This is the "effective coverage" weighting.
MODALITY_ADEQUACY: dict[Modality, float] = {
    Modality.SURFACE_RADAR: 1.00,
    Modality.EO_IR: 0.75,
    Modality.IMAGERY: 0.60,
    Modality.RF: 0.45,
    Modality.AIS: 0.35,
}

DEFAULT_QUERY_MODALITIES = [
    Modality.SURFACE_RADAR,
    Modality.EO_IR,
    Modality.AIS,
    Modality.RF,
]

# --------------------------------------------------------------------------------------
# Mission clock
# --------------------------------------------------------------------------------------
MISSION_START = t(3, 30)
MISSION_END = t(6, 0)
MISSION_NOW = t(5, 40)

# The single planted total-blackout window (Case B). Every sensing modality is blind here.
BLACKOUT_REGION = "grid_b7"
BLACKOUT_WINDOW = (t(4, 7), t(4, 11))

# Sector Alpha planted true-absence window (Case A).
TRUE_ABSENCE_REGION = "sector_alpha"
TRUE_ABSENCE_WINDOW = (t(4, 0), t(4, 20))

# Radar handover gap on the Bravo -> Alpha transit (used by the multi-hop case).
HANDOVER_GAP = (t(4, 52), t(5, 12))


def minutes(n: float) -> timedelta:
    return timedelta(minutes=n)
