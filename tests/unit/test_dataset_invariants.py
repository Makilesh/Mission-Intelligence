"""The planted failure cases are the ground truth of the whole evaluation. Guard them."""
from __future__ import annotations

from app.dataset import world
from app.dataset.world import t
from app.models.schemas import Modality


def test_no_contacts_inside_planted_absence_window(records):
    """Case A is only meaningful if the world really is empty there."""
    start, end = world.TRUE_ABSENCE_WINDOW
    alpha = set(world.atomic_regions("sector_alpha")) | {"sector_alpha"}
    offenders = [
        r.record_id
        for r in records
        if r.region in alpha and start <= r.timestamp <= end and not r.is_absence_report
    ]
    assert offenders == []


def test_nothing_recorded_inside_blackout(records):
    """Case B: if nothing observed Grid B7, nothing can have been recorded there."""
    start, end = world.BLACKOUT_WINDOW
    offenders = [
        r.record_id
        for r in records
        if r.region == world.BLACKOUT_REGION and start <= r.timestamp < end
    ]
    assert offenders == []


def test_contradiction_cluster_present(records):
    ids = {r.record_id for r in records}
    assert {"RADAR-221", "AIS-1312", "EO-1042", "MR-014"} <= ids


def test_multihop_chain_present(records):
    by_id = {r.record_id: r for r in records}
    assert by_id["RADAR-210"].track_id == "T-42"
    assert by_id["RADAR-221"].track_id == "T-88"
    assert by_id["AIS-1310"].vessel_name == "MV KESTREL"
    assert by_id["MR-021"].text.count("T-42") >= 1


def test_traps_present_and_labelled(records):
    traps = [r for r in records if "trap" in r.tags]
    assert len(traps) >= 5
    annex = [r for r in traps if r.region == "sector_alpha_training_annex"]
    assert annex, "need a lexically similar but out-of-region distractor"
    stale = [r for r in traps if "stale" in r.tags]
    assert stale, "need at least one stale high-similarity distractor"


def test_all_modalities_represented(records):
    present = {r.modality for r in records}
    for m in [
        Modality.SURFACE_RADAR,
        Modality.AIS,
        Modality.EO_IR,
        Modality.RF,
        Modality.MISSION_REPORT,
        Modality.STANDING_ORDER,
        Modality.IMAGERY,
        Modality.TERRAIN,
    ]:
        assert m in present


def test_record_ids_unique(records):
    ids = [r.record_id for r in records]
    assert len(ids) == len(set(ids))


def test_region_hierarchy_is_consistent():
    leaves = world.atomic_regions("sector_alpha")
    assert leaves == ["grid_a1", "grid_a2", "grid_a3", "grid_b7"]
    assert world.region_matches("grid_b7", "sector_alpha")
    assert not world.region_matches("grid_b1", "sector_alpha")
    # The training annex is a lexical trap, not a child of Sector Alpha.
    assert not world.region_matches("sector_alpha_training_annex", "sector_alpha")


def test_ledger_covers_every_sensing_modality(ledger):
    mods = {e.modality for e in ledger.entries}
    assert {
        Modality.SURFACE_RADAR,
        Modality.AIS,
        Modality.EO_IR,
        Modality.RF,
        Modality.IMAGERY,
    } <= mods


def test_ledger_and_records_are_independent(ledger, records):
    """Sanity: the ledger asserts coverage for windows that contain zero records."""
    rep = ledger.check("sector_alpha", (t(4, 0), t(4, 20)))
    contacts = [
        r for r in records
        if r.region.startswith(("grid_a", "grid_b7", "sector_alpha"))
        and t(4, 0) <= r.timestamp <= t(4, 20)
        and not r.is_absence_report
    ]
    assert rep.covered_fraction > 0.9 and contacts == []
