"""Phase 2/3 gate: the ledger must distinguish observed-empty from never-looked."""
from __future__ import annotations

import pytest

from app.dataset.world import t
from app.models.schemas import CoverageStatus, Modality, TimeRange


def test_case_a_sector_alpha_is_observed(ledger, tr):
    """Case A: Sector Alpha 04:00-04:20 is essentially fully observed by radar."""
    rep = ledger.check("sector_alpha", tr((4, 0), (4, 20)))
    assert rep.status is CoverageStatus.OBSERVED
    assert rep.covered_fraction >= 0.9
    assert rep.absence_claim_supported is True
    assert rep.absence_block_reason == ""


def test_case_b_blind_window_blocks_absence(ledger, tr):
    """Case B: the 04:07-04:11 window must never support an absence claim."""
    rep = ledger.check("sector_alpha", tr((4, 7), (4, 11)))
    assert rep.absence_claim_supported is False
    assert rep.status is not CoverageStatus.OBSERVED
    assert rep.covered_fraction < 0.85
    assert "grid_b7" in rep.blind_subregions


def test_blackout_grid_is_not_observed_not_unknown(ledger, tr):
    """An asserted blind window reports NOT_OBSERVED (we know we did not look)."""
    rep = ledger.check("grid_b7", tr((4, 7), (4, 11)))
    assert rep.status is CoverageStatus.NOT_OBSERVED
    assert rep.covered_fraction == 0.0
    assert rep.no_information_fraction == 0.0  # we have entries; they say "blind"


def test_no_ledger_entry_is_unknown(ledger, tr):
    """A region the ledger has never heard of is UNKNOWN, not NOT_OBSERVED."""
    rep = ledger.check("grid_z9", tr((4, 0), (4, 20)))
    assert rep.status is CoverageStatus.UNKNOWN
    assert rep.covered_fraction == 0.0
    assert rep.no_information_fraction == 1.0
    assert rep.absence_claim_supported is False


def test_missing_interval_matches_planted_blackout(ledger, tr):
    rep = ledger.check("sector_alpha", tr((4, 0), (4, 20)))
    assert rep.missing_intervals, "the 4-minute Grid B7 blackout must be surfaced"
    start, end = rep.missing_intervals[0]
    assert start == t(4, 7)
    assert end == t(4, 11)


def test_missing_modalities_reported(ledger, tr):
    """AIS drops out 04:40-04:52; RF drops out 04:40-05:00."""
    rep = ledger.check("sector_alpha", tr((4, 41), (4, 51)))
    assert Modality.AIS in rep.missing_modalities
    assert Modality.RF in rep.missing_modalities
    assert Modality.SURFACE_RADAR not in rep.missing_modalities


def test_sector_charlie_ais_only_cannot_support_absence(ledger, tr):
    """AIS sees cooperative traffic only: it can never establish absence on its own."""
    rep = ledger.check("sector_charlie", tr((4, 0), (5, 0)))
    assert rep.absence_claim_supported is False
    assert Modality.SURFACE_RADAR in rep.missing_modalities
    assert rep.covered_fraction < 0.5
    assert rep.no_information_fraction > 0  # no EO/IR or RF entries exist at all


def test_degraded_window_is_visible(ledger, tr):
    rep = ledger.check("sector_alpha", tr((5, 40), (5, 50)))
    assert Modality.SURFACE_RADAR in rep.degraded_modalities
    assert rep.covered_fraction < 1.0


def test_coverage_is_monotone_in_window_growth(ledger):
    """Extending a window into a blind interval must not increase coverage."""
    narrow = ledger.check("sector_alpha", TimeRange(start=t(4, 0), end=t(4, 7)))
    wide = ledger.check("sector_alpha", TimeRange(start=t(4, 0), end=t(4, 11)))
    assert wide.covered_fraction < narrow.covered_fraction


def test_sensor_dropout_reduces_coverage(ledger, tr):
    window = tr((4, 0), (4, 20))
    before = ledger.check("sector_alpha", window)
    after = ledger.with_sensor_dropout("radar_01", window).check("sector_alpha", window)
    assert after.covered_fraction < before.covered_fraction
    assert after.absence_claim_supported is False


def test_coverage_loss_sweep_is_monotone(ledger, tr):
    window = tr((4, 0), (4, 20))
    fractions = []
    for kept in (1.0, 0.8, 0.6, 0.4, 0.2):
        lg = ledger.with_coverage_loss(kept, region="sector_alpha")
        fractions.append(lg.check("sector_alpha", window).covered_fraction)
    assert fractions == sorted(fractions, reverse=True)


def test_documents_cannot_create_coverage(ledger, tr):
    """Mission reports are not sensors: requesting them yields no coverage of their own."""
    rep = ledger.check(
        "sector_alpha", tr((4, 0), (4, 20)), modalities=[Modality.MISSION_REPORT]
    )
    # Non-sensing modalities fall back to the default sensing set rather than inventing
    # coverage from documents.
    assert all(m.value in {m2.value for m2 in rep.requested_modalities} for m in [Modality.SURFACE_RADAR])
    assert Modality.MISSION_REPORT not in rep.requested_modalities


@pytest.mark.parametrize(
    "region,window,expect_absence",
    [
        ("sector_alpha", ((4, 0), (4, 20)), True),
        ("sector_alpha", ((4, 7), (4, 11)), False),
        ("grid_b7", ((4, 7), (4, 11)), False),
        ("sector_charlie", ((4, 0), (5, 0)), False),
        ("grid_a1", ((4, 0), (4, 20)), True),
    ],
)
def test_absence_gate_matrix(ledger, region, window, expect_absence):
    rep = ledger.check(region, TimeRange(start=t(*window[0]), end=t(*window[1])))
    assert rep.absence_claim_supported is expect_absence
