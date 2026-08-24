"""Phase 5 gate: deterministic extraction of times, regions, entities and intent."""
from __future__ import annotations

import pytest

from app.dataset.world import t
from app.models.schemas import Modality, QueryIntent
from app.reasoning.decomposition import decompose, extract_entities, extract_region


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Were there any surface contacts in Sector Alpha between 04:00 and 04:20?", QueryIntent.ABSENCE_CHECK),
        ("Can we conclude that no contacts were present in Sector C?", QueryIntent.ABSENCE_CHECK),
        ("What vessel was detected near Grid B7?", QueryIntent.IDENTITY_RESOLUTION),
        ("Is the vessel detected at 05:20 the same vessel tracked at 04:00?", QueryIntent.ASSOCIATION),
        ("Why are these two sources reporting different vessel identities?", QueryIntent.EXPLAIN_DISAGREEMENT),
    ],
)
def test_intent_detection(question, expected):
    assert decompose(question).intent is expected


def test_explicit_window_extraction():
    plan = decompose("Were there any contacts in Sector Alpha between 04:07 and 04:11?")
    assert plan.time_range is not None
    assert plan.time_range.start == t(4, 7)
    assert plan.time_range.end == t(4, 11)


def test_relative_window_extraction():
    plan = decompose("What contacts were observed in Sector Alpha during the last 20 minutes?")
    assert plan.time_range is not None
    assert (plan.time_range.end - plan.time_range.start).total_seconds() == 20 * 60


def test_bare_last_hour_extraction():
    plan = decompose("Were there any suspicious RF emissions in this region during the last hour?")
    assert plan.time_range is not None
    assert (plan.time_range.end - plan.time_range.start).total_seconds() == 3600


def test_after_anchor_is_open_ended():
    plan = decompose("Was vessel V-17 observed after 04:00?")
    assert plan.time_range is not None
    assert plan.time_range.start == t(4, 0)
    assert plan.time_range.end > plan.time_range.start


def test_region_aliases():
    assert extract_region("contacts in Sector Alpha") == "sector_alpha"
    assert extract_region("contacts in Sector A") == "sector_alpha"
    assert extract_region("near Grid B7") == "grid_b7"
    assert extract_region("Sector C") == "sector_charlie"
    # The trap region must not be mistaken for Sector Alpha.
    assert extract_region("the Sector Alpha Training Annex") == "sector_alpha_training_annex"


def test_entity_extraction():
    assert "V-17" in extract_entities("Was vessel V-17 observed after 04:00?")
    assert "412700111" in extract_entities("Which vessel is MMSI 412700111?")
    assert "T-88" in extract_entities("Where is track T-88 now?")


def test_association_anchors_are_chronological():
    plan = decompose("Is the vessel detected at 05:20 the same vessel tracked at 04:00?")
    assert plan.comparison_targets == ["04:00", "05:20"]


def test_every_sensing_modality_gets_its_own_subquery():
    """Independent modality retrieval: no single blended query may mask a channel."""
    plan = decompose("Were there any contacts in Sector Alpha between 04:00 and 04:20?")
    covered = {m for sq in plan.subqueries for m in sq.preferred_modalities}
    for modality in (Modality.SURFACE_RADAR, Modality.AIS, Modality.EO_IR, Modality.RF):
        assert modality in covered


def test_modality_preference_is_soft_by_default():
    """Parent-filter prevention: an inferred modality must not become a hard filter."""
    plan = decompose("Were there any radar contacts in Sector Alpha between 04:00 and 04:20?")
    assert plan.hard_modalities == []
    assert all(sq.hard_modalities == [] for sq in plan.subqueries)


def test_explicit_operator_restriction_becomes_hard():
    plan = decompose("Using only radar, were there contacts in Sector Alpha at 04:10?")
    assert Modality.SURFACE_RADAR in plan.hard_modalities
