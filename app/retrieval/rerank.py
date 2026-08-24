"""Metadata-aware reranking.

Parent-filter failure prevention (spec section 8): metadata derived from query
classification is applied here as a **soft preference**, never as a retrieval filter. A
record from the "wrong" modality or region is down-weighted, not removed — so a
misclassified query degrades gracefully instead of losing the only relevant evidence.

The only hard constraint is `SubQuery.hard_modalities`, which is populated exclusively
when the *operator* explicitly restricts the sources.
"""
from __future__ import annotations

from datetime import datetime

from app.config import SETTINGS
from app.dataset import world
from app.models.schemas import Modality, RetrievedDoc, SourceRecord, SubQuery
from app.retrieval.sparse import tokenize


def _region_affinity(record_region: str, query_region: str | None) -> float:
    if not query_region:
        return 0.5  # neutral: the query did not constrain region
    if record_region == query_region:
        return 1.0
    if world.region_matches(record_region, query_region):
        return 1.0
    # A record whose region *contains* the queried region is still relevant.
    if world.region_matches(query_region, record_region):
        return 0.85
    parent_a = world.parent_of(record_region)
    parent_b = world.parent_of(query_region) or query_region
    if parent_a and parent_a == parent_b:
        return 0.4  # sibling grid in the same sector
    return 0.0


def _time_affinity(ts: datetime, subquery: SubQuery) -> float:
    if subquery.time_range is None:
        return 0.5
    tr = subquery.time_range
    if tr.start <= ts <= tr.end:
        return 1.0
    delta_min = min(abs((ts - tr.start).total_seconds()), abs((ts - tr.end).total_seconds())) / 60.0
    if delta_min <= 5:
        return 0.8
    if delta_min <= 20:
        return 0.5
    if delta_min <= 60:
        return 0.25
    return 0.05


def _modality_affinity(modality: Modality, subquery: SubQuery) -> float:
    if not subquery.preferred_modalities:
        return 0.6
    if modality in subquery.preferred_modalities:
        return 1.0
    # Soft, not exclusive: context documents keep a real chance of surfacing.
    if modality in (Modality.MISSION_REPORT, Modality.STANDING_ORDER):
        return 0.55
    return 0.35


def _entity_affinity(record: SourceRecord, subquery: SubQuery) -> float:
    if not subquery.entities:
        return 0.5
    haystack = " ".join(
        [record.text.lower()]
        + [e.lower() for e in record.entities]
        + [str(v).lower() for v in (record.track_id, record.mmsi, record.vessel_name) if v]
    )
    hits = sum(1 for e in subquery.entities if e.lower() in haystack)
    return min(1.0, hits / len(subquery.entities))


def _lexical_affinity(text: str, query: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(text))
    return len(q & d) / len(q)


def rerank(
    candidates: list[RetrievedDoc],
    records: dict[str, SourceRecord],
    subquery: SubQuery,
) -> list[RetrievedDoc]:
    cfg = SETTINGS.retrieval
    if not candidates:
        return []
    max_fusion = max(c.fusion_score for c in candidates) or 1.0

    out: list[RetrievedDoc] = []
    for cand in candidates:
        record = records.get(cand.record_id)
        if record is None:
            continue
        # The ONE hard constraint, and only when the operator asked for it.
        if subquery.hard_modalities and record.modality not in subquery.hard_modalities:
            continue

        region = _region_affinity(record.region, subquery.region)
        time_aff = _time_affinity(record.timestamp, subquery)
        modality = _modality_affinity(record.modality, subquery)
        entity = _entity_affinity(record, subquery)
        lexical = _lexical_affinity(record.text, subquery.text)

        score = (
            cfg.w_fusion * (cand.fusion_score / max_fusion)
            + cfg.w_region * region
            + cfg.w_time * time_aff
            + cfg.w_modality * modality
            + cfg.w_entity * entity
            + cfg.w_lexical * lexical
        )
        why = list(cand.why)
        if region >= 1.0 and subquery.region:
            why.append(f"region match {record.region}")
        elif region == 0.0 and subquery.region:
            why.append(f"region mismatch {record.region} (down-weighted, not filtered)")
        if time_aff >= 1.0 and subquery.time_range:
            why.append("inside requested window")
        elif time_aff <= 0.25 and subquery.time_range:
            why.append("far outside requested window")
        if entity >= 1.0 and subquery.entities:
            why.append("entity match")

        out.append(
            cand.model_copy(
                update={
                    "rerank_score": round(score, 6),
                    "score": round(score, 6),
                    "why": why,
                }
            )
        )
    out.sort(key=lambda d: -d.rerank_score)
    return out
