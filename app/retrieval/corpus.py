"""Corpus loading and the searchable text projection of a heterogeneous record."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import SYNTHETIC_DIR
from app.dataset import world
from app.models.schemas import SourceRecord


def searchable_text(record: SourceRecord) -> str:
    """Flatten a record into the string that gets indexed.

    Structured attributes are appended in natural language so that BM25 can match on
    identifiers (MMSI, track id) and the dense encoder gets region/modality context.
    """
    region = world.REGIONS.get(record.region)
    region_name = region.name if region else record.region.replace("_", " ").title()
    parts = [
        record.text,
        f"source {record.modality.value.replace('_', ' ')} sensor {record.sensor}",
        f"region {region_name} {record.region}",
        f"time {record.timestamp.strftime('%H:%M')}Z {record.timestamp.strftime('%Y-%m-%d')}",
    ]
    if region and region.parent:
        parts.append(f"within {world.REGIONS[region.parent].name}")
    if record.entities:
        parts.append("entities " + ", ".join(record.entities))
    for label, value in (
        ("track", record.track_id),
        ("mmsi", record.mmsi),
        ("vessel", record.vessel_name),
        ("object", record.object_type),
        ("classification", record.classification),
    ):
        if value:
            parts.append(f"{label} {value}")
    if record.heading is not None:
        parts.append(f"heading {int(record.heading)} degrees")
    if record.speed is not None:
        parts.append(f"speed {record.speed} knots")
    if record.frequency_mhz is not None:
        parts.append(f"frequency {record.frequency_mhz} MHz")
    if record.is_absence_report:
        parts.append("negative report no contacts detected absence")
    return ". ".join(p for p in parts if p)


class Corpus:
    def __init__(self, records: list[SourceRecord]) -> None:
        self.records = records
        self.by_id: dict[str, SourceRecord] = {r.record_id: r for r in records}
        self.ids: list[str] = [r.record_id for r in records]
        self.texts: list[str] = [searchable_text(r) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def get(self, record_id: str) -> SourceRecord | None:
        return self.by_id.get(record_id)

    @classmethod
    def load(cls, path: Path | None = None) -> "Corpus":
        path = path or (SYNTHETIC_DIR / "records.json")
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([SourceRecord(**row) for row in rows])


_CORPUS: Corpus | None = None


def get_corpus(reload: bool = False) -> Corpus:
    global _CORPUS
    if _CORPUS is None or reload:
        _CORPUS = Corpus.load()
    return _CORPUS


def set_corpus(corpus: Corpus) -> None:
    """Swap the process-wide corpus (used by retrieval-poisoning injection)."""
    global _CORPUS
    _CORPUS = corpus
