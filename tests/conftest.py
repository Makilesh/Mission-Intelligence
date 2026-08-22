"""Shared fixtures. The dataset is generated on demand so the suite is self-contained."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import COVERAGE_DIR, SYNTHETIC_DIR
from app.coverage.ledger import CoverageLedger
from app.dataset import generator
from app.dataset.world import t
from app.models.schemas import SourceRecord, TimeRange


def _ensure_dataset() -> None:
    if not (SYNTHETIC_DIR / "records.json").exists() or not (COVERAGE_DIR / "ledger.json").exists():
        generator.generate(write=True)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> None:
    _ensure_dataset()


@pytest.fixture(scope="session")
def records() -> list[SourceRecord]:
    _ensure_dataset()
    rows = json.loads(Path(SYNTHETIC_DIR / "records.json").read_text(encoding="utf-8"))
    return [SourceRecord(**r) for r in rows]


@pytest.fixture(scope="session")
def ledger() -> CoverageLedger:
    _ensure_dataset()
    return CoverageLedger.from_json()


@pytest.fixture(scope="session")
def tr():  # noqa: ANN201
    def _make(a: tuple[int, int], b: tuple[int, int]) -> TimeRange:
        return TimeRange(start=t(*a), end=t(*b))

    return _make
