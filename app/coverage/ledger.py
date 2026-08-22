"""The Coverage Ledger.

The most important component in the system. It answers, independently of any retrieval
index, the question: *what did we actually observe, where, when, and with what?*

Design rules enforced here:

1. Coverage is NEVER inferred from documents or retrieval hits. The ledger is loaded from
   its own store and can be queried on its own.
2. "No ledger entry" (UNKNOWN / no information) is structurally distinct from an asserted
   NOT_OBSERVED. Both are distinct from OBSERVED with zero detections.
3. Coverage is rasterised over (atomic sub-region x time slot) so that a blind pocket
   inside a large sector cannot be averaged away silently.
4. Modality adequacy is applied: AIS sees only cooperative traffic, so AIS-only coverage
   can never support a confident absence claim about non-cooperative vessels.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from app.config import COVERAGE_DIR, SETTINGS
from app.dataset import world
from app.dataset.world import MODALITY_ADEQUACY, atomic_regions
from app.models.schemas import (
    CoverageEntry,
    CoverageReport,
    CoverageStatus,
    ModalityCoverage,
    Modality,
    SubRegionCoverage,
    TimeRange,
)


def _to_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _runs(flags: Sequence[bool], start: datetime, step: timedelta) -> list[tuple[datetime, datetime]]:
    """Convert a boolean per-slot mask into a list of maximal (start, end) intervals."""
    out: list[tuple[datetime, datetime]] = []
    run_start: int | None = None
    for i, flag in enumerate(flags):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            out.append((start + run_start * step, start + i * step))
            run_start = None
    if run_start is not None:
        out.append((start + run_start * step, start + len(flags) * step))
    return out


class CoverageLedger:
    """Queryable store of coverage assertions."""

    def __init__(self, entries: Iterable[CoverageEntry] | None = None) -> None:
        self.entries: list[CoverageEntry] = list(entries or [])
        self._by_region: dict[str, list[CoverageEntry]] = defaultdict(list)
        self._reindex()

    # ---------------------------------------------------------------- construction ----
    def _reindex(self) -> None:
        self._by_region.clear()
        for e in self.entries:
            self._by_region[e.region].append(e)

    @classmethod
    def from_json(cls, path: Path | None = None) -> "CoverageLedger":
        path = path or (COVERAGE_DIR / "ledger.json")
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CoverageEntry(**row) for row in rows)

    def add(self, entry: CoverageEntry) -> None:
        self.entries.append(entry)
        self._by_region[entry.region].append(entry)

    def clone(self) -> "CoverageLedger":
        return CoverageLedger(deepcopy(self.entries))

    # ------------------------------------------------------------------- injection ----
    def with_sensor_dropout(
        self,
        sensor: str,
        time_range: TimeRange | None = None,
        regions: Sequence[str] | None = None,
        reason: str = "Injected sensor dropout",
    ) -> "CoverageLedger":
        """Return a new ledger where `sensor` is blind over the given window/regions.

        Used by the failure-injection framework. Entries are *split* rather than deleted,
        so the ledger keeps asserting NOT_OBSERVED (we know we did not look) instead of
        silently degrading into UNKNOWN.
        """
        target_regions: set[str] | None = None
        if regions:
            target_regions = set()
            for r in regions:
                target_regions.update(atomic_regions(r))

        out: list[CoverageEntry] = []
        counter = 0
        for e in self.entries:
            if e.sensor != sensor or (target_regions and e.region not in target_regions):
                out.append(e)
                continue
            window = time_range or e.time_range
            overlap = e.time_range.intersection(window)
            if overlap is None:
                out.append(e)
                continue
            pieces: list[tuple[datetime, datetime, CoverageStatus, float, str]] = []
            if e.time_start < overlap.start:
                pieces.append(
                    (e.time_start, overlap.start, e.coverage_status, e.coverage_confidence, e.reason)
                )
            pieces.append(
                (overlap.start, overlap.end, CoverageStatus.NOT_OBSERVED, 0.99, reason)
            )
            if overlap.end < e.time_end:
                pieces.append(
                    (overlap.end, e.time_end, e.coverage_status, e.coverage_confidence, e.reason)
                )
            for s, en, st, conf, rsn in pieces:
                counter += 1
                out.append(
                    CoverageEntry(
                        entry_id=f"{e.entry_id}-INJ{counter}",
                        region=e.region,
                        time_start=s,
                        time_end=en,
                        modality=e.modality,
                        sensor=e.sensor,
                        coverage_status=st,
                        coverage_confidence=conf,
                        reason=rsn,
                    )
                )
        return CoverageLedger(out)

    def with_coverage_loss(self, fraction_kept: float, region: str | None = None) -> "CoverageLedger":
        """Artificially reduce coverage to `fraction_kept` of each entry's duration.

        Used by the confidence-calibration sweep (100/80/60/40/20 %).
        """
        fraction_kept = max(0.0, min(1.0, fraction_kept))
        targets: set[str] | None = set(atomic_regions(region)) if region else None
        out: list[CoverageEntry] = []
        for e in self.entries:
            if targets is not None and e.region not in targets:
                out.append(e)
                continue
            if e.coverage_status in (CoverageStatus.NOT_OBSERVED, CoverageStatus.UNKNOWN):
                out.append(e)
                continue
            total = (e.time_end - e.time_start).total_seconds()
            keep = total * fraction_kept
            cut = e.time_start + timedelta(seconds=keep)
            if keep > 0:
                out.append(e.model_copy(update={"time_end": cut}))
            if keep < total:
                out.append(
                    e.model_copy(
                        update={
                            "entry_id": f"{e.entry_id}-LOSS",
                            "time_start": cut,
                            "time_end": e.time_end,
                            "coverage_status": CoverageStatus.NOT_OBSERVED,
                            "coverage_confidence": 0.99,
                            "reason": f"Injected coverage loss (kept {fraction_kept:.0%})",
                        }
                    )
                )
        return CoverageLedger(out)

    # ----------------------------------------------------------------------- query ----
    def entries_for(
        self,
        region: str,
        time_range: TimeRange,
        modalities: Sequence[Modality] | None = None,
    ) -> list[CoverageEntry]:
        grids = set(atomic_regions(region))
        wanted = set(modalities) if modalities else None
        out = []
        for grid in grids:
            for e in self._by_region.get(grid, []):
                if wanted and e.modality not in wanted:
                    continue
                if e.time_range.overlaps(time_range):
                    out.append(e)
        return sorted(out, key=lambda e: (e.region, e.modality.value, e.time_start))

    def check(
        self,
        region: str,
        time_range: TimeRange | tuple[datetime, datetime] | tuple[str, str],
        modalities: Sequence[Modality | str] | None = None,
    ) -> CoverageReport:
        """The core API. Returns an explicit, independently computed coverage report."""
        cfg = SETTINGS.coverage
        tr = self._normalise_range(time_range)
        mods: list[Modality] = [
            m if isinstance(m, Modality) else Modality(str(m).lower())
            for m in (modalities or world.DEFAULT_QUERY_MODALITIES)
        ]
        # Documents are not sensors: they can never create coverage.
        sensing = [m for m in mods if m in MODALITY_ADEQUACY]
        if not sensing:
            sensing = list(world.DEFAULT_QUERY_MODALITIES)

        grids = atomic_regions(region)
        step = timedelta(seconds=cfg.resolution_seconds)
        n_slots = max(1, math.ceil(tr.duration_seconds / cfg.resolution_seconds))
        slot_starts = [tr.start + i * step for i in range(n_slots)]

        # weight[grid][modality][slot] and a parallel "has information" mask
        weight: dict[str, dict[Modality, list[float]]] = {
            g: {m: [0.0] * n_slots for m in sensing} for g in grids
        }
        # "informed" is tracked per (grid, modality, slot): a cell with no ledger entry is
        # genuine no-information, not an assertion of blindness.
        informed: dict[str, dict[Modality, list[bool]]] = {
            g: {m: [False] * n_slots for m in sensing} for g in grids
        }
        degraded_mods: set[Modality] = set()
        used_entries: list[CoverageEntry] = []
        quality_samples: list[float] = []
        sensors: set[str] = set()

        for grid in grids:
            for e in self._by_region.get(grid, []):
                if e.modality not in weight[grid]:
                    continue
                if not e.time_range.overlaps(tr):
                    continue
                used_entries.append(e)
                sensors.add(e.sensor)
                w = cfg.status_weight.get(e.coverage_status.value, 0.0)
                if e.coverage_status is CoverageStatus.DEGRADED:
                    degraded_mods.add(e.modality)
                if w > 0:
                    quality_samples.append(e.coverage_confidence)
                for i, s0 in enumerate(slot_starts):
                    mid = s0 + step / 2
                    if e.time_start <= mid < e.time_end:
                        informed[grid][i] = True
                        cur = weight[grid][e.modality][i]
                        weight[grid][e.modality][i] = max(cur, w)

        # ---- effective coverage per (grid, slot): best capable modality wins ----------
        eff: dict[str, list[float]] = {}
        for grid in grids:
            eff[grid] = [
                max(
                    (weight[grid][m][i] * MODALITY_ADEQUACY.get(m, 0.0) for m in sensing),
                    default=0.0,
                )
                for i in range(n_slots)
            ]

        total_cells = len(grids) * n_slots
        covered_fraction = sum(sum(v) for v in eff.values()) / total_cells
        no_info_cells = sum(1 for g in grids for i in range(n_slots) if not informed[g][i])
        no_information_fraction = no_info_cells / total_cells

        # ---- per-modality reporting (raw, no adequacy weighting) ---------------------
        per_modality: list[ModalityCoverage] = []
        missing_modalities: list[Modality] = []
        best_modality_fraction = 0.0
        for m in sensing:
            frac = sum(sum(weight[g][m]) for g in grids) / total_cells
            best_modality_fraction = max(best_modality_fraction, frac)
            # A slot is "missing" for this modality when it does not fully cover the whole
            # queried region during that slot.
            mask = [
                (sum(weight[g][m][i] for g in grids) / len(grids)) < 0.999
                for i in range(n_slots)
            ]
            mod_sensors = sorted(
                {e.sensor for e in used_entries if e.modality is m and e.coverage_status
                 not in (CoverageStatus.NOT_OBSERVED, CoverageStatus.UNKNOWN)}
            )
            per_modality.append(
                ModalityCoverage(
                    modality=m,
                    covered_fraction=round(frac, 4),
                    missing_intervals=_runs(mask, tr.start, step),
                    sensors=mod_sensors,
                    degraded=m in degraded_mods,
                )
            )
            if frac < 0.05:
                missing_modalities.append(m)

        # ---- per-subregion reporting -------------------------------------------------
        per_subregion: list[SubRegionCoverage] = []
        blind_subregions: list[str] = []
        max_blind_fraction = 0.0
        for grid in grids:
            blind_mask = [eff[grid][i] <= 0.0 for i in range(n_slots)]
            blind_fraction = sum(blind_mask) / n_slots
            max_blind_fraction = max(max_blind_fraction, blind_fraction)
            if blind_fraction > 0:
                blind_subregions.append(grid)
            per_subregion.append(
                SubRegionCoverage(
                    region=grid,
                    covered_fraction=round(sum(eff[grid]) / n_slots, 4),
                    blind_fraction=round(blind_fraction, 4),
                    missing_intervals=_runs(blind_mask, tr.start, step),
                )
            )

        # ---- aggregate blind intervals: any sub-region blind at that instant ----------
        agg_mask = [any(eff[g][i] <= 0.0 for g in grids) for i in range(n_slots)]
        missing_intervals = _runs(agg_mask, tr.start, step)

        # ---- status ------------------------------------------------------------------
        status = self._status(
            used_entries=used_entries,
            covered_fraction=covered_fraction,
            degraded_mods=degraded_mods,
        )

        # ---- absence gate ------------------------------------------------------------
        absence_ok = True
        block = ""
        if covered_fraction < cfg.absence_coverage_threshold:
            absence_ok = False
            block = (
                f"effective coverage {covered_fraction:.0%} is below the "
                f"{cfg.absence_coverage_threshold:.0%} threshold required to assert absence"
            )
        elif max_blind_fraction >= cfg.absence_max_blind_subregion_fraction:
            absence_ok = False
            worst = max(per_subregion, key=lambda s: s.blind_fraction)
            block = (
                f"sub-region {worst.region} was unobserved for "
                f"{worst.blind_fraction:.0%} of the requested window"
            )

        return CoverageReport(
            region=region,
            time_range=tr,
            requested_modalities=sensing,
            status=status,
            covered_fraction=round(min(1.0, max(0.0, covered_fraction)), 4),
            best_modality_fraction=round(min(1.0, best_modality_fraction), 4),
            coverage_quality=round(
                sum(quality_samples) / len(quality_samples) if quality_samples else 0.0, 4
            ),
            no_information_fraction=round(no_information_fraction, 4),
            missing_intervals=missing_intervals,
            missing_modalities=missing_modalities,
            degraded_modalities=sorted(degraded_mods, key=lambda m: m.value),
            per_modality=per_modality,
            per_subregion=per_subregion,
            blind_subregions=blind_subregions,
            contributing_sensors=sorted(sensors),
            ledger_entries=sorted({e.entry_id for e in used_entries}),
            absence_claim_supported=absence_ok,
            absence_block_reason=block,
        )

    # ------------------------------------------------------------------- internals ----
    @staticmethod
    def _normalise_range(
        time_range: TimeRange | tuple[datetime, datetime] | tuple[str, str]
    ) -> TimeRange:
        if isinstance(time_range, TimeRange):
            return time_range
        start, end = time_range
        if isinstance(start, str) and len(start) <= 5 and ":" in start:
            # "04:00" shorthand resolves against the mission day.
            hh, mm = (int(x) for x in start.split(":"))
            hh2, mm2 = (int(x) for x in str(end).split(":"))
            return TimeRange(start=world.t(hh, mm), end=world.t(hh2, mm2))
        return TimeRange(start=_to_dt(start), end=_to_dt(end))  # type: ignore[arg-type]

    @staticmethod
    def _status(
        used_entries: list[CoverageEntry],
        covered_fraction: float,
        degraded_mods: set[Modality],
    ) -> CoverageStatus:
        cfg = SETTINGS.coverage
        if not used_entries:
            return CoverageStatus.UNKNOWN
        if covered_fraction >= cfg.observed_threshold:
            return CoverageStatus.OBSERVED
        if covered_fraction <= cfg.partial_threshold:
            positive = [
                e
                for e in used_entries
                if cfg.status_weight.get(e.coverage_status.value, 0.0) > 0
            ]
            if not positive:
                asserted = [
                    e for e in used_entries if e.coverage_status is CoverageStatus.NOT_OBSERVED
                ]
                return CoverageStatus.NOT_OBSERVED if asserted else CoverageStatus.UNKNOWN
            return CoverageStatus.PARTIALLY_OBSERVED
        active = {
            e.coverage_status
            for e in used_entries
            if cfg.status_weight.get(e.coverage_status.value, 0.0) > 0
        }
        if active == {CoverageStatus.DEGRADED}:
            return CoverageStatus.DEGRADED
        return CoverageStatus.PARTIALLY_OBSERVED

    # ------------------------------------------------------------------- utilities ----
    def timeline(
        self,
        region: str,
        time_range: TimeRange,
        modalities: Sequence[Modality] | None = None,
        buckets: int = 24,
    ) -> list[dict]:
        """Bucketed coverage for the operator timeline widget."""
        step = time_range.duration_seconds / buckets
        out = []
        for i in range(buckets):
            s = time_range.start + timedelta(seconds=step * i)
            e = time_range.start + timedelta(seconds=step * (i + 1))
            rep = self.check(region, TimeRange(start=s, end=e), modalities)
            out.append(
                {
                    "start": s,
                    "end": e,
                    "covered_fraction": rep.covered_fraction,
                    "status": rep.status.value,
                }
            )
        return out

    def sensors(self) -> list[str]:
        return sorted({e.sensor for e in self.entries})

    def summary(self) -> dict:
        by_status: dict[str, int] = defaultdict(int)
        for e in self.entries:
            by_status[e.coverage_status.value] += 1
        return {
            "entries": len(self.entries),
            "regions": sorted(self._by_region),
            "sensors": self.sensors(),
            "by_status": dict(by_status),
        }


_LEDGER: CoverageLedger | None = None


def get_ledger(reload: bool = False) -> CoverageLedger:
    global _LEDGER
    if _LEDGER is None or reload:
        _LEDGER = CoverageLedger.from_json()
    return _LEDGER


def set_ledger(ledger: CoverageLedger) -> None:
    """Swap the process-wide ledger (used by failure injection and the API)."""
    global _LEDGER
    _LEDGER = ledger
