"""Typed domain models. Everything that crosses a component boundary is a Pydantic model."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------
class CoverageStatus(str, Enum):
    OBSERVED = "OBSERVED"
    PARTIALLY_OBSERVED = "PARTIALLY_OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class Modality(str, Enum):
    SURFACE_RADAR = "surface_radar"
    AIS = "ais"
    EO_IR = "eo_ir"
    RF = "rf"
    IMAGERY = "imagery"
    MISSION_REPORT = "mission_report"
    STANDING_ORDER = "standing_order"
    TERRAIN = "terrain"


# Modalities that can actually observe the world. Mission reports and standing orders are
# documents *about* observations, so they must never create coverage on their own.
SENSING_MODALITIES = {
    Modality.SURFACE_RADAR,
    Modality.AIS,
    Modality.EO_IR,
    Modality.RF,
    Modality.IMAGERY,
}


class EvidenceState(str, Enum):
    PRESENCE = "PRESENCE"
    OBSERVED_ABSENCE = "OBSERVED_ABSENCE"
    UNOBSERVED = "UNOBSERVED"
    PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
    CONTRADICTION = "CONTRADICTION"
    STALE = "STALE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class AnswerState(str, Enum):
    PRESENCE = "PRESENCE"
    OBSERVED_ABSENCE = "OBSERVED_ABSENCE"
    UNKNOWN = "UNKNOWN"
    CONTRADICTION = "CONTRADICTION"


class ContradictionDimension(str, Enum):
    IDENTITY = "identity"
    POSITION = "position"
    HEADING = "heading"
    SPEED = "speed"
    CLASSIFICATION = "classification"
    TIMESTAMP = "timestamp"


class SubQueryType(str, Enum):
    RETRIEVE_PRESENCE = "retrieve_presence"
    RETRIEVE_TRACK = "retrieve_track"
    RETRIEVE_CURRENT_DETECTION = "retrieve_current_detection"
    RETRIEVE_IDENTITY = "retrieve_identity"
    RETRIEVE_TRAJECTORY = "retrieve_trajectory"
    RETRIEVE_CONTEXT = "retrieve_context"
    RETRIEVE_ORDERS = "retrieve_orders"


class QueryIntent(str, Enum):
    PRESENCE_CHECK = "presence_check"
    ABSENCE_CHECK = "absence_check"
    IDENTITY_RESOLUTION = "identity_resolution"
    ASSOCIATION = "association"
    EXPLAIN_DISAGREEMENT = "explain_disagreement"
    SUMMARY = "summary"


# --------------------------------------------------------------------------------------
# Time / geography primitives
# --------------------------------------------------------------------------------------
class TimeRange(BaseModel):
    start: datetime
    end: datetime

    @field_validator("end")
    @classmethod
    def _ordered(cls, v: datetime, info):  # type: ignore[no-untyped-def]
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("TimeRange.end must be >= start")
        return v

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end

    def intersection(self, other: "TimeRange") -> Optional["TimeRange"]:
        if not self.overlaps(other):
            return None
        return TimeRange(start=max(self.start, other.start), end=min(self.end, other.end))

    def label(self) -> str:
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"


class Region(BaseModel):
    """A named region. Atomic regions are the leaves used for coverage rasterisation."""

    region_id: str
    name: str
    atomic: bool = True
    children: list[str] = Field(default_factory=list)
    parent: Optional[str] = None
    centroid: tuple[float, float] | None = None
    terrain: str = "open_water"
    notes: str = ""


# --------------------------------------------------------------------------------------
# Coverage ledger
# --------------------------------------------------------------------------------------
class CoverageEntry(BaseModel):
    """One assertion about what a sensor did (or did not) observe. Independent of retrieval."""

    entry_id: str
    region: str
    time_start: datetime
    time_end: datetime
    modality: Modality
    sensor: str
    coverage_status: CoverageStatus
    coverage_confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(start=self.time_start, end=self.time_end)


class ModalityCoverage(BaseModel):
    modality: Modality
    covered_fraction: float
    missing_intervals: list[tuple[datetime, datetime]] = Field(default_factory=list)
    sensors: list[str] = Field(default_factory=list)
    degraded: bool = False


class SubRegionCoverage(BaseModel):
    region: str
    covered_fraction: float
    blind_fraction: float
    missing_intervals: list[tuple[datetime, datetime]] = Field(default_factory=list)


class CoverageReport(BaseModel):
    """Result of an independent coverage query. This is never derived from documents."""

    region: str
    time_range: TimeRange
    requested_modalities: list[Modality]
    status: CoverageStatus
    covered_fraction: float = Field(ge=0.0, le=1.0)
    best_modality_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Mean coverage_confidence of the ledger entries that contributed. Sensor *quality*,
    #: kept separate from the geometric/temporal fraction so the two never get conflated.
    coverage_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Fraction of the (sub-region x time) volume for which the ledger holds no entry at
    #: all. This is "we do not know whether we looked", distinct from an asserted
    #: NOT_OBSERVED ("we know we did not look").
    no_information_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    blind_subregions: list[str] = Field(default_factory=list)
    missing_intervals: list[tuple[datetime, datetime]] = Field(default_factory=list)
    missing_modalities: list[Modality] = Field(default_factory=list)
    degraded_modalities: list[Modality] = Field(default_factory=list)
    per_modality: list[ModalityCoverage] = Field(default_factory=list)
    per_subregion: list[SubRegionCoverage] = Field(default_factory=list)
    contributing_sensors: list[str] = Field(default_factory=list)
    ledger_entries: list[str] = Field(default_factory=list)
    absence_claim_supported: bool = False
    absence_block_reason: str = ""

    def human_summary(self) -> str:
        pct = round(self.covered_fraction * 100)
        gaps = ", ".join(
            f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in self.missing_intervals
        )
        base = f"{self.status.value}: {pct}% of {self.region} over {self.time_range.label()}"
        if gaps:
            base += f"; gaps: {gaps}"
        if self.missing_modalities:
            base += f"; no data from: {', '.join(m.value for m in self.missing_modalities)}"
        return base


# --------------------------------------------------------------------------------------
# Source records / documents
# --------------------------------------------------------------------------------------
class SourceRecord(BaseModel):
    """A single heterogeneous observation record, normalised for indexing."""

    record_id: str
    modality: Modality
    sensor: str
    timestamp: datetime
    region: str
    text: str
    reliability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    position: tuple[float, float] | None = None
    heading: float | None = None
    speed: float | None = None
    object_type: str | None = None
    classification: str | None = None
    frequency_mhz: float | None = None
    mmsi: str | None = None
    vessel_name: str | None = None
    track_id: str | None = None
    is_absence_report: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    record_id: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    why: list[str] = Field(default_factory=list)
    subquery_id: str | None = None


# --------------------------------------------------------------------------------------
# Query decomposition
# --------------------------------------------------------------------------------------
class SubQuery(BaseModel):
    subquery_id: str
    type: SubQueryType
    text: str
    entities: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    region: str | None = None
    preferred_modalities: list[Modality] = Field(default_factory=list)
    hard_modalities: list[Modality] = Field(default_factory=list)
    rationale: str = ""


class QueryPlan(BaseModel):
    raw_question: str
    intent: QueryIntent
    entities: list[str] = Field(default_factory=list)
    region: str | None = None
    time_range: TimeRange | None = None
    preferred_modalities: list[Modality] = Field(default_factory=list)
    hard_modalities: list[Modality] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    requested_relationship: str | None = None
    subqueries: list[SubQuery] = Field(default_factory=list)
    decomposer: str = "deterministic"
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------
class Evidence(BaseModel):
    evidence_id: str
    source_id: str
    source: Modality
    sensor: str
    claim: str
    state: EvidenceState
    region: str
    time_range: TimeRange
    reliability: float = Field(ge=0.0, le=1.0)
    recency: float = Field(default=1.0, ge=0.0, le=1.0)
    retrieval_score: float = 0.0
    entities: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id,
            "claim": self.claim,
            "source": self.source.value,
            "sensor": self.sensor,
            "state": self.state.value,
            "region": self.region,
            "time_range": self.time_range.label(),
            "reliability": round(self.reliability, 3),
        }


class ContradictionClaim(BaseModel):
    evidence_id: str
    source: Modality
    sensor: str
    value: Any
    reliability: float


class Contradiction(BaseModel):
    contradiction_id: str
    dimension: ContradictionDimension
    entity: str | None
    region: str
    time_range: TimeRange
    claims: list[ContradictionClaim]
    reason: str
    severity: float = Field(ge=0.0, le=1.0)
    severity_label: Literal["low", "moderate", "high"] = "low"
    resolvable: bool = False


# --------------------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------------------
class ConfidenceBreakdown(BaseModel):
    coverage: float
    source_reliability: float
    recency: float
    retrieval_agreement: float
    evidence_support: float
    contradiction_penalty: float
    stale_penalty: float
    absence_coverage_multiplier: float = 1.0
    unknown_ceiling_applied: bool = False
    raw_score: float = 0.0
    confidence: float = 0.0

    def explain(self) -> list[str]:
        parts = [
            f"coverage={self.coverage:.2f}",
            f"reliability={self.source_reliability:.2f}",
            f"recency={self.recency:.2f}",
            f"agreement={self.retrieval_agreement:.2f}",
            f"evidence_support={self.evidence_support:.2f}",
        ]
        if self.contradiction_penalty:
            parts.append(f"contradiction_penalty=-{self.contradiction_penalty:.2f}")
        if self.stale_penalty:
            parts.append(f"stale_penalty=-{self.stale_penalty:.2f}")
        if self.absence_coverage_multiplier < 1.0:
            parts.append(f"absence_coverage_multiplier={self.absence_coverage_multiplier:.2f}")
        if self.unknown_ceiling_applied:
            parts.append("unknown_ceiling_applied")
        return parts


# --------------------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------------------
class TraceStage(BaseModel):
    name: str
    latency_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    trace_id: str
    question: str
    stages: list[TraceStage] = Field(default_factory=list)
    retrieved: list[RetrievedDoc] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    reasoning_latency_ms: float = 0.0

    def stage(self, name: str) -> TraceStage | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None


# --------------------------------------------------------------------------------------
# Final response
# --------------------------------------------------------------------------------------
class CoverageSummary(BaseModel):
    region: str
    time_range: str
    coverage_fraction: float
    status: CoverageStatus
    modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    missing_intervals: list[str] = Field(default_factory=list)


class MissionAnswer(BaseModel):
    answer: str
    state: AnswerState
    confidence: float = Field(ge=0.0, le=1.0)
    coverage: CoverageSummary
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    confidence_breakdown: ConfidenceBreakdown | None = None
    plan: QueryPlan | None = None
    trace: QueryTrace | None = None
    operator_view: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
