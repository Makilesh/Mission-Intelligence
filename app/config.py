"""Central configuration. All tunables live here so experiments are reproducible."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
COVERAGE_DIR = DATA_DIR / "coverage"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "evaluation"

DATASET_VERSION = "mission-synth-v1.0.0"
QUERY_SET_VERSION = "golden-v1.0.0"

# Mission clock. Everything in the synthetic world happens inside this window.
MISSION_START = datetime(2026, 8, 22, 3, 30, tzinfo=timezone.utc)
MISSION_END = datetime(2026, 8, 22, 6, 0, tzinfo=timezone.utc)
# "now" for the operator; used for recency scoring and relative time phrases.
MISSION_NOW = datetime(2026, 8, 22, 5, 40, tzinfo=timezone.utc)

RANDOM_SEED = 20260822


class CoverageConfig(BaseModel):
    """Weights and thresholds for the coverage ledger."""

    status_weight: dict[str, float] = Field(
        default_factory=lambda: {
            "OBSERVED": 1.0,
            "PARTIALLY_OBSERVED": 0.5,
            "DEGRADED": 0.4,
            "NOT_OBSERVED": 0.0,
            "UNKNOWN": 0.0,
        }
    )
    observed_threshold: float = 0.95
    partial_threshold: float = 0.05
    # An OBSERVED_ABSENCE claim needs at least this much coverage...
    absence_coverage_threshold: float = 0.85
    # ...and no atomic sub-region may be blind for more than this fraction of the window.
    absence_max_blind_subregion_fraction: float = 0.35
    # Sub-interval granularity (seconds) used when rasterising coverage.
    resolution_seconds: int = 30


class RetrievalConfig(BaseModel):
    dense_model: str = os.getenv("DENSE_MODEL", "all-MiniLM-L6-v2")
    dense_backend: Literal["auto", "sentence_transformers", "hashing"] = os.getenv(
        "DENSE_BACKEND", "auto"
    )  # type: ignore[assignment]
    embedding_dim_fallback: int = 384
    top_k_dense: int = 25
    top_k_sparse: int = 25
    rrf_k: int = 60
    final_k: int = 12
    rerank: bool = True
    # Soft-preference weights for the metadata-aware reranker (parent-filter prevention:
    # metadata mismatches DOWN-WEIGHT, they never exclude).
    w_fusion: float = 1.0
    w_region: float = 0.35
    w_time: float = 0.35
    w_modality: float = 0.20
    w_entity: float = 0.45
    w_lexical: float = 0.25


class ConfidenceConfig(BaseModel):
    w_coverage: float = 0.40
    w_reliability: float = 0.22
    w_recency: float = 0.13
    w_agreement: float = 0.15
    w_evidence: float = 0.10
    contradiction_weight: float = 0.55  # scales contradiction severity into a penalty
    unknown_ceiling: float = 0.45  # a claim under UNKNOWN state can never be confident
    stale_penalty: float = 0.15
    max_evidence_saturation: int = 6


class LLMConfig(BaseModel):
    provider: Literal["deterministic", "anthropic", "openai"] = os.getenv(
        "LLM_PROVIDER", "deterministic"
    )  # type: ignore[assignment]
    model: str = os.getenv("LLM_MODEL", "claude-opus-5")
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: float = 30.0


class Settings(BaseModel):
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset_version: str = DATASET_VERSION
    query_set_version: str = QUERY_SET_VERSION


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, SYNTHETIC_DIR, COVERAGE_DIR, INDEX_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
