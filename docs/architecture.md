# Architecture

## Text diagram

```text
                    ┌──────────────────────────┐
                    │      Operator Query      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Query Decomposition      │   deterministic rules:
                    │ + Entity/Time/Region     │   times, regions, identifiers,
                    │ Extraction               │   intent, modality preference
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼─────────────────┐
                │                │                 │      run concurrently
                ▼                ▼                 ▼      (asyncio.gather)
       ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
       │ Coverage Ledger│ │ Dense Search │ │ Sparse Search │
       │  (independent) │ │ MiniLM+FAISS │ │     BM25      │
       └───────┬────────┘ └──────┬───────┘ └───────┬───────┘
               │                 │                 │
               │                 └────────┬────────┘
               │                          ▼
               │                 ┌────────────────┐
               │                 │ RRF Fusion     │  ranks, not scores
               │                 │ + Reranking    │  metadata = soft preference
               │                 └───────┬────────┘
               │                         │
               └──────────────┬──────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Evidence Aggregator  │  PRESENCE
                    │                      │  OBSERVED_ABSENCE
                    │  a negative report   │  UNOBSERVED
                    │  is only an absence  │  PARTIAL_COVERAGE
                    │  if the LEDGER says  │  CONTRADICTION
                    │  we actually looked  │  STALE
                    └──────────┬───────────┘  LOW_CONFIDENCE
                               ▼
                    ┌──────────────────────┐
                    │ Contradiction Engine │  identity · position · heading
                    │  cross-source only   │  speed · classification · time
                    │  never resolved      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Deterministic        │  coverage · reliability · recency
                    │ Confidence /         │  agreement · evidence count
                    │ Coverage Calculation │  − contradiction · − staleness
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ LLM Reasoning Layer  │  phrasing only
                    │ + Answer Synthesis   │  grounding-validated
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Operator Response    │
                    │ + Evidence + Gaps    │
                    └──────────────────────┘
```

The one structural rule the diagram encodes: **nothing flows from the retrieval path into
the coverage path.** The ledger is reachable from the query and from the evidence
classifier, never from a retrieval result.

## Components

### `app/coverage/ledger.py` — the Coverage Ledger
Stores `CoverageEntry` assertions: `(region, time_start, time_end, modality, sensor,
coverage_status, coverage_confidence, reason)`. `check(region, time_range, modalities)`
rasterises the query volume over **(atomic sub-region × time slot)** and returns a
`CoverageReport` with the covered fraction, per-modality and per-sub-region breakdowns,
missing intervals, missing modalities, degraded modalities, a `no_information_fraction`,
and the absence gate verdict.

Three distinctions the ledger refuses to blur:

1. `NOT_OBSERVED` (we know we did not look) vs `UNKNOWN` (no entry exists at all).
2. Geometric/temporal coverage (`covered_fraction`) vs sensor quality (`coverage_quality`).
3. Coverage by *any* modality vs coverage by a modality *capable of the claim*
   (`MODALITY_ADEQUACY`: radar 1.00, EO/IR 0.75, imagery 0.60, RF 0.45, AIS 0.35).

### `app/retrieval/` — hybrid retrieval
Dense (`sentence-transformers/all-MiniLM-L6-v2` + `faiss.IndexFlatIP`, with a deterministic
TF-IDF+SVD / numpy fallback), sparse (`BM25Okapi` with a tokenizer that preserves `T-88`
and `V-17`), fused with Reciprocal Rank Fusion (`k=60`). Every decomposed sub-question is
retrieved for independently and merged round-robin, so a narrow channel cannot starve a
broad one.

Reranking applies region, time, modality, entity and lexical affinity as **soft
preferences**. The only hard constraint is `SubQuery.hard_modalities`, populated solely
when the operator says something like "using only radar".

### `app/evidence/` — evidence state model
`classify()` assigns one of seven states per record. The critical branch: a record with
`is_absence_report=True` calls the ledger for its own region and the queried window, and is
promoted to `OBSERVED_ABSENCE` only if `absence_claim_supported` is true.

`aggregate()` additionally materialises **gap evidence** — every unobserved interval and
every missing modality becomes a first-class `UNOBSERVED` evidence record sourced from
`coverage_ledger`, so a gap is something the answer can cite rather than something it
silently lacks.

### `app/contradiction/engine.py`
Clusters co-located, near-simultaneous evidence and compares six dimensions. Two gates
prevent false positives: contradictions must be **cross-source** (two AIS reports naming
two vessels is two vessels), and kinematic comparisons require **co-location** (both
reports carrying positions that agree to within tolerance). Severity =
`magnitude × weakest-source-quality × dimension-weight`, so two unreliable sources
disagreeing scores low and two trusted sources disagreeing scores high. `resolvable` is
always `False`.

### `app/confidence/model.py`
`decide_state()` picks the answer state by fixed precedence (in-scope contradiction →
presence → ledger-supported absence → unknown). `calculate_confidence()` combines coverage,
source reliability, recency, retrieval agreement and evidence count, then applies a
contradiction penalty, a staleness penalty, an absence-specific coverage multiplier and an
UNKNOWN ceiling. Monotonicity in coverage is unit-tested.

### `app/reasoning/`
`decomposition.py` (rules), `association.py` (multi-hop track association: kinematics,
transit plausibility, identity, custody), `llm.py` (three providers + grounding validator),
`pipeline.py` (orchestration, tracing, multi-hop expansion).

For association questions the pipeline performs a **second retrieval hop**: hop 4
(identity) cannot be planned until hop 2 (which contact) has resolved, so the anchor
contact is chosen first and then a targeted identity query is issued against its region and
time.

### `app/observability.py`
Every query produces a `QueryTrace` with per-stage latencies:
`decomposition → dense_sparse_retrieval → coverage_check → fusion_rerank →
evidence_classification → contradiction_detection → confidence_calculation →
[multihop_expansion] → [association_analysis] → llm_synthesis`, plus totals and a
structured JSON log line.
