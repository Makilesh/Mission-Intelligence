# BUILD PLAN — Mission Intelligence Coverage-Aware Retrieval & Reasoning System

> Self-authored execution prompt derived from `p1.md`. This is the contract I hold myself to.

## Prime directive

**An empty retrieval result is not evidence of absence.** Observation coverage is first-class,
persisted, and queried *independently of the retrieval index*. Every answer must be one of
`PRESENCE | OBSERVED_ABSENCE | UNKNOWN | CONTRADICTION`, and the difference between
"we looked and saw nothing" and "we never looked" must be structurally impossible to blur.

## Non-negotiable engineering rules

1. No hardcoded answers, no hardcoded confidence, no LLM-self-reported confidence.
2. Coverage is NEVER inferred from retrieval hits.
3. Contradictions are surfaced, never resolved by majority vote or top-1 retrieval score.
4. Deterministic layers (time math, geo filtering, confidence, contradiction rules, evidence
   state) run in Python. The LLM only phrases what the deterministic layer already decided,
   and is fenced by a post-hoc grounding validator.
5. Every component runs offline and deterministically (seeded). LLM providers are pluggable;
   the default provider is a deterministic template synthesizer so evaluation is reproducible.
6. Retrieval is benchmarked WITHOUT the generator, separately from end-to-end answers.
7. Category filters from query classification are *soft preferences*, never hard constraints
   (parent-filter failure prevention).
8. Typed Pydantic models everywhere; async where it buys parallelism.

## Phase order (do not reorder; do not proceed on a failing foundational test)

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | Synthetic multimodal dataset + planted failure cases A–E | dataset self-consistency tests |
| 2 | Coverage ledger (independent, queryable) | ledger unit tests |
| 3 | Coverage tests + planted absence/blind-window tests | blind window detected, absence distinct from unknown |
| 4 | Dense (ST/FAISS w/ offline fallback) + BM25 + RRF + metadata reranker | retrieval-only benchmark runs |
| 5 | Query decomposition (deterministic + optional LLM) | sub-query extraction tests |
| 6 | Evidence state model + aggregator | state classification tests |
| 7 | Contradiction engine | true vs false contradiction severity tests |
| 8 | Deterministic confidence model | monotonic in coverage (proved by test) |
| 9 | LLM reasoning layer + grounding guard | no fabricated evidence IDs |
| 10 | Evaluation harness, 50 golden questions | all metrics computed, fabrication rate = 0 |
| 11 | Failure injection framework | 5 injectors behave as specified |
| 12 | FastAPI service | all 7 endpoints |
| 13 | Streamlit operator UI | coverage bar, timeline, evidence, contradictions, trace |
| 14 | Docker + README + RESULTS.md + DECISIONS_LOG.md + architecture diagram | full test suite green |

After every phase: run tests → record metrics → append to `DECISIONS_LOG.md`.

## Key design decisions locked before coding

- **Region model is hierarchical**: sectors decompose into atomic grids. Coverage is computed as a
  spatio-temporal fraction over (atomic region x time), per modality, then aggregated.
- **Absence gate**: an `OBSERVED_ABSENCE` claim requires (a) covered_fraction >= 0.85 AND
  (b) no atomic sub-region blind for >= 50% of the queried window. Otherwise -> `UNKNOWN`.
  This is what makes Demo 1 and Demo 2 differ even though both target Sector Alpha.
  *(Revised during Phase 10 to 35%: at 50% a point-in-time question at 04:09, widened to
  +/-5 min, could still claim an observed absence with grid_b7 dark for 40% of it. See
  D2.5 in DECISIONS_LOG.md.)*
- **Coverage status weights**: OBSERVED 1.0, PARTIALLY_OBSERVED 0.5, DEGRADED 0.4,
  NOT_OBSERVED 0.0, UNKNOWN 0.0. `NOT_OBSERVED` (asserted blind) is distinct from `UNKNOWN`
  (no ledger entry at all) — the ledger must be able to say "I have no idea".
- **Confidence** is a weighted deterministic score, multiplied by an absence-specific coverage
  term so that absence claims decay hard as coverage decays; monotonicity is unit-tested.
