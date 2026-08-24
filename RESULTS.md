# Results

All numbers below are produced by `python scripts/run_evaluation.py` and
`python scripts/run_failure_injection.py`, and are reproducible from a clean checkout
(`python -m app.dataset.generator && python scripts/build_index.py`).

## Reproducibility record

Every benchmark run records its full environment. This run:

| Field | Value |
|---|---|
| Dataset version | `mission-synth-v1.0.0` |
| Query set version | `golden-v1.0.0` |
| Documents indexed | 144 |
| Coverage ledger entries | 117 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Dense backend | `faiss.IndexFlatIP` |
| Sparse backend | `rank_bm25.BM25Okapi` |
| Retrieval config | RRF `k=60`, `top_k_dense=25`, `top_k_sparse=25`, `final_k=12`, rerank on |
| LLM provider / model | `deterministic` / `template-synthesis`, temperature 0.0 |
| Python / platform | 3.12.10 / Windows-11 |

> The default reasoning provider is a deterministic template synthesiser. That is a
> deliberate methodological choice, not a shortcut: retrieval and coverage metrics must not
> move when a model provider changes. Switching `LLM_PROVIDER` to `anthropic` or `openai`
> routes phrasing through a real model, still fenced by the grounding validator.

---

## 1. Retrieval-only benchmark (no generator)

Measured as `query → retrieval → evidence`, with the LLM entirely out of the loop. Ground
truth (`relevant_ids`) is computed by an oracle over the corpus — region, window, modality —
which the retriever plays no part in defining.

| Metric | Decomposed pipeline | Single flat query (baseline) |
|---|---|---|
| Recall@5 | **0.487** | 0.407 |
| Recall@10 | **0.618** | 0.568 |
| Recall@20 | **0.726** | 0.663 |
| Precision@5 | 0.404 | — |
| Precision@10 | 0.302 | — |
| Precision@20 | 0.190 | — |
| MRR | **0.704** | 0.675 |
| Benchmark wall time | 3.4 s for 50 questions | — |

Decomposing into per-modality sub-questions and retrieving for each independently is worth
**+0.08 recall@5, +0.06 recall@20 and +0.03 MRR** over a single blended query. The gain is
largest at small `k`, which is where it matters: it is the difference between a channel
appearing in the evidence set and being crowded out.

**Trap rate@10 = 0.50.** Half the planted distractors *are* retrieved. This is reported
deliberately rather than tuned away: the traps are lexically near-identical to genuine
records, and a dense retriever should surface them. What matters is what happens next —
see the trap citation rate below.

---

## 2. End-to-end benchmark (50 golden questions)

| Metric | Value | Target |
|---|---|---|
| **Fabrication rate on planted absences** | **0.000** | **0%** |
| Fabrication rate overall | 0.000 | — |
| Answer-state accuracy | 1.000 (50/50) | — |
| Unknown recall (questions whose truth is "we cannot know") | 1.000 | — |
| Observed-absence accuracy | 1.000 | — |
| Coverage classification accuracy | 1.000 | — |
| Blind-window detection | 1.000 | — |
| Contradiction recall | 1.000 | — |
| Contradiction false-positive rate | 0.000 | — |
| Ungrounded evidence citations | 0.000 | — |
| Trap citation rate (trap supporting a claim) | 0.000 | — |
| Gap reporting rate (gap required and reported) | 1.000 | — |
| Mean evidence coverage | 0.785 | — |
| Mean / p95 latency | 45 ms / 57 ms | seconds-level |

### By category

| Category | n | State accuracy |
|---|---|---|
| presence | 9 | 1.00 |
| blind_window | 7 | 1.00 |
| planted_absence | 6 | 1.00 |
| partial_coverage | 6 | 1.00 |
| contradiction | 5 | 1.00 |
| multi_hop | 4 | 1.00 |
| stale | 4 | 1.00 |
| trap | 4 | 1.00 |
| coverage | 4 | 1.00 |
| false_contradiction | 1 | 1.00 |

### Mean confidence by answer state

| State | n | Mean confidence |
|---|---|---|
| `OBSERVED_ABSENCE` | 7 | 0.81 |
| `PRESENCE` | 13 | 0.63 |
| `CONTRADICTION` | 11 | 0.46 |
| `UNKNOWN` | 19 | 0.27 |

The ordering is the point: the system is most confident when it both looked and saw
nothing, and least confident when it could not look.

### On the 100% state accuracy

This is a synthetic world of 144 documents whose golden set was authored alongside it, so
100% should be read as "the planted failure modes are all handled", not as evidence of
general capability. Two guards keep it honest:

* Expectations for the planted cases (A–E) are **hand-pinned** in `app/evaluation/golden.py`
  so no oracle bug can quietly redefine the right answer.
* During development the harness caught six cases where the *expectation* was wrong rather
  than the system — questions where contacts genuinely existed in a partially covered
  window, and the correct answer was `PRESENCE` with a coverage caveat, not `UNKNOWN`.
  Those were corrected against the world data (see the Phase 10 entries in
  `DECISIONS_LOG.md`), not by relaxing the system.

It also took four genuine engineering fixes to get there, each documented: modality- and
entity-scoped claims, contradiction scope-gating, a dataset self-consistency invariant, and
a second retrieval hop for association questions.

---

## 3. Confidence calibration under injected coverage loss

Coverage is artificially reduced to 100 / 80 / 60 / 40 / 20 % of the queried window and the
same question is re-asked.

| Injected level | Coverage fraction | State | Confidence |
|---|---|---|---|
| 100 % | 0.95 | `OBSERVED_ABSENCE` | **0.84** |
| 80 % | 0.75 | `UNKNOWN` | **0.394** |
| 60 % | 0.55 | `UNKNOWN` | **0.349** |
| 40 % | 0.39 | `UNKNOWN` | **0.312** |
| 20 % | 0.20 | `UNKNOWN` | **0.270** |

*(Sector Alpha, 04:00–04:20; the other two calibration questions behave the same way.)*

| Metric | Value |
|---|---|
| Pearson(confidence, coverage) | **0.68** |
| Spearman(confidence, coverage) | **0.78** |
| Monotonicity violations | **0** |

Two honest notes:

1. **Spearman is the more meaningful figure here.** The curve has a deliberate step at the
   absence gate: crossing below 85 % coverage flips the state to `UNKNOWN` and drops
   confidence sharply. Pearson penalises that step for being non-linear; the step is the
   intended behaviour.
2. The first version of the model applied a flat ceiling to `UNKNOWN` confidence, which
   made "unknown at 75 % coverage" numerically identical to "unknown at 20 %" (Spearman
   0.61). That flattening was itself a small instance of the failure this system exists to
   prevent, so the ceiling was changed to scale with coverage. Spearman 0.61 → **0.78**.

Monotonicity is additionally proved by unit test, not just observed:
`test_confidence_is_monotone_in_coverage` and
`test_confidence_strictly_decreases_with_fraction_only`.

---

## 4. Failure injection

| Injection | Before | After | Verdict |
|---|---|---|---|
| **Sensor dropout** (`radar_01` off 04:00–04:20) | `OBSERVED_ABSENCE`, cov 0.95, conf 0.84 | `UNKNOWN`, cov 0.71, conf 0.45 | ✅ coverage ↓, confidence ↓, absence claim withdrawn |
| **Stale data** (previous-mission sweep, identical wording) | `OBSERVED_ABSENCE`, conf 0.845 | `OBSERVED_ABSENCE`, conf 0.841 | ✅ classified `STALE`, excluded from the claim, small recency penalty |
| **False contradiction** (two unreliable sources) | no contradictions | identity 0.11 **low**, heading 0.09 **low**; state unchanged | ✅ flagged, low severity, does not hijack the answer |
| **True contradiction** (radar vs AIS, both reliable) | `PRESENCE`, conf 0.73 | `CONTRADICTION`, conf 0.43, heading 0.80 **high** | ✅ high severity, state changes, confidence drops |
| **Retrieval poisoning** (5 near-duplicates claiming "full coverage confirmed, Grid B7 included", correct region *and* window) | `UNKNOWN`, cov 0.75 | `UNKNOWN`, cov 0.75 | ✅ retrieved, downgraded to `PARTIAL_COVERAGE`, no absence asserted |

The poisoning case is the sharpest test in the suite. The injected records carry the right
region, the right timestamp, high stated reliability, and text explicitly asserting that
Grid B7 was covered. They rank into the evidence set. They change nothing — because
coverage is not something a document is allowed to assert.

---

## 5. Latency

| Stage | Typical |
|---|---|
| Decomposition | 0.3 ms |
| Dense + sparse retrieval (concurrent with coverage check) | 40–45 ms |
| Coverage check | 0.03 ms |
| Fusion + rerank | 0.02 ms |
| Evidence classification | 2–3 ms |
| Contradiction detection | 0.1 ms |
| Confidence calculation | 0.05 ms |
| Synthesis (deterministic provider) | 0.04 ms |
| **Total (mean / p95)** | **45 ms / 57 ms** |

Retrieval dominates, and it is already run concurrently with the coverage lookup via
`asyncio.gather`. Every deterministic operation — time arithmetic, geographic filtering,
confidence, contradiction rules, metric computation — is Python, outside any model call.
With a hosted LLM provider, add one network round-trip; the pipeline issues exactly one LLM
call per query, and none at all in the default configuration.

---

## 6. Test suite

74 tests, all passing.

| Suite | Tests | Covers |
|---|---|---|
| `tests/unit/test_coverage_ledger.py` | 17 | absence gate matrix, `NOT_OBSERVED` vs `UNKNOWN`, blind-window detection, dropout, coverage-loss monotonicity |
| `tests/unit/test_dataset_invariants.py` | 10 | planted cases intact, no contact in the absence window, nothing in the blackout, region hierarchy, ledger/corpus independence |
| `tests/unit/test_decomposition.py` | 17 | intent, time expressions, region aliases (including the trap annex), entity extraction, soft vs hard modality filters |
| `tests/unit/test_evidence_and_confidence.py` | 11 | evidence states, stale/low-confidence, detections vs documents, confidence monotonicity, absence decay, contradiction penalty |
| `tests/failure_injection/test_injections.py` | 8 | all five injectors, severity ordering, contradictions never resolved |
| `tests/integration/test_api.py` | 11 | all seven endpoints, trace completeness, coverage endpoint independence, ingest |

---

## 7. What this does not establish

* The corpus is synthetic and small (144 records, 117 coverage entries). Retrieval numbers
  would not transfer to a real archive.
* The 0.00 fabrication rate is a property of the **pipeline's structure** — the ledger gate,
  the evidence state machine, the deterministic state decision. With a hosted LLM doing the
  phrasing, the grounding validator still rejects any answer citing unknown evidence IDs,
  but prose-level faithfulness would need its own evaluation.
* Coverage adequacy weights, severity thresholds and the absence gate (≥ 0.85 coverage and
  no sub-region blind for ≥ 35 % of the window) are hand-chosen and documented, not fitted
  to any sensor model.
* Contradiction detection is rule-based across six dimensions. It will not catch a
  disagreement that only exists in free-text nuance.
