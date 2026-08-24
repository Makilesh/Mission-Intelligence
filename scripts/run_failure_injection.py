"""Run every failure injection and report the before/after deltas.

    python scripts/run_failure_injection.py --quiet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR  # noqa: E402
from app.coverage.ledger import get_ledger  # noqa: E402
from app.dataset.world import t  # noqa: E402
from app.evaluation import failure_injection as FI  # noqa: E402
from app.models.schemas import TimeRange  # noqa: E402
from app.reasoning.pipeline import answer_question  # noqa: E402
from app.retrieval.hybrid import HybridRetriever, get_retriever  # noqa: E402

QUESTIONS = {
    "sensor_dropout": "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
    "stale_data": "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
    "false_contradiction": "What contacts were observed in Grid A1 between 05:20 and 05:30?",
    "true_contradiction": "What contacts were observed in Grid A2 between 05:20 and 05:30?",
    "retrieval_poisoning": "Were there any contacts in Sector Alpha between 04:07 and 04:11?",
}


def _run(question, retriever, ledger, corpus):  # noqa: ANN001
    return asyncio.run(
        answer_question(question, retriever=retriever, ledger=ledger, corpus=corpus)
    )


def _summarise(answer) -> dict:  # noqa: ANN001
    return {
        "state": answer.state.value,
        "confidence": answer.confidence,
        "coverage_fraction": answer.coverage.coverage_fraction,
        "contradictions": [
            {"dimension": c["dimension"], "severity": c["severity"], "label": c["severity_label"]}
            for c in answer.contradictions
        ],
        "gaps": len(answer.gaps),
        "claim_evidence": answer.meta.get("claim_evidence_ids", [])[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.quiet:
        logging.disable(logging.INFO)

    base_retriever = get_retriever()
    base_ledger = get_ledger()
    results = []

    for name in FI.ALL_INJECTIONS:
        question = QUESTIONS[name]
        before = _run(question, base_retriever, base_ledger, base_retriever.corpus)

        if name == "sensor_dropout":
            injection = FI.sensor_dropout(
                base_ledger, "radar_01", TimeRange(start=t(4, 0), end=t(4, 20))
            )
            retriever, ledger, corpus = base_retriever, injection.ledger, base_retriever.corpus
        else:
            injection = FI.ALL_INJECTIONS[name]()
            corpus = FI.apply_records(base_retriever.corpus, injection.records)
            retriever = HybridRetriever(corpus, embedder=base_retriever.embedder)
            ledger = base_ledger

        after = _run(question, retriever, ledger, corpus)
        results.append(
            {
                "injection": name,
                "description": injection.description,
                "expectation": injection.expectation,
                "question": question,
                "before": _summarise(before),
                "after": _summarise(after),
                "delta": {
                    "confidence": round(after.confidence - before.confidence, 4),
                    "coverage_fraction": round(
                        after.coverage.coverage_fraction - before.coverage.coverage_fraction, 4
                    ),
                    "state_changed": before.state.value != after.state.value,
                },
            }
        )
        print(f"\n=== {name} ===")
        print(f"  {injection.description}")
        print(f"  expect: {injection.expectation}")
        print(f"  before: {json.dumps(_summarise(before))}")
        print(f"  after : {json.dumps(_summarise(after))}")

    path = EVAL_DIR / "failure_injection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
