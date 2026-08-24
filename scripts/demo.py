"""Run the five required demonstration scenarios and print operator views.

    python scripts/demo.py --quiet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.coverage.ledger import get_ledger  # noqa: E402
from app.dataset.world import t  # noqa: E402
from app.evaluation import failure_injection as FI  # noqa: E402
from app.models.schemas import TimeRange  # noqa: E402
from app.reasoning.pipeline import answer_question  # noqa: E402
from app.retrieval.hybrid import get_retriever  # noqa: E402

DEMOS = [
    ("Demo 1 — Observed absence",
     "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
     "No contacts observed; coverage ~95%; high confidence."),
    ("Demo 2 — Unknown (blind window)",
     "Were there any contacts in Sector Alpha between 04:07 and 04:11?",
     "Cannot determine; insufficient coverage; must NOT say 'no contacts detected'."),
    ("Demo 3 — Contradiction",
     "What vessel was detected near Grid B7?",
     "Sources disagree (AIS V-17 vs mission report V-21); not reconciled."),
    ("Demo 4 — Multi-hop association",
     "Is the vessel detected at 05:20 the same vessel tracked at 04:00?",
     "Kinematic + custody + identity reasoning over the retrieved chain."),
]

DEMO5 = (
    "Demo 5 — Sensor dropout",
    "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
    "Same question, radar_01 disabled: coverage down, confidence down, answer more uncertain.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the structured response too")
    args = parser.parse_args()
    if args.quiet:
        logging.disable(logging.INFO)

    retriever = get_retriever()
    ledger = get_ledger()

    for title, question, expectation in DEMOS:
        answer = asyncio.run(answer_question(question, retriever=retriever, ledger=ledger))
        print("\n" + "=" * 96)
        print(f"{title}\nQ: {question}\nExpected: {expectation}\n")
        print(answer.operator_view)
        if args.json:
            print("\nSTRUCTURED RESPONSE:")
            print(
                json.dumps(
                    {
                        "answer": answer.answer,
                        "state": answer.state.value,
                        "confidence": answer.confidence,
                        "coverage": answer.coverage.model_dump(mode="json"),
                        "evidence": answer.evidence[:4],
                        "gaps": answer.gaps,
                        "contradictions": answer.contradictions,
                    },
                    indent=2,
                )
            )

    # ---- Demo 5: same question, one sensor removed ------------------------------------
    title, question, expectation = DEMO5
    window = TimeRange(start=t(4, 0), end=t(4, 20))
    before = asyncio.run(answer_question(question, retriever=retriever, ledger=ledger))
    injection = FI.sensor_dropout(ledger, "radar_01", window)
    after = asyncio.run(answer_question(question, retriever=retriever, ledger=injection.ledger))

    print("\n" + "=" * 96)
    print(f"{title}\nQ: {question}\nExpected: {expectation}\n")
    print(f"BEFORE  state={before.state.value:17s} coverage={before.coverage.coverage_fraction:.0%}"
          f"  confidence={before.confidence:.2f}")
    print(f"AFTER   state={after.state.value:17s} coverage={after.coverage.coverage_fraction:.0%}"
          f"  confidence={after.confidence:.2f}")
    print(f"DELTA   coverage {after.coverage.coverage_fraction - before.coverage.coverage_fraction:+.2f}"
          f"   confidence {after.confidence - before.confidence:+.2f}\n")
    print(after.operator_view)


if __name__ == "__main__":
    main()
