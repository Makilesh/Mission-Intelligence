"""Run the full evaluation suite and print a human-readable summary.

Usage:
    python scripts/run_evaluation.py                 # everything
    python scripts/run_evaluation.py --retrieval     # retrieval-only benchmark
    python scripts/run_evaluation.py --calibration   # confidence calibration sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import golden, harness  # noqa: E402


def _print(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="retrieval-only benchmark")
    parser.add_argument("--end-to-end", action="store_true", help="end-to-end benchmark")
    parser.add_argument("--calibration", action="store_true", help="confidence calibration")
    parser.add_argument("--quiet", action="store_true", help="suppress per-query trace logs")
    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.INFO)

    path = golden.write()
    print(f"golden set: {path} ({len(golden.build())} questions)")

    if not (args.retrieval or args.end_to_end or args.calibration):
        report = harness.run_all()
        _print("environment", report["environment"])
        _print("retrieval-only", report["retrieval_only"]["summary"])
        _print("end-to-end", report["end_to_end"]["summary"])
        _print("calibration", report["calibration"]["summary"])
        print(f"\nfull report -> {report['report_path']}")
        return

    if args.retrieval:
        _print("retrieval-only", harness.run_retrieval_benchmark()["summary"])
    if args.end_to_end:
        _print("end-to-end", harness.run_end_to_end()["summary"])
    if args.calibration:
        _print("calibration", harness.run_calibration()["summary"])


if __name__ == "__main__":
    main()
