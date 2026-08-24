"""Generate the synthetic mission dataset and the coverage ledger.

The two artefacts are produced by independent code paths on purpose: the ledger must never
be derivable from the records, or "no record" would silently become "no coverage".

    python scripts/generate_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import COVERAGE_DIR, SYNTHETIC_DIR  # noqa: E402
from app.dataset.generator import generate  # noqa: E402


def main() -> None:
    records, coverage = generate(write=True)
    manifest = json.loads((SYNTHETIC_DIR / "manifest.json").read_text(encoding="utf-8"))

    print(f"records          : {len(records)}")
    print(f"coverage entries : {len(coverage)}")
    print(f"by modality      : {json.dumps(manifest['modality_counts'])}")
    print(f"records          -> {SYNTHETIC_DIR / 'records.json'}")
    print(f"coverage ledger  -> {COVERAGE_DIR / 'ledger.json'}")
    print(f"manifest         -> {SYNTHETIC_DIR / 'manifest.json'}")
    print("\nplanted cases:")
    for name, detail in manifest["planted_cases"].items():
        print(f"  {name}: {json.dumps(detail)}")


if __name__ == "__main__":
    main()
