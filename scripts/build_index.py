"""Build (and warm) the hybrid retrieval index, printing the reproducibility manifest."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.hybrid import get_retriever  # noqa: E402


def main() -> None:
    t0 = time.time()
    retriever = get_retriever(reload=True)
    path = retriever.save_manifest()
    print(json.dumps(retriever.build_info, indent=2))
    print(f"manifest -> {path}")
    print(f"built in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
