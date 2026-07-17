"""Build the standalone original Curio graph Kaggle candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import build_notebook


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "submissions" / "curio-graph-v16"
NOTEBOOK_PATH = CANDIDATE_DIR / "submission.ipynb"


def main() -> None:
    previous = os.environ.get("CURIO_BUILD_EXPLORER")
    os.environ["CURIO_BUILD_EXPLORER"] = "graph"
    try:
        notebook = build_notebook.build()
    finally:
        if previous is None:
            os.environ.pop("CURIO_BUILD_EXPLORER", None)
        else:
            os.environ["CURIO_BUILD_EXPLORER"] = previous

    notebook["cells"][0]["source"] = (
        "# Curio Graph v16 — Original ARC-AGI-3 Candidate\n\n"
        "Generated from `agent/my_agent.py` by "
        "`scripts/build_curio_candidate.py`. The graph explorer is baked "
        "into the Kaggle run command; do not edit this notebook directly."
    )
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"[build_curio_candidate] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
