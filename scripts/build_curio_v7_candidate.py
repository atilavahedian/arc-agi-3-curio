"""Build the isolated thread-safe original Curio v7 Kaggle candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import build_notebook


ROOT = Path(__file__).resolve().parents[1]
AGENT_SOURCE = ROOT / "baselines" / "curio_v7_threadsafe.py"
CANDIDATE_DIR = ROOT / "submissions" / "curio-v7-threadsafe"
NOTEBOOK_PATH = CANDIDATE_DIR / "submission.ipynb"


def main() -> None:
    previous_source = build_notebook.AGENT_SRC
    previous_explorer = os.environ.pop("CURIO_BUILD_EXPLORER", None)
    build_notebook.AGENT_SRC = AGENT_SOURCE
    try:
        notebook = build_notebook.build()
    finally:
        build_notebook.AGENT_SRC = previous_source
        if previous_explorer is not None:
            os.environ["CURIO_BUILD_EXPLORER"] = previous_explorer

    notebook["cells"][0]["source"] = (
        "# Curio v7 Threadsafe — Original ARC-AGI-3 Ablation\n\n"
        "Generated from `baselines/curio_v7_threadsafe.py` by "
        "`scripts/build_curio_v7_candidate.py`. The policy is the frozen "
        "original v7 source plus isolated `GameAction` payloads; do not edit "
        "this notebook directly."
    )
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1), encoding="utf-8")
    print(
        "[build_curio_v7_candidate] Wrote "
        f"{NOTEBOOK_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
