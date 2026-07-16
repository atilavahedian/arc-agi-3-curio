"""Validate a Kaggle ARC-AGI-3 notebook submission directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DATASETS = {
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path)
    args = parser.parse_args()

    root = args.submission_dir.resolve()
    metadata_path = root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file.endswith(".ipynb"):
        raise SystemExit("kernel-metadata.json must name a .ipynb code_file")

    notebook_path = root / code_file
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise SystemExit(f"invalid or empty notebook: {notebook_path}")

    if metadata.get("enable_internet") is not False:
        raise SystemExit("competition notebook must have internet disabled")
    if metadata.get("enable_gpu") is not True:
        raise SystemExit("Duck v15 requires a GPU")
    if metadata.get("machine_shape") != "NvidiaRtxPro6000":
        raise SystemExit("Duck v15 requires machine_shape=NvidiaRtxPro6000")
    if "arc-prize-2026-arc-agi-3" not in metadata.get("competition_sources", []):
        raise SystemExit("ARC-AGI-3 competition source is missing")

    datasets = set(metadata.get("dataset_sources", []))
    missing = sorted(REQUIRED_DATASETS - datasets)
    if missing:
        raise SystemExit(f"required dataset source(s) missing: {missing}")

    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    required_markers = (
        "KAGGLE_IS_COMPETITION_RERUN",
        "OperationMode.COMPETITION",
        "taaf_grafts.composite",
        '"efficiency": True',
        '"retry_guard": True',
        '"shortcircuit": True',
        '"banking": True',
        "submission.parquet",
    )
    absent = [marker for marker in required_markers if marker not in source]
    if absent:
        raise SystemExit(f"submission notebook marker(s) missing: {absent}")

    print(f"validated: {notebook_path}")
    print(f"kernel: {metadata['id']}")
    print(f"datasets: {len(datasets)}; cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
