"""Validate the standalone original Curio graph Kaggle candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "submissions" / "curio-graph-v16"
WRITE_MARKER = "%%writefile /tmp/my_agent.py\n"


def fail(message: str) -> None:
    raise SystemExit(message)


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_dir", nargs="?", type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()

    candidate = args.submission_dir.resolve()
    metadata_path = candidate / "kernel-metadata.json"
    if not metadata_path.is_file():
        fail(f"missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file.endswith(".ipynb"):
        fail("kernel-metadata.json must name a .ipynb code_file")
    if Path(code_file).name != code_file:
        fail("kernel-metadata.json code_file must be a local filename")
    actual_entries = {path.name for path in candidate.iterdir()}
    expected_entries = {"kernel-metadata.json", code_file}
    if actual_entries != expected_entries:
        fail("candidate directory has unexpected entries: "
             f"{sorted(actual_entries - expected_entries)}")
    notebook_path = candidate / code_file
    if not notebook_path.is_file():
        fail(f"missing notebook: {notebook_path}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if notebook.get("nbformat") != 4 or not cells:
        fail(f"invalid or empty notebook: {notebook_path}")
    kaggle_notebook = notebook.get("metadata", {}).get("kaggle", {})
    if kaggle_notebook.get("isInternetEnabled") is not False:
        fail("notebook metadata must disable internet")
    if kaggle_notebook.get("isGpuEnabled") is not True:
        fail("notebook metadata must enable the Kaggle GPU runtime")

    if metadata.get("is_private") is not True:
        fail("candidate kernel must remain private")
    if metadata.get("enable_internet") is not False:
        fail("competition notebook must have internet disabled")
    if metadata.get("enable_gpu") is not True:
        fail("Curio graph candidate metadata must enable the Kaggle GPU runtime")
    if metadata.get("competition_sources") != [
            "arc-prize-2026-arc-agi-3"]:
        fail("competition_sources must contain only the official ARC-AGI-3 source")
    for source_kind in ("dataset_sources", "kernel_sources", "model_sources"):
        if metadata.get(source_kind, []):
            fail(f"original Curio candidate requires empty {source_kind}")

    texts = [cell_text(cell) for cell in cells]
    write_cells = [text for text in texts if text.startswith(WRITE_MARKER)]
    if len(write_cells) != 1:
        fail("expected exactly one /tmp/my_agent.py write cell")
    embedded_agent = write_cells[0][len(WRITE_MARKER):]
    source_agent = (ROOT / "agent" / "my_agent.py").read_text(encoding="utf-8")
    if embedded_agent != source_agent:
        fail("embedded my_agent.py does not exactly match agent/my_agent.py")

    source = "\n".join(texts)
    run_cells = [text for text in texts
                 if "KAGGLE_IS_COMPETITION_RERUN" in text
                 and "python main.py --agent myagent" in text]
    if len(run_cells) != 1:
        fail("expected exactly one competition run cell")
    if "CURIO_EXPLORER=graph python main.py --agent myagent" \
            not in run_cells[0]:
        fail("competition run cell does not bake CURIO_EXPLORER=graph")
    if "CURIO_GENERIC_ONLY=1" in run_cells[0]:
        fail("competition run cell must not enable the generic-only ablation")

    required_markers = (
        "KAGGLE_IS_COMPETITION_RERUN",
        "submission.parquet",
        "OPERATION_MODE=online",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        fail(f"candidate notebook marker(s) missing: {missing}")

    forbidden_markers = (
        "taaf_grafts", "qwen", "vllm", "duck-v15", "driessmit1/",
        "thtennant/",
    )
    lower_source = source.lower()
    present = [marker for marker in forbidden_markers
               if marker.lower() in lower_source]
    if present:
        fail(f"external/copy candidate marker(s) present: {present}")

    agent_sha = hashlib.sha256(source_agent.encode()).hexdigest()
    notebook_sha = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    print(f"validated: {notebook_path}")
    print(f"kernel: {metadata['id']}")
    print(f"agent sha256: {agent_sha}")
    print(f"notebook sha256: {notebook_sha}")
    print(f"external sources: 0; cells: {len(cells)}; graph mode: baked")


if __name__ == "__main__":
    main()
