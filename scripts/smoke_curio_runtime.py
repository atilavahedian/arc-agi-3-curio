"""Smoke a generated Curio notebook through Kaggle's agent registry shape.

This deliberately stops before opening an ARC gateway connection.  It proves
that the exact agent embedded in the candidate notebook can be copied into a
fresh framework tree, imported through the notebook-generated ``agents``
registry, instantiated in its declared profile, and returns legal actions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "submissions" / "curio-graph-v16"
DEFAULT_FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"
WRITE_MARKER = "%%writefile /tmp/my_agent.py\n"
REGISTRY_PATH = "agents/__init__.py"


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def embedded_sources(notebook_path: Path) -> tuple[str, str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    texts = [cell_text(cell) for cell in notebook.get("cells", [])]
    agents = [text[len(WRITE_MARKER):] for text in texts
              if text.startswith(WRITE_MARKER)]
    run_cells = [text for text in texts if REGISTRY_PATH in text]
    if len(agents) != 1 or len(run_cells) != 1:
        raise SystemExit("candidate must contain one agent and one registry cell")

    run_cell = run_cells[0]
    registry_anchor = run_cell.index(REGISTRY_PATH)
    write_start = run_cell.index('f.write("""', registry_anchor) \
        + len('f.write("""')
    write_end = run_cell.index('""")', write_start)
    registry = run_cell[write_start:write_end]
    if "'myagent': MyAgent" not in registry:
        raise SystemExit("candidate registry does not expose myagent")
    return agents[0], registry


def frame(actions: list[int], grid: list[list[int]]) -> FrameData:
    return FrameData(
        frame=[grid],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=actions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--framework-dir", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument(
        "--profile",
        choices=("graph-hardened", "legacy", "legacy-hardened"),
        default="graph-hardened",
        help="runtime assertions appropriate to the embedded Curio generation",
    )
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    metadata = json.loads(
        (candidate / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = candidate / metadata["code_file"]
    agent_source, registry_source = embedded_sources(notebook_path)
    compile(agent_source, "/tmp/my_agent.py", "exec")
    compile(registry_source, REGISTRY_PATH, "exec")

    framework = args.framework_dir.resolve()
    if not (framework / "agents" / "agent.py").is_file():
        raise SystemExit(
            f"framework not found at {framework}; run make setup first")

    with tempfile.TemporaryDirectory(prefix="curio-registry-smoke-") as tmp:
        sandbox = Path(tmp) / "ARC-AGI-3-Agents"
        shutil.copytree(framework, sandbox)
        (sandbox / "agents" / "templates" / "my_agent.py").write_text(
            agent_source, encoding="utf-8")
        (sandbox / REGISTRY_PATH).write_text(
            registry_source, encoding="utf-8")

        # Import only from the fresh copy, just as the Kaggle run does after
        # copying the official framework into /kaggle/working.
        for name in list(sys.modules):
            if name == "agents" or name.startswith("agents."):
                del sys.modules[name]
        sys.path.insert(0, str(sandbox))
        previous = os.environ.get("CURIO_EXPLORER")
        if args.profile == "graph-hardened":
            os.environ["CURIO_EXPLORER"] = "graph"
        else:
            os.environ.pop("CURIO_EXPLORER", None)
        try:
            registry = importlib.import_module("agents")
            AgentClass = registry.AVAILABLE_AGENTS["myagent"]
            agent = AgentClass(
                card_id="runtime-smoke",
                game_id="runtime-smoke",
                agent_name="MyAgent.runtime-smoke",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
            grid = [[0 for _x in range(64)] for _y in range(64)]
            if args.profile == "graph-hardened":
                if not agent._gx_on:
                    raise SystemExit(
                        "candidate did not instantiate in graph mode")
                agent._policy = lambda _grid, _latest: (_ for _ in ()).throw(
                    RuntimeError("registry smoke fault"))
                simple = agent.choose_action([], frame([0, 4, 6], grid))
                if simple is not GameAction.ACTION4:
                    raise SystemExit(
                        f"illegal/non-deterministic simple fallback: {simple}")
            elif args.profile == "legacy":
                if hasattr(agent, "_gx_on"):
                    raise SystemExit(
                        "legacy candidate unexpectedly contains graph mode")
                simple = agent.choose_action([], frame([1, 2, 3, 4], grid))
                if simple.value not in {1, 2, 3, 4}:
                    raise SystemExit(f"legacy agent returned illegal action: {simple}")
            else:
                if hasattr(agent, "_gx_on"):
                    raise SystemExit(
                        "hardened legacy candidate unexpectedly contains graph mode")
                normal = agent.choose_action([], frame([1, 2, 3, 4], grid))
                if normal.value not in {1, 2, 3, 4}:
                    raise SystemExit(
                        f"hardened legacy agent returned illegal action: {normal}")
                if getattr(agent, "_runtime_faults", 0) != 0:
                    raise SystemExit(
                        "hardened legacy normal smoke unexpectedly used fallback")

                simple_fault_agent = AgentClass(
                    card_id="runtime-simple-fault-smoke",
                    game_id="runtime-simple-fault-smoke",
                    agent_name="MyAgent.runtime-simple-fault-smoke",
                    ROOT_URL="",
                    record=False,
                    arc_env=None,
                )
                simple_fault_agent._choose_action = (
                    lambda _frames, _latest: (_ for _ in ()).throw(
                        RuntimeError("legacy simple smoke fault")))
                simple = simple_fault_agent.choose_action(
                    [], frame([0, 4, 6], grid))
                if simple is not GameAction.ACTION4:
                    raise SystemExit(
                        f"illegal/non-deterministic simple fallback: {simple}")
                if getattr(simple_fault_agent, "_runtime_faults", 0) != 1:
                    raise SystemExit(
                        "simple exception fallback did not record exactly one fault")

            click_agent = AgentClass(
                card_id="runtime-click-smoke",
                game_id="runtime-click-smoke",
                agent_name="MyAgent.runtime-click-smoke",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
            if args.profile == "graph-hardened":
                click_agent._choose_action = (
                    lambda _frames, _latest: (_ for _ in ()).throw(
                        ValueError("click smoke fault")))
            elif args.profile == "legacy-hardened":
                click_agent._choose_action = (
                    lambda _frames, _latest: (_ for _ in ()).throw(
                        ValueError("legacy click smoke fault")))
            grid[19][37] = 7
            click = click_agent.choose_action([], frame([6], grid))
            coords = (click.action_data.x, click.action_data.y)
            expected_coords = (
                (32, 32) if args.profile == "legacy-hardened" else (37, 19)
            )
            if click is not GameAction.ACTION6 or coords != expected_coords:
                raise SystemExit(
                    f"invalid frame-derived click fallback: {click} {coords}")
            if args.profile == "legacy-hardened" \
                    and getattr(click_agent, "_runtime_faults", 0) != 1:
                raise SystemExit(
                    "click exception fallback did not record exactly one fault")
        finally:
            if previous is None:
                os.environ.pop("CURIO_EXPLORER", None)
            else:
                os.environ["CURIO_EXPLORER"] = previous
            sys.path.remove(str(sandbox))
            for name in list(sys.modules):
                if name == "agents" or name.startswith("agents."):
                    del sys.modules[name]

    print(f"registry smoke: {metadata['id']} -> myagent/MyAgent")
    if args.profile == "graph-hardened":
        print("graph mode: enabled; simple fallback: ACTION4")
        print("click fallback: ACTION6 at current-frame pixel (37, 19)")
    elif args.profile == "legacy":
        print(f"legacy mode: enabled; legal simple action: ACTION{simple.value}")
        print("legacy click policy: ACTION6 at current-frame pixel (37, 19)")
    else:
        print(
            "legacy-hardened mode: enabled; normal action: "
            f"ACTION{normal.value}; simple exception fallback: ACTION4"
        )
        print("click exception fallback: ACTION6 at frame center (32, 32)")


if __name__ == "__main__":
    main()
