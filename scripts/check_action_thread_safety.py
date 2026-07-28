"""Reproduce and verify Curio's threaded ``GameAction`` payload isolation.

The official ARC-AGI-3 Swarm runs every game in a Python thread.  Arcengine's
``GameAction`` members are Enum singletons, so its stock ``action_data`` field
is process-global.  This probe forces the precise interleaving that used to
send one game's ACTION6 coordinates to another game, then repeats the same
official ``Agent.do_action_request`` path with Curio loaded.

Run from the repository root after ``make setup``::

    .venv/bin/python scripts/check_action_thread_safety.py --iterations 1000

An alternate standalone agent source can be checked with ``--agent-source``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path
from threading import Barrier, Thread
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not FRAMEWORK.exists():
    raise SystemExit(f"Framework not found at {FRAMEWORK}. Run `make setup` first.")
sys.path.insert(0, str(FRAMEWORK))

from arcengine import GameAction  # noqa: E402
from agents.agent import Agent  # noqa: E402


class CaptureEnvironment:
    """Small environment double that records the exact official request."""

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "action": action.value,
            "data": dict(data or {}),
            "reasoning": dict(reasoning or {}),
        }


class ProbeAgent:
    """Only the attributes used by the official do_action_request method."""

    def __init__(self) -> None:
        self.arc_env = CaptureEnvironment()

    @staticmethod
    def _convert_raw_frame_data(raw: dict[str, Any]) -> dict[str, Any]:
        return raw


def load_curio(agent_source: Path) -> None:
    """Import Curio exactly as the notebook does, installing its guard."""
    spec = importlib.util.spec_from_file_location(
        "curio_action_thread_probe", agent_source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {agent_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def request(action: GameAction) -> dict[str, Any]:
    """Use the unmodified official framework request implementation."""
    return Agent.do_action_request(ProbeAgent(), action)  # type: ignore[arg-type]


def coords_for(role: str, iteration: int) -> tuple[int, int]:
    if role == "first":
        return iteration % 31, (iteration * 3) % 32
    return 63 - (iteration % 31), 63 - ((iteration * 5) % 32)


def threaded_probe(
    *, patched: bool, iterations: int, agent_source: Path
) -> dict[str, Any]:
    if patched:
        load_curio(agent_source)

    first_set = Barrier(2, timeout=15)
    second_set = Barrier(2, timeout=15)
    first_read = Barrier(2, timeout=15)
    cycle_done = Barrier(2, timeout=15)
    counts = {
        "first_action_mismatches": 0,
        "second_action_mismatches": 0,
        "first_reasoning_mismatches": 0,
        "second_reasoning_mismatches": 0,
    }
    failures: list[str] = []

    def first_worker() -> None:
        try:
            for iteration in range(iterations):
                intended = coords_for("first", iteration)
                action = GameAction.ACTION6
                action.set_data({"x": intended[0], "y": intended[1]})
                action.reasoning = {"owner": "first", "iteration": iteration}
                first_set.wait()
                second_set.wait()
                sent = request(action)
                got = (sent["data"]["x"], sent["data"]["y"])
                if got != intended:
                    counts["first_action_mismatches"] += 1
                if action.reasoning != {
                    "owner": "first", "iteration": iteration
                }:
                    counts["first_reasoning_mismatches"] += 1
                first_read.wait()
                cycle_done.wait()
        except Exception as exc:  # pragma: no cover - diagnostic failure path
            failures.append(f"first: {exc!r}")

    def second_worker() -> None:
        try:
            for iteration in range(iterations):
                first_set.wait()
                intended = coords_for("second", iteration)
                action = GameAction.ACTION6
                action.set_data({"x": intended[0], "y": intended[1]})
                action.reasoning = {"owner": "second", "iteration": iteration}
                second_set.wait()
                first_read.wait()
                sent = request(action)
                got = (sent["data"]["x"], sent["data"]["y"])
                if got != intended:
                    counts["second_action_mismatches"] += 1
                if action.reasoning != {
                    "owner": "second", "iteration": iteration
                }:
                    counts["second_reasoning_mismatches"] += 1
                cycle_done.wait()
        except Exception as exc:  # pragma: no cover - diagnostic failure path
            failures.append(f"second: {exc!r}")

    threads = [Thread(target=first_worker), Thread(target=second_worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("threaded probe deadlocked")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "mode": "thread-fixed" if patched else "thread-baseline",
        "iterations": iterations,
        **counts,
    }


def process_worker(role: str, ready: Any, peer_ready: Any, output: Any) -> None:
    """A separate interpreter has its own Enum members and payloads."""
    intended = coords_for(role, 0)
    action = GameAction.ACTION6
    action.set_data({"x": intended[0], "y": intended[1]})
    ready.set()
    if not peer_ready.wait(timeout=15):
        output.put((role, intended, None, "peer timed out"))
        return
    sent = request(action)
    got = (sent["data"]["x"], sent["data"]["y"])
    output.put((role, intended, got, None))


def process_probe() -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    first_ready = ctx.Event()
    second_ready = ctx.Event()
    output = ctx.Queue()
    processes = [
        ctx.Process(
            target=process_worker,
            args=("first", first_ready, second_ready, output),
        ),
        ctx.Process(
            target=process_worker,
            args=("second", second_ready, first_ready, output),
        ),
    ]
    for process in processes:
        process.start()
    results = [output.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError(
            f"process probe failed: {[process.exitcode for process in processes]}"
        )
    mismatches = sum(
        1 for _role, intended, got, error in results
        if error is not None or tuple(intended) != tuple(got or ())
    )
    return {
        "mode": "process-baseline",
        "workers": len(processes),
        "action_mismatches": mismatches,
    }


def subprocess_probe(
    mode: str, iterations: int, agent_source: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--iterations",
            str(iterations),
            "--agent-source",
            str(agent_source),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def full_probe(iterations: int, agent_source: Path) -> dict[str, Any]:
    # Separate interpreters are essential: once Curio replaces the descriptor,
    # it correctly remains installed for the life of that worker process.
    baseline = subprocess_probe("thread-baseline", iterations, agent_source)
    fixed = subprocess_probe("thread-fixed", iterations, agent_source)
    processes = subprocess_probe("process-baseline", iterations, agent_source)
    passed = (
        baseline["first_action_mismatches"] == iterations
        and baseline["first_reasoning_mismatches"] == iterations
        and baseline["second_action_mismatches"] == 0
        and fixed["first_action_mismatches"] == 0
        and fixed["second_action_mismatches"] == 0
        and fixed["first_reasoning_mismatches"] == 0
        and fixed["second_reasoning_mismatches"] == 0
        and processes["action_mismatches"] == 0
    )
    return {
        "passed": passed,
        "baseline": baseline,
        "fixed": fixed,
        "processes": processes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "thread-baseline", "thread-fixed", "process-baseline"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--agent-source",
        type=Path,
        default=ROOT / "agent" / "my_agent.py",
        help="Curio source whose thread-local payload guard should be loaded",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    agent_source = args.agent_source.resolve()
    if not agent_source.is_file():
        raise SystemExit(f"--agent-source does not exist: {agent_source}")

    if args.mode == "thread-baseline":
        result = threaded_probe(
            patched=False,
            iterations=args.iterations,
            agent_source=agent_source,
        )
    elif args.mode == "thread-fixed":
        result = threaded_probe(
            patched=True,
            iterations=args.iterations,
            agent_source=agent_source,
        )
    elif args.mode == "process-baseline":
        result = process_probe()
    else:
        result = full_probe(args.iterations, agent_source)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.mode == "all" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
