"""Full-set sweep with per-level score attribution.

Runs `agent/my_agent.py` over a set of official games -- by default all of
them -- in parallel worker processes, and records for every level: actions
spent, the human baseline, and the resulting level score. That per-level
detail is what the competition metric is actually made of, so it is the only
view that shows *where* score is being lost.

Metric recap (arc_agi/scorecard.py):

    level_score = min((baseline / actions) ** 2 * 100, 115)      # 0 if unwon
    game_score  = min(sum(level_score * i) / sum(i),             # i = 1-based
                      completed_weight / total_weight * 100)     # <= 100
    aggregate   = mean(game_score over games)                    # <= 100

The aggregate is already a percentage-point score on the competition's
0--100 scale.  It is a public-game development measurement, not an estimate
of the leaderboard score: Kaggle evaluates 110 separate unseen games.

Usage:
    .venv/bin/python scripts/sweep.py --max-steps 1000 --jobs 6
    .venv/bin/python scripts/sweep.py --games cn04,ft09 --max-steps 2000
    .venv/bin/python scripts/sweep.py --agent-source baselines/curio_v7_threadsafe.py
    .venv/bin/python scripts/sweep.py --out runs/baseline.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

# Every official public game id, in the repo's canonical order.
ALL_GAMES = [
    "tu93", "ar25", "re86", "su15", "m0r0", "cn04", "ft09", "tr87", "sc25",
    "lp85", "dc22", "sp80", "ka59", "g50t", "sb26", "lf52", "bp35", "s5i5",
    "r11l", "sk48", "wa30", "vc33", "ls20", "cd82", "tn36",
]


def _load_agent_class(agent_source: Path):
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", agent_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load agent source: {agent_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def game_score(level_scores: list[float], completed: list[bool]) -> float:
    """Reproduce EnvironmentScoreCalculator.to_score for a single game."""
    if not level_scores:
        return 0.0
    total = sum(s * (i + 1) for i, s in enumerate(level_scores))
    total_w = sum(i + 1 for i in range(len(level_scores)))
    max_w = sum(i + 1 for i, s in enumerate(level_scores) if s > 0)
    if total_w == 0:
        return 0.0
    return min(total / total_w, max_w / total_w * 100.0)


def run_one(game_id: str, max_steps: int, seed: str,
            env: dict[str, str], agent_source: str) -> dict[str, Any]:
    """Play one game in this process and return its per-level detail."""
    os.environ.update(env)
    os.environ["CURIO_SEED"] = seed
    os.environ.setdefault("MPLBACKEND", "agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3-mpl")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    import arc_agi
    from arc_agi import OperationMode

    started = time.time()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    AgentCls = _load_agent_class(Path(agent_source))
    AgentCls.MAX_ACTIONS = min(AgentCls.MAX_ACTIONS, max_steps)

    # The agent and framework log heavily; keep the sweep output readable.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        arc_env = arc.make(game_id)
        agent = AgentCls(
            card_id="local-sweep", game_id=game_id,
            agent_name=f"MyAgent.sweep.{game_id}", ROOT_URL="http://localhost",
            record=False, arc_env=arc_env, tags=["local-sweep"])
        agent.main()
        card = json.loads(arc.get_scorecard().model_dump_json())

    final = agent.frames[-1]
    run: dict[str, Any] = {}
    for env_scores in card.get("environments", []):
        for candidate in env_scores.get("runs", []):
            if candidate.get("level_scores"):
                run = candidate
                break

    level_scores = [float(s) for s in run.get("level_scores") or []]
    return {
        "game": game_id,
        "state": str(final.state),
        "levels_completed": int(final.levels_completed),
        "actions": int(agent.action_counter),
        "score": float(run.get("score", 0.0)),
        "level_scores": level_scores,
        "level_actions": list(run.get("level_actions") or []),
        "level_baselines": list(run.get("level_baseline_actions") or []),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", default=None,
                   help="Comma-separated ids. Default: all 25 official games.")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Per-game action cap.")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="Parallel worker processes.")
    p.add_argument("--seed", default=os.environ.get("CURIO_SEED", "0"))
    p.add_argument(
        "--agent-source",
        type=Path,
        default=ROOT / "agent" / "my_agent.py",
        help="standalone agent source to evaluate (default: agent/my_agent.py)",
    )
    p.add_argument("--out", default=None, help="Write the full result JSON here.")
    p.add_argument("--label", default="", help="Free-text label stored in the JSON.")
    args = p.parse_args()

    games = ([g.strip() for g in args.games.split(",") if g.strip()]
             if args.games else list(ALL_GAMES))
    agent_source = args.agent_source.resolve()
    if not agent_source.is_file():
        raise SystemExit(f"--agent-source does not exist: {agent_source}")
    passthrough = {k: v for k, v in os.environ.items() if k.startswith("CURIO_")}

    started = time.time()
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_one,
                g,
                args.max_steps,
                args.seed,
                passthrough,
                str(agent_source),
            ): g
            for g in games
        }
        for fut in as_completed(futures):
            game = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # a crash must not hide the rest
                res = {"game": game, "state": f"ERROR: {exc}", "score": 0.0,
                       "levels_completed": 0, "actions": 0, "level_scores": [],
                       "level_actions": [], "level_baselines": [], "seconds": 0}
            results[game] = res
            print(f"  {res['game']:6} score={res['score']:6.2f} "
                  f"levels={res['levels_completed']:2} "
                  f"actions={res['actions']:6} {res['state']} "
                  f"({res['seconds']}s)", flush=True)

    ordered = [results[g] for g in games if g in results]
    aggregate = sum(r["score"] for r in ordered) / len(ordered) if ordered else 0.0

    print("\n===== per-game =====")
    for r in sorted(ordered, key=lambda r: -r["score"]):
        detail = " ".join(
            f"L{i + 1}:{a}/{b}" for i, (a, b) in enumerate(
                zip(r["level_actions"], r["level_baselines"]))) or "-"
        print(f"  {r['game']:6} {r['score']:6.2f}  {detail}")

    print(f"\nPUBLIC-GAME SCORE = {aggregate:.4f}%")
    print(f"games={len(ordered)} cap={args.max_steps} seed={args.seed} "
          f"wall={time.time() - started:.0f}s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "label": args.label,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "agent_source": str(agent_source),
            "aggregate": aggregate,
            "public_game_score_percent": aggregate,
            "games": ordered,
        }, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
