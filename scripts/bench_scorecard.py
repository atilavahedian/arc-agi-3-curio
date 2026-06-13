"""Bench companion: same runs as scripts/bench.sh, plus the full scorecard.

Runs agent/my_agent.py on the fixed 6-game bench set for one CURIO_SEED and
prints, per game: levels completed, per-level actions, per-level scores and
the aggregate scorecard score — the efficiency numbers bench.sh discards.

Usage:
    CURIO_SEED=0 .venv/bin/python scripts/bench_scorecard.py --max-steps 4000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

GAMES = ["lp85", "vc33", "ls20", "ft09", "tr87", "cn04"]


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-steps", type=int, default=4000)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger().setLevel(logging.WARNING)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    MyAgentCls = load_my_agent_class()
    MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    seed = os.environ.get("CURIO_SEED", "0")
    for game_id in GAMES:
        env = arc.make(game_id)
        agent = MyAgentCls(
            card_id="local-dev", game_id=game_id,
            agent_name=f"MyAgent.bench.{game_id}", ROOT_URL="http://localhost",
            record=False, arc_env=env, tags=["local-dev"])
        agent.main()
        final = agent.frames[-1]
        print(f"GAME {game_id} seed={seed} levels={final.levels_completed} "
              f"actions={agent.action_counter} state={final.state}")

    sc = arc.get_scorecard()
    data = json.loads(sc.model_dump_json())
    for env_scores in data.get("environments", []):
        for run in env_scores.get("runs", []):
            print(f"SCORE {run.get('id')} seed={seed} "
                  f"score={run.get('score'):.2f} "
                  f"levels={run.get('levels_completed')} "
                  f"level_actions={run.get('level_actions')} "
                  f"level_scores={[round(s, 2) for s in (run.get('level_scores') or [])]} "
                  f"baselines={run.get('level_baseline_actions')}")
    print(f"AGGREGATE seed={seed} score={data.get('score', sc.score)}")


if __name__ == "__main__":
    main()
