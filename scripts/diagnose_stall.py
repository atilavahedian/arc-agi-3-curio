"""Explain where a game's action budget actually goes.

The 25-game baseline shows most games clearing level 1 at or below the human
baseline and then spending ~9,900 actions on level 2 without completing it.
Aggregate benchmarks cannot say *why*. This script instruments a single run
and reports, per level:

  * which policy head emitted each action (family solver vs generic core);
  * how many distinct masked states the run visited, and when the last new
    one appeared -- a flat novelty curve means the explorer is looping;
  * the most-repeated actions, so a stuck cycle is visible directly;
  * deaths, resets and how often the chosen action changed nothing.

Usage:
    .venv/bin/python scripts/diagnose_stall.py --game sb26 --max-steps 3000
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))


def load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instrument(AgentCls) -> None:
    """Record the emitting head, level, and state novelty for every action."""
    heads = [n for n in dir(AgentCls)
             if n.endswith("_policy") and n != "_policy"]

    def wrap_head(name, orig):
        def wrapper(self, *a, **k):
            result = orig(self, *a, **k)
            if result is not None:
                self._diag_head = name
            return result
        return wrapper

    for name in heads:
        setattr(AgentCls, name, wrap_head(name, getattr(AgentCls, name)))

    orig_choose = AgentCls.choose_action

    def choose_action(self, frames, latest_frame):
        self._diag_head = "dispatch/reset"
        before_states = len(getattr(self, "_state_visits", {}) or {})
        action = orig_choose(self, frames, latest_frame)
        after_states = len(getattr(self, "_state_visits", {}) or {})
        data = getattr(action, "_data", None) or {}
        key = str(action.value)
        if data.get("x") is not None:
            key = f"6:{data['x']},{data['y']}"
        self._diag_log.append({
            "i": self.action_counter,
            "level": latest_frame.levels_completed,
            "state": str(latest_frame.state),
            "head": getattr(self, "_diag_head", "?"),
            "action": key,
            "novel": after_states > before_states,
        })
        return action

    AgentCls.choose_action = choose_action

    orig_init = AgentCls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._diag_log = []
        self._diag_head = "?"

    AgentCls.__init__ = __init__


def report(log: list[dict[str, Any]], game: str) -> dict[str, Any]:
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in log:
        by_level[rec["level"]].append(rec)

    print(f"\n===== {game}: {len(log)} actions across "
          f"{len(by_level)} level(s) =====")
    summary: dict[str, Any] = {"game": game, "total_actions": len(log),
                               "levels": {}}
    for level in sorted(by_level):
        recs = by_level[level]
        heads = Counter(r["head"] for r in recs)
        actions = Counter(r["action"] for r in recs)
        novel_idx = [r["i"] for r in recs if r["novel"]]
        deaths = sum(1 for r in recs if "GAME_OVER" in r["state"])
        last_novel = novel_idx[-1] if novel_idx else None
        # Actions spent after the final new state: pure wasted budget.
        wasted = (recs[-1]["i"] - last_novel) if last_novel else len(recs)

        print(f"\n-- level {level}: {len(recs)} actions, "
              f"{len(novel_idx)} novel states, {deaths} game-overs")
        print(f"   last new state at action {last_novel}; "
              f"{wasted} actions after it produced nothing new")
        print(f"   heads: {dict(heads.most_common(6))}")
        print(f"   top actions: {dict(actions.most_common(8))}")
        print(f"   distinct actions used: {len(actions)}")
        summary["levels"][str(level)] = {
            "actions": len(recs), "novel_states": len(novel_idx),
            "deaths": deaths, "last_novel_at": last_novel,
            "wasted_after_last_novel": wasted,
            "heads": dict(heads), "distinct_actions": len(actions),
            "top_actions": dict(actions.most_common(10)),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seed", default=os.environ.get("CURIO_SEED", "0"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.environ["CURIO_SEED"] = args.seed
    os.environ.setdefault("MPLBACKEND", "agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3-mpl")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    import arc_agi
    from arc_agi import OperationMode

    module = load_agent_module()
    AgentCls = module.MyAgent
    instrument(AgentCls)
    AgentCls.MAX_ACTIONS = min(AgentCls.MAX_ACTIONS, args.max_steps)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        env = arc.make(args.game)
        agent = AgentCls(
            card_id="local-diag", game_id=args.game,
            agent_name=f"MyAgent.diag.{args.game}", ROOT_URL="http://localhost",
            record=False, arc_env=env, tags=["local-diag"])
        agent.main()

    final = agent.frames[-1]
    print(f"{args.game}: state={final.state} "
          f"levels={final.levels_completed} actions={agent.action_counter}")
    summary = report(agent._diag_log, args.game)
    summary["final_state"] = str(final.state)
    summary["levels_completed"] = int(final.levels_completed)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
