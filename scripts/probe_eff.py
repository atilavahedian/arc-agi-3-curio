"""Instrumented probe: which policy branch emits each action, per level.

Usage:
    CURIO_SEED=0 .venv/bin/python scripts/probe_eff.py --game vc33 --max-steps 1200
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--max-steps", type=int, default=1200)
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    mod = load_my_agent_class()
    MyAgent = mod.MyAgent
    MyAgent.MAX_ACTIONS = args.max_steps

    # per-level counters of which branch produced the action
    stats: dict[int, Counter] = {}
    level_first_engage: dict[str, int] = {}

    orig_policy = MyAgent._policy
    orig_novelty = MyAgent._novelty_policy
    orig_lattice = MyAgent._lattice_policy
    orig_editor = MyAgent._editor_policy
    orig_port = MyAgent._port_policy
    orig_choose = MyAgent.choose_action

    def policy(self, grid, latest_frame):
        lvl = latest_frame.levels_completed
        c = stats.setdefault(lvl, Counter())
        self._branch = None
        act = orig_policy(self, grid, latest_frame)
        c[self._branch or "?"] += 1
        return act

    def novelty(self, grid, latest_frame):
        if getattr(self, "_branch", "x") is None:
            self._branch = ("warmup" if self.action_counter <
                            mod.NOVELTY_WARMUP and not self._warmup_skip
                            else "novelty")
        return orig_novelty(self, grid, latest_frame)

    def lattice(self, grid, latest_frame):
        r = orig_lattice(self, grid, latest_frame)
        if r is not None and getattr(self, "_branch", "x") is None:
            self._branch = "lattice"
        return r

    def editor(self, grid, latest_frame):
        r = orig_editor(self, grid, latest_frame)
        if r is not None and getattr(self, "_branch", "x") is None:
            self._branch = "editor"
        return r

    def port(self, grid, latest_frame):
        r = orig_port(self, grid, latest_frame)
        if r is not None and getattr(self, "_branch", "x") is None:
            self._branch = "port"
        return r

    def step(self, action):
        if getattr(self, "_branch", "x") is None:
            self._branch = "plan"
        return orig_step(self, action)
    orig_step = MyAgent._step

    last = {"lvl": 0, "n": 0}

    def choose(self, frames, latest_frame):
        act = orig_choose(self, frames, latest_frame)
        lvl = latest_frame.levels_completed
        if lvl != last["lvl"]:
            print(f"LEVEL {last['lvl']} -> {lvl} at action {self.action_counter}"
                  f" (+{self.action_counter - last['n']})")
            last["lvl"], last["n"] = lvl, self.action_counter
        return act

    MyAgent._policy = policy
    MyAgent._novelty_policy = novelty
    MyAgent._lattice_policy = lattice
    MyAgent._editor_policy = editor
    MyAgent._port_policy = port
    MyAgent._step = step
    MyAgent.choose_action = choose

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    agent = MyAgent(
        card_id="local-dev", game_id=args.game,
        agent_name=f"probe.{args.game}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["local-dev"])
    agent.main()
    final = agent.frames[-1]
    print(f"FINAL levels={final.levels_completed} actions={agent.action_counter} "
          f"state={final.state}")
    for lvl in sorted(stats):
        print(f"  level {lvl}: {dict(stats[lvl])}")
    print(f"game_overs={agent._game_overs} "
          f"click_sigs={len(agent._click_effects)} "
          f"rules={agent._movement_rules()} "
          f"attr_events={agent._attr_events}")


if __name__ == "__main__":
    main()
