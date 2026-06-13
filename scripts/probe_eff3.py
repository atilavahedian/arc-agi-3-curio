"""Trust-timeline probe: when do rules/avatar/register become trusted,
and what does the attr planner spend its plans on.

Usage:
    CURIO_SEED=0 .venv/bin/python scripts/probe_eff3.py --game ls20 --max-steps 500
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_module():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    mod = load_module()
    MyAgent = mod.MyAgent
    MyAgent.MAX_ACTIONS = args.max_steps

    orig_choose = MyAgent.choose_action
    seen = {"rules": {}, "avatar": None, "reg": None, "ready": None}

    def choose(self, frames, latest_frame):
        act = orig_choose(self, frames, latest_frame)
        n = self.action_counter
        for a, d in self._movement_rules().items():
            if a not in seen["rules"]:
                seen["rules"][a] = n
                print(f"  rule {a}->{d} trusted at action {n}")
        if seen["avatar"] is None and self._avatar_signature() is not None:
            seen["avatar"] = n
            print(f"  avatar sig trusted at action {n}")
        if seen["reg"] is None and self._attr_register() is not None:
            seen["reg"] = n
            print(f"  attr register formed at action {n} "
                  f"(events={self._attr_events})")
        if seen["ready"] is None:
            rules = self._movement_rules()
            avail = self._avail - {0, 6}
            if avail and all(a in rules or self._act_uses[a] >= mod.RULE_TRIES
                             for a in avail) and len(rules) >= 2:
                seen["ready"] = n
                print(f"  attr-planner READY at action {n} "
                      f"uses={dict(self._act_uses)} rules={rules}")
        return act

    orig_plan_attr = MyAgent._plan_attr_route

    def plan_attr(self, grid, avatar, rules, reg):
        out = orig_plan_attr(self, grid, avatar, rules, reg)
        print(f"  [plan@{self.action_counter}] attr plan -> "
              f"{'None' if out is None else len(out)} steps")
        return out

    orig_probe = MyAgent._probe_tiles

    def probe_tiles(self, grid, avatar, rules):
        out = orig_probe(self, grid, avatar, rules)
        if out is not None:
            print(f"  [plan@{self.action_counter}] probe_tiles -> {len(out)}")
        return out

    MyAgent.choose_action = choose
    MyAgent._plan_attr_route = plan_attr
    MyAgent._probe_tiles = probe_tiles

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    agent = MyAgent(
        card_id="local-dev", game_id=args.game,
        agent_name=f"probe.{args.game}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["local-dev"])
    agent.main()
    final = agent.frames[-1]
    print(f"FINAL levels={final.levels_completed} actions={agent.action_counter}")


if __name__ == "__main__":
    main()
