"""Port-planner timeline probe for cn04-family games."""
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
    p.add_argument("--game", default="cn04")
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    mod = load_module()
    MyAgent = mod.MyAgent
    MyAgent.MAX_ACTIONS = args.max_steps

    seen = {"color": None, "first": None, "bench": None}
    orig_port = MyAgent._port_policy
    orig_choose = MyAgent.choose_action

    def port(self, grid, latest_frame):
        r = orig_port(self, grid, latest_frame)
        n = self.action_counter
        if seen["color"] is None and self._port_color() is not None:
            seen["color"] = n
            print(f"  port color trusted at {n}: {self._port_color()}")
        if r is not None and seen["first"] is None:
            seen["first"] = n
            print(f"  first port action at {n}")
        if self._pc_benched is not None and seen["bench"] != n:
            seen["bench"] = n
            print(f"  BENCHED at {n} strikes={self._pc_strikes}")
        return r

    last = {"lvl": 0}

    def choose(self, frames, latest_frame):
        act = orig_choose(self, frames, latest_frame)
        if latest_frame.levels_completed != last["lvl"]:
            print(f"LEVEL -> {latest_frame.levels_completed} "
                  f"at {self.action_counter}")
            last["lvl"] = latest_frame.levels_completed
            seen["first"] = None
        return act

    orig_fail = MyAgent._pc_fail

    def fail(self, sol):
        print(f"  pc_fail at {self.action_counter} "
              f"strikes->{self._pc_strikes + 1}")
        return orig_fail(self, sol)

    MyAgent._port_policy = port
    MyAgent.choose_action = choose
    MyAgent._pc_fail = fail

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    agent = MyAgent(
        card_id="local-dev", game_id=args.game,
        agent_name=f"probe.{args.game}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["local-dev"])
    agent.main()
    print(f"FINAL levels={agent.frames[-1].levels_completed} "
          f"actions={agent.action_counter} strikes={agent._pc_strikes} "
          f"benched={agent._pc_benched} votes={dict(agent._port_votes)}")


if __name__ == "__main__":
    main()
