"""Debug harness: run MyAgent on ls20 in-process and dump ATTR model state.

Usage: .venv/bin/python scripts/debug_attr.py [steps] [game]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "scripts"))

import arc_agi
from arc_agi import OperationMode

from play_local import load_my_agent_class  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
GAME = sys.argv[2] if len(sys.argv) > 2 else "ls20"

logging.basicConfig(level=logging.WARNING)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
MyAgent = load_my_agent_class()
MyAgent.MAX_ACTIONS = STEPS

env = arc.make(GAME)
agent = MyAgent(
    card_id="local-dev", game_id=GAME, agent_name=f"dbg.{GAME}",
    ROOT_URL="http://localhost", record=False, arc_env=env, tags=["dbg"],
)

orig_choose = agent.choose_action
last = {"n": -10**9}


def spy_choose(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    n = agent.action_counter
    if n - last["n"] >= 250:
        last["n"] = n
        reg = agent._attr_register()
        shifty = sum(1 for s, fx in agent._tile_fx.items()
                     if any(a != b for a, b in fx.items()))
        print(f"--- step {n} levels={latest_frame.levels_completed}"
              f" state={latest_frame.state}", flush=True)
        print(f"    rules={agent._movement_rules()}"
              f" warm={n < 500}", flush=True)
        print(f"    attr_events={agent._attr_events}"
              f" reg_cells={0 if reg is None else len(reg)}"
              f" tile_fx={len(agent._tile_fx)} shifty={shifty}", flush=True)
        print(f"    pad_reject={{ {', '.join(f'{k}:{len(v)}' for k, v in agent._pad_reject.items())} }}"
              f" refills={len(agent._refill_sigs)}", flush=True)
        print(f"    budget={agent._budget_state()}"
              f" walls={len(agent._walls)} plan={len(agent._plan)}"
              f" deaths={agent._game_overs}", flush=True)
    return action


agent.choose_action = spy_choose
agent.main()
final = agent.frames[-1]
reg = agent._attr_register()
print(f"\nFINAL levels={final.levels_completed} state={final.state} "
      f"actions={agent.action_counter}")
print(f"attr_events={agent._attr_events} "
      f"reg={'None' if reg is None else sorted(reg)}")
print(f"tile_fx={len(agent._tile_fx)} classes")
for sig, fx in list(agent._tile_fx.items())[:10]:
    changing = sum(1 for a, b in fx.items() if a != b)
    print(f"  sig(len {len(sig)}, {len(set(sig))} colors) "
          f"entries={len(fx)} changing={changing}")
print(f"pad_reject={agent._pad_reject}")
print(f"budget={agent._budget_state()} obs={agent._budget_obs.most_common(4)}")
