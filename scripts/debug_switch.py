"""Debug harness: run MyAgent on dc22 in-process and dump SWITCH model state.

Usage: .venv/bin/python scripts/debug_switch.py [steps] [game] [period]
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

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
GAME = sys.argv[2] if len(sys.argv) > 2 else "dc22"
PERIOD = int(sys.argv[3]) if len(sys.argv) > 3 else 50

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
orig_switch = agent._switch_policy
last = {"n": -10**9, "sw": 0}


def spy_switch(grid, latest_frame):
    out = orig_switch(grid, latest_frame)
    if out is not None:
        last["sw"] += 1
    return out


agent._switch_policy = spy_switch


def spy_choose(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    n = agent.action_counter
    if n - last["n"] >= PERIOD:
        last["n"] = n
        confirmed = {s: agent._switch_cycle(s) for s in agent._sw_recs}
        confirmed = {s: c for s, c in confirmed.items() if c is not None}
        print(f"--- step {n} levels={latest_frame.levels_completed}"
              f" state={latest_frame.state} deaths={agent._game_overs}",
              flush=True)
        print(f"    rules={agent._movement_rules()}"
              f" frames={agent._frames_diffed}"
              f" port={agent._port_color()}"
              f" reg={agent._attr_register() is not None}"
              f" act_uses={dict(agent._act_uses)}", flush=True)
        recs = {s: (len(r['events']), len(r['mask']), r['overlap'])
                for s, r in agent._sw_recs.items()}
        print(f"    recs(uses,mask,overlap)={recs}", flush=True)
        print(f"    confirmed={[(s, len(c[0]), len(c[1])) for s, c in confirmed.items()]}"
              f" probes={dict(agent._sw_probe)}", flush=True)
        print(f"    stepped={len(agent._stepped_sigs)}"
              f" blocked={sum(1 for v in agent._block_sigs.values() if v >= 2)}"
              f" bans={len(agent._fall_bans)}"
              f" budget={agent._budget_state()}"
              f" rem={None if agent.frames[-1].is_empty() else agent._budget_remaining(agent.frames[-1].frame[-1])}",
              flush=True)
        print(f"    sw_actions={last['sw']} plan={len(agent._sw_plan)}"
              f" strikes={agent._sw_strikes} benched={agent._sw_benched}"
              f" nogoal={agent._sw_nogoal is not None}"
              f" hud={len(agent._hud_mask)}", flush=True)
    return action


agent.choose_action = spy_choose
agent.main()
final = agent.frames[-1]
print(f"FINAL levels={final.levels_completed} actions={agent.action_counter}"
      f" state={final.state} sw_actions={last['sw']}")
