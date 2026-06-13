"""Debug harness: run MyAgent on cn04 in-process and dump PORT model state.

Usage: .venv/bin/python scripts/debug_port.py [steps] [game]
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
GAME = sys.argv[2] if len(sys.argv) > 2 else "cn04"

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
last = {"n": -10**9, "lv": -1}


def spy_choose(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    n = agent.action_counter
    lv = latest_frame.levels_completed
    if n - last["n"] >= 100 or lv != last["lv"]:
        last["n"] = n
        last["lv"] = lv
        print(f"--- step {n} levels={lv} state={latest_frame.state}", flush=True)
        print(f"    rules={agent._movement_rules()}"
              f" port={agent._port_color()} fb={agent._fb_color()}"
              f" xformv={dict(agent._xform_votes)}", flush=True)
        pp = {pid: (len(p['cells']), sorted(agent._pc_pieces and []) or
                    len(p['ports']), len(p['pats']), p['stack'])
              for pid, p in agent._pc_pieces.items()}
        print(f"    pieces(cells,ports,pats,stack)={pp}"
              f" sel={agent._pc_sel}", flush=True)
        print(f"    sol={'yes' if agent._pc_solution else 'no'}"
              f" failed={len(agent._pc_failed_cfgs)}"
              f" unreach={len(agent._pc_unreachable)}"
              f" strikes={agent._pc_strikes}"
              f" benched={agent._pc_benched}"
              f" bounce={agent._pc_bounce} idprobe={agent._pc_idprobe}"
              f" deaths={agent._game_overs}", flush=True)
    return action


agent.choose_action = spy_choose
agent.main()
print(f"FINAL: levels={last['lv']} deaths={agent._game_overs} "
      f"steps={agent.action_counter}")
