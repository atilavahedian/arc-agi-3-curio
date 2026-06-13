"""Verbose per-step trace of the port planner on cn04 level 2+."""
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

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
logging.basicConfig(level=logging.WARNING)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
MyAgent = load_my_agent_class()
MyAgent.MAX_ACTIONS = STEPS

env = arc.make("cn04")
agent = MyAgent(
    card_id="x", game_id="cn04", agent_name="dbg", ROOT_URL="http://localhost",
    record=False, arc_env=env, tags=["dbg"],
)

orig_choose = agent.choose_action
lines = [0]


def spy(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    lv = latest_frame.levels_completed
    if lv >= 1 and lines[0] < 200:
        lines[0] += 1
        from agent_pat import describe  # noqa
    if lv >= 1 and lines[0] < 200:
        pass
    return action


def spy2(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    lv = latest_frame.levels_completed
    if lv >= 1 and lines[0] < 220:
        lines[0] += 1
        import my_agent_mod  # noqa
    return action


def spy3(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    lv = latest_frame.levels_completed
    if lv >= 1 and lines[0] < 220:
        lines[0] += 1
        sel = agent._pc_sel
        sol = agent._pc_solution
        pieces = agent._pc_pieces
        import my_agent_helpers  # noqa
    return action


# simplest: inline state dump
def dump(frames, latest_frame):
    action = orig_choose(frames, latest_frame)
    lv = latest_frame.levels_completed
    if lv >= 1 and lines[0] < 260:
        lines[0] += 1
        import importlib.util
        if "uam" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "uam", ROOT / "agent" / "my_agent.py")
            uam = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(uam)
            sys.modules["uam"] = uam
        mod = sys.modules["uam"]
        pieces = agent._pc_pieces
        info = []
        for pid, p in sorted(pieces.items()):
            pos = mod.bbox_min(p["cells"]) if p["cells"] else None
            info.append(f"{pid}:pos={pos} np={len(p['ports'])}"
                        f" pats={len(p['pats'])} st={p['stack']}")
        sol = agent._pc_solution
        soltxt = ""
        if sol:
            parts = []
            for pid, (pat, pos) in sorted(sol.items()):
                p = pieces.get(pid)
                cur = (mod.pat_norm(p["cells"], p["ports"]),
                       mod.bbox_min(p["cells"])) if p and p["cells"] else None
                okp = "P" if cur and cur[0] == pat else "p"
                oko = "O" if cur and cur[1] == pos else "o"
                parts.append(f"{pid}@{pos}{okp}{oko}")
            soltxt = " ".join(parts)
        print(f"[{agent.action_counter:4d}] lv={lv} act={action.value}"
              f"{getattr(action.action_data, 'x', '')},"
              f"{getattr(action.action_data, 'y', '')}"
              f" sel={agent._pc_sel} xfs={agent._pc_xform_spent}"
              f" bn={agent._pc_bounce} strikes={agent._pc_strikes}"
              f" fail={len(agent._pc_failed_cfgs)}"
              f" unr={len(agent._pc_unreachable)}"
              f" bench={agent._pc_benched} | {' | '.join(info)}"
              f" | SOL {soltxt}", flush=True)
    return action


agent.choose_action = dump
agent.main()
print(f"FINAL levels={agent._best_levels} deaths={agent._game_overs}")
