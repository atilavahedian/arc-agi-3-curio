"""Instrumented bp35 run: drive MyAgent manually and dump slide-model state
every step.

Usage: CURIO_SEED=0 .venv/bin/python scripts/debug_slide.py [steps]
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode

logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

spec = importlib.util.spec_from_file_location(
    "user_agent_module", ROOT / "agent" / "my_agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MyAgent = module.MyAgent

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("bp35")
agent = MyAgent(card_id="dbg", game_id="bp35", agent_name="dbg",
                ROOT_URL="http://localhost", record=False, arc_env=env,
                tags=["dbg"])
MyAgent.MAX_ACTIONS = steps

raw0 = env.reset()
frames = [agent._convert_raw_frame_data(raw0)]
import arcengine
from arcengine import GameAction, GameState

fr_prev = None
orig_sp = MyAgent._slide_policy
def spy_sp(self, grid, lf):
    r = orig_sp(self, grid, lf)
    if STEP[0] >= 49 and STEP[0] <= 56:
        p_ = self._sl_pitch()
        ph_ = self._sl_phase_xy(p_) if p_ else None
        rt = None
        if p_ and ph_ and self._sl_av is not None:
            cur_ = self._sl_tile(self._sl_av, p_, ph_)
            mv_ = self._sl_movers()
            if len(mv_) == 2:
                rt = self._sl_route(grid, cur_, self._sl_grav, p_, ph_, mv_)
        print(f"   [sp] step={STEP[0]} ret={r} pitch={p_} ph={ph_} grav={self._sl_grav} route={rt if rt is None else (rt[1], rt[0][:4])} strikes={self._sl_strikes} walk={len(self._sl_walk)} benched={self._sl_benched}")
    return r
MyAgent._slide_policy = spy_sp
orig_learn = MyAgent._learn
def spy_learn(self, grid, leveled=False, frames_=None):
    before = {a: dict(v) for a, v in self._move_votes.items()}
    orig_learn(self, grid, leveled, frames_)
    for act, v in self._move_votes.items():
        for d, n in v.items():
            if n != before.get(act, {}).get(d, 0) and abs(d[0]) == 8:
                print(f"   !!! vote {act}->{d} at step {STEP[0]} pa={self._prev_action}")
MyAgent._learn = spy_learn
STEP = [0]
for i in range(steps):
    STEP[0] = i
    latest = frames[-1]
    action = agent.choose_action(frames, latest)
    a = agent
    why = getattr(action, "reasoning", None)
    print(f"#{i:4d} act={agent._prev_action!r:10} ev={a._sl_events} "
          f"eng={a._sl_engaged} cam={a._sl_cam} av={a._sl_av} "
          f"grav={a._sl_grav} plan={len(a._sl_plan)} "
          f"strikes={a._sl_strikes} lvl={latest.levels_completed} "
          f"st={latest.state.name} why={why}")
    if action == GameAction.RESET and not str(action.action_data):
        pass
    if i in (61, 62, 63):
        import copy as _copy
        globals().setdefault('saved', {})[i] = _copy.deepcopy(a._prev_grid)
    if i == 64:
        m = module
        ga, gb = saved[62], saved[63]   # post-7 grid, post-4 grid
        groups = {}
        for color, delta, cells in m.moved_objects(ga, gb):
            groups.setdefault(delta, []).append((color, len(cells)))
        print("  A->B groups:", {d: v[:5] for d, v in groups.items()})
        groups = {}
        for color, delta, cells in m.moved_objects(gb, saved[61] if False else ga):
            groups.setdefault(delta, []).append((color, len(cells)))
        print("  B->A groups:", {d: v[:5] for d, v in groups.items()})
    if i == 60:
        # gate forensics
        lf = latest
        print("  avail:", lf.available_actions, [type(x) for x in (lf.available_actions or [])][:2])
        print("  events:", a._sl_events, "engaged:", a._sl_engaged)
        print("  pitch:", a._sl_pitch(), "av:", a._sl_av)
        print("  movers:", a._sl_movers())
        print("  move_votes:", {k: v.most_common(3) for k, v in a._move_votes.items()})
        import arcengine as _ae
        print("  6 in avail:", _ae.GameAction.ACTION6.value in set(lf.available_actions or []))
        sp = a._slide_policy(a._prev_grid if False else [list(r) for r in fr_prev], lf) if False else None
    fr = agent.do_action_request(action)
    if fr is None:
        print("env returned None")
        break
    frames.append(fr)
    agent.action_counter += 1
    if fr.state is GameState.WIN:
        print("WIN!")
        break

a = agent
print("events:", a._sl_events, "engaged:", a._sl_engaged)
print("grav votes:", a._sl_grav_votes, "grav:", a._sl_grav)
print("walk classes:", len(a._sl_walk))
for c in list(a._sl_walk)[:10]:
    print("  walk", c)
print("block:", a._sl_block.most_common(10))
print("clickfx:", a._sl_clickfx)
print("probe:", a._sl_probe.most_common(10))
print("deadly:", a._sl_deadly)
print("phase:", a._sl_phase)
print("pitch:", a._sl_pitch(), "rules:", a._movement_rules())
print("map size:", len(a._sl_map))
print("levels:", frames[-1].levels_completed)
