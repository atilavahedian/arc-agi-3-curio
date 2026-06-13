"""Probe cn04 level 2: win L1 scripted, then test 0003 movement clamp."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("cn04")

GLYPH = ".0123456789abcdef"


def show(frame, label, rows=None):
    g = frame.frame[-1]
    print(f"== {label}  state={frame.state} levels={frame.levels_completed}")
    if rows:
        for y, row in enumerate(g):
            if rows[0] <= y <= rows[1]:
                print(f"{y:2d} " + "".join(GLYPH[v + 1] for v in row))


def act(env, a, x=None, y=None):
    action = GameAction.from_id(a) if a else GameAction.RESET
    data = {"x": x, "y": y} if x is not None else None
    return env.step(action, data=data)


f = act(env, 0)
# win L1: rotate once, 9 right, 7 down (wrong type), rot 180, 5 left 2 up
f = act(env, 5)
for i in range(9):
    f = act(env, 4)
for i in range(7):
    f = act(env, 2)
f = act(env, 5)
f = act(env, 5)
for i in range(5):
    f = act(env, 3)
for i in range(2):
    f = act(env, 1)
show(f, "L2 start", rows=(0, 63))

# select 0003 (display ~47,24) and press down 8 times
f = act(env, 6, 47, 24)
for i in range(8):
    f = act(env, 2)
    g = f.frame[-1]
    cells = [(x, y) for y in range(64) for x in range(64)
             if g[y][x] == 0 and x > 36 and y > 2]
    ys = [y for _x, y in cells]
    rng = f"{min(ys)}..{max(ys)}" if ys else "none"
    print(f"down {i+1}: black-body y-range {rng}  state={f.state}")
show(f, "after downs", rows=(25, 60))
