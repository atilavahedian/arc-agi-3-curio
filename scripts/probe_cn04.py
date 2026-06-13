"""Ground-truth probe: drive cn04 with a scripted action list and dump frames.

Usage: .venv/bin/python scripts/probe_cn04.py
"""
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


def show(frame, label, full=False, rows=None):
    g = frame.frame[-1]
    print(f"== {label}  state={frame.state} levels={frame.levels_completed} "
          f"nframes={len(frame.frame)}")
    if full or rows:
        for y, row in enumerate(g):
            if rows and not (rows[0] <= y <= rows[1]):
                continue
            print(f"{y:2d} " + "".join(GLYPH[v + 1] for v in row))


def act(env, a, x=None, y=None):
    action = GameAction.from_id(a) if a else GameAction.RESET
    data = {"x": x, "y": y} if x is not None else None
    return env.step(action, data=data)


f = act(env, 0)
show(f, "RESET")

# A selected (black), ports at bottom anchors (14,26),(20,26).
# Rotate once: ports left at anchors (11,14),(11,20)  [verified earlier]
f = act(env, 5)
show(f, "rot1")
# B port anchors: (38,35),(38,41). delta = (27,21) -> 9 right, 7 down
for i in range(9):
    f = act(env, 4)
for i in range(7):
    f = act(env, 2)
show(f, "aligned?", rows=(28, 52))
# extra action to trigger next_level if win predicate passed
f = act(env, 6, 5, 60)
show(f, "after settle click")

# wrong type pairing: rotate 180 and re-align swapped
f = act(env, 5)
f = act(env, 5)
show(f, "rot 180", rows=(30, 50))
# ports now at (53,35),(53,41); move (-15,-6) -> 5 left, 2 up
for i in range(5):
    f = act(env, 3)
for i in range(2):
    f = act(env, 1)
show(f, "re-aligned", rows=(30, 50))
f = act(env, 6, 5, 60)
show(f, "after settle click 2", rows=(0, 40))
