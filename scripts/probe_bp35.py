"""Probe bp35 mechanics: verify rendering, avatar screen anchor, vertical
scroll translations, click coordinate mapping, lava clock, budget bar row 63,
and the 15-action level-1 ground truth.

Usage: .venv/bin/python scripts/probe_bp35.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

logging.basicConfig(level=logging.WARNING)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("bp35")

frame = env.reset()


def grid_of(fr):
    return [list(map(int, row)) for row in fr.frame[-1]]


def show(g, y0=0, y1=64):
    for y in range(y0, y1):
        print(f"{y:2d} " + "".join(f"{v:X}" if v < 16 else "?" for v in g[y]))


def vshift(a, b):
    """Detect uniform vertical shift: b == a shifted by (0, dy), cell level."""
    best = None
    for dy in range(-60, 61):
        if dy == 0:
            continue
        ok = bad = 0
        for y in range(63):  # row 63 = budget bar, exempt
            sy = y + dy
            if 0 <= sy < 63:
                for x in range(64):
                    if a[sy][x] == b[y][x]:
                        ok += 1
                    else:
                        bad += 1
        if ok > 1000 and bad <= (ok + bad) * 0.05:
            if best is None or bad < best[2]:
                best = (dy, ok, bad)
    return best


def act6(x, y):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})


def act(n):
    return env.step(GameAction.from_id(n))


g0 = grid_of(frame)
print("avail:", frame.available_actions, "state:", frame.state)
counts = Counter()
for row in g0:
    counts.update(row)
print("color counts:", counts.most_common())
print("=== start frame ===")
show(g0)

# avatar color: qtzthuktsgl? find the small sprite around rows 36-41
# print rows 30..45 done above; now act
seq = []
prev = g0
fr = frame


def step_and_report(label, fn):
    global prev, fr
    fr = fn()
    g = grid_of(fr)
    sh = vshift(prev, g)
    nch = sum(1 for y in range(64) for x in range(64) if prev[y][x] != g[y][x])
    print(f"{label}: frames={len(fr.frame)} changed={nch} vshift={sh} "
          f"state={fr.state} levels={fr.levels_completed} "
          f"score={fr.score if hasattr(fr, 'score') else '?'}")
    prev = g
    return g


# Ground truth plan for level 1 (engine grid): 4x RIGHT (last one floats up),
# click block above, 3x LEFT, click above, click above, RIGHT, click above,
# 2x LEFT onto gem = 15 actions.
# Avatar screen anchor: rows 36-41 (gravity up). Click "above" = the tile at
# avatar screen bbox shifted by -6 in y.
AV_COLOR = None
g = prev


def find_avatar(g):
    # avatar is the sprite containing color natvcboyxnk (eye); just scan for
    # the smallest component of the avatar body color around rows 30..45.
    # We learn its colors from the start frame at (3,23)->screen x 18..23.
    cells = [(x, y) for y in range(64) for x in range(64)
             if g[y][x] in AV_COLORS]
    if not cells:
        return None
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return min(xs), min(ys), max(xs), max(ys)


# learn avatar colors from start frame: tile (3,23), cam=(0,102): screen
# x=18..23, y=36..41
AV_COLORS = {9, 11}
print("avatar colors:", AV_COLORS)
print("avatar bbox at start:", find_avatar(g0))


def click_above(g, label=""):
    bb = find_avatar(g)
    cx = (bb[0] + bb[2]) // 2
    cy = bb[1] - 3  # one tile up: avatar rows 37-40, tile above = rows 30-35
    return step_and_report(f"CLICK_ABOVE({cx},{cy}){label}",
                           lambda: act6(cx, cy))


g = prev
seq15 = []
for i in range(4):
    g = step_and_report(f"R{i+1}", lambda: act(4))
print("avatar:", find_avatar(g))
g = click_above(g)            # 5: destroy (7,19), float to (7,16)
print("avatar:", find_avatar(g))
bb = find_avatar(g)
g = step_and_report("CLICK_LEFT3", lambda: act6(bb[0] - 16, bb[1] + 1))
# 6: destroy (4,16) three tiles left
print("avatar:", find_avatar(g))
for i in range(3):
    g = step_and_report(f"L{i+1}", lambda: act(3))
print("avatar:", find_avatar(g))   # expect tile x=4
g = click_above(g)            # 10: destroy (4,15), float to (4,13)
print("avatar:", find_avatar(g))
g = click_above(g)            # 11: destroy (4,12), float to (4,10)
print("avatar:", find_avatar(g))
g = step_and_report("R", lambda: act(4))   # 12: (5,10)
g = click_above(g)            # 13: destroy (5,9), float to (5,8)
print("avatar:", find_avatar(g))
g = step_and_report("L1", lambda: act(3))  # 14
g = step_and_report("L2", lambda: act(3))  # 15: gem!
print("FINAL:", fr.state, "levels:", fr.levels_completed)
print("=== level 2 start ===")
show(g)
