"""Probe dc22 mechanics directly: click each button repeatedly and dump the
changed-cell masks, plus footprint info around the avatar.

Usage: .venv/bin/python scripts/probe_dc22.py
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
env = arc.make("dc22")

frame = env.reset()


def grid_of(fr):
    return fr.frame[-1]


def diff(a, b):
    out = []
    for y in range(64):
        for x in range(64):
            if a[y][x] != b[y][x]:
                out.append((x, y))
    return out


def act6(x, y):
    a = GameAction.ACTION6
    a.set_data({"x": x, "y": y})
    return env.step(a)


def act(n):
    return env.step(GameAction.from_id(n))


g0 = grid_of(frame)
counts = Counter()
for row in g0:
    counts.update(row)
print("color counts:", counts.most_common())

# find avatar color 14
av = [(x, y) for y in range(64) for x in range(64) if g0[y][x] == 14]
print("avatar cells:", av)
goal = [(x, y) for y in range(64) for x in range(64) if g0[y][x] == 11]
print("goal cells:", goal)

# buttons: color 9 and 8 blobs (from sprite defs)
for color in (8, 9, 6, 15):
    cells = [(x, y) for y in range(64) for x in range(64) if g0[y][x] == color]
    if cells:
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        print(f"color {color}: n={len(cells)} bbox=({min(xs)},{min(ys)})-({max(xs)},{max(ys)})")

# click button color 9 (blrmbx) center repeatedly
cells9 = [(x, y) for y in range(64) for x in range(64) if g0[y][x] == 9]
cx = sum(x for x, _ in cells9) // len(cells9)
cy = sum(y for _, y in cells9) // len(cells9)
print(f"\nclicking color-9 button at ({cx},{cy}) 4 times:")
prev = g0
for i in range(4):
    frame = act6(cx, cy)
    g = grid_of(frame)
    d = diff(prev, g)
    interior = [c for c in d if 3 <= c[0] < 61 and 3 <= c[1] < 61]
    print(f"  click {i}: nframes={len(frame.frame)} changed={len(d)} interior={len(interior)} "
          f"bbox={None if not interior else (min(x for x,_ in interior), min(y for _,y in interior), max(x for x,_ in interior), max(y for _,y in interior))}")
    prev = g

cells8 = [(x, y) for y in range(64) for x in range(64) if prev[y][x] == 8]
cx8 = sum(x for x, _ in cells8) // len(cells8)
cy8 = sum(y for _, y in cells8) // len(cells8)
print(f"\nclicking color-8 button at ({cx8},{cy8}) 4 times:")
for i in range(4):
    frame = act6(cx8, cy8)
    g = grid_of(frame)
    d = diff(prev, g)
    interior = [c for c in d if 3 <= c[0] < 61 and 3 <= c[1] < 61]
    print(f"  click {i}: nframes={len(frame.frame)} changed={len(d)} interior={len(interior)} "
          f"bbox={None if not interior else (min(x for x,_ in interior), min(y for _,y in interior), max(x for x,_ in interior), max(y for _,y in interior))}")
    prev = g

# dump the avatar's surroundings: what does the floor look like
av = [(x, y) for y in range(64) for x in range(64) if prev[y][x] == 14]
ax = min(x for x, _ in av)
ay = min(y for _, y in av)
print(f"\navatar anchor=({ax},{ay});  16x16 window around it:")
for y in range(max(0, ay - 6), min(64, ay + 10)):
    print("   ", "".join(f"{prev[y][x]:X}" if prev[y][x] >= 0 else "?"
                         for x in range(max(0, ax - 6), min(64, ax + 10))))

# movement probe: step in each direction once and report
for n in (1, 2, 3, 4):
    before = grid_of(env.frames[-1]) if hasattr(env, "frames") else prev
    frame = act(n)
    g = grid_of(frame)
    av2 = [(x, y) for y in range(64) for x in range(64) if g[y][x] == 14]
    ax2 = min(x for x, _ in av2) if av2 else -1
    ay2 = min(y for _, y in av2) if av2 else -1
    print(f"ACTION{n}: avatar=({ax2},{ay2}) nframes={len(frame.frame)}")
    prev = g
