"""Zero-bench probe for the ATTRACTOR-HERD family (su15-style).

Drives the engine using ONLY the rendered grid (no sprite tags), to prove a
generic head can: (1) find a goal anchor + movable blobs from pixels, and
(2) win by walking a blob into the goal via successive attractor clicks.

Usage: .venv/bin/python scripts/probe_su15.py [game] [maxlevels]
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

GAME = sys.argv[1] if len(sys.argv) > 1 else "su15"
MAXLVL = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def grid_of(raw):
    a = np.array(raw.frame)
    g = a[-1] if a.ndim == 3 else a
    return np.array(g)


def components(g):
    """4-connected same-color components of non-background cells."""
    bg = collections.Counter(int(v) for row in g for v in row).most_common(1)[0][0]
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if seen[y, x] or g[y, x] == bg:
                continue
            col = g[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
            comps.append({
                "color": int(col), "size": len(cells),
                "cx": sum(xs)/len(xs), "cy": sum(ys)/len(ys),
                "bbox": (min(xs), min(ys), max(xs)+1, max(ys)+1),
                "cells": cells,
            })
    return comps, bg


def find_env_game(env, cls):
    for attr in dir(env):
        o = getattr(env, attr, None)
        if o is not None and o.__class__.__name__ == cls:
            return o
    return None


def busy(game):
    return getattr(game, "vsfwpngmx", False)


def settle(env, game, cap=40):
    raw = None
    for _ in range(cap):
        raw = env.step(GameAction.ACTION6, {"x": -1, "y": -1})
        if raw is None or (game is not None and not busy(game)):
            break
    return raw


def click(env, game, x, y):
    raw = env.step(GameAction.ACTION6, {"x": int(round(x)), "y": int(round(y))})
    if raw is not None and game is not None and busy(game):
        raw = settle(env, game)
    return raw


def _solidity(c):
    x0, y0, x1, y1 = c["bbox"]
    area = max(1, (x1 - x0) * (y1 - y0))
    return c["size"] / area  # 1.0 = perfectly filled rectangle


def _square(c):
    x0, y0, x1, y1 = c["bbox"]
    w, h = x1 - x0, y1 - y0
    return min(w, h) / max(1, max(w, h))


def goal_and_movers(g):
    """Pixel-only heuristic for the attractor-herd family.

    The playfield is a 64x64 board with thin decoration bars at top/bottom
    edges, optional dotted guide cells (size-1 specks), small compact MOVER
    blobs, and a medium solid GOAL zone. Strategy:
      * drop edge-hugging decoration bars (bbox spans most of a board edge)
      * drop size-1 specks (dotted guides / sub-pixel noise)
      * GOAL = the largest remaining solid, square-ish blob (size ~16..120)
      * MOVERS = remaining compact solid blobs (size ~4..16), nearest first
    """
    comps, bg = components(g)
    if not comps:
        return None, []
    H, W = g.shape

    def is_bar(c):
        x0, y0, x1, y1 = c["bbox"]
        w, h = x1 - x0, y1 - y0
        spans = (w >= W - 4) or (h >= H - 4)
        edge = y0 <= 1 or y1 >= H - 1 or x0 <= 1 or x1 >= W - 1
        return spans and edge

    live = [c for c in comps if c["size"] >= 2 and not is_bar(c)]
    if not live:
        return None, []
    goal_cands = [c for c in live
                  if 16 <= c["size"] <= 120 and _solidity(c) >= 0.5
                  and _square(c) >= 0.6]
    goal = (max(goal_cands, key=lambda c: c["size"]) if goal_cands
            else max(live, key=lambda c: c["size"]))
    gx0, gy0, gx1, gy1 = goal["bbox"]

    def overlaps_goal(c):
        cx, cy = c["cx"], c["cy"]
        return gx0 - 1 <= cx <= gx1 + 1 and gy0 - 1 <= cy <= gy1 + 1

    movers = [c for c in live
              if c is not goal and 4 <= c["size"] <= 16
              and _solidity(c) >= 0.55 and _square(c) >= 0.5
              and not overlaps_goal(c)]
    # nearest mover first (fewest clicks to herd in)
    movers.sort(key=lambda c: (c["cx"]-goal["cx"])**2 + (c["cy"]-goal["cy"])**2)
    return goal, movers


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(GAME)
    raw = env.reset()
    cls = GAME[0].upper() + GAME[1:]
    game = find_env_game(env, cls)
    total_clicks = 0
    last_level = 0
    inert = set()         # mover centers (rounded) proven non-responsive
    SEL = getattr(game, "kacsjmxae", 8) if game else 8
    for outer in range(400):
        g = grid_of(raw)
        lvl = raw.levels_completed or 0
        if g.size == 0:
            raw = click(env, game, 32, 32); total_clicks += 1
            continue
        if lvl >= MAXLVL:
            print(f"REACHED level {lvl} in {total_clicks} clicks")
            return
        if lvl != last_level:
            print(f"-- advanced to level {lvl} at click {total_clicks} --")
            last_level = lvl
            inert.clear()
        goal, movers = goal_and_movers(g)
        movers = [m for m in movers if (round(m["cx"]), round(m["cy"])) not in inert]
        if goal is None or not movers:
            inert.clear()  # reset memory; geometry may have changed
            raw = click(env, game, 32, 32); total_clicks += 1
            continue
        gcx, gcy = goal["cx"], goal["cy"]
        m = movers[0]
        mcx, mcy = m["cx"], m["cy"]
        key = (round(mcx), round(mcy))
        dx, dy = gcx - mcx, gcy - mcy
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 1.0:
            tx, ty = gcx, gcy
        else:
            r = min(SEL, dist)
            tx, ty = mcx + dx/dist*r, mcy + dy/dist*r
        # local patch around the candidate BEFORE the click
        col = m["color"]
        bx0, by0, bx1, by1 = m["bbox"]
        raw = click(env, game, tx, ty); total_clicks += 1
        if raw is None:
            print("terminal/None at click", total_clicks)
            return
        g2 = grid_of(raw)
        # responsiveness: is the candidate's blob still exactly where it was?
        # decorations stay put -> mark inert; movers shift -> keep.
        if g2.size and g2.shape == g.shape:
            still_there = False
            comps2, _bg2 = components(g2)
            for c in comps2:
                if c["color"] == col and abs(c["cx"]-mcx) < 1.0 and abs(c["cy"]-mcy) < 1.0:
                    still_there = True
                    break
            if still_there:
                inert.add(key)
    print(f"did not reach level {MAXLVL}; last_level={last_level}, clicks={total_clicks}")


if __name__ == "__main__":
    main()
