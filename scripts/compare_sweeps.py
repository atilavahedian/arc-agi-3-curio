"""Diff two `scripts/sweep.py` result files.

Promotion rule of thumb: a change is only worth keeping if the aggregate rises
and no game regresses from a win to a non-win. Because a per-game score is
capped at 100 and a slow level completion scores ~0 (the metric squares
baseline/actions), aggregate movement is dominated by wins gained or lost --
so the per-game table below is the part to read, not just the total.

Usage:
    .venv/bin/python scripts/compare_sweeps.py runs/before.json runs/after.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--eps", type=float, default=0.005,
                   help="Per-game delta below which a game counts unchanged.")
    args = p.parse_args()

    before, after = load(args.before), load(args.after)
    bg = {g["game"]: g for g in before["games"]}
    ag = {g["game"]: g for g in after["games"]}

    print(f"before: {before.get('label') or args.before}")
    print(f"after : {after.get('label') or args.after}\n")

    gains, losses = [], []
    for game in sorted(set(bg) | set(ag)):
        b, a = bg.get(game), ag.get(game)
        bs = b["score"] if b else 0.0
        as_ = a["score"] if a else 0.0
        delta = as_ - bs
        bwin = bool(b and "WIN" in b["state"])
        awin = bool(a and "WIN" in a["state"])
        flag = ""
        if bwin and not awin:
            flag = "  <-- LOST WIN"
            losses.append(game)
        elif awin and not bwin:
            flag = "  <-- NEW WIN"
            gains.append(game)
        if abs(delta) >= args.eps or flag:
            print(f"  {game:6} {bs:6.2f} -> {as_:6.2f}  ({delta:+6.2f})"
                  f"  actions {b['actions'] if b else 0:6} ->"
                  f" {a['actions'] if a else 0:6}{flag}")

    ba, aa = before["aggregate"], after["aggregate"]
    print(f"\nAGGREGATE {ba:.4f} -> {aa:.4f}  ({aa - ba:+.4f})")
    print(f"leaderboard estimate {ba / 100:.4f} -> {aa / 100:.4f}")
    if gains:
        print(f"new wins:  {', '.join(gains)}")
    if losses:
        print(f"LOST wins: {', '.join(losses)}   <-- blocks promotion")
    print("VERDICT:", "REGRESSION" if losses else
          ("IMPROVED" if aa > ba + 1e-9 else "neutral"))


if __name__ == "__main__":
    main()
