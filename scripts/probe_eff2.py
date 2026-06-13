"""Deep probe: click-by-click affordance picture for one game.

Usage:
    CURIO_SEED=0 .venv/bin/python scripts/probe_eff2.py --game vc33 --max-steps 900
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_module():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--max-steps", type=int, default=900)
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    mod = load_module()
    MyAgent = mod.MyAgent
    MyAgent.MAX_ACTIONS = args.max_steps

    orig_choose = MyAgent.choose_action
    log = []  # (n, level, kind, sig_known_state)

    def choose(self, frames, latest_frame):
        grid = mod.grid_of(latest_frame)
        act = orig_choose(self, frames, latest_frame)
        pa = self._prev_action
        if pa and pa.startswith("6:") and grid is not None:
            x, y = map(int, pa[2:].split(","))
            sig = mod.signature_under(mod.components(grid), (x, y))
            eff = self._click_effects.get(sig)
            if eff is None:
                kind = "new"
            else:
                ch, tr = eff
                pp = (ch + 1) / (tr + 2)
                if tr >= mod.CLICK_PROBES and pp < mod.CLICK_DEAD:
                    kind = "DEADCLICK"
                elif pp > 0.5:
                    kind = "productive"
                else:
                    kind = f"probe{tr}"
            log.append((self.action_counter, latest_frame.levels_completed,
                        kind))
        elif pa and pa.isdigit() and pa != "0":
            log.append((self.action_counter,
                        latest_frame.levels_completed, f"act{pa}"))
        return act

    MyAgent.choose_action = choose

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    agent = MyAgent(
        card_id="local-dev", game_id=args.game,
        agent_name=f"probe.{args.game}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["local-dev"])
    agent.main()
    final = agent.frames[-1]
    print(f"FINAL levels={final.levels_completed} actions={agent.action_counter}")
    per_level: dict[int, Counter] = {}
    for n, lvl, kind in log:
        per_level.setdefault(lvl, Counter())[kind] += 1
    for lvl in sorted(per_level):
        c = per_level[lvl]
        # collapse probes
        probes = sum(v for k, v in c.items() if k.startswith("probe"))
        acts = sum(v for k, v in c.items() if k.startswith("act"))
        print(f"  level {lvl}: new={c.get('new', 0)} probes={probes} "
              f"productive={c.get('productive', 0)} "
              f"DEAD={c.get('DEADCLICK', 0)} simple={acts}")
    lib = agent._click_effects
    prod = {s: e for s, e in lib.items() if (e[0] + 1) / (e[1] + 2) > 0.5}
    dead = {s: e for s, e in lib.items()
            if e[1] >= mod.CLICK_PROBES
            and (e[0] + 1) / (e[1] + 2) < mod.CLICK_DEAD}
    print(f"library: {len(lib)} sigs, {len(prod)} productive, {len(dead)} dead")
    print("productive:", sorted(prod.values()))
    tries = Counter(e[1] for e in lib.values())
    print("tries histogram:", dict(sorted(tries.items())))


if __name__ == "__main__":
    main()
