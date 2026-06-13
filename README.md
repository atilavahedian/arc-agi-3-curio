# Curio — a world-modeling agent for ARC-AGI-3

Curio is a general agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition, where an agent is dropped into unseen interactive grid games
with no instructions and must explore, infer the rules, and win.

Most entries start from random action search. Curio instead **learns a world
model at play time** — figuring out what it controls, what each action does,
where the walls are, and what the goal is — then plans toward it.

## Results

Measured locally against the public game set (the offline `arc-agi` engine,
which mirrors the Kaggle scoring metric), 3 seeds, fixed action budget:

| Metric | Value |
|---|---|
| Public games with progress | 13 of 25 |
| Games fully won | `ft09` — all 6 levels, **below the human action baseline** |
| All-25 efficiency scorecard | 8.74 |
| Held-out generalization (8 never-tuned games) | 0.087 (first non-zero) |

The scoring metric is `(baseline_actions / actions_taken)² × 100`, level-index
weighted — so *fast* wins are worth far more than slow ones. `ft09` is solved
exactly (via a GF(2) linear-algebra solver) rather than by search.

## How it works

Curio is a single hand-written Python policy (CPU-only, no training) composed
of cooperating capabilities, each added and verified independently:

- **Perception** — connected-component segmentation + rotation-canonical shape
  signatures; HUD/status-bar masking so a ticking counter doesn't make every
  frame look novel.
- **Control discovery** — rigid-group, multi-color avatar detection and
  movement-rule voting from frame diffs; soft-wall mapping; BFS route planning.
- **Puzzle solvers** — a lattice/recolor model with an exact GF(2) solver
  (constraint puzzles), attribute-state product-graph planning (lock-and-key
  games), and a cursor/editor model with analogy-based goal inference.
- **Memory & efficiency** — a persistent click-affordance library and
  first-discovery speed tuning, because the metric squares efficiency.

## Honest status

This is a mid-field research project with one top-tier component (`ft09`) and a
clearly measured gap: capabilities built by studying public games don't yet
generalize to unseen ones (held-out score 0.087). The roadmap to a competitive
score is in [`ROADMAP_0.6.md`](ROADMAP_0.6.md) — it centers on rebuilding the
game-agnostic exploration core, the only component whose gains transfer to the
hidden evaluation set, with an honest probability assessment.

## Layout

```
agent/my_agent.py     the agent (the one file that defines the policy)
scripts/bench.sh      3-seed benchmark across a fixed game set
scripts/bench_scorecard.py   per-level efficiency scoring (the Kaggle metric)
scripts/             per-game probe/analysis tooling
backups/             verified capability checkpoints (the agent's evolution)
ROADMAP_0.6.md       engineering + research roadmap
```

## Credit

Built on the official [ARC-AGI-3 Kaggle starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
and [agents framework](https://github.com/arcprize/ARC-AGI-3-Agents) by the ARC Prize team.
