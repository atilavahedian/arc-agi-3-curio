# Curio — a world-modeling agent for ARC-AGI-3

Curio is a general agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition, where an agent is dropped into unseen interactive grid games
with no instructions and must explore, infer the rules, and win.

Most entries start from random action search. Curio instead **learns a world
model at play time** — figuring out what it controls, what each action does,
where the walls are, and what the goal is — then plans toward it.

## Current evidence

The latest competition submission scored **0.18**. Source-hash and timestamp
auditing tied that run to commit `29e278a`; it predates multiple later solver
capabilities. The current head has not yet received a leaderboard score.

The current source has independently verified complete local wins on six
official campaigns: `re86` (8 levels, 552 actions), `cn04` (328 actions),
`ft09` (124), `tr87` (242), `sc25` (147), and `ka59` (7 levels, 313
actions). These checks prove those implementations against the official local
environments; they do not predict hidden-game generalization.

The scoring metric squares `human_baseline_actions / agent_actions`, caps the
per-level value at 1.15, weights later levels more heavily, and averages across
games. Fast wins are therefore worth much more than slow exploration.

## How it works

Curio is a single hand-written Python policy (CPU-only, no training) composed
of cooperating capabilities, each added and verified independently:

- **Perception** — connected-component segmentation + rotation-canonical shape
  signatures; HUD/status-bar masking so a ticking counter doesn't make every
  frame look novel.
- **Control discovery** — rigid-group, multi-color avatar detection and
  movement-rule voting from frame diffs; soft-wall mapping; BFS route planning.
- **Puzzle solvers** — a lattice/recolor model with an exact GF(2) solver,
  attribute-state product-graph planning, structural port assembly, typed
  editor-rule synthesis, visual spell-program execution, and a frame-derived
  kinetic push simulator with weighted state-space planning.
- **Memory & efficiency** — a persistent click-affordance library and
  first-discovery speed tuning, because the metric squares efficiency.
- **Graph exploration** — a state graph with salience-ranked action frontiers,
  learned transition edges, safe click-instance pruning, and reset-aware
  backtracking for games that do not match a proved solver.

## Honest status

The 0.18 result is far below the goal. Public-game solvers are useful regression
proofs, but the main research gap is still unseen-game generalization. Current
work therefore combines narrowly gated exact solvers with an original generic
online-learning and graph-planning core. No external solver notebook, model,
dataset, or kernel is part of the Curio candidate.

## Layout

```
agent/my_agent.py     the agent (the one file that defines the policy)
scripts/bench.sh      3-seed benchmark across a fixed game set
scripts/bench_scorecard.py   per-level efficiency scoring (the Kaggle metric)
scripts/             per-game probe/analysis tooling
backups/             verified capability checkpoints (the agent's evolution)
submissions/curio-graph-v16/ generated original Kaggle candidate
```

## Credit

Built on the official [ARC-AGI-3 Kaggle starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
and [agents framework](https://github.com/arcprize/ARC-AGI-3-Agents) by the ARC Prize team.
