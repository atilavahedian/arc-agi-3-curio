# Curio — a world-modeling agent for ARC-AGI-3

Curio is a general agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition, where an agent is dropped into unseen interactive grid games
with no instructions and must explore, infer the rules, and win.

Most entries start from random action search. Curio instead **learns a world
model at play time** — figuring out what it controls, what each action does,
where the walls are, and what the goal is — then plans toward it.

## Current evidence

The latest completed competition submission scored **0.18%**. A fresh pull of
the private Kaggle kernel proved that its embedded agent matches the current
repository source byte-for-byte. The run completed successfully, so 0.18% is
the current agent's real unseen-game result, not a stale-package artifact.

The current source has independently verified complete local wins on seven
official campaigns: `tu93` (9 levels, 211 actions), `re86` (8 levels, 552
actions), `cn04` (328 actions), `ft09` (124), `tr87` (242), `sc25` (147),
and `ka59` (7 levels, 313 actions). These checks prove those implementations
against the official local environments; they do not predict hidden-game
generalization.

The scoring metric squares `human_baseline_actions / agent_actions`, caps the
per-level value at 1.15, weights later levels more heavily, and averages across
games. Fast wins are therefore worth much more than slow exploration.

**The leaderboard maximum is 100%, not 1.00.** A game's score is capped at
`completed_weight / total_weight * 100`, which is exactly 100 when every level
is completed, and the cross-game mean is reported directly in percentage
points. In practice local scoring is close to binary: a win scores ~100, and a
game that completes a level or two usually scores under 3 because the squared
efficiency term crushes anything slow.

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
  kinetic push simulator with weighted state-space planning. A time-expanded
  maze planner learns actor identities, activation ranges, and motion queues
  from settled frame transitions.
- **Memory & efficiency** — a persistent click-affordance library and
  first-discovery speed tuning, because the metric squares efficiency.
- **Graph exploration** — a state graph with salience-ranked action frontiers,
  learned transition edges, safe click-instance pruning, and reset-aware
  backtracking for games that do not match a proved solver.

## Honest status

The 0.18 result is far below the goal, and the measurements in `RESULTS.md`
explain why more precisely than "generalization is hard".

Ablation settles which half of the agent matters. With every family head
disabled (`CURIO_GENERIC_ONLY=1`) the agent scores 0.14% on the 25 public games
and wins nothing, against 29.49% for the full agent. Kaggle evaluates a separate
set of 110 unseen games (55 public-leaderboard and 55 private-leaderboard), so
the public-game result cannot be used as a leaderboard estimate. The unchanged
0.18% result after adding public-family wins is evidence that those specialized
heads did not transfer; it is not evidence that the hidden games are variants
of the public families.

The remaining wall is not brittleness. Six heads were traced to their exact
bail-out on the first level they fail, and the games stop on mechanics no head
models — peg solitaire, conserved-transfer networks, remote manipulators.
Because a game is capped at `completed_weight/total_weight*100`, slow partial
progress contributes little on the public development set. Further unseen-game
score requires genuinely general online mechanism discovery, while complete
public-game wins remain useful regression tests rather than hidden-score proof.

No external solver notebook, model, dataset, or kernel is part of the Curio
candidate; `scripts/validate_curio_candidate.py` enforces that on every build.

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
