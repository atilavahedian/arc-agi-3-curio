# Curio — Verified Results

This file records measured results for the original Curio agent. Local public
environments are regression and development evidence; they are not a prediction
of the hidden ARC-AGI-3 leaderboard score.

Last updated: 2026-07-17.

## Official Kaggle evidence

The latest completed original-Curio submission scored **0.18**. Its submitted
agent matches repository commit `29e278a`, not the current head. The current
agent has several later, locally verified capabilities and has not yet received
an official hidden score.

The existing generated notebook is intentionally considered stale until the
next capability batch is finished. Rebuild and identity validation are required
before the next Kaggle push.

## Clean 25-game sweep

Source identity: commit `c6b7f9169851f2cf0f813e51a16e17dfb5c94a69`.

Configuration:

```text
CURIO_EXPLORER=graph
CURIO_SEED=0
MAX_ACTIONS=10000
```

The framework loop is inclusive, so capped runs report 10,001 actions.

| Game | Levels | Actions | Final state |
|---|---:|---:|---|
| tu93 | 5 | 10001 | NOT_FINISHED |
| ar25 | 1 | 10001 | NOT_FINISHED |
| re86 | 8 | 552 | WIN |
| su15 | 1 | 10001 | NOT_FINISHED |
| m0r0 | 1 | 10001 | NOT_FINISHED |
| cn04 | 6 | 328 | WIN |
| ft09 | 6 | 124 | WIN |
| tr87 | 6 | 242 | WIN |
| sc25 | 6 | 147 | WIN |
| lp85 | 1 | 10001 | NOT_FINISHED |
| dc22 | 4 | 10001 | NOT_FINISHED |
| sp80 | 1 | 10001 | NOT_FINISHED |
| ka59 | 0 | 10001 | NOT_FINISHED |
| g50t | 1 | 10001 | GAME_OVER |
| sb26 | 1 | 10001 | NOT_FINISHED |
| lf52 | 2 | 10001 | NOT_FINISHED |
| bp35 | 1 | 10001 | NOT_FINISHED |
| s5i5 | 1 | 10001 | NOT_FINISHED |
| r11l | 1 | 10001 | NOT_FINISHED |
| sk48 | 0 | 10001 | NOT_FINISHED |
| wa30 | 0 | 10001 | NOT_FINISHED |
| vc33 | 2 | 10001 | NOT_FINISHED |
| ls20 | 2 | 10001 | NOT_FINISHED |
| cd82 | 1 | 10001 | NOT_FINISHED |
| tn36 | 1 | 10001 | NOT_FINISHED |

Measured total: **59 levels**, **5 complete wins**, aggregate local scorecard
`21.011111569871918`.

## Protected complete wins

The current graph-backtracking head `c7977cb` preserves these exact seed-0
results at a 1,000-action cap:

| Game | Result | Actions |
|---|---:|---:|
| cn04 | 6/6 WIN | 328 |
| ft09 | 6/6 WIN | 124 |
| tr87 | 6/6 WIN | 242 |
| sc25 | 6/6 WIN | 147 |

`re86` also remains an 8-level WIN in 552 actions on the fixed proxy runs.

## Kinetic-push promotion gate

Commit `e3170f0` adds an original frame-derived selected-piece, collision,
socket, force-surface, and countdown model with closed-loop weighted planning.
It contains no game or level dispatch, fixed board coordinates, or scripted
solution path.

Exact official seed-0 result from the committed `main` tree:

| Game | Result | Actions | Scorecard |
|---|---:|---:|---:|
| ka59 | 7/7 WIN | 313 | 100.0 |

Verification at promotion:

- full unit suite: 120/120;
- focused kinetic-push tests: 8/8;
- all 25 official opening frames checked: only `ka59` passed the compound
  action-and-pixel gate;
- protected `re86`, `cn04`, `ft09`, `tr87`, and `sc25` wins unchanged;
- `git diff --check`: clean;
- independent read-only review: approved.

## Graph-backtracking promotion gate

Commit `c7977cb` adds branch-complete, fatal-edge-safe backtracking. RESET is a
true last resort after 256 consecutive non-novel states; exact fatal edges need
four identical state/action deaths before they are excluded. This avoids
mistaking timer or concurrent-hazard deaths for causal evidence.

Fixed 18-game proxy, 1,000 actions/game:

| Seed | Pushed baseline | `c7977cb` |
|---:|---:|---:|
| 0 | 11.570994662830218 | 11.570994662830218 |
| 7 | 11.572613173241251 | 11.572231892877130 |
| 42 | 11.568548011882740 | 11.568548011882740 |
| **Mean** | **11.570718615984737** | **11.570591522530030** |

Completion counts are identical for every game and seed. Two seeds match
exactly; seed 7 differs by 0.00038 after `bp35` confirms one deterministic
fatal edge. The change is treated as proxy-neutral, not as evidence of a hidden
score gain.

Verification at promotion:

- graph tests: 33/33;
- full unit suite: 112/112;
- `git diff --check`: clean;
- independent read-only review: no blocker;
- protected complete wins: unchanged.

## Next measured targets

- `dc22`: current clean sweep is 4 levels. An original, frame-derived remote
  manipulator prototype reaches the level-5 upper branch; production handoff
  with the existing switch planner is in progress.
- `tu93`: repair late-stage avatar/goal tracking and promote the proved
  time-aware hazard planner without regressing its early campaign levels.

## Reproduction

Run a single public game:

```bash
CURIO_EXPLORER=graph CURIO_SEED=0 make play-local GAME=cn04 STEPS=1000
```

Run the fixed three-seed proxy:

```bash
CURIO_EXPLORER=graph STEPS=1000 SEEDS="0 7 42" scripts/bench_held.sh
```

Run unit tests:

```bash
MPLCONFIGDIR=/tmp/arc3-mpl .venv/bin/python -m unittest discover -s tests
```
