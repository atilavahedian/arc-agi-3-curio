# Curio — Verified Results

This file records measured results for the original Curio agent. Local public
environments are regression and development evidence; they are not a prediction
of the hidden ARC-AGI-3 leaderboard score.

Last updated: 2026-07-27.

## Official Kaggle evidence

The latest completed original-Curio submission (`54980829`, submitted
2026-07-25) scored **0.18**. A fresh pull of the private Kaggle kernel proved
that its embedded agent matches the current repository source byte-for-byte:
SHA-256 `e7132bb6659a48ed84e80c0d73aa5dc09c85ad110b6c701d51a07e6ba4fa61c9`.
The kernel completed successfully, and the candidate contains no dataset,
kernel, or model sources beyond the official competition source. The low score
is therefore a generalization result, not a stale-package or runtime-failure
artifact.

## Where the score actually is (measured 2026-07-25)

`scripts/sweep.py` records per-level actions, baselines and scores, so score
loss can be attributed instead of guessed.

Full official set, seed 0, 10,000-action cap:

| Configuration | Public-game score |
|---|---:|
| head before this work | 28.9361% |
| head after the sort-parser fix | 29.4916% |
| `CURIO_GENERIC_ONLY=1` | 0.1433% |

**The metric tops out at 100%.** A per-game score is
`min(weighted_level_avg, completed_weight/total_weight*100)`, and that cap is
exactly 100 when every level is completed. The cross-game mean is reported
directly in percentage points; it is not divided by 100. This is observed, not
just derived: `ft09` beats the human baseline on all six levels (124 actions
against 208) and still scores exactly 100.00%.

Two facts follow, and they set the priorities.

**Scoring heavily favors complete, efficient play.** The seven full wins
contribute 28.0000 points to the 25-game mean; all partial campaigns together
contribute the remaining 1.4916 points (5.1% of the measured 29.4916%). The
metric squares `baseline/actions`, so a level completed in 9,990 actions is
worth ~0. A game's score is also capped at
`completed_weight/total_weight*100` — completing 1 of 8 levels caps that game
at 2.8 no matter how fast it was. Partial progress matters, but complete wins
dominate this public-game development measurement.

**The family heads carry the public-game score.** The generic core wins zero
public games alone and scores 0.1433%, versus 29.4916% for the complete agent.
Kaggle evaluates a separate set of 110 unseen games, half for each leaderboard,
so neither number predicts the hidden result. The unchanged 0.18% Kaggle score
after adding public-family wins shows that those gains did not transfer and
invalidates the earlier hidden-variant assumption.

## Generic-core exploration: measured dead ends

Recorded so none of these is retried on intuition.

The dominant visible symptom is an unsolved-level wall: 17 of 18 non-winning
games spend at least 8,000 actions on their first unsolved level, including 15
that spend at least 9,000. Among the 16 non-winners with any completed level,
only three complete every solved prefix at or below the human baseline.
Instrumented with `scripts/diagnose_stall.py`, two mechanisms were confirmed
and two fixes were measured and rejected.

*Novelty trap.* On lp85 level 2 the explorer emits the click key `6:20,16` 610
times from 134 distinct states. A click that reliably changes the board makes
every successor state look novel, so per-state `_tried` never blocks the key and
`_steps_since_novelty` never trips. Adding level-scoped click-repetition
balancing to `_use_balance` (penalty zero below the cap, so orderings stay
bit-identical) measured *worse*: 34.0154 against 34.0305 on a fixed 12-game
subset. Reverted.

*Frontier exhaustion.* `_gx_untried` returns 0 or 1 candidates in 88% of calls
on a stalled level (4,429 empty of 7,963). The documented escape is
RESET-backtracking, bounded by `GX_RESET_CAP`. Raising that bound from 4 to 32
to 128 produced a bit-identical 34.0154 every time, because the branch never
executes at all: instrumenting `_gx_emit_reset` over lp85, tn36 and s5i5 at
3,000 actions each recorded `gx_resets=0` and no reset reasons on any of them.
The stall gate is unreachable in practice, so tuning it cannot help.

## Family-head robustness: what is and is not available

Six heads were traced to the exact `return None` that fires on their first
unsolved level, by `sys.settrace` restricted to each head's code object.

Landed: the assignment head's welded-ring and border-named-door relaxations
(sb26 2.78 -> 16.67, aggregate +0.5556, all seven wins byte-identical).

Diagnosed, verified by prototype, not landed — each buys only capped partial
credit:

| Game | Head | Finding | Value if landed |
|---|---|---|---:|
| su15 | `_herd_policy` | movers are 1x1 sprites the blob parser misreads; HUD band not excluded | +0.18 |
| ls20 | `_plan_attr_route` | wall rule records no-motion but not displacement | +0.07 |
| dc22 | `_switch_policy` | level 5 is not a switch-planning problem at all | none |
| vc33 | none fires | a conserved-transfer network; no head models it; levels 4-7 add swap gates | none |
| lf52 | none fires | peg solitaire; no head models the verb | none |

The pattern is consistent and worth stating plainly: these games stop on
*unmodelled mechanics*, not on brittleness. Because a game is capped at
`completed_weight/total_weight*100`, clearing 3 of 8 levels can never exceed
16.67, so slow partial progress moves the public-game mean only slightly.
Landing every remaining item above is worth roughly +0.25 public percentage
points. That is useful local evidence, not a hidden-score prediction.

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

## Dynamic node-maze promotion gate

Commit `9a4d78f` replaces rare-color identity guesses with stable visual glyph
and goal identities, learns actor activation and motion from settled frame
transitions, and plans in joint player/actor state space. The dynamic state
graph is confined to the parsed lattice domain.

Exact official seed-0 result from the combined committed `main` tree:

| Game | Result | Actions | Scorecard |
|---|---:|---:|---:|
| tu93 | 9/9 WIN | 211 | 100.0 |

Verification at promotion:

- combined full unit suite: 127/127;
- focused graph/slide tests: 40/40;
- all 25 official opening frames checked: only `tu93` passed the complete
  slide-head gate;
- combined official gate preserved `ka59`, `re86`, `cn04`, `ft09`, `tr87`,
  and `sc25` exact wins;
- no game/level dispatch, fixed coordinates/colors, or scripted paths;
- `git diff --check`: clean;
- independent read-only review: approved.

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
- `sk48`: the frame-only opening probe established a telescoping-arm movement
  grammar and visible target/instruction layout; the interaction rule is still
  unproved, so no production head has been promoted.

## Reproduction

Run a single public game:

```bash
CURIO_EXPLORER=graph CURIO_SEED=0 make play-local GAME=cn04 STEPS=1000
```

Run the fixed three-seed regression subset. Despite its historical filename,
this is not held-out or an unseen-generalization estimate because several games
in the set have since been explicitly tuned:

```bash
CURIO_EXPLORER=graph STEPS=1000 SEEDS="0 7 42" scripts/bench_held.sh
```

Run unit tests:

```bash
MPLCONFIGDIR=/tmp/arc3-mpl .venv/bin/python -m unittest discover -s tests
```
