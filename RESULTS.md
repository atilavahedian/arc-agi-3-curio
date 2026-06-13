# Honest Scoreboard — Curio v7

Measured numbers for the Curio ARC-AGI-3 agent. Every figure below is from
a real local bench run; nothing is estimated or extrapolated.

Bench command shape (per game set):

```
cd ARC-AGI-3-Kaggle-Starter
CURIO_SEED=<s> make play-local GAME=<csv> STEPS=4000 2>&1 | grep -E "levels=|Aggregate"
```

Date: 2026-06-13. Budget: 4000 steps/game. All runs at seed 0.

---

## Game sets

- **FIT (7)** — reverse-engineered; used only to guard regressions, NOT to
  judge generalization: `lp85 vc33 ls20 ft09 tr87 cn04 dc22`
- **HELD-18** — honest proxy, never tuned on:
  `tu93 ar25 re86 su15 m0r0 sc25 sp80 ka59 g50t sb26 lf52 bp35 s5i5 r11l sk48 wa30 cd82 tn36`

---

## HELD-18 — full agent (default, CURIO_GENERIC_ONLY unset), seed 0

| game | levels |   | game | levels |   | game | levels |
|------|:------:|---|------|:------:|---|------|:------:|
| tu93 | 0 |   | sc25 | 0 |   | s5i5 | 1 |
| ar25 | 1 |   | sp80 | 1 |   | r11l | 1 |
| re86 | 0 |   | ka59 | 0 |   | sk48 | 0 |
| su15 | 0 |   | g50t | 1 |   | wa30 | 0 |
| m0r0 | 1 |   | sb26 | 0 |   | cd82 | 1 |
|      |   |   | lf52 | 1 |   | tn36 | 1 |
|      |   |   | bp35 | 0 |   |      |   |

**Aggregate scorecard score: `0.07802407296184444`**

(9 of 18 games reach level 1; none reach level 2 within budget.)

---

## HELD-18 — generic core only (CURIO_GENERIC_ONLY=1), seed 0

Family heads disabled: lattice/GF2, editor, attribute-state, port-align,
switch — and their gates. Only the generic core runs (object perception,
movement-rule voting, BFS routing, novelty exploration, HUD masking,
affordance).

| game | levels |   | game | levels |   | game | levels |
|------|:------:|---|------|:------:|---|------|:------:|
| tu93 | 0 |   | sc25 | 0 |   | s5i5 | 1 |
| ar25 | 1 |   | sp80 | 1 |   | r11l | 1 |
| re86 | 0 |   | ka59 | 0 |   | sk48 | 0 |
| su15 | 0 |   | g50t | 0 |   | wa30 | 0 |
| m0r0 | 1 |   | sb26 | 0 |   | cd82 | 1 |
|      |   |   | lf52 | 1 |   | tn36 | 1 |
|      |   |   | bp35 | 0 |   |      |   |

**Aggregate scorecard score: `0.03518127148821125`**

(8 of 18 games reach level 1; the only per-game regression vs. full is
`g50t` 1 → 0 — a family head was carrying that level.)

---

## Ablation summary

| configuration              | HELD-18 aggregate | games at L1+ |
|----------------------------|:-----------------:|:------------:|
| full agent (default)       | 0.07802407296184444 | 9 |
| generic core only          | 0.03518127148821125 | 8 |
| **family-head contribution** | **+0.04284280147363319** | **+1** |

The five family heads roughly **double** the held-set aggregate score
(0.078 vs 0.035) over the generic substrate alone. Most of that lift is
concentrated: on this proxy set at seed 0 the family heads add exactly one
extra level (`g50t`), while the generic core carries the other eight L1
games on its own. This is the honest floor — the family heads help, but the
generic perception+planning substrate is doing most of the held-set work.

---

## Regression floor (FIT set, toggle OFF, seed 0)

Confirms default behavior is unchanged by adding the toggle.

| game | required        | measured            | ok |
|------|-----------------|---------------------|:--:|
| ft09 | 6 levels, WIN   | 6 levels, state WIN | ✓ |
| cn04 | >= 5            | 5                   | ✓ |
| tr87 | >= 3            | 3                   | ✓ |
| ls20 | >= 2            | 2                   | ✓ |
| vc33 | >= 2            | 2                   | ✓ |
| lp85 | >= 1            | 1                   | ✓ |
| dc22 | >= 4            | 4                   | ✓ |

FIT-set aggregate (toggle OFF, seed 0): `30.996644710050123`.
**Floor intact** — the ablation toggle is bit-identical to the prior agent
when `CURIO_GENERIC_ONLY` is unset.
