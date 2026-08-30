# Router/ensemble v2 salvage rejection

Date: 2026-08-30

The only changed public game in the rejected dense-board router was the
four-cardinal-plus-click family.  Its level-start profiles showed one highly
fragmented state with 100 foreground components.  I reconstructed the
smallest plausible structural salvage: retain the original observable gate,
but require at most 64 connected foreground components.  This removes the
fragmented level without using any game/level identifier, coordinate, colour,
or scripted path.

The salvage was evaluated on the affected game for all three required policy
seeds, with the same pinned environment layout (`environment_seed=12345`),
2,000-action cap, graph mode, and the current unmodified source at `f49a51f`:

| policy seed | baseline graph | fragment-cap salvage | delta |
| ---: | ---: | ---: | ---: |
| 0 | 13.6268410440 | 13.9506588984 | +0.3238178544 |
| 7 | 14.1439739289 | 13.3367827102 | -0.8071912187 |
| 42 | 14.3621588521 | 13.8954035539 | -0.4667552982 |

The salvage remains seed-unstable and materially regresses seeds 7 and 42.
Therefore no protected-win rerun or broader gate was justified: there is no
seed-stable positive candidate to promote.  The temporary implementation was
removed.  This is evidence-only; no source policy change, Kaggle submission,
external solver, competitor code, or push was used.
