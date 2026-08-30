# Router/ensemble v2 rejection

Date: 2026-08-30

This audit tested an original-only, geometry-based arbitration hypothesis: on
a dense frame exposing exactly the four cardinal actions plus click, route the
generic graph explorer through the existing legacy novelty ranker.  A second
revision latched that decision only on a level-start frame, so later repaint
frames could not change the policy identity.  The implementation contained no
game or level identifiers, fixed coordinates, fixed colours, or scripted path.

The hypothesis was rejected.  The corrected comparison used the same current
`f49a51f` source tree, official local ARC runtime, 18-game nonwinner gate,
2,000-action cap, `CURIO_EXPLORER=graph`, and three seeds.  The unmodified
source was run in a detached worktree at the same commit.  Results are
aggregate public-game percentage points:

| seed | unmodified graph | level-start dense router | delta |
| ---: | ---: | ---: | ---: |
| 0 | 2.0732590763 | 2.1605221663 | +0.0872630900 |
| 7 | 2.1033695617 | 2.0808285787 | -0.0225409830 |
| 42 | 2.1074789513 | 2.0702295122 | -0.0372494391 |
| mean | 2.0947025298 | 2.1038600858 | +0.0091575560 |

The positive mean is not an acceptable gate: two of three seeds regress, and
the seed-0 lift is concentrated in one focused board.  Focused runs on that
board measured router/legacy-equivalent scores of 15.1976%, 13.7382%, and
13.6917% for seeds 0, 7, and 42 respectively; this does not establish a
seed-robust causal improvement.  An independent opening-frame scan also found
no public opening satisfying the dense profile, so the rule has no reliable
observable public trigger at level start.

Local focused unit tests for the temporary profile predicate passed 9/9, and
the source compiled, but those tests do not outweigh the failed pooled gate.
The temporary implementation and tests were removed; this file is the only
commit from this lane.  No Kaggle submission, external solver, competitor
code, or push was used.
