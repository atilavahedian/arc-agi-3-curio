# Frozen original baselines

This directory contains provenance-auditable Curio sources used for controlled
Kaggle ablations. They are not imported by the main agent.

## `curio_v7_threadsafe.py`

- Policy source: the repository's pre-external Curio v7 snapshot,
  `backups/curio_v7_salvage.py`.
- Frozen source SHA-256:
  `c45126dfc15719c1ddd6b56718e3075bc53ff7bcab911b1c21d99e81fa9ca0db`.
- Policy changes: none.
- Runtime change: the same thread-local `GameAction.action_data` and
  `GameAction.reasoning` isolation used by the current Curio agent.
- Patched SHA-256:
  `9270deee1c6e25ffa8fe537494b05f13e861cece19276019b4b28a8763e9ac68`.
- External datasets, models, kernels, and solver sources: none.

The alternate source exists to test whether the simpler original policy
generalizes better to hidden games after removing the official Swarm runtime's
cross-thread action-payload race. It must be packaged as a separate private
kernel so it cannot overwrite the current Curio candidate.

Verify the forced-interleaving repair with:

```sh
.venv/bin/python scripts/check_action_thread_safety.py \
  --agent-source baselines/curio_v7_threadsafe.py --iterations 10000
```

## `curio_v7_fault_contained.py`

- Base policy: the frozen thread-safe v7 source above, SHA-256
  `9270deee1c6e25ffa8fe537494b05f13e861cece19276019b4b28a8763e9ac68`.
- Policy changes on healthy calls: none.
- Runtime change: a top-level exception boundary clears transient plans and
  returns a deterministic advertised action, so one unexpected policy fault
  cannot terminate the game thread.
- Delta from the frozen base: 117 insertions, 0 deletions.
- Hardened SHA-256:
  `0153b60a69647fb4dee093cbc41c1a464e348e58ea73a72d1757e3b9c113be69`.
- Reproducible evidence: `runs/v7_fault_containment.json`.
- External datasets, models, kernels, and solver sources: none.

The evidence includes 3,129 action-for-action healthy-path comparisons with
zero fallbacks, plus a 10,000-iteration forced race probe. These are local
engineering gates, not a hidden-score claim.
