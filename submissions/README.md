# Submission candidates

The original Curio notebook remains in `notebooks/`. Promotion candidates live
here so experimental kernels cannot overwrite the proven Curio submission or
its Kaggle history. Only independently developed Curio candidates belong in
this directory.

## Curio Graph v16 (original)

`curio-graph-v16/` is the current fully original candidate. Its notebook is
generated directly from `agent/my_agent.py`, bakes `CURIO_EXPLORER=graph` into
the Kaggle run command, and uses only the official ARC-AGI-3 competition source.
Its dataset, model, and kernel source lists are intentionally empty.

Build, validate, push, and check it with:

```sh
make package-curio-graph-v16
make verify-curio-graph-v16
make submit-curio-graph-v16
make status-curio-graph-v16
```

The validator requires the embedded agent to exactly match the repository
source and rejects external solver, model, dataset, and kernel markers.

## Curio v7 Threadsafe (original ablation)

`curio-v7-threadsafe/` isolates the frozen pre-external Curio v7 policy and
adds only the verified thread-local `GameAction` payload repair. It leaves the
later graph explorer off and has its own private Kaggle kernel identity, so it
cannot overwrite the current candidate or its history.

Build, validate, push, and check it with:

```sh
make package-curio-v7-threadsafe
make verify-curio-v7-threadsafe
make submit-curio-v7-threadsafe
make status-curio-v7-threadsafe
```

Validation requires a byte-exact match to
`baselines/curio_v7_threadsafe.py`, no external sources, no graph-mode
environment override, a legal registry smoke, and the forced threaded action
payload test.

## Curio v7 Hardened (original ablation)

`curio-v7-hardened/` is a second private kernel identity for the same frozen
v7 lineage. It embeds `baselines/curio_v7_fault_contained.py`: the thread-safe
v7 source plus a top-level exception boundary that leaves successful policy
actions unchanged and emits a deterministic advertised action only when an
unexpected hidden-game parser fault occurs. The graph explorer remains off.

Build, validate, push, and check it with:

```sh
make package-curio-v7-hardened
make verify-curio-v7-hardened
make submit-curio-v7-hardened
make status-curio-v7-hardened
```

Validation requires a byte-exact embedded source, private original-only
metadata, no external sources or graph-mode override, a registry smoke that
checks both the untouched legacy path and injected simple/click exceptions,
and a 1,000-iteration forced threaded payload test.
