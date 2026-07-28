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
