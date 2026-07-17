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
