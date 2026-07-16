# Submission candidates

The original Curio notebook remains in `notebooks/`. High-score candidates live
here so experimental kernels cannot overwrite the proven Curio submission or
its Kaggle history.

## Duck v15

`duck-v15/` packages the strongest currently audited public Duck variant. It
uses the milestone-winning Qwen 3.6 27B Python-tool harness and adds four safe
grafts:

- an action-efficiency reminder based on stalls and state revisits;
- retry protection around transient analyzer failures;
- early termination of provable no-op action-batch tails;
- win banking, which replays a completed solution after pruning wasted actions.

Validate and push it with:

```sh
make verify-duck-v15
make submit-duck-v15
make status-duck-v15
```

The kernel is private, uses the competition's RTX Pro 6000 runtime, keeps
internet disabled, and references the public CC0/MIT source and model datasets
listed in its `kernel-metadata.json`.
