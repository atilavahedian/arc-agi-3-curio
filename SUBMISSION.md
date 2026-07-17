# ARC-AGI-3 submission runbook

The current original candidate is **Curio Graph v16**. It is packaged separately
at `submissions/curio-graph-v16/` so it cannot overwrite the historical default
Curio kernel.

## What Curio Graph v16 contains

- the exact current `agent/my_agent.py` source;
- `CURIO_EXPLORER=graph` baked into the Kaggle run command;
- the official ARC-AGI-3 competition framework and wheels at runtime;
- no external dataset, model, or kernel sources;
- no external solver, model, dataset, or kernel source.

The kernel is private and has internet disabled. The validator compares the
embedded agent byte-for-byte with the repository source before any push.

## Local evidence

The promotion gate uses the offline ARC engine with 1,000 actions per game and
seeds 0, 7, and 42. On all three seeds:

- `vc33`: 2 levels, improved from the previous 0–1 range;
- `ft09`: 6 levels and a full win in exactly 124 actions;
- `dc22`: 4 levels;
- `lp85`: 1 level;
- `su15`: at least 1 level;
- `sc25`: unchanged at 0.

The 18-game held safety pass showed no level regression in seed 0. These are
local measurements, not a promise of a hidden leaderboard score or first place.

## Build and verify

From the repository root:

```sh
make package-curio-graph-v16
make verify-curio-graph-v16
```

The generated notebook is tracked at
`submissions/curio-graph-v16/submission.ipynb`. Do not edit it manually; edit
`agent/my_agent.py` and rebuild.

## Push the private Kaggle kernel

Ensure `.kaggle/access_token` contains the project-local Kaggle API token, then:

```sh
make submit-curio-graph-v16
make status-curio-graph-v16
```

The push creates or updates this private kernel:

<https://www.kaggle.com/code/atilavahedian/arc3-curio-graph-v16-original>

Wait for the kernel status to become complete. A kernel push does not consume a
competition submission by itself.

## Enter it in the competition

After the kernel completes successfully, open its Output/Submission panel,
select `submission.parquet`, and choose **Submit to Competition**. That final
action consumes one daily submission slot.

Only submit if the account has accepted the current competition rules and meets
the competition's eligibility requirements. Never commit `.kaggle/access_token`.

## Historical default Curio notebook

`notebooks/` remains the legacy config-A package with graph mode off. Its old
`make notebook`, `make submit`, and `make status` targets are kept for
reproducibility; they are not the current v16 recommendation.
