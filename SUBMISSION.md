# SUBMISSION — how to submit Curio to ARC Prize 2026 (ARC-AGI-3)

This is the exact, ordered checklist to get the agent onto the Kaggle
leaderboard. Everything is already wired; you mostly run two commands and click
one button. Read the **Eligibility caveat** at the bottom first.

Competition: <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
Kaggle username baked into the kernel: **atilavahedian**

---

## What ships

- **Config A (default).** `CURIO_EXPLORER` unset, `CURIO_GENERIC_ONLY` unset.
- This is the **default in code** — both toggles read `os.environ.get(...)` and
  default to OFF, so the executed path is byte-identical to the verbatim Curio v7
  agent. **Kaggle does not propagate local env vars into the kernel**, so the
  shipped behavior is whatever the source file defaults to — which is config A.
  No action needed to "select" it; it is the default.
- The graph explorer (`CURIO_EXPLORER=graph`) and the generic-only ablation
  (`CURIO_GENERIC_ONLY=1`) remain **local-only toggles** and are NOT active in
  the submission. They lose to config A by ~2x on the held-out proxy (see below),
  so they are kept behind env toggles per the regression-floor rule.

### Honest measured score

All numbers below are **measured locally** with the offline `arc-agi` engine
(mirrors the Kaggle scoring metric), STEPS=4000. Nothing is extrapolated to the
hidden set.

| Config | What | HELD-18 proxy (seed 0) | Games at L1+ |
|---|---|---|---|
| **A (shipped)** | default, family heads + v7 novelty fallback | **0.0780** | 9 |
| B | explorer alone (`CURIO_EXPLORER=graph CURIO_GENERIC_ONLY=1`) | 0.0405 | 9 |
| C | hybrid (`CURIO_EXPLORER=graph`) | 0.0391 | 11 |

- **Shipped held-18 proxy: ~0.078** (seed 0); ~0.079 on seed 7 — a stable
  0.078–0.079 band.
- The explorer reaches *more* levels (config C: 11 games at L1+) but "wins
  slowly": the extra completed levels carry bloated action counts, and the
  `(baseline/actions)² × 100` cap-115 metric never recovers the cost. So A wins.
- **This is a mid-field research score, not a winning one. 0.6 is NOT
  approached and was never expected.** The held-18 set is an honest internal
  proxy; the true hidden-set score is unknown and not claimed.

### FIT regression floor (hard constraint, default config, seed 0) — all met

`ft09=6 WIN (124 actions)`, `cn04=5`, `tr87=3`, `dc22=4`, `ls20=2`, `vc33=2`,
`lp85=1`; aggregate **30.997**. Re-measured this stage, identical to prior.

---

## Step-by-step submission

Run all commands from the repo root:
`/Users/atilavahedian/Claude/arc3/ARC-AGI-3-Kaggle-Starter`

### 0. One-time: accept the competition rules

You cannot submit until you have clicked **"I Understand and Accept"** on the
competition rules. Do this in a browser while signed in as `atilavahedian`:

- Open <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules>
- Click **I Understand and Accept** at the bottom.

(If you skip this, `make submit` / the push will be rejected with a rules error.)

### 1. One-time: install your Kaggle API token

The Makefile reads the token from a project-local file (gitignored, never
committed):

1. Go to <https://www.kaggle.com/settings> → **API** → **Create New Token**.
   This downloads `kaggle.json` containing your API key.
2. Copy *just the key value* into a one-line file at:
   `/Users/atilavahedian/Claude/arc3/ARC-AGI-3-Kaggle-Starter/.kaggle/access_token`

   ```sh
   mkdir -p .kaggle
   # paste the token string (the "key" field from kaggle.json) as the only line:
   printf '%s' 'YOUR_KAGGLE_API_TOKEN' > .kaggle/access_token
   ```

   The token is the `"key"` value inside the downloaded `kaggle.json`, not the
   whole JSON.

### 2. Build the notebook (sanity check before pushing)

```sh
make notebook
```

This regenerates `notebooks/submission.ipynb` from `agent/my_agent.py` with
config A baked in (no env toggles in the run cell). Expect:
`[build_notebook] Wrote notebooks/submission.ipynb  (accelerator: t4)`.

### 3. Push the kernel to Kaggle

```sh
make submit
```

`make submit` runs `make notebook` again (so the notebook is always fresh) and
pushes `notebooks/` to your account as a private kernel. It will refuse to push
if the username placeholder is still present — it is already set to
`atilavahedian`, so this guard passes.

Track the run:

```sh
make status
```

Wait for the kernel to finish running (it executes the agent against the games
via the gateway sidecar and produces `submission.parquet`).

### 4. Submit to the competition (the button)

The kernel push uploads/runs the notebook but does **not** auto-enter it on the
leaderboard. After the kernel run completes successfully:

1. Open your kernel on Kaggle:
   <https://www.kaggle.com/code/atilavahedian/arc-prize-2026-arc-agi-3-starter>
2. Click **Submit to Competition** (top-right of the kernel page, or via the
   **Output** tab → the competition submission panel).
3. Confirm `submission.parquet` is the selected output file, then click
   **Submit**.

That's it — the score appears on your leaderboard entry once Kaggle finishes the
competition rerun.

---

## Caveats

- **Submission limit: 5 per day.** Don't burn them on accidental re-pushes.
  Each `Submit to Competition` click counts; rebuilding/pushing the kernel does
  not (only the final submit does).
- **Eligibility.** Per your account notes, 18+ platform eligibility is the
  gating item (you turn 18 on 2026-08-12). Do **not** submit until you are
  eligible / have a consent path cleared. This file is the runbook for *when*
  you submit, not a green light to submit now.
- **Do not commit `.kaggle/access_token`.** It is gitignored. Same for `.venv/`,
  `vendor/`, `environment_files/`, and `*.log`.
- **`notebooks/submission.ipynb` is a build artifact** (gitignored, regenerated
  by `make notebook`). The canonical source is `agent/my_agent.py` +
  `scripts/build_notebook.py`. Never hand-edit the notebook.

---

## Reproduce the numbers

```sh
cd /Users/atilavahedian/Claude/arc3/ARC-AGI-3-Kaggle-Starter

# Shipped config (A), held-18 proxy:
CURIO_SEED=0 make play-local \
  GAME=tu93,ar25,re86,su15,m0r0,sc25,sp80,ka59,g50t,sb26,lf52,bp35,s5i5,r11l,sk48,wa30,cd82,tn36 \
  STEPS=4000

# FIT regression floor (default config):
CURIO_SEED=0 make play-local GAME=lp85,vc33,ls20,ft09,tr87,cn04,dc22 STEPS=4000
```
