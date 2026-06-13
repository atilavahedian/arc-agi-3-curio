I have all the context I need from the five analyst deliverables. Let me write the roadmap.

# ARC-AGI-3 Engineering Roadmap to ~0.6

## 1. Honest verdict

**0.6 is reachable only as a thin-tail outcome (~10-15% probability), and only if one specific bet pays off: that a game-agnostic graph-exploration core — not the existing per-game family modules, and not an LLM — can be made near-SOTA-general on unseen games.** The headline 8.74 public score is a public-overfit artifact; the number that actually scores on the hidden set is the frozen-8 generalization estimate of **0.087**, with seed-0 solving literally zero held-out levels. So 0.6 is not a 2-3x climb, it is a ~6-12x climb on the metric that counts, against a 1156-team field where #1 is 1.21 and the median is 0.21, under a hard constraint that you get **zero hidden-set feedback until 2026-08-12**. The conditions under which 0.6 becomes plausible are all-or-nothing: (a) the generic graph explorer matures to the ~0.30-0.40 band that the published SOTA-general method (Just-Explore, 3rd on Preview, pure CPU) demonstrably reaches; (b) your self-built held-out proxy turns out to be honest rather than optimistic; (c) post-Aug-12 real-feedback iteration plus absorbed open-source drops adds the final increment; and (d) the field's 0.6 bar hasn't risen too far by November. The honest plan-of-record is **0.35**; 0.6 is the stretch, not the target. The single highest-leverage decision is to **kill the per-game-family iteration loop now** — it has saturated (8.74 public / 0.087 hidden proves it does not transfer) — and reinvest 100% of effort into the only component whose gains transfer to the hidden set: the generic explorer.

## 2. Chosen architecture

**Primary spine: a training-free, game-agnostic graph-based explorer (pure CPU Python), with Curio v7's family modules demoted to opportunistic, high-confidence-gated boosters, and a strictly-gated local LLM as an optional meta-controller layered on top — additive, never on the hot path.**

### The spine (this is the product)
A Just-Explore-style global state-graph explorer, built by refactoring machinery Curio already has rather than greenfield:
- **Perception:** reuse Curio's `components()` 4-connected segmentation, `shape_signature()` rotation-canonical hashing, `moved_objects()`/`recolored_objects()` frame-diff extraction, and the HUD-mask machinery (`_update_hud_mask`, `_masked_hash`). Status-bar/HUD masking is a large state-dedup win — keep it.
- **State graph:** nodes = perceptual masked-frame hashes; edges = `(state_hash, action) -> state_hash`. Deterministic outcomes are cached and **never re-tried** (this is what beats random by 3-5x within the interaction budget).
- **Action selection:** salience-tiered prioritization — exhaust high-salience untested actions first (by segment size/morphology/color), then shortest-path (BFS frontier) navigation to the nearest state with an untested high-salience action, escalating the salience threshold only when needed.
- **ACTION6 click taming:** convert the 4096-way click space into ~tens of prioritized candidates via salience tiers over segmented components; snap clicks to object centroids, never raw eyeballed pixels.
- **Correctness invariants (the only thing you can validate blind):** full frontier coverage, no infinite loops, deterministic caching, and — critically — **correct reset handling** (mark reset-inducing actions as "tested"; the published reset-handling bug silently cost the authors ~4 levels and a rank). These invariants are verifiable on public games and game-agnostic by construction.

### The boosters (demoted, not deleted)
Keep Curio's crown-jewel exact solvers as callable tools, but re-gate them so they **only engage on high-confidence structural signature match**, and the strong generic spine runs otherwise: `_plan_route` BFS, `_plan_attr_route`, `_lattice_solve`/`_lattice_gf2` (GF(2) exact solver), `_plan_switch_route`, `_pc_solve` port pairing, the editor rewrite engine, the affordance library (`_click_effects`/`_afford_rank`), and FASTPATH win-path replay. These are exact and sub-baseline-efficient where they fire — but a misfiring module must never hijack a novel game (the seed-0=0 signature). Falsification machinery (`_strikes`/`_benched`) rejects a bad engagement within a few probe actions.

### The optional LLM layer (build last, ship only if it beats pure search on the proxy)
- **Model:** `gpt-oss-20b` (MXFP4, OpenAI, Apache-2.0) — the user's preferred American open-weight model, already run locally. Vendored as an **offline Kaggle Dataset**. Target the **ARC-AGI-3-exclusive RTX 6000 Ada (g4-standard-48, 48GB, native MXFP4, internet forced off** — which exactly matches the offline-notebook eval rule). T4x2 is the fallback tier (gated cadence only). `gpt-oss-120b` does **not fit** (62GB > 48GB) — use it only as an offline prep tool on your own hardware, never in the submission.
- **Role:** the LLM is a **low-frequency meta-controller**, never a per-step actor (per-step LLM is the proven <1% failure mode and quadratically self-defeating against the `(baseline/actions)^2` metric). It is invoked only at discrete junctures the spine flags as novel/uncertain: game entry (after ~30 frames of probing → "which family / what is the win condition?"), stuck escalation (when `_steps_since_novelty > patience` or a module benches → "re-classify / nominate the goal"), and goal nomination (when a solver needs a target). It **proposes, the symbolic layer disposes and verifies**; every hypothesis is falsifiable by the engine and rejected by existing strike/bench logic, so a hallucination costs a few probe actions, never a derailment.
- **Loop discipline (event-driven, not step-driven):** 3 tiers — Tier 0 deterministic executor (no model, ~95-99% of steps, runs Curio's BFS/replay for free); Tier 1 cheap reactive call (single short JSON action, only when the executor is genuinely undecided and budget remains); Tier 2 rare deliberative call on state-change events only (returns a symbolic **plan/macro** that Tier 0 compiles and executes for ~15-40 steps with zero further calls). The biggest lever is **amortization: ≤1 LLM call per ~15-40 executed actions.**
- **State encoding:** never raw 64x64 rasters (~4k-8k tokens/frame, and LLMs misread them). Layered, cheapest-first: **object list** from segmentation (~150-400 tokens, default), **frame diffs** as transition records (~30-120 tokens, the workhorse for learning action semantics), ASCII/downsampled map only on demand. Target <800 tokens/deliberative call, <200/reactive. Pass semantic color names, not bare ints.
- **Budget & safety:** `reasoning_effort=low/minimal` + hard `max_tokens` cap (the documented gpt-oss thinking-block / `num_predict` quirk is the #1 latency risk — uncapped, a 1s call balloons to 10s+). `temperature=0`, fixed seed, JSON-grammar-constrained decoding for determinism across the two Kaggle runs. Hard per-game call cap (~8-15 deliberative + ~30 reactive) and a wall-clock guard; on exhaustion or timeout, **fall back to pure spine** so the notebook never DNFs and the floor is preserved. Validate every returned action against `latest_frame.available_actions` — on parse failure, fall back to the deterministic choice, **never** a blind fixed ACTION5 (which could be a death key).

**Why this beats both extremes:** pure-LLM-on-pixels can't sustain 100s of steps of state tracking (the <1% reality) and is offline-infeasible at scale; pure-Curio's entire 0.087 gap is generalization failure from brittle family gates. The spine carries general coverage on unseen games (the transferable asset); the LLM supplies the one thing Curio structurally lacks and the one thing LLMs are actually good at — few-shot recognition of an unseen game's family/goal — routing to exact solvers that execute cheaply.

## 3. Phased timeline (keyed to real dates; today is 2026-06-13)

### P0 — Honest re-baseline & instrumentation (now → Jun 22, while you still have help)
- Stop trusting 8.74. Build a **held-out generalization harness as the only scoreboard**: split the 25 public games into a "fit" set (the 7 reverse-engineered families) and a "held" set (the other 18, against which no module was built); run ≥3 seeds; report the held-18 scorecard. This reproduces the frozen-8=0.087 result at larger n and becomes your blind proxy.
- **Ablate every family module OFF** and measure the generic core (movement-rule + BFS + novelty) alone on held-18 — that lone number is your true floor.
- Stand up the wall-clock budget manager skeleton early (the 6-9h cap over many private games is a real DNF risk).
- **Expected:** held-18 proxy ≈ 0.05-0.20; generic-core-only ≈ 0.03-0.10. This is the real starting line.

### P1 — Rebuild the generic core as the product (Jun 22 → Jul 31, solo, ~5-6 wks) — MAKE-OR-BREAK
- Replace the "novelty + 2-rule BFS" fallback with the full graph-based explorer described in §2 (segmentation → masked-hash state graph → salience-tiered frontier exploration → systematic RESET backtracking → deterministic caching → **correct reset handling**). This is refactor-and-strengthen, not greenfield — Curio already has the parts.
- Validate **only** on held-18 with strict leave-families-out discipline; treat any proxy gain with suspicion.
- **June 30 — Milestone-#1 winners' code drops** (Tufa Labs et al. must open-source to claim the $25K). Mine it immediately; port any *general, game-agnostic* technique (graph bookkeeping, salience heuristics, click-candidate taming). Do **not** port game-specific hacks.
- **Expected:** held-18 proxy ≈ 0.20-0.35 if the explorer matures (paper hit ~30/52 median levels on 6 games; private-set dilution is the unknown). Everything downstream multiplies off this.

### P2 — Re-gate family modules + LLM layer build (Aug 1 → Aug 11, still blind)
- Re-gate lattice/attr/port/editor/switch to engage **only on high-confidence signature match**, spine runs otherwise. Finalize the wall-clock/per-game budget manager.
- Build the LLM meta-controller plumbing per §2 (offline gpt-oss-20b on RTX6000, read-only state-digest serializers, soft-gate override, verify→strike loop). Ship it **only if** it beats pure search on the held proxy — likely a coin flip; keep a pure-spine submission toggle ready either way.
- **Expected:** held-18 proxy ≈ 0.25-0.40 (family boosters add ~+0.03-0.08 where structure recurs; spine carries the rest).

### P3 — FIRST REAL SIGNAL: submit Aug 12, blind period ends (Aug 12 → Sep 20)
- **Aug 12 (18th birthday, also the submission unlock — no slack):** first legitimate hidden-set submission. Now you have ground truth and 5 subs/day.
- Diff hidden score vs held-18 proxy to calibrate how badly the proxy lied; iterate the generic core against real feedback. This is where most actual gain toward 0.6 (if any) happens.
- **Expected:** first submission realistically 0.15-0.35; with ~5 weeks of real-feedback iteration, plausible reach 0.30-0.45.

### P4 — Milestone #2 push + stretch to final (Sep 20 → Nov 2)
- **Sep 30 — Milestone-#2 open-source drop:** absorb general techniques again.
- Squeeze **efficiency** — the metric is `(baseline/actions)^2` capped at 115, so winning a level in half the baseline actions nearly maxes it. This is pure CPU-search optimization, exactly where Curio already excels (ft09 won sub-baseline). Per-level efficiency tuning + broader generality.
- **Expected:** best realistic final 0.35-0.50; strong-execution tail ~0.55. 0.6 remains a stretch the current data does not support.

## 4. What to build first (strict order)

1. **The held-out harness (P0, days 1-3).** Nothing else matters until you can measure generalization honestly. The proxy *is* the project's instrument — without it you optimize blind in the worst sense.
2. **The generic-core ablation number (P0).** Know your true floor with all family modules off.
3. **The wall-clock budget manager skeleton (P0).** A deep search that works locally can time out and silently score 0 on later hidden games — build the guard before the search, not after.
4. **The graph-based explorer spine (P1).** Highest-certainty transferable win. Reset handling and deterministic caching first (correctness invariants you can verify blind), then salience-tiered frontier exploration, then ACTION6 click taming.
5. **Re-gating of family modules (P2).** Only after the spine is strong, so a misfiring module can never hijack a novel game.
6. **The LLM layer (P2, last).** Pure-additive, behind falsification and a wall-clock guard, shipped only if it beats pure search on the proxy. If it's net-negative (likely, given latency), deprioritize without regret — worst case equals the spine's score, never worse.

The discipline that makes this work: **every component above #4 must be validated only on the held-18 proxy with leave-families-out, and every LLM directive must be falsifiable by the engine** — so the worst case is always bounded to the current floor.

## 5. Probability and likeliest outcome (plainly)

- **Probability of hitting 0.6: ~10-15%.** It is a thin tail, not a base case. You'd need a ~6-12x climb on the honest 0.087 number, to out-engineer ~1150 teams (including Tufa Labs) solo as a first-timer with no compute moat, after 8 weeks of blind optimization against a proxy that held-out generalization proxies systematically over-predict — getting your first real number only 12 weeks into the build with ~7 weeks left, against a 0.6 bar that will likely have *risen* by November.
- **Likeliest actual outcome: a final hidden-set score of ~0.25-0.40** — upper-middle of the leaderboard, clearly above the 0.21 median, well below the 0.59-0.68 top-10 pack. Most probable path: P0 reveals the true floor is ~0.09 and forces the reframe; the P1 generic-explorer rebuild is responsible for the bulk of the gain (proxy to ~0.20-0.30); family boosters add ~+0.05 on recurring structure; the Aug-12 first submission lands ~0.20-0.30 and reveals the proxy was somewhat optimistic; ~5-7 weeks of real-feedback iteration plus the open-source drops push the final to ~0.30-0.40, with efficiency-tuning (Curio's existing strength) providing the last increment.
- **Bottom line:** treat **0.35 as the honest plan-of-record and 0.6 as upside.** A 0.30-0.40 solo result is genuinely respectable and a strong credibility artifact. The path is real but narrow, and it runs entirely through the game-agnostic explorer — not the family modules that produced the misleading 8.74, and not the LLM, which is permitted by the compute envelope but unproven as a generalization lever until you can finally submit on 2026-08-12.

The five analyst deliverables relevant to this roadmap and the agent under analysis live at:
- `/Users/atilavahedian/Claude/arc3/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py` (Curio v7, the agent to refactor)
- `/Users/atilavahedian/Claude/arc3/ARC-AGI-3-Agents/agents/agent.py` and `/Users/atilavahedian/Claude/arc3/ARC-AGI-3-Agents/agents/templates/llm_agents.py` (runtime API and the per-step LLM anti-pattern to avoid)