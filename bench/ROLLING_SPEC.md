# Rolling diffusion-forcing schedule — spec for review

**Status:** proposal, nothing implemented.
**One-line:** emit one frame per dynamics forward instead of five, by keeping W
frames in flight at staggered noise levels, without retraining.

**Read section 5 first.** It contains a scope limitation that decides whether
this is worth building at all, and it is the thing I most want challenged.

---

## 1. What is being proposed

Today `next_latent` denoises one frame with `num_steps` sequential forwards at
`T=1`, then spends a fifth forward committing it to the KV cache.
Five weight streams of 3.147 GB to perform four denoising steps on one frame.

Rolling keeps `W` frames in flight simultaneously at different points on the τ
ladder. Each dynamics forward advances *every* in-flight frame by one step and
retires the most-denoised one. Per emitted frame: **one forward over `W+1`
timesteps of tokens**, instead of **five forwards over one timestep each**.

The token count is identical at `W=4` (5 x 290 either way). The weight traffic
is one fifth. That asymmetry is the entire proposal.

---

## 2. Measured basis

From the refit after `sdpa_cudnn` (log `bi2a2sns6`), the frame cost is affine
to within 2%:

    t(B) = 12.88 ms  +  10.21 ms x B

- **12.88 ms fixed** — streaming 3.147 GB of bf16 weights, five times per
  frame, 15.7 GB/frame. Scales with **forwards**, not with tokens.
- **10.21 ms/stream marginal** — real per-stream compute. Scales with
  **denoising steps**.

Rolling attacks the first term and leaves the second alone. Step distillation
attacks the second and leaves the first alone. They are orthogonal and compose.

---

## 3. Enabling facts, verified in the code

Every one of these was checked, not assumed.

1. **τ is per-frame, not per-call.** `Dynamics.__call__` takes
   `tau_indices (B, T)` and `step_indices (B, T)`, and builds
   `shortcut_token (B, T, 1, d_model)` (`models.py:1289, 1326, 1327`). Each
   timestep in a block carries its own noise level. This is the load-bearing
   fact and it is already exercised: the non-cached debug path in
   `next_latent` concatenates *different* τ for prefill / decode / current
   segments (`generation.py:159-162`).

2. **The ladder does not write the cache.** `refinement_step` calls
   `dynamics(..., caches=caches)` and discards the returned cache
   (`generation.py:167-170`, the `_` in `(h_seq, _)`). Cache updates are
   functional, so the current frame's K/V are written, attended, and thrown
   away. In-flight frames therefore need no cache support — they are
   recomputed every step, which is what already happens.

3. **The commit pass stores frames NOISED, not clean.** After the ladder,
   `latent_noised_caching = latent_t_final * tau_ctx + (1 - tau_ctx) * noise`
   with `tau_ctx_target = 0.9` (`generation.py:203`, `46`). The cache holds
   context at τ=0.9, never at τ=1. Rolling must preserve this exactly.

4. **The cache already handles `T > 1` query blocks.** `prefill` runs
   `T = T_ctx` against caches. `_get_ordered_kv_inplace(query_len)` in
   `bench/patches.py` builds the causal mask as
   `causal = age >= (query_len - 1 - i)`, which is correct for a block of
   `query_len` queries appended after the cache — it was written general.

5. **Space attention is per-frame and stays linear in T.** Space layers operate
   with T folded into the batch axis, so a `T=W+1` block does `W+1` independent
   `290 x 290` attentions, not one `(W+1)*290` squared. **No quadratic blowup.**
   (This was the risk that would have killed the design; it does not apply.)

---

## 4. Design

### 4.1 Block layout

One forward per emitted frame, over `T = W+1` timesteps:

    position   0          1          2        ...   W
    frame      I          I+1        I+2      ...   I+W
    role       COMMIT     in-flight  ...            freshly admitted noise
    tau        tau_ctx    tau[W]     tau[W-1] ...   tau[0] = 0

- Position 0 is the frame that finished denoising on the previous step,
  re-noised to `tau_ctx` exactly as the current commit pass does. Its *output*
  is discarded; it is present only so its K/V get written to the cache.
- Positions 1..W are in flight, ordered oldest (most denoised) to newest.
- Absolute positions I..I+W are chronological, so causal time attention is
  already correct: frame I+2 attends to I+1 and I, never to I+3.

### 4.2 Per-step update

For each in-flight position `k` in 1..W, holding ladder index `s_k = W - k`:

    beta = schedule.beta_values[s_k]
    latent[k] <- beta * latent[k] + (1 - beta) * x_pred[k]

This is byte-for-byte the same Euler update as `refinement_step`
(`generation.py:181`). Every frame receives exactly `W` updates with exactly
the same `beta` sequence as today's ladder, in the same order. **The per-frame
sampler is unchanged.** What changes is only what the *context* looks like
while those updates happen (see 8.1).

### 4.3 Cache handling — the one subtle part

The forward writes `W+1` entries into the ring buffer and returns a cache with
`index += W+1`. We want only position 0 retained.

**Rewind the index to `index + 1`.** Slots for positions 1..W keep stale data,
but `_get_ordered_kv_inplace` masks on `written = age <= index - 1`, so any
slot beyond the index is already excluded from attention, and the next step
overwrites them. No buffer surgery, no extra kernel.

This works with the shipped `no_roll_kv` patch and needs the released
`KVCache.update` for the `T > 1` wrapping case (`fast_kv_write` deliberately
falls back to the original when `T != 1`, which is exactly right here).

### 4.4 Warmup and drain

- **Warmup:** after prefill, the window is empty. Admit one noise frame per
  step for `W` steps before the first emission. Cost: `W` extra forwards once
  per session, against a 134 ms prefill. Negligible.
- **Drain:** at end of stream, run `W` steps with no new admissions.
- **Steady state** is the only regime that matters for throughput.

---

## 5. SCOPE LIMITATION — read this before anything else

**Rolling requires knowing future actions, and interactive play does not have
them.**

Frame `I+k` is conditioned on action `a_{I+k}`. At the moment we start
denoising it, that action is `k` steps in the future. In closed-loop
interactive use — a human or a policy reacting to each rendered frame — those
actions do not exist yet. Denoising them against a placeholder and substituting
the real action at commit time would mean `W-1` of the `W` denoising steps were
conditioned on the wrong action.

So rolling is valid for:

- **Open-loop generation from a known action sequence.** This includes the
  entire existing evaluation path: `quality_rollout.py` replays recorded VPT
  actions, and FVD is computed against those. Also: offline video synthesis,
  scripted rollouts, dataset generation, replay.

and NOT valid for:

- **Closed-loop interactive play**, which is the headline use case for a world
  model. There, `W` must be 1, which is exactly the current sampler.

**Partial mitigation, unproven:** an action-prediction head or a policy that
can commit to `W` actions ahead would restore it, at the cost of being wrong
when the user does something unexpected. A `W=2` variant costs one frame of
action lag (~13 ms) and recovers part of the win. Neither is specified here.

**This is the review question.** If the target is interactive serving, this
proposal is worth much less than section 9 suggests, and step distillation —
which has no such restriction — should absorb the effort instead.

---

## 6. Code changes

Small and contained. No model changes.

| file | change |
|---|---|
| `dreamer/generation.py` | new `rolling_latents()` beside `next_latent`; do not modify the existing function |
| `bench/patches.py` | `rolling_schedule` patch to route `latent_rollout` through it, so it A/Bs like every other patch |
| `bench/profile_inference.py` | `--rolling-window W`; a `rolling` stage next to `ladder` |
| `bench/quality_rollout.py` | already takes `--patches`; nothing to do |

`DenoiseSchedule` needs no change — `beta_values` and `tau_indices` are already
the per-step tables rolling indexes into.

Estimated: **2-3 days** to a measurable prototype, most of it in the cache
index-rewind and the warmup/drain edges.

---

## 7. Correctness plan

Same standard as every other patch in this repo: a gate that fails loudly.

1. **`W=1` must reproduce the released sampler exactly.** With a window of one,
   rolling degenerates to the current ladder. This is the cheapest possible
   regression test and it must be bitwise-identical, not approximately equal.
   If it is not, the block layout or the beta indexing is wrong.
2. **Cache-state assertion.** After each rolling step, the cache index must
   have advanced by exactly 1 and the K/V at the newest slot must equal a
   reference commit pass on the same finalized latent. Direct comparison
   against `next_latent`'s commit.
3. **Action alignment assertion.** Assert the action at block position `k` is
   `a_{I+k}`. An off-by-one here is silent, produces plausible video, and would
   be caught by nothing else.
4. **Firing counter.** Same pattern as `sdpa_cudnn`: hard-fail if the rolling
   path never executes, so an accidental fallback cannot be reported as a win.

---

## 8. Quality gate

### 8.1 What actually changes, and the risk

The per-frame sampler is identical (4.2). What differs is the context each
denoising step sees:

- **Today:** all context at exactly `tau_ctx = 0.9`.
- **Rolling:** committed context at `tau_ctx = 0.9`, plus `k-1` in-flight
  frames ahead of it at intermediate τ forming a monotone staircase.

The argument that this is safe: the model was **trained with diffusion forcing,
i.e. independent per-frame noise levels**, so mixed-τ context is in
distribution. The argument for caution: training samples τ *independently*,
while rolling produces a *monotone staircase* — a thin subset of the training
distribution. Neither today's uniform-`tau_ctx` context nor rolling's staircase
is "the" training distribution; both are special cases.

This is an empirical question and it is the main quality risk.

### 8.2 The protocol needs fixing first

The current FVD gate resolves **~28 FVD (~6%)** at n=6 (see the control-arm
section of `RESULTS.md`). That was adequate for asking "did exact patches break
anything" and is **not** adequate for certifying a change to the sampling
procedure, which could plausibly cost 2-4%.

Before this can be gated, the protocol needs: more windows and more seeds to
shrink the floor, and reporting of **unpaired** statistics with an
**exact-patch control arm** included every time — the paired-by-seed test is
invalid for a chaotic autoregressive rollout and produced a false positive at
t(5)=3.59 once already.

Gate criterion: **rolling at `W=4` must not differ from `steps=4` by more than
the exact-patch control arm differs from baseline, on FVD, drift-at-horizon,
std ratio and motion.**

---

## 9. Projected performance

From the measured decomposition. **Projections, not measurements** — nothing is
implemented.

| config | fwd/frame | B=1 | vs baseline | B=16 agg | vs baseline |
|---|---:|---:|---:|---:|---:|
| today (5 fwd, 4 steps) | 5 | 23.09 ms | 1.94x | 90.8 fps | 1.49x |
| rolling W=4 | 1 | **12.79 ms** | **3.50x** | 96.4 fps | 1.59x |
| rolling + distill to 2 steps | 1 | 8.70 ms | 5.14x | 159.1 fps | 2.62x |
| rolling + distill to 1 step | 1 | 6.66 ms | 6.72x | 235.6 fps | 3.88x |

**Rolling is a latency play, not a throughput play.** It is 1.8x at B=1 and
only 1.06x at B=16, because batching already amortises the weight streaming
that rolling removes. Anyone serving at batch 16 gains almost nothing.

Combined with the action-causality limit in section 5, the honest summary is:
**rolling helps single-stream open-loop generation a lot, and interactive
batched serving almost not at all.**

---

## 10. Risks and kill criteria

| risk | detection | kill criterion |
|---|---|---|
| action causality makes it inapplicable | design review, section 5 | if the target is interactive, stop now |
| staircase-τ context is out of distribution | FVD gate, 8.2 | exceeds the control-arm floor |
| `W=1` does not reproduce the released sampler | test 7.1 | any mismatch — indicates a design error, not a tuning issue |
| index rewind interacts badly with ring wrap | test 7.2 across a wrap | any cache mismatch |
| one big forward compiles worse than five small | measure `rolling` stage | under 1.5x at B=1 |
| gain evaporates at serving batch | already known, section 9 | not a kill, a scoping fact |

---

## 11. Recommendation

Build it **only if open-loop generation is a real target**. It is 2-3 days for
a projected 1.8x single-stream, it needs no training compute, no data pipeline
and no new checkpoint, and it composes cleanly with distillation later.

If the target is interactive closed-loop serving at batch, **skip it** and put
the effort into step distillation, which cuts the marginal term that batching
cannot amortise and carries no causality restriction.

The protocol upgrade in 8.2 is worth doing regardless — it currently cannot
certify any sampling change, including distillation.
