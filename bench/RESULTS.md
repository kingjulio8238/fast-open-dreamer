# Open Dreamer inference optimization — measured results

All numbers from `reactor-team/open-dreamer` @ `4f3ab34` (the released inference
code, imported as `pipeline`), on Modal H100 80GB HBM3, JAX 0.4.38, random-init
weights unless stated.

**This document was adversarially audited; 14 defects were found and fixed.**
Provenance conventions used throughout:

- *(log X)* — captured in `tasks/X.output`, machine-checkable.
- *(terminal)* — produced in the operator's shell and pasted into the working
  session, **not** captured in an artifact log. Real, but you take the
  transcript's word for it. Re-run to promote it to *(log)*.

## Headline

**44.35 → 26.83 ms/frame at batch 1: 1.65×, 22.5 → 37.3 fps.**
(Cross-instance; realistic range **1.55×–1.75×** — see the variance section.)

Both sides are the released sampling setting (4 denoising steps), so this is
like-for-like with **no change to model behaviour**. Stack: `no_remat`,
`fast_kv_write`, `no_roll_kv`, `bf16_weights`, `block_attn` with cuDNN
(log `b7xzmi618`).

At B=16 the same stack gives **221.25 ms/frame = 72.3 fps aggregate**. No
steps=4 B=16 baseline was measured, so no batch speedup is claimed here.

The patch-firing counters for that run were `42 split, 264 dense, 0 tag/shape
mismatch, 0 degenerate splits skipped`, which is exactly
9 encoder × 1 trace × 2 batches + 6 decoder × 2 traces × 2 batches = 42, and
22 dynamics × 6 traces × 2 batches = 264. Every tokenizer space layer took the
split; every dynamics layer stayed dense, as intended.

### A larger number was withdrawn

An earlier version of this document led with **2.73×** (16.22 ms), which
included cutting the sampler from 4 to 2 denoising steps. At n=2 seeds that cut
looked free — the FVD gap was 1.2× the seed spread, i.e. "not distinguishable".

**Re-run at n=6 seeds, it is not free.** Paired by seed:

| metric | steps=2 | steps=4 | paired t(5) | seeds favouring steps=4 |
|---|---:|---:|---:|---:|
| FVD | 545.7 | **514.4** | **2.97** (crit. 2.571) | **6 / 6** |
| std ratio | 0.9594 | **0.9648** | **−5.20** | **6 / 6** |
| motion/gt | 0.9087 | **0.9445** | −2.34 | 5 / 6 |

FVD is significant at p<0.05 and std ratio far beyond it; steps=4 wins on
**every seed** for two of three metrics. The n=2 result was not evidence of
equivalence, it was absence of power — exactly what "not proven different" is
supposed to warn about, and the warning was correct.

So **steps=2 is measurably worse and is no longer recommended.** 16.22 ms /
61.6 fps remains reproducible and is reported below for completeness, but it
buys 1.65× more speed for a real quality cost.

## Ablation ladder (steps=2, one run, one instance — log `bq8903z0q`)

| patch | B=1 ms | Δ | cum | fps | B=16 ms | fps agg |
|---|---:|---:|---:|---:|---:|---:|
| baseline (as released) | 29.02 | — | 1.00× | 34.5 | 170.79 | 93.7 |
| +`no_remat` | 28.80 | −0.22 | 1.01× | 34.7 | 170.29 | 94.0 |
| +`fast_kv_write` | 27.20 | −1.61 | 1.07× | 36.8 | 167.20 | 95.7 |
| +`no_roll_kv` | 26.99 | −0.20 | 1.08× | 37.0 | 165.32 | 96.8 |
| +`bf16_weights` | 17.51 | **−9.48** | 1.66× | 57.1 | 157.11 | 101.8 |
| +`block_attn(cudnn)` | **16.22** | −1.29 | **1.79×** | **61.6** | **136.76** | **117.0** |

At B=16: 170.79 → 136.76 ms, **93.7 → 117.0 fps aggregate (1.25×)**. The batch
gain is smaller because B=16 was already near 40% of peak, where weight
bandwidth is amortised over tokens and `bf16_weights` buys much less.

`bf16_weights` dominates. The trace explains why: with `param_dtype=float32` and
`dtype=bfloat16`, XLA emits a standalone convert pass over all 1.57 B parameters
rather than fusing the cast into the GEMM — **39.3% of GPU kernel time**,
dropping to **2.2%** under bf16 storage.

## Exactness

| patch | exact? | evidence |
|---|---|---|
| `no_remat` | yes | identity substitution for `nnx.remat` |
| `fast_kv_write` | yes | T==1 makes the wrap branch provably dead |
| `no_roll_kv` | yes | on-device, 40 writes across 2 ring wraps, max diff 2.38e-07 |
| `block_attn` | yes | on-device: decoder 2.29e-05, **encoder 6.88e-05** absolute (float32 reduction order) |
| `bf16_weights` | **no** | quality-gated below |
| `fp8_weights` | **no** | 0.0375 rel. error — but accuracy was never the blocker; it is also non-functional and 37% slower, see the fp8 section |

### Independent exactness audit

Two of three commissioned reviewers returned nothing, so the exactness brief was
executed directly. Results *(terminal)*:

**`no_roll_kv` — exhaustive, not sampled.** The released mask (roll + valid +
causal) was replayed against the patch formula (`age`/`causal`/`written`) for
**every** `(window, index, query_len)` state across 9 window sizes from 1 to 192,
covering warmup, `index == window` exactly, and multiple wraps. **0 mismatches.**

**`fast_kv_write` — the dead-branch claim is provable.** `write_idx + 1 > window`
occurs in 0 of 199,000 `(window, index)` states, since `write_idx = index mod W`
is at most `W-1`. The wrapping branch cannot execute at T=1.

**`_detect_block_split` — sound under brute force.** Every boolean mask up to
S=4 (66,064 masks): 12 accepted, **0** where the implied split differs from the
mask. At production S=1432, six adversarial families were correctly rejected —
3-block (dynamics with agent tokens), a 2-block with one bit flipped, a
split shifted by one row, banded/local-window, causal, and block-diagonal —
while the genuine encoder and decoder masks were accepted at p=512. The final
`np.array_equal(m, ref)` makes acceptance imply exact equivalence.

**`_attn_core_split` — op-by-op identical to the original.** Both reduce to the
same sequence: `to_q` → rearrange → `to_kv` → rearrange → scale → qk_norm
(*both* `qknorm` and `quest` paths present) → RoPE → SDPA → rearrange →
`to_out` → dropout. The only deviations are the two-call split and hardcoding
`start_pos=0`, which is what the original computes when `cache is None`. That
premise was verified three ways in the reference `models.py`: SpaceSelfAttention
constructs its GQA with `is_causal=False`; it passes `None` as the cache
positional arg; and `BlockCausalLayer` only ever hands caches to *time* layers
(`cache_i = ... if caches is not None and is_time_layer else None`).

**RoPE ordering** is applied to the full q/k before slicing, so the second block
keeps absolute positions p..S-1.

The on-device numbers (decoder 2.29e-05, encoder 6.88e-05 absolute in float32)
are float32 reduction order, not algebra: the numpy replay of the same split
agrees to **1e-12** in exact arithmetic.

## Tokenizer attention (`block_attn`) — log `bz495jr82`

Both tokenizer masks are 2-block around the latent/patch boundary, so the masked
attention splits into two **unmasked** calls, which makes it flash-eligible.

| stage | dense | block, XLA SDPA | block, cuDNN flash | cuDNN vs dense |
|---|---:|---:|---:|---:|
| encode (8 frames) B=1 | 42.61 ms | 33.72 | **20.29** | **2.10×** |
| decode B=1 | 3.36 | 3.02 | **2.13** | **1.58×** |
| encode (8 frames) B=8 | 322.49 | 259.15 | **154.64** | **2.09×** |
| decode B=8 | 18.13 | 15.45 | **8.06** | **2.25×** |

Removing the mask is necessary but not sufficient.
`jax.nn.dot_product_attention(implementation=None)` uses the XLA reference path,
which materializes the score matrix mask or no mask — the XLA-split column gains
only **1.11–1.26×**. cuDNN is what turns it into flash attention.

MFU note: the block_attn columns' derived TFLOP/s and MFU were computed with
`dec_flops()` still assuming a dense S×S score matrix, overstating them by
**3.0%**. `dec_flops` now takes a `block_split` flag. **Millisecond timings are
unaffected.**

**Memory.** Encoding 8 windows × 32 frames requests
`8*32 * 24 * 1432² * 4 = 50,396,135,424 B = 46.9 GiB` of fp32 scores. An OOM at
exactly that byte count was observed *(terminal)*; artifact logs here only print
the projected size. Flash attention removes the allocation entirely.

**Not applied to the dynamics.** Its `wm_agent` mask is 2-block but splits at
p=1 — the action token attends only to itself — so the split would peel off a
single-query attention: an extra launch plus a concatenate to save 1/290 of the
score entries. Skipped **a priori** by a `min_block=8` guard on that reasoning.
An earlier cross-run pair suggested a 0.7 ms regression, but those runs differed
in SDPA backend as well as in this flag, so it does not support the decision and
is not cited as evidence.

## Where the remaining time goes

`analyze_trace.py`, two traces:

| class | baseline (fp32, 5 fwd) | optimized (bf16, 3 fwd) |
|---|---:|---:|
| dtype convert | 80.51 ms — 39.3% | 1.76 ms — 2.2% |
| GEMM | 69.63 ms — 34.0% | 43.80 ms — **55.9%** |
| fusion/other | 31.16 ms — 15.2% | 21.09 ms — 26.9% |
| copy/transpose | 14.74 ms — 7.2% | 6.49 ms — 8.3% |
| reduce/norm | 5.48 ms — 2.7% | 4.43 ms — 5.7% |
| attention/softmax | 3.28 ms — 1.6% | 0.71 ms — 0.9% |

The kernel-name heuristic initially missed `nvjet_*` — cuBLAS 12.x's GEMM
family, which carries no `gemm` substring. **Within the optimized trace**,
correcting for it moved GEMM from an apparent 17.3% to the real 55.9%. (In the
baseline trace the same correction is 8.5% → 34.0%. Quoting 8.5% → 55.9% would
splice two traces and roughly double the apparent size of the fix.)

**Not launch-bound — but not saturated either.** An earlier version of this
document claimed ~98.8% occupancy. That was wrong, and the error was mine: the
statistic reported as "occupancy" dropped all idle in the 10 µs–1 ms band while
keeping the multi-millisecond inter-iteration gaps in the denominator. Corrected
(`busy / (wall − gaps>1 ms)`):

| trace | in-iteration occupancy | in-iteration idle | of which sub-10 µs |
|---|---:|---:|---:|
| baseline (fp32, 5 fwd) | **80.4%** | 19.6% | 1.9% |
| optimized (bf16, 3 fwd) | **93.6%** | 6.4% | 1.2% |

The **launch-bound conclusion survives** — sub-10 µs idle, which is what
per-kernel launch overhead can explain, is 1.2–1.9% in both. But the headroom is
6.4% on the optimized config, not ~1%, and it sits in 10 µs–1 ms gaps (sync
points, host work between dispatches), not in launch latency. The claim that the
raw idle "is 4 multi-millisecond host gaps" holds only for the optimized trace
(71.9% of its idle); in the baseline those 4 gaps are just 27.7%, with 180 gaps
of 0.1–1 ms and 602 of 10–100 µs sitting *inside* iterations.

Independently, every GPU kernel in the trace carries a `cuda_graph_id`, so XLA
already uses CUDA graphs internally. That observation is from inspecting the
trace directly, not from any harness code.

### Fusion attribution (`resolve_fusions.py`, HLO ↔ trace join)

**Config caveat:** this join is from log `b3uxndq1l`, which ran the XLA-reference
SDPA **and** had dynamics block_attn enabled — neither is in the shipped config.
Rows 1–2 (`patches.py:368`, 30.5% of the table) are the XLA SDPA arm and **do
not exist** under cuDNN. The `%` column is **share of resolved fusion time
(~44 ms)**, not share of GPU time — unlike the class table above.

| ms | % of fusion time | calls | source / op |
|---:|---:|---:|---|
| 7.85 | 17.8% | 385 | `patches.py:368` convert (XLA SDPA arm — not in shipped config) |
| 5.58 | 12.7% | 1360 | `patches.py:368` dot_general (SDPA einsums — same) |
| 5.23 | 11.9% | 305 | `models.py:402` dot_general (attention QKV projections) |
| 2.46 | 5.6% | 140 | `models.py:400` dot_general |
| 2.12 | 4.8% | 15 | `models.py:1336` concatenate (dynamics token assembly) |
| 2.08 | 4.7% | 440 | `models.py:332` dot_general (MLP `fc_out`) |
| 2.01 | 4.6% | 250 | `models.py:439` dot_general |
| 1.63 | 3.7% | 450 | `models.py:609` reduce_sum (RMSNorm) |
| 1.28 | 2.9% | 80 | `generation.py:185` while (the tau-ladder scan) |

**Actionable finding, observed not inferred:** `loop_pad_fusion_124` at
`models.py:332` has output shape `bf16[296,...]`, and the optimized HLO contains
58 instances of `bf16[296,15360]`. XLA pads the token axis **290 → 296** for the
GEMM. `S = 1 action + 1 shortcut + 256 spatial + 32 register = 290`; `n_register`
38 (→296) or 30 (→288) would remove the pad. The pad also appears in the
shipped-config traces, so it is not an artifact of the HLO run. Untested, and it
changes the architecture, so it belongs to a retrain rather than to serving.

## fp8 weights — REJECTED

Both operands quantized to e4m3, written in the shape XLA's GemmRewriter is
documented to fold into one cuBLASLt fp8 GEMM. Quantization worked: 120 of 156
Linear kernels converted, dynamics weights **3.15 GB → 1.60 GB (1.97×)**.

It is slower. **This is the only fully controlled A/B in this document** — both
arms from log `bf954nn59`, same container, same instance, same command but one
patch:

| stage | bf16 | fp8 | |
|---|---:|---:|---|
| dyn_fwd B=1 | 6.31 ms | 8.85 | **1.40× slower** |
| ladder B=1 | 14.85 | 22.28 | **1.50× slower** |
| frame B=1 | 17.67 | 24.20 | **1.37× slower** |
| dyn_fwd B=16 | 45.24 | 49.88 | 1.10× slower |
| ladder B=16 | 124.62 | 139.02 | 1.12× slower |
| frame B=16 | 140.58 | 154.70 | 1.10× slower |

**Zero fp8/e4m3 kernels were emitted** — confirmed two independent ways: no
kernel name in the trace contains `fp8`/`e4m3`/`e5m2`, and the in-container check
reports `{"fp8_total_ms": 0, "verdict": "NO fp8 kernels"}`. XLA did not fold the
pattern, so dequantization runs as ordinary ops before an unchanged bf16 GEMM.

| class | fp8 run |
|---|---:|
| fusion/other | 46.83 ms — 42.9% |
| GEMM | 42.23 ms — 38.7% |
| reduce/norm | 11.03 ms — **10.1%** |
| copy/transpose | 6.55 ms — 6.0% |
| attention/softmax | 1.51 ms — 1.4% |
| dtype convert | 1.01 ms — 0.9% |

The largest **non-GEMM** kernel is `loop_multiply_fusion_1` at 15.95 ms over 440
calls (the largest kernel overall is an `nvjet_*` GEMM at 16.05 ms). The full
multiply set is 32.98 ms = **30.3% of GPU time**.

Mechanism, **inferred and not directly traced**: the multiply fusions are
`kernel.astype(bf16) * scale`, and their cost is consistent with XLA
materializing the full dequantized bf16 weight matrix — reading 1 byte per
parameter and writing 2, after which the GEMM reads those 2 bytes back. Net
weight traffic rises, matching achieved bandwidth falling to 296 GB/s despite
half-size weights. No HLO dump was taken for the fp8 config; `dump_hlo` on it
would settle this cheaply.

Convert kernels are only 0.9%, so an earlier hypothesis that this reintroduced
the `bf16_weights` convert problem was wrong — the mechanism is the dequantize
*multiply*, not the dtype convert.

To make fp8 pay, the fused cuBLASLt kernel must actually appear in the trace.
That likely means a supported fp8 path (flax's fp8 ops with delayed scaling)
rather than hand-writing the pattern, plus statically calibrated activation
scales to remove the amax reduction from the hot path.

Accuracy if revisited: e4m3 gives **0.0375** relative error on a representative
dynamics GEMM vs **0.0029** for bf16 — 12.9× worse, needing its own FVD gate.

## Quality gate for steps 4 → 2 — n=6 seeds, log `bxjtjjqtm`

Long-horizon rollout, 8 windows × 96 frames, **6 seeds**, FVD vs
ground-truth-decoded, paired by seed:

| metric | steps=2 | steps=4 | mean paired diff | paired t(5) | seeds favouring 4 |
|---|---:|---:|---:|---:|---:|
| FVD | 545.7 | **514.4** | +31.3 | **2.97** | **6/6** |
| motion/gt | 0.9087 | **0.9445** | −0.036 | −2.34 | 5/6 |
| std ratio | 0.9594 | **0.9648** | −0.0055 | **−5.20** | **6/6** |

Two-sided critical t at α=0.05, df=5 is 2.571. FVD clears it; std ratio clears
it by a wide margin; motion falls just short but agrees in direction. Per-seed
FVD differences (steps2 − steps4): +6.0, +49.7, +68.4, +24.7, +1.4, +37.7 —
**every one positive.**

**Conclusion: steps=4 is significantly better than steps=2.** The earlier n=2
study reported "1.2× the noise floor, not distinguishable"; that was correct as
stated and wrong as a basis for action. n=2 gives 1 degree of freedom and had
essentially no power to detect a 31-unit gap. Adding four seeds cost one GPU run
and reversed the recommendation.

**steps=1 remains clearly rejected** — it was 4.9–5.4× the noise floor at n=2 on
every metric and visibly loses texture over ~90 frames.

Caveats that still stand: 48 clips per arm (8 × 96 ÷ 16) against the repo's own
`eval_fvd`, which uses 256 videos, and all windows come from **one** VPT
episode. Absolute FVD is not comparable to published numbers; only the paired
comparison is. A larger, multi-episode study could move the effect size, though
6/6 sign agreement makes a direction flip unlikely.

A teacher-forced one-step study (steps 1/2/4/8, 8 trials) found no step-count
effect, but it is *(terminal)* and superseded: one-step accuracy cannot see
compounding, which is precisely where the difference lives.

## Test quality — a note on what "PASS" is worth

Two gates in this repo passed while testing nothing, and both were caught only
by deliberately trying to break them:

1. `check_block_attn_equivalence` originally inlined its own split and never
   called `_attn_core_split`. It passed on code that was not the patch.
2. `check_kv_equivalence` drove both arms from a single cache already written
   by the patched `update`, so a wrong write index corrupted both sides
   identically. `fast_kv_write` had no working gate at all.

A third instance appeared while closing those out. The new dispatch test set
`sa.attn._block_spec`, but `_space_attn_call` reads `self._block_spec` — and
`sa.attn` is the inner `GroupedQueryAttention`, not the `SpaceSelfAttention`
the tag belongs on. Both dispatch cases silently exercised the mask-detection
fallback instead of the tag, and both produced correct output, so the test
passed. It was caught only because the stale-tag case asserts on the
`shape_mismatch` **counter** rather than on the output: the fallback derives the
same split from the same mask, so output equality cannot distinguish "the guard
rejected the stale tag" from "the guard never ran".

A fourth followed immediately, from the fix itself: the corrected dispatch test
deliberately provokes a stale tag, which left `shape_mismatch = 1` in the
process-global counters after the gate returned. That counter is read by the
run's own no-op check *and* by the decision of whether to charge split or dense
FLOPs — so a test asserting the guard works would have silently disabled the
FLOP correction for every subsequent measurement. The gate now snapshots and
restores the counters.

The general lesson, which cost four separate incidents here: **a passing test
that produces the right answer through the wrong path is indistinguishable from
a working one unless it asserts on the path — and a test that asserts on shared
state must put that state back.**

## Guard against the silent no-op

A monkeypatch that quietly fails to apply produces unpatched numbers under a
patched label. That happened here: `block_attn` originally sat at the
`GroupedQueryAttention` level, where the mask arrives as a *tracer* inside
`nnx.remat`, so value-based detection failed and it no-opped on all 34 attention
calls while reporting plausible timings.

The harness now refuses to report a run where `block_attn` never fired
(`SystemExit(2)`), prints how many calls took each path, and records a
`shape_mismatch` counter for tags that do not match the runtime token count.

Reading the counters correctly — and this is worth stating precisely, because
an earlier version of this section said "every block_attn result logs
`N split, 0 dense`", which is true only for `bench_attn` (it skips the dynamics
stages entirely) and misleading everywhere else. A reviewer reasoned from that
sentence to the conclusion that the `min_block` guard must be bypassed, since
`dense == 0` with 22 untagged dynamics layers present would have no other
explanation. The logic was right; the sentence was wrong.

The counters are arithmetically checkable, which makes them the strongest
no-op evidence available:

| run | logged | predicted |
|---|---|---|
| ablation, steps=2 | `24 split, 220 dense` | decoder 6 × 2 traces × 2 batches = 24; dynamics 22 × 5 × 2 = 220 |
| full stack, steps=4 | `42 split, 264 dense` | encoder 9 × 1 × 2 + decoder 6 × 2 × 2 = 42; dynamics 22 × 6 × 2 = 264 |
| attention A/B | `15 split, 0 dense` | encoder 9 + decoder 6 = 15; dynamics never runs |

Every figure matches a count derived independently from the layer structure.
The `dense` column is the **untagged dynamics staying dense by design**, not
failures — under a guard bypass those calls would appear in `split` instead.

## Variance — two kinds, do not conflate them

**Within an instance.** Every ablation arm ran sequentially in one container on
one GPU, so instance variance does not apply to the ladder. The floor there is
the p10–p90 spread the harness reports per stage:

| step | Δ | spread | verdict |
|---|---:|---:|---|
| +`no_remat` | −0.22 | 1.49 | **within noise** |
| +`fast_kv_write` | −1.61 | 0.71 | above noise |
| +`no_roll_kv` | −0.20 | 0.57 | **within noise** |
| +`bf16_weights` | −9.48 | 1.47 | above noise |
| +`block_attn(cudnn)` | −1.29 | 0.64 | above noise |

Three of five steps are individually resolvable; two are not. `no_remat` and
`no_roll_kv` are kept because they are **exact and free**, not because their
deltas are trustworthy — each could be zero. Their combined −0.42 ms is ~2.5% of
the final frame; the headline does not depend on them.

**Across instances.** Measured GEMM/HBM ceilings differ per instance: 742.5
TFLOP/s / 2353 GB/s on the benchmarking instance *(terminal)*, and 819.8 / 2890
on another *(log `b5yd5jzr3`)*. Any comparison spanning two runs inherits this,
Any comparison spanning two runs inherits this — and the **1.65× headline
does**: 44.35 ms is log `bxnc23q70` and 26.83 ms is log `b7xzmi618`, separate
Modal apps and therefore separate instances.

The best direct evidence for the size of that effect is two measurements of an
*identical* configuration landing at 16.22 and 17.67 ms (the ablation's final
arm and `bench_fp8`'s bf16 control) — **9.0% apart**. Propagating that, the
1.65× headline has a realistic range of roughly **1.55×–1.75×**. The
within-run ablation deltas are not affected, since every arm shares a container.

**MFU caveat.** All MFU percentages here use 742.5 TFLOP/s. The harness prints
these as "peak (assumed)" for good reason — it is a measured ceiling *for one
instance*, not a constant. Against 819.8 every MFU figure would be ~1.10× lower
(decode B=8 52.8% → 47.8%). **Treat MFU as indicative and the millisecond
timings as the result.**

## Reproducing

```bash
modal run bench/modal_bench.py::gpu_probe            # re-measure ceilings on YOUR instance
modal run bench/modal_bench.py::ablate --batch "1 16" --steps 2
modal run bench/modal_bench.py::bench_attn --batch "1 8"
modal run bench/modal_bench.py::bench_fp8 --batch "1 16" --steps 2
modal run bench/modal_bench.py::download_checkpoint  # 7.9 GB, public, ungated
modal run bench/modal_bench.py::quality_rollout --steps "1 2 4" --seeds "0 1"
modal run bench/modal_bench.py::quality_steps --steps "1 2 4 8" --trials 8
modal run bench/modal_bench.py::dump_hlo
modal volume get open-dreamer-results / ./bench/results
python bench/analyze_trace.py bench/results/trace/<run>
python bench/resolve_fusions.py bench/results/hlo bench/results/th
```

## Known gaps

- `quality_steps` and the numpy exactness replay have no preserved artifacts.
- No HLO dump for the fp8 config, so its mechanism is inferred.
- The fusion attribution is from a non-shipped configuration.
- FVD n=2 seeds, 48 clips — underpowered for the steps 2 vs 4 question.
- Two of three commissioned reviewers returned no findings; the exactness brief
  was executed directly instead (see the audit above). It is self-review, not
  independent review — weaker evidence than an outside attack would have been.
- `no_remat` exactness rests on the argument that `jax.checkpoint` is
  semantically the identity in a forward-only computation. Not separately tested.

---

# Tier 1′ — batching across streams

## The blocker, and what it costs to remove

`KVCache.index` is a **scalar** shared by the whole batch (`models.py:43`,
`index  # scalar integer (i32)`). Every row of a batch must therefore sit at the
same rollout step, so a served batch can only be formed from sessions that
started together and never diverge. That is the definition of static batching,
and it is what makes continuous batching impossible rather than merely awkward.

`ragged_kv` in `bench/patches.py` makes the index shape `(B,)`. Three things
have to change together, and missing any one of them is silently wrong rather
than loud:

1. **`update`** — a scalar `dynamic_update_slice` becomes a `vmap` of per-row
   writes at `index % window_size`.
2. **`get_ordered_kv`** — the causal/written mask has to be built per row, from
   `idx.reshape(-1, 1, 1, 1)` rather than a scalar.
3. **RoPE `start_pos`** — this is the trap. The released code is
   `jnp.outer(jnp.arange(T) + start_pos, inv_freq)`, and `jnp.outer`
   **flattens its arguments**. Passing a `(B,)` `start_pos` does not raise; it
   silently produces a `(B*T, dim)` table and every row gets the wrong angle.
   The patched version branches on `sp.ndim` and builds `(B, T)` explicitly.

## The measurement that decides how much this is worth

Aggregate throughput, one `frame` step, recommended stack, H100 (log
`bzce85sia`). B=32 OOMs in prefill (20.7 GB single allocation).

|  B | frame ms | single-stream fps | aggregate fps | MFU |
|---:|---------:|------------------:|--------------:|----:|
|  1 |   25.716 |              38.9 |          38.9 | 26.3% |
|  2 |   39.437 |              25.4 |          50.7 | 34.3% |
|  4 |   65.482 |              15.3 |          61.1 | 41.3% |
|  8 |  116.586 |               8.6 |          68.6 | 46.4% |
| 16 |  220.085 |               4.5 |          72.7 | 49.2% |

The curve is almost exactly affine — a two-parameter fit lands within 2% at
every point, and within 0.1% for B>=4:

    t(B) = 13.33 ms  +  12.93 ms x B

That decomposition is the whole story of this model's inference:

- **13.33 ms fixed.** Work done once per step no matter how many streams share
  it: streaming 3.147 GB of bf16 weights through five dynamics forwards, 15.7
  GB/frame, ~1180 GB/s or 50% of the measured 2353 GB/s ceiling. At B=1 this is
  **52% of the frame**.
- **12.93 ms/stream marginal.** 5.02 TFLOP of real per-stream compute at 388
  TFLOP/s, 52% MFU.

So the batching ceiling is `1000/12.93` = **77.4 fps aggregate, 1.99x over
B=1** — and B=8 already captures 89% of it, B=16 94%. There is no configuration
in which batching is worth more than 2x, because the fixed half is all it can
amortise.

**Batching buys throughput and spends latency.** Single-stream frame time goes
from 25.7 ms to 220 ms. For a world model driven by a human or a policy in the
loop, that is the wrong trade past about B=4.

---

# Tier 2 — fp8/int8 W8A16 GEMM with in-register dequant — REJECTED

## Why a custom kernel at all

The earlier XLA-level attempt (`fp8_weights` in `bench/patches.py`) wrote
`dot(convert(w_fp8) * scale, x)` and XLA materialised the dequantised bf16
matrix in HBM *before* the GEMM. Traffic went up, `loop_multiply_fusion_*`
became 30% of GPU time, zero fp8 kernels were emitted, and the frame got 1.37x
**slower**. A Pallas kernel is the only way to keep the dequant in registers.

`bench/fp8_gemm.py` implements it. Two dead ends before a working kernel:
`float8_e4m3fn -> bf16` and `-> f32 -> bf16` both hit
`LLVM ERROR: Unsupported rounding mode for conversion` in this JAX's Triton.
Switched to **int8 with per-output-channel symmetric scales** — identical 1
byte/weight (the bottleneck is weight *bytes*, not weight *precision*), a
universally supported conversion, and more accurate for weights than
per-tensor e4m3. Measured rel err 0.0071 across every shape, consistent with
int8 quantisation and stable, so the kernel is numerically sound.

## A prediction stated in advance, and falsified

The file predicted fp8 would help `fc_in`/`fc_out` (28-43% of peak bandwidth,
84% of parameters) and do little for `to_q`/`to_kv` (1-6%, latency-bound).
The first measurement at bm=16/bn=64/bk=64 inverted it exactly:

| shape | M=296 | M=4736 |
|---|---:|---:|
| to_q  | **1.04x** | 0.20x |
| to_kv | 0.78x | 0.44x |
| fc_in | 0.25x | **0.14x** |
| fc_out| 0.44x | 0.19x |

`fc_in`, predicted to gain most, was worst. So the cost was not weight bytes —
it was the tile. Two concrete causes: bm=16 is below the m=64 that H100 `wgmma`
issues at, and an `(m, n)` grid with full-K blocks re-walks the weight matrix
once per m-block (19x at M=296, 296x at M=4736) where cuBLAS walks it once.

## The SRAM budget that justified bm=16 was imaginary

The docstring argued bm=16 from a 228 KB/SM budget. That was wrong: in the
Triton backend a `BlockSpec` is a **block pointer**, and `a_ref[:, sl]` lowers
to a load of just that slice, so the full-K block is never materialised. The
sweep proves it — bm=128 with K=1920 (a nominal 480 KB "block") compiles and
runs fine. bm=16 was self-imposed for no reason.

Sweeping tiles at M=296 (log `b9dtdqdri`), best per shape:

| shape | cuBLAS | best Pallas tile | best time | speedup |
|---|---:|---|---:|---:|
| to_q  1920x1920  | 58.3 us | bm=64 bn=64 bk=64   | 48.6 us | **1.20x** |
| fc_out 7680x1920 | 72.9 us | bm=64 bn=128 bk=128 | 133.6 us | 0.55x |
| fc_in 1920x15360 (M=4736) | 520.8 us | bm=128 bn=128 bk=64 | 714.1 us | 0.70x |

Better tiling moved `fc_in` from 0.14x to 0.70x and `fc_out` from 0.44x to
0.55x — a 3-5x improvement over my first kernel, and still a loss. `bm=320`,
the one tile that would cover M=296 in a single m-block and remove the re-read
entirely, fails to lower at all.

## Why this is rejected rather than iterated

Two independent reasons, either sufficient.

**The prize is capped, and small.** The Tier 1′ fit says the frame is
`13.33 ms fixed + 12.93 ms x B`, where the fixed part *is* the weight stream.
A **perfect** W8 kernel halves only that:

| B | frame | fixed share | perfect-W8 speedup |
|--:|------:|------------:|-------------------:|
| 1 | 26.3 ms | 51% | 1.34x |
| 4 | 65.0 ms | 20% | 1.11x |
| 16 | 220.2 ms | 6% | **1.03x** |

Batching and quantisation attack the **same 13.33 ms**, so they are
substitutes, not additives. Anyone who batches has already collected what fp8
was going to pay, and 1.03x does not justify a hand-written GEMM.

**The remaining gap is a research-grade kernel.** Closing 0.55x -> 1.0x on
`fc_out` means matching cuBLAS `nvjet_*` with async copies, warp specialisation,
software pipelining and swizzled layouts — the Marlin/CUTLASS feature set,
which Pallas-on-Triton at this JAX version does not readily express. That is
weeks of work for a ceiling of 1.34x in the one regime (B=1) where it helps.

The single measured win, `to_q` at 1.20x, does not survive scrutiny either:
cuBLAS's own time for that shape moved 83.4 -> 58.3 us between two runs, so at
these sizes run-to-run variance is comparable to the effect. `to_q` is also
7.4 MB of 3.147 GB of weights.

**Rejected on evidence, same as the XLA fp8 attempt.** The kernel, the sweep
and the numbers are kept in `bench/fp8_gemm.py` so the result is reproducible
and the negative is auditable.

## What ragged indices cost, and what they buy

**Cost.** Same-container A/B at B=8, recommended stack, `decode`/`prefill`/
`encode` skipped (log `b2fyjwe06`):

| arm | dyn_fwd | ladder | frame | aggregate |
|---|---:|---:|---:|---:|
| A `fast_kv_write` + `no_roll_kv` (scalar index) | 24.299 ms | 107.877 ms | 116.733 ms | 68.5 fps |
| B `ragged_kv` (per-row index) | 23.650 ms | 114.490 ms | 122.915 ms | 65.1 fps |

**5.3% slower.** All of it is in `ladder` (+6.1%) — the vmap'd per-row
`dynamic_update_slice` — while `dyn_fwd` is marginally *faster*. Arm A
reproduces the independent batch sweep to 0.13% (116.733 vs 116.586 ms), so
this is a real 5.3% and not run-to-run noise.

**Buy.** `bench/scheduler.py` implements slot admission/eviction over the
ragged cache and compares it against the static policy a scalar index forces,
under Poisson arrivals with lognormal session lengths (CV 0.6), driven by the
measured `t(B)` above. Continuous batching is charged the full 5.3% penalty;
static is not.

The mechanism is padding: a rollout compiled at width B computes idle slots
rather than skipping them, so a step costs `t(B)` whether 8 slots are busy or
1, and `fps = occupancy x B / t(B)`. Occupancy is then purely a scheduling
property. At 85% offered load:

| width | policy | occupancy | fps | p95 wait |
|---:|---|---:|---:|---:|
| 4 | static | 62.7% | 38.2 | 135.6 s |
| 4 | continuous | 98.5% | 57.1 | 21.8 s |
| 8 | static | 51.2% | 34.8 | 141.4 s |
| 8 | continuous | 92.1% | **59.4** | 10.4 s |
| 16 | static | 44.2% | 31.9 | 156.1 s |
| 16 | continuous | 85.8% | 58.7 | **7.0 s** |

Static batching does best at *small* width (cohorts drain sooner), so the fair
comparison is best-config against best-config. Over 5 seeds:

    best static 38.3-39.6 fps   best continuous 59.1-66.9 fps   1.59x mean
    (min 1.51x, max 1.75x, 5/5 seeds favour continuous, ranges disjoint)

So `ragged_kv` gives up 5.3% of peak to recover far more than that in
occupancy, and cuts p95 admission wait by ~20x. The 1.59x is a **scheduling**
result, not a kernel one, and it is bounded above by the 1.99x batching ceiling
— which is why it is worth doing and also why it is the last large win
available on this axis without changing the model.

### The one thing to check before trusting the 1.59x

The simulation's cost model is measured, but the *policy* comparison is
simulated, not run on a GPU. Its load-bearing assumption is that session
lengths vary (`--length-cv 0.6`). At `--length-cv 0` the gain collapses to
1.00-1.19x, because a uniform cohort drains all at once and static batching
loses nothing. If served sessions really are fixed-length and synchronised,
static batching is fine and `ragged_kv` is a 5.3% loss for nothing.

---

# How much is left below the GEMMs? Bounded by deleting the ops.

GEMMs are 55.9% of GPU time and cuBLAS already beats anything hand-written
here (see the Tier 2 rejection). That leaves 44.1% — fusion/other 26.9%,
copy/transpose 8.3%, reduce/norm 5.7% — which no work so far has touched, and
which is where fused kernels normally win at low batch.

Rather than write a fused kernel and then discover it was worth 2% (which is
how the fp8 effort was spent), the prize was bounded first by **deleting the
ops**. `no_norm` and `no_swiglu_gate` in `bench/patches.py` are deliberately
WRONG — they remove real computation — so the frame times under them are
speed-of-light numbers no kernel can beat.

B=1, recommended stack, one container (log `blaepzd4f`):

| arm | frame B=1 | vs control | what it bounds |
|---|---:|---:|---|
| control | 26.341 ms | — | |
| `norm_bf16` (real candidate) | 26.454 ms | **+0.4%** | fp32→bf16 norm compute |
| `no_norm` (delete every RMSNorm) | 24.119 ms | **−8.4%** | *every* norm kernel |
| `no_swiglu_gate` (delete `u*silu(v)`) | 25.155 ms | **−4.5%** | a fused SwiGLU |
| both | 24.491 ms | −7.0% | the two combined |

The combined arm being *slower* than `no_norm` alone puts the noise floor at
~1.5%, so read these as ceilings of ~8% / ~4% / ~8%, not as additive.

**Two conclusions, both negative and both cheap to have obtained.**

`norm_bf16` was a real hypothesis and it is dead. Every RMSNorm is built with a
hardcoded `dtype=jnp.float32` (`models.py:305, 373, 374, 563`) while
activations are bf16 — structurally the same defect as the
`param_dtype=float32`/`dtype=bfloat16` mismatch that `bf16_weights` fixes for
−9.48 ms. Retyping all 200 norm layers (120 dynamics + 80 tokenizer, counter
confirmed) changes **nothing**. XLA was evidently already handling the
reduction well regardless of the declared dtype. One arm, hypothesis falsified,
no kernel written.

**Every fused kernel that could be written for this model is bounded at ~8%,
and realistically captures about half of that.** Epilogue-fusing norm into the
preceding GEMM, a Pallas norm+residual kernel, a fused SwiGLU, a persistent
megakernel — none can beat deleting the op, and deleting *all* of them is 8%.
Against 2.23x from removing three of five forwards, kernel work is not where
the value is. **This is the measured answer to "how far down does it pay to
go": not very.**

## Hardware axis — blocked on the toolchain, not on the hardware

The frame splits into a bandwidth-bound half and a compute-bound half, so the
device is a first-class variable. `GPU_TYPE` is now `BENCH_GPU`-overridable.
Holding the *achieved* efficiency measured on H100 (50% of HBM, 52% of MFU):

| device | HBM GB/s | bf16 TFLOP/s | fixed | marginal | B=1 ms | B=1 fps | vs H100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H100 | 2353 | 742 | 13.34 | 13.00 | 26.35 | 38.0 | 1.00x |
| H200 | 4800 | 990 | 6.54 | 9.75 | 16.29 | 61.4 | 1.61x |
| B200 | 8000 | 2250 | 3.92 | 4.29 | 8.22 | 121.7 | **3.20x** |

**Optimistic bounds, not predictions** — M=290 is harder to saturate on a
bigger machine, so achieved efficiency will fall.

Measuring it is blocked: a B200 provisions fine (183359 MiB, log `bqinilwxu`)
but every run aborts with `Unsupported conversion from bf16 to f16 / LLVM
ERROR: Unsupported rounding mode for conversion`, with and without the cuDNN
attention path. The image pins **jax 0.4.38 / jaxlib 0.4.38** (Dec 2024),
which predates Blackwell support. This is a bounded image rebuild (jax >= 0.5,
CUDA 12.8+), not a property of the model — and it would also let this fork's
own `dreamer/parallel.py` import, which currently needs a `jax.sharding.AxisType`
that does not exist in 0.4.38.

## ROI, measured rather than asserted

| lever | gain | status | layer |
|---|---:|---|---|
| distillation, 5 → 2 forwards | **2.23x** | not started | algorithm |
| B200 | up to 3.20x | **blocked**, toolchain | hardware |
| tensor parallel, TP=2 | ~1.7x (at 2x the GPUs) | untested | deployment |
| continuous batching | 1.59x under real arrivals | **built + verified** | scheduling |
| every fusion kernel, combined | **<=8%**, ~4% realistic | **measured ceiling** | kernels |
| RMSNorm fp32 → bf16 | **0%** | measured, dead | kernels |
| fp8/int8 GEMM | negative | measured twice, dead | kernels |

The value is at the top of the stack, not the bottom. Going "all the way down
to the silicon" is exactly where the least is left, and that is now a
measurement rather than an opinion.

---

# Profiling the SHIPPED config — and the largest win since `bf16_weights`

Every earlier trace in this document was of something other than the shipped
configuration: the kernel-class table came from a steps=2 run, and the fusion
attribution came from log `b3uxndq1l`, which had the XLA SDPA arm *and*
dynamics block_attn, neither of which ships. Log `bi9diven2` is the first trace
of what is actually served — steps=4, bf16 weights, cuDNN block_attn, B=1, H100.

    frame B=1  26.136 ms   25.9% MFU
    in-iteration occupancy 95.6%   (4.4% idle, 1.5% of it sub-10us)
    18165 kernels, mean 6.26 us
    memory: peak 11.77 GB, in use 4.75 GB, largest single alloc 0.27 GB
    dynamics KV cache resident 0.342 GB (window 192, 8 time layers)

| class | ms | % |
|---|---:|---:|
| GEMM | 66.90 | 58.8% |
| fusion/other | 26.02 | 22.9% |
| copy/transpose | 9.20 | 8.1% |
| reduce/norm | 6.98 | 6.1% |
| dtype convert | 2.71 | 2.4% |
| attention/softmax | 1.95 | 1.7% |

## What the fusion→source join found

| ms | % of resolved | calls | source / op |
|---:|---:|---:|---|
| 10.21 | 16.7% | 605 | `models.py:402` dot_general |
| **10.18** | **16.6%** | 750 | **`models.py:439` convert_element_type** |
| **9.57** | **15.6%** | 2045 | **`models.py:439` dot_general** |
| 3.47 | 5.7% | 25 | `models.py:1336` concatenate (token assembly) |
| 3.17 | 5.2% | 740 | `models.py:332` dot_general |
| 2.62 | 4.3% | 750 | `models.py:609` reduce_sum |
| 1.26 | 2.0% | 545 | `models.py:439` transpose |

`models.py:439` is `jax.nn.dot_product_attention`, and it totals **21.0 ms of
113.5 ms GPU-busy — 18.5% of all GPU time**, with `gemm_fusion_dot_832_0`
carrying shape `f32[290,...]`: the score matrix is being materialised in fp32.

The cause was already written down in this document and not acted on:
**`implementation=None` does not mean "pick the best backend", it means the XLA
reference path.** `block_attn` never fixed this for dynamics — it reports
"tagged 15 tokenizer + **0 dynamics**", because the dynamics space mask is a
`(1,1,290,290)` tracer whose 1/290 split is degenerate. So all 30 layers x 5
forwards of dynamics attention ran the reference path, while the
`attention/softmax` class showed only 1.95 ms — that 1.95 ms was the
*tokenizer's* real cuDNN calls, and it made the dynamics cost invisible under
"GEMM" and "fusion".

## `sdpa_cudnn` — measured

Same container, one patch (log `bl9krd2pl`). 314 call sites took cuDNN, 0 were
rejected; the profiler hard-fails if that count is ever 0, so an all-fallback
run cannot be misreported as applied.

| stage | control | `sdpa_cudnn` | |
|---|---:|---:|---|
| frame B=1 | 27.053 ms | **24.005 ms** | 1.13x |
| dyn_fwd B=8 | 24.410 ms | 18.396 ms | 1.33x |
| ladder B=8 | 109.607 ms | 88.963 ms | 1.23x |
| frame B=8 | 116.474 ms | **95.082 ms** | **1.22x** |
| frame B=8 MFU | 46.5% | **57.0%** | |

**Correctness.** Acceptance is not correctness — a backend that ignored the
mask would also accept and would also be fast. `check_sdpa_cudnn_equivalence`
tests both dynamics mask regimes against the reference the released code was
getting by default, plus a negative control:

    space mask         rel 5.49e-03
    causal + window    rel 5.41e-03
    mask-actually-applied separation 2.75e+00   (vs tol 3e-2)

The first two are the same order as `block_attn`'s 5.10e-03 and are bf16
accumulation-order differences. The third is the load-bearing one: masked and
unmasked cuDNN outputs differ by 90x the tolerance, so the mask is genuinely
being applied.

## The roofline moved

Re-measured batch curve (log `bi2a2sns6`), still affine:

| | fixed | marginal | ceiling |
|---|---:|---:|---:|
| before | 13.33 ms | 12.93 ms | 77.4 fps |
| **after** | 12.88 ms | **10.21 ms** | **97.9 fps** |

The fixed term barely moves (−3%) and the **marginal term drops 21%**. That
matters more than the headline: the marginal term is per-stream compute, the
one batching cannot amortise and quantisation cannot touch. Every previous win
in this document attacked the fixed half.

| B | frame | agg fps | real-time streams/H100 |
|---:|---:|---:|---:|
| 1 | 24.278 ms | 41.2 | 2.1 |
| 2 | 33.794 | 59.2 | 3.0 |
| 4 | 52.718 | 75.9 | 3.8 |
| 8 | 93.079 | 85.9 | 4.3 |
| 16 | 177.190 | **90.3** | **4.5** |

Peak memory at B=16 is 23.38 GB of 80.77, largest single alloc 0.68 GB.

**New headline: 44.35 -> 24.278 ms at B=1 = 1.83x, 22.5 -> 41.2 fps. At B=16,
90.3 fps aggregate = 4.00x the released baseline.**

## What this says about the earlier "kernels are dry" conclusion

That conclusion was measured and it was also scoped too narrowly. The `no_norm`
/ `no_swiglu_gate` ablation correctly bounded *fusion* work at ~8%, and that
still holds. But it bounded the wrong thing: the biggest remaining cost was not
a fusion that needed writing, it was **a backend that was never selected**. The
ablation could not have found it, because deleting norms says nothing about
which SDPA kernel runs.

The lesson for what remains: prefer profiling the *shipped* configuration and
attributing time to source lines over reasoning about kernel classes in the
abstract. The kernel-class table said "GEMM 58.8%, attention 1.7%" and that was
true and completely misleading.

---

# Is it 4x, and is it the same quality? No, and not proven.

Both halves of the headline needed checking and both came back qualified.

## The 4.00x was not an apples-to-apples comparison

It compared the **optimized engine at B=16** (90.3 fps aggregate) against the
**released baseline at B=1** (22.3 fps). Those are different quantities:
throughput-under-batching versus single-stream latency. The released code has a
batch dimension and batches perfectly well, so crediting the optimization stack
with the batching gain is double counting.

This document already warned about exactly this — "no steps=4 B=16 baseline was
measured, so no batch speedup is claimed here" — and then the claim was made
anyway. The missing measurement has now been taken (log `bp7js0475`):

| config | baseline | optimized | like-for-like |
|---|---:|---:|---:|
| B=1 | 44.763 ms (22.3 fps) | 24.278 ms (41.2 fps) | **1.84x** |
| B=16 | 263.348 ms (60.8 fps agg) | 177.190 ms (90.3 fps agg) | **1.49x** |

**The defensible engine speedup is 1.84x at B=1 and 1.49x at B=16.** The
released code already reaches 60.8 fps aggregate at B=16 on its own, and the
gain shrinks with batch because `bf16_weights` — the largest single patch —
attacks the fixed weight-streaming term that batching already amortises.

"4x" survives only as a *deployment* statement — one H100 serving 90.3 fps
aggregate against a naive single-stream 22.3 — and it should be labelled as
such, never as the engine's speedup.

## Quality: measured for the first time, and not clean

Every speed benchmark in this document runs on **randomly-initialised weights**
(`profile_inference.py` says so in its docstring), and every prior FVD run
varied only `--steps`. The optimization stack had therefore never been put in
front of a real checkpoint. `quality_rollout.py` now takes `--patches`, so it
can be.

Real checkpoint, steps=4, 8 windows, 96-frame autoregressive rollout, n=6
paired seeds, baseline vs the full stack (logs `bp7js0475`, `bhks0ndyh`):

| metric | baseline | optimized | delta | paired t(5) | seeds worse |
|---|---:|---:|---:|---:|---:|
| **FVD** (lower better) | 514.38 | 528.52 | **+14.14** | **2.28** | **5 / 6** |
| std ratio (higher better) | 0.9648 | 0.9653 | +0.0005 | 0.13 | 2 / 6 |
| motion vs gt | 0.9433 | 0.9483 | +0.0050 | 0.27 | 2 / 6 |
| drift at horizon | 0.01066 | 0.01070 | +0.00004 | 0.19 | 3 / 6 |

Critical |t(5)| at p<0.05 is 2.571, so **FVD does not reach significance — but
it is close (t=2.28, p~0.07) and 5 of 6 seeds move the same way.** That is not
a clean bill of health. It is the same signature as the steps=2 question, where
n=2 looked fine and n=6 showed a real regression; the honest reading here is
"suggestive of a small real cost, underpowered to confirm", and by the effect
size (Cohen's d 0.93) it would take roughly n=12 to settle.

**The good news is specific and it is the thing that was actually at risk.**
Drift at horizon is identical to four decimal places. That was the real worry:
a ~5e-03 relative difference per attention call, compounded through 30 layers x
5 forwards x 96 autoregressive frames where each output becomes the next
input, could plausibly have diverged. It does not. Nor does variance collapse
(std ratio) or motion.

## Which patch is responsible

Three of the six are exact and cannot be:

    no_remat        identity in a forward-only computation
    fast_kv_write   verified write max|diff| 0.00e+00
    no_roll_kv      verified read max|diff| 2.38e-07

The candidates are `bf16_weights` (weight rounding, never quality-gated —
flagged as "verify against a real checkpoint before shipping" from the start
and this is the first time that was done), and `block_attn` / `sdpa_cudnn`
(both ~5e-03 relative, bf16 accumulation order).

**Untested and the obvious next step:** rerun the n=6 gate with `bf16_weights`
dropped. If FVD returns to baseline, the stack splits cleanly into an
exact-plus-cuDNN tier that is free, and a `bf16_weights` tier that trades ~2.7%
FVD for the largest single speedup. That is a decision worth making explicitly
rather than by default.

## Corrected summary

| claim | status |
|---|---|
| 1.84x at B=1, like-for-like | **measured** |
| 1.49x at B=16, like-for-like | **measured** |
| 4.00x engine speedup | **withdrawn** — conflated batching with the stack |
| 90.3 fps aggregate on one H100 | measured, deployment framing only |
| no long-horizon drift from the stack | **measured**, n=6, the main risk cleared |
| identical quality | **not established** — FVD +14.1, 5/6 seeds, p~0.07 |
