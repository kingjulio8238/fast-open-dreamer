# Why the dynamics forward reaches only 27% of HBM bandwidth

`dyn_fwd` at B=1 moves 3.147 GB of bf16 weights in 6.55 ms = 637 GB/s against a
measured 2353 GB/s ceiling. This diagnostic asked whether that is a fixable
tiling artefact (M=290 is not a multiple of 64) or intrinsic to the shapes.

**Answer: neither, quite. It is driven by weight-matrix SIZE, not by M.**
Run: `bench/gemm_probe.py` on H100, log `bjkdz8657`.

## Achieved weight-stream bandwidth at M≈290

| GEMM | weights | achieved | % of 2353 GB/s |
|---|---:|---:|---:|
| `fc_in` 1920×15360 | 59.0 MB | 660–1020 GB/s | 28–43% |
| `fc_out` 7680×1920 | 29.5 MB | 370–530 | 16–23% |
| `to_q` / `to_out` 1920×1920 | 7.4 MB | 100–130 | **4–6%** |
| `to_kv` 1920×384 | 1.5 MB | 20–30 | **~1%** |

An 8× efficiency spread from matrix size alone. Small matrices finish before the
memory system reaches steady state, so they never approach peak regardless of M.

The same ops at large M climb their efficiency curve steeply — `to_q` goes
33 TFLOP/s at M=256 to **334 TFLOP/s at M=4640**. That is why B=16 already
reaches 45–50% MFU where B=1 sits at 19%.

## Three cheap fixes tested and REJECTED

**1. Padding the token axis 290 → 296.** Already happening. The optimized HLO
emits `bf16[296,15360]` (58 instances) — XLA pads for us. Changing `n_register`
from 32 to 38 would remove the pad *kernels* (~2.7% of GPU time), not improve
any GEMM.

**2. Fusing `to_q` + `to_kv` into one (1920, 2304) GEMM.** Expected a win from
halving small-GEMM launches. It is **38–57% slower**:

```
M=290: separate q+kv  67.8 us | fused qkv 106.2 us  (-56.6%)
M=296: separate q+kv  61.3 us | fused qkv  84.8 us  (-38.4%)
```

The concatenated matrix tiles worse than two separate calls, and the output
slice costs more than the saved launch. Firmly refuted.

**3. Fusing the SwiGLU chain.** Worth ~9% of the MLP block
(full chain 0.109 ms vs GEMMs-only 0.099 ms at M=296). Real but minor; the
9.1 MB intermediate is 1.36 GB/frame of avoidable round-trip, which is small
next to 15.7 GB of weight traffic.

## Methodological caution

Summing isolated GEMM timings predicts **11.34 ms per forward**; the real
`dyn_fwd` is **6.55 ms**. XLA already overlaps layers and keeps weights warmer
in L2 than isolated timing suggests. **Microbenchmarks overstate what per-op
fixes can recover** — treat the table above as relative structure, not as a
budget to be summed.

## Consequence for the optimisation plan

There is no cheap shape-level win. The 3.6× of headroom against the bandwidth
roofline is real, but it is the small-matrix effect, and the only two things
that address it are:

- **larger effective M** (batching across concurrent streams), which moves every
  GEMM up the curve measured above; and
- **fewer weight bytes** (fp8/int quantisation with in-register dequantisation),
  which helps the small matrices most.

Both are real projects. The "do the easy tiling work first" option does not
exist.
