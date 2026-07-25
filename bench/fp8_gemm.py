"""W8A16 GEMM in Pallas: fp8 weights, bf16 activations, dequant inside the kernel.

Why a custom kernel is required. The measured bottleneck is weight bytes:
`dyn_fwd` streams 3.147 GB per forward and reaches 637 GB/s of 2353. Halving
the weight bytes is the direct attack. But XLA cannot express weight-only
quantization -- the earlier attempt (bench/patches.py `fp8_weights`) wrote

    dot(convert(w_fp8) * scale, x)

and XLA materialised the full dequantised bf16 matrix in HBM before the GEMM,
so traffic went UP and the frame got 1.37x SLOWER. `loop_multiply_fusion_*`
was 30% of GPU time and zero fp8 kernels were emitted.

A Pallas kernel fixes exactly that: the fp8 tile is loaded from HBM as 1
byte/element, converted to bf16 in SRAM, and fed to the tensor cores. HBM never
sees the bf16 form.

SRAM budget drives the tiling. H100 gives ~228 KB/SM. With full-K blocks:
    A tile  bm x K x 2 bytes    bm=16, K=1920 ->  61 KB
    W tile  K x bn x 1 byte     bn=64,  K=1920 -> 123 KB
    total ~184 KB, fits.
bm=16 is small, but M=290 at batch 1 is small anyway -- and the diagnostic
(bench/GEMM_DIAGNOSTIC.md) says these GEMMs are limited by weight streaming,
not by M.

Prediction worth stating before measuring: this should help `fc_in`/`fc_out`,
which already reach 28-43% of peak bandwidth and are 84% of the parameters,
and do little for `to_q`/`to_kv`, which sit at 1-6% and are latency-bound
rather than bandwidth-bound. If fp8 helps everything equally, that prediction
is wrong and the model of the bottleneck needs revisiting.

MEASURED, and the prediction was WRONG in the informative direction. At
bm=16/bn=64/bk=64 the kernel is numerically fine (rel err 0.0071, pure int8
quantisation) but loses to cuBLAS almost everywhere:

    to_q  M=296  1.04x |  to_q  M=4736  0.20x
    to_kv M=296  0.78x |  to_kv M=4736  0.44x
    fc_in M=296  0.25x |  fc_in M=4736  0.14x
    fc_out M=296 0.44x |  fc_out M=4736 0.19x

`fc_in` -- predicted to benefit MOST -- is the worst. That inverts the
prediction, so the cost is not weight bytes here; it is the tiling. bm=16 is
below the m=64 that H100 wgmma needs, so the tensor cores run at a fraction of
issue rate, and the (m,n) grid re-walks the weight matrix once per m-block
(19x at M=296, 296x at M=4736) instead of the once cuBLAS manages.

The SRAM budget above, which is what motivated bm=16, is also probably wrong:
in the Triton backend a BlockSpec is a *block pointer*, and `a_ref[:, sl]`
lowers to a load of just that slice, so the full-K block is never materialised.
`--sweep` tests exactly that by trying tiles the SRAM budget said were illegal.
If bm=64/128 runs at all, the budget was imaginary and bm=16 was self-imposed.
"""
from __future__ import annotations

import argparse
import functools
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# int8, not e4m3. Triton in this JAX build cannot convert float8_e4m3fn at all
# -- "LLVM ERROR: Unsupported rounding mode for conversion" -- neither to bf16
# nor via f32. int8 gives the identical 1 byte/weight (which is the entire
# point, since the bottleneck is weight BYTES not weight precision), its
# integer->float conversion is universally supported, and per-output-channel
# symmetric int8 is typically MORE accurate for weights than per-tensor e4m3.
INT8_MAX = 127.0


def _w8a16_kernel(a_ref, w_ref, s_ref, o_ref, *, bk: int):
    """One (bm, bn) output tile.

    a_ref (bm, K) bf16, w_ref (K, bn) int8, o_ref (bm, bn) bf16.
    The K loop lives inside the kernel so the int8 tile is dequantised in SRAM.
    s_ref (bn,) holds the per-output-channel scale, applied once to the
    accumulator rather than per K-step.
    """
    K = a_ref.shape[1]

    def body(i, acc):
        sl = pl.dslice(i * bk, bk)
        a = a_ref[:, sl]                            # (bm, bk) bf16
        w = w_ref[sl, :]                            # (bk, bn) int8, 1 byte/elem
        # Dequant in registers on a tile already in SRAM: HBM only ever sees
        # 1 byte per weight, which is the entire point of the kernel.
        wb = w.astype(jnp.bfloat16)
        return acc + jax.lax.dot_general(
            a, wb, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)

    acc = jax.lax.fori_loop(0, K // bk, body,
                            jnp.zeros((a_ref.shape[0], w_ref.shape[1]), jnp.float32))
    o_ref[...] = (acc * s_ref[...][None, :]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("bm", "bn", "bk"))
def w8a16_matmul(a, w_i8, scales, *, bm=16, bn=64, bk=64):
    """a (M,K) bf16 @ dequant(w_i8 (K,N), scales (N,)) -> (M,N) bf16."""
    M, K = a.shape
    _, N = w_i8.shape
    m_pad = (-M) % bm
    if m_pad:
        a = jnp.pad(a, ((0, m_pad), (0, 0)))
    Mp = a.shape[0]

    out = pl.pallas_call(
        functools.partial(_w8a16_kernel, bk=bk),
        grid=(Mp // bm, N // bn),
        in_specs=[
            pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, bn), lambda i, j: (0, j)),
            pl.BlockSpec((bn,), lambda i, j: (j,)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((Mp, N), jnp.bfloat16),
    )(a, w_i8, scales)
    return out[:M] if m_pad else out


def quantize(w_bf16):
    """Per-output-channel symmetric int8; returns (w_int8, scales (N,) f32)."""
    w32 = w_bf16.astype(jnp.float32)
    amax = jnp.max(jnp.abs(w32), axis=0)                      # (N,)
    scales = jnp.maximum(amax, 1e-12) / INT8_MAX
    q = jnp.clip(jnp.round(w32 / scales[None, :]), -INT8_MAX, INT8_MAX)
    return q.astype(jnp.int8), scales.astype(jnp.float32)


def best_ms(fn, *a, w=10, n=20):
    for _ in range(w):
        jax.block_until_ready(fn(*a))
    b = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(fn(*a))
        b = min(b, time.perf_counter() - t)
    return b * 1e3


SHAPES = [
    ("to_q    1920x1920", 1920, 1920),
    ("to_kv   1920x384", 1920, 384),
    ("fc_in   1920x15360", 1920, 15360),
    ("fc_out  7680x1920", 7680, 1920),
]


# The two shapes that decide it: `fc_in` is 84% of the parameters and was the
# worst result, `to_q` was the only one that broke even. If a better tile does
# not move fc_in, no tile will.
SWEEP_SHAPES = [("fc_in   1920x15360", 1920, 15360), ("to_q    1920x1920", 1920, 1920)]

# bm=64/128 is what the SRAM budget in the docstring claimed was impossible
# (bm=128 x K=1920 x 2B = 480 KB against ~228 KB/SM). Including them is the
# test of whether that budget was real. Failures are caught and reported rather
# than aborting the sweep, since "this tile does not compile" is itself the
# answer for those rows.
SWEEP_TILES = [
    (16, 64, 64),      # the measured baseline, for reference
    (16, 128, 64),
    (32, 128, 64),
    (64, 64, 64),
    (64, 128, 64),
    (64, 128, 128),
    (128, 64, 64),
    (128, 128, 64),
    (128, 128, 128),
    (256, 64, 64),
    (256, 128, 64),
    # bm=320 covers M=296 in a single m-block, so the weight matrix is walked
    # exactly once -- the one configuration that removes the re-read entirely.
    (320, 64, 64),
    (320, 128, 64),
]


def sweep(ms=(296, 4736), shapes=SWEEP_SHAPES):
    """Is the kernel slow because of the tile, or because Pallas cannot express
    a competitive GEMM here? Sweeping tiles separates the two."""
    ref = jax.jit(lambda a, b: a @ b)
    key = jax.random.PRNGKey(0)

    for name, K, N in shapes:
        for M in ms:
            a = jax.random.normal(key, (M, K), jnp.bfloat16)
            w = jax.random.normal(key, (K, N), jnp.bfloat16) * (K ** -0.5)
            wq, sc = quantize(w)
            t_ref = best_ms(ref, a, w)
            print(f"\n{name}  M={M}   cuBLAS bf16 {t_ref * 1e3:.1f} us")
            print(f"  {'bm':>5}{'bn':>6}{'bk':>6}{'pallas':>12}{'speedup':>10}")
            for bm, bn, bk in SWEEP_TILES:
                if N % bn or K % bk:
                    continue
                try:
                    f = functools.partial(w8a16_matmul, bm=bm, bn=bn, bk=bk)
                    jax.block_until_ready(f(a, wq, sc))
                    t = best_ms(f, a, wq, sc)
                    flag = "  <-- beats cuBLAS" if t < t_ref else ""
                    print(f"  {bm:>5}{bn:>6}{bk:>6}{t * 1e3:9.1f} us"
                          f"{t_ref / t:9.2f}x{flag}")
                except Exception as e:
                    msg = str(e).split("\n")[0][:60]
                    print(f"  {bm:>5}{bn:>6}{bk:>6}    {type(e).__name__}: {msg}")
            del a, w, wq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[296, 4736])
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep tiles instead of benchmarking one config")
    args = ap.parse_args()

    print(f"jax {jax.__version__}  device {jax.devices()[0].device_kind}")

    if args.sweep:
        # fc_out is the other 84%-of-params GEMM and has a different K (7680),
        # so it exercises the K loop far harder than fc_in does.
        shapes = SWEEP_SHAPES + [("fc_out  7680x1920", 7680, 1920)]
        return sweep(ms=tuple(args.ms), shapes=shapes)

    print(f"tiles bm={args.bm} bn={args.bn} bk={args.bk}\n")

    ref = jax.jit(lambda a, b: a @ b)
    key = jax.random.PRNGKey(0)

    print(f"{'shape':>22}{'M':>7}{'cuBLAS bf16':>14}{'Pallas W8(int8)':>17}"
          f"{'speedup':>10}{'rel err':>10}")
    for name, K, N in SHAPES:
        for M in args.ms:
            if N % args.bn:
                print(f"  {name:>20} N={N} not divisible by bn={args.bn}, skipped")
                continue
            a = jax.random.normal(key, (M, K), jnp.bfloat16)
            w = jax.random.normal(key, (K, N), jnp.bfloat16) * (K ** -0.5)
            wq, sc = quantize(w)

            try:
                got = w8a16_matmul(a, wq, sc, bm=args.bm, bn=args.bn, bk=args.bk)
                exp = ref(a, w)
                err = float(jnp.linalg.norm((got - exp).astype(jnp.float32))
                            / jnp.linalg.norm(exp.astype(jnp.float32)))
                t_ref = best_ms(ref, a, w)
                t_p = best_ms(lambda x, y, z: w8a16_matmul(
                    x, y, z, bm=args.bm, bn=args.bn, bk=args.bk), a, wq, sc)
                print(f"{name:>22}{M:>7}{t_ref * 1e3:11.1f} us{t_p * 1e3:12.1f} us"
                      f"{t_ref / t_p:9.2f}x{err:10.4f}")
            except Exception as e:
                print(f"{name:>22}{M:>7}   FAILED: {type(e).__name__}: {str(e)[:80]}")
            del a, w, wq


if __name__ == "__main__":
    main()
