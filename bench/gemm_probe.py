"""Why does the dynamics forward reach only 27% of HBM bandwidth?

`dyn_fwd` at B=1 moves 3.147 GB of bf16 weights in 6.55 ms = 637 GB/s against a
measured 2353 GB/s ceiling, at an arithmetic intensity of 289 FLOP/byte versus a
316 ridge. So it is just short of bandwidth-bound and achieving a fifth of it.
Two candidate explanations, with very different consequences:

  SHAPE      M=290 is a bad tile for H100 wgmma (which wants multiples of 64),
             so the GEMMs run at low occupancy. Fixable by padding the token
             axis or changing n_register -- cheap, no numerics change.

  INTRINSIC  low-M GEMMs simply cannot saturate HBM because there is not enough
             work in flight to hide the weight stream. Then the only lever is
             moving fewer bytes (quantization), and Tier 1 is not worth doing.

This sweeps M across the model's REAL Linear shapes and reports achieved
TFLOP/s and, more importantly, achieved weight-streaming bandwidth. If a nearby
M (256, 320) is much faster than 290, it is shape. If the whole low-M range is
flat and far below peak, it is intrinsic.

Also measures the SwiGLU chain separately: fc_in -> split -> silu -> mul ->
fc_out materializes a (M, 7680) intermediate twice per layer, which is
avoidable traffic a fused kernel would remove.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

# (name, K, N) for one dynamics layer at d_model=1920, SwiGLU mlp_ratio 4.
DYN_SHAPES = [
    ("to_q    1920x1920", 1920, 1920),
    ("to_kv   1920x384", 1920, 384),
    ("to_out  1920x1920", 1920, 1920),
    ("fc_in   1920x15360", 1920, 15360),
    ("fc_out  7680x1920", 7680, 1920),
]
DEC_SHAPES = [
    ("dec to_q  1024x1024", 1024, 1024),
    ("dec fc_in 1024x8192", 1024, 8192),
    ("dec fc_out 4096x1024", 4096, 1024),
]


def best_ms(fn, *args, warmup=20, iters=30):
    """Best-of-N with a proper clock-ramp warmup, blocking every iteration."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+",
                    default=[64, 128, 256, 288, 290, 296, 320, 512, 1024, 4640, 8192])
    ap.add_argument("--peak-tflops", type=float, default=742.5)
    ap.add_argument("--peak-bw", type=float, default=2353.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"device: {getattr(dev, 'device_kind', dev)}   jax {jax.__version__}")
    print(f"assumed peaks: {args.peak_tflops} TFLOP/s bf16, {args.peak_bw} GB/s\n")

    mm = jax.jit(lambda a, b: a @ b)
    results = {"peaks": [args.peak_tflops, args.peak_bw], "sweeps": {}}

    for title, shapes in (("DYNAMICS (d_model 1920)", DYN_SHAPES),
                          ("DECODER (d_model 1024)", DEC_SHAPES)):
        print("=" * 96)
        print(f"{title}   — achieved TFLOP/s and weight-stream GB/s vs M")
        print("=" * 96)
        hdr = f"  {'M':>6}" + "".join(f"{n.split()[0]:>18}" for n, _, _ in shapes)
        print(hdr)
        for M in args.ms:
            row_t, row_b = [], []
            for name, K, N in shapes:
                a = jax.random.normal(jax.random.PRNGKey(0), (M, K), jnp.bfloat16)
                b = jax.random.normal(jax.random.PRNGKey(1), (K, N), jnp.bfloat16)
                ms = best_ms(mm, a, b)
                tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
                # Weight-stream bandwidth: the (K,N) matrix is the dominant read
                # at low M, and is what the model is actually limited by.
                gbs = K * N * 2 / (ms * 1e-3) / 1e9
                row_t.append(tflops)
                row_b.append(gbs)
                results["sweeps"].setdefault(name, []).append(
                    {"M": M, "ms": ms, "tflops": tflops, "weight_gbs": gbs})
                del a, b
            print(f"  {M:>6}" + "".join(f"{t:8.0f}TF{g/1000:6.2f}TB" for t, g in zip(row_t, row_b)))
        print()

    # ---- the shape question, stated directly -------------------------------
    print("=" * 96)
    print("SHAPE vs INTRINSIC")
    print("=" * 96)
    for name in [n for n, _, _ in DYN_SHAPES]:
        rows = {r["M"]: r for r in results["sweeps"][name]}
        near = {m: rows[m]["weight_gbs"] / 1e3 for m in (256, 288, 290, 296, 320) if m in rows}
        big = max(rows[m]["weight_gbs"] for m in rows) / 1e3
        if near:
            spread = (max(near.values()) - min(near.values())) / max(near.values()) * 100
            best_m = max(near, key=near.get)
            print(f"  {name:22} M=290 {near.get(290, 0):5.2f} TB/s | "
                  f"best nearby M={best_m} {near[best_m]:5.2f} TB/s "
                  f"({spread:4.1f}% spread) | best any M {big:5.2f} TB/s")
    print()
    print("  Read: a large spread across 256..320 means SHAPE (pad the token axis).")
    print("  A flat low band that only rises at large M means INTRINSIC (quantize).")

    # ---- SwiGLU chain: is the materialized intermediate worth fusing? -------
    print()
    print("=" * 96)
    print("SwiGLU CHAIN (one dynamics MLP block)")
    print("=" * 96)
    M = 296
    k1 = jax.random.PRNGKey(0)
    x = jax.random.normal(k1, (M, 1920), jnp.bfloat16)
    w_in = jax.random.normal(k1, (1920, 15360), jnp.bfloat16)
    w_out = jax.random.normal(k1, (7680, 1920), jnp.bfloat16)

    @jax.jit
    def chain(x, wi, wo):
        pre = x @ wi
        u, v = jnp.split(pre, 2, axis=-1)
        return (u * jax.nn.silu(v)) @ wo

    @jax.jit
    def gemms_only(x, wi, wo):
        return (x @ wi)[:, :7680] @ wo

    t_chain = best_ms(chain, x, w_in, w_out)
    t_gemms = best_ms(gemms_only, x, w_in, w_out)
    inter = M * 15360 * 2 / 1e6
    print(f"  M={M}: full chain {t_chain:6.3f} ms | GEMMs only {t_gemms:6.3f} ms | "
          f"elementwise+split costs {t_chain - t_gemms:6.3f} ms "
          f"({(t_chain - t_gemms) / t_chain * 100:.0f}%)")
    print(f"  intermediate is {inter:.1f} MB; x30 layers x5 forwards = "
          f"{inter * 30 * 5 / 1000:.2f} GB/frame of avoidable round-trip")
    print(f"  -> fusing the whole block would save at most "
          f"{(t_chain - t_gemms) * 30 * 5:.1f} ms/frame at this M")
    results["swiglu"] = {"M": M, "chain_ms": t_chain, "gemms_ms": t_gemms}

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
