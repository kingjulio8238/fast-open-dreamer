"""Stage-by-stage inference profiler for Open Dreamer.

Measures the real serving path (autoregressive single-frame generation) and
breaks it into the units that actually cost time, so each optimisation can be
attributed to a number:

  prefill    one dynamics forward over T_ctx context frames (once per session)
  dyn_fwd    one dynamics forward at T=1 with KV cache  -- the atomic unit;
             a shortcut rollout runs (num_steps + 1) of these per frame
  ladder     the full tau-ladder (`next_latent`): scan of num_steps + commit pass
  decode     one tokenizer-decoder forward at T=1 with KV cache
  frame      end-to-end `next_frame` = ladder + decode
  kv_roll    isolated `KVCache.get_ordered_kv`, which rolls the whole KV buffer

Runs with randomly-initialised weights -- no checkpoint required, since shapes
and dtypes are what determine performance.

Usage:
    python bench/profile_inference.py --batch 1
    python bench/profile_inference.py --batch 1 4 16 --steps 4 --param-dtype bfloat16
    python bench/profile_inference.py --batch 1 --trace /tmp/od_trace   # -> Perfetto
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from flax import nnx

from bench.prod_config import (
    dynamics_config, tokenizer_config, H, W, N_LATENTS,
    NUM_BINARY_ACTIONS, CATEGORICAL_ACTION_DIM, MODEL_PKG,
)

# `MODEL_PKG` is "pipeline" when running against the released inference repo
# (reactor-team/open-dreamer) and "dreamer" when running against this fork. The
# model/generation code is byte-identical between them apart from the policy
# head, so the same harness benchmarks both.
import importlib
_actions = importlib.import_module(f"{MODEL_PKG}.actions")
_generation = importlib.import_module(f"{MODEL_PKG}.generation")
_models = importlib.import_module(f"{MODEL_PKG}.models")
_parallel = importlib.import_module(f"{MODEL_PKG}.parallel")

Actions = _actions.Actions
DenoiseSchedule = _generation.DenoiseSchedule
next_frame = getattr(_generation, "next_frame", None)
next_latent = _generation.next_latent
Dynamics, Tokenizer, KVCache = _models.Dynamics, _models.Tokenizer, _models.KVCache
MeshRules = _parallel.MeshRules

# Peak dense bf16 TFLOP/s and HBM GB/s. Used only to report MFU / bandwidth
# utilisation; override with --peak-tflops / --peak-bw for an unlisted device.
DEVICE_PEAKS = {
    "NVIDIA H100": (990.0, 3350.0),
    "NVIDIA H200": (990.0, 4800.0),
    "NVIDIA B200": (2250.0, 8000.0),
    "NVIDIA GB200": (2500.0, 8000.0),
    "NVIDIA A100": (312.0, 2039.0),
    "NVIDIA L40S": (362.0, 864.0),
    "NVIDIA RTX 4090": (165.0, 1008.0),
    "NVIDIA RTX 5090": (210.0, 1792.0),
}


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def bench(fn, *args, warmup: int = 3, iters: int = 20):
    """Return (median_ms, p10_ms, p90_ms, compile_s). Blocks on every output."""
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup - 1):
        jax.block_until_ready(fn(*args))

    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    n = len(samples)
    return (statistics.median(samples), samples[max(0, n // 10)],
            samples[min(n - 1, (9 * n) // 10)], compile_s)


# --------------------------------------------------------------------------
# analytic cost, for MFU / bandwidth attribution
# --------------------------------------------------------------------------

def matmul_params(d, n_heads, n_kv, mlp_ratio):
    hd = d // n_heads
    h = int(d * mlp_ratio)
    return 2 * d * d + d * 2 * (n_kv * hd) + d * (2 * h) + h * d


def n_time_layers(depth, every, off):
    return sum(1 for i in range(depth) if (i + off) % every == 0)


def dyn_flops(cfg, B, T=1):
    S = 1 + 1 + N_LATENTS // cfg.packing_factor + cfg.n_register
    n_t = n_time_layers(cfg.depth, cfg.time_every, cfg.time_layer_offset)
    n_s = cfg.depth - n_t
    gemm = 2 * B * T * S * matmul_params(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_ratio) * cfg.depth
    space = n_s * 4 * S * S * cfg.d_model * B * T
    t_eff = min(T, cfg.context_length) if T > 1 else cfg.context_length
    time_ = n_t * 4 * T * t_eff * cfg.d_model * B * S
    return gemm + space + time_


def dec_flops(cfg, B, T=1, block_split: bool = False):
    """Decoder FLOPs. `block_split=True` accounts for block_attn.

    Without this flag the space-attention term assumes a dense S x S score
    matrix. Under block_attn the decoder computes only p^2 + (S-p)*S entries
    (latents attend to latents; patches attend to everything), which is 23%
    fewer. Reporting dense FLOPs for a run that took the split path overstates
    achieved TFLOP/s and MFU by ~3%.
    """
    n_patches = (cfg.H // cfg.patch_size) * (cfg.W // cfg.patch_size)
    S = cfg.n_latents + n_patches
    n_t = n_time_layers(cfg.depth, cfg.time_every, cfg.time_layer_offset)
    n_s = cfg.depth - n_t
    gemm = 2 * B * T * S * matmul_params(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, 4.0) * cfg.depth
    gemm += 2 * B * T * S * (cfg.d_bottleneck * cfg.d_model + cfg.d_model * cfg.d_patch)
    entries = S * S
    if block_split:
        p = cfg.n_latents
        entries = p * p + (S - p) * S
    space = n_s * 4 * entries * cfg.d_model * B * T
    time_ = n_t * 4 * T * min(T, cfg.context_length or T) * cfg.d_model * B * S
    return gemm + space + time_


def param_bytes(model, itemsize_override: int | None = None):
    _, state, _ = nnx.split(model, nnx.Param, ...)
    total = 0
    for leaf in jax.tree.leaves(state):
        total += leaf.size * (itemsize_override or leaf.dtype.itemsize)
    return total


def param_count(model):
    _, state, _ = nnx.split(model, nnx.Param, ...)
    return sum(leaf.size for leaf in jax.tree.leaves(state))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_actions(B, T=None, key=None):
    shape = (B,) if T is None else (B, T)
    return Actions(
        binary=jnp.zeros(shape + (NUM_BINARY_ACTIONS,), dtype=jnp.int32),
        categorical=jnp.full(shape, CATEGORICAL_ACTION_DIM // 2, dtype=jnp.int32),
        continuous=None,
    )


def build_models(args):
    mesh_rules = MeshRules()
    dcfg = dynamics_config(dtype=args.dtype, param_dtype=args.param_dtype)
    tcfg = tokenizer_config(dtype=args.dtype, param_dtype=args.param_dtype)
    print("building models (random init) ...", flush=True)
    dynamics = Dynamics(dcfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(0))
    tokenizer = Tokenizer(tcfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(0))
    return dynamics, tokenizer, dcfg, tcfg


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=4, help="tau-ladder denoising steps (4 = shortcut)")
    ap.add_argument("--ctx", type=int, default=192, help="dynamics KV window (frames of history)")
    ap.add_argument("--dtype", default="bfloat16", help="activation/compute dtype")
    ap.add_argument("--param-dtype", default="float32", help="weight storage dtype (repo default: float32)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--peak-bw", type=float, default=None, help="HBM GB/s")
    ap.add_argument("--trace", default=None, help="dump a JAX profiler trace to this dir")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip: prefill dyn_fwd ladder decode frame kv_roll")
    ap.add_argument("--encode-frames", type=int, default=8,
                    help="frames per encode call; the dense path OOMs above ~5 at B=8")
    ap.add_argument("--only-add", nargs="*", default=[],
                    help="stages to run even if listed in --skip")
    ap.add_argument("--block-attn-min", type=int, default=8,
                    help="skip block splits where the smaller block is under this "
                         "many tokens; the dynamics mask splits at 1 and loses")
    ap.add_argument("--block-attn-impl", default=None,
                    choices=[None, "cudnn", "xla"],
                    help="SDPA backend for the block-split path; 'cudnn' = flash")
    ap.add_argument("--patches", nargs="*", default=[],
                    help="optimisations from bench/patches.py: no_roll_kv "
                         "fast_kv_write no_remat block_attn bf16_weights")
    args = ap.parse_args()

    if args.patches:
        from bench import patches as _patches
        code_patches = [p for p in args.patches if p != "bf16_weights"]
        if code_patches:
            _patches.apply(code_patches, pkg=MODEL_PKG)
            # Never benchmark a patch that changes the answer.
            if "no_roll_kv" in code_patches:
                _patches.check_kv_equivalence(pkg=MODEL_PKG)
            if "block_attn" in code_patches:
                _patches.set_block_attn_impl(args.block_attn_impl)
                _patches.check_block_attn_equivalence(pkg=MODEL_PKG)
                print(f"block_attn SDPA implementation: "
                      f"{args.block_attn_impl or 'default (XLA reference)'}")
        print(f"patches active : {args.patches}")

    dev = jax.devices()[0]
    dev_name = getattr(dev, "device_kind", str(dev))
    peak_tf, peak_bw = None, None
    for k, (tf, bw) in DEVICE_PEAKS.items():
        if k.split()[-1].lower() in dev_name.lower():
            peak_tf, peak_bw = tf, bw
            break
    peak_tf = args.peak_tflops or peak_tf
    peak_bw = args.peak_bw or peak_bw

    print("=" * 84)
    print(f"device        : {dev_name}  ({len(jax.devices())} visible)")
    print(f"peak (assumed): {peak_tf} TFLOP/s bf16, {peak_bw} GB/s HBM"
          if peak_tf else "peak          : unknown -- pass --peak-tflops/--peak-bw for MFU")
    print(f"dtype         : compute={args.dtype}  params={args.param_dtype}")
    print(f"denoise steps : {args.steps}  ->  {args.steps + 1} dynamics forwards per frame")
    print("=" * 84)

    dynamics, tokenizer, dcfg, tcfg = build_models(args)
    _fp8_stats_pending = None
    if "bf16_weights" in args.patches:
        from bench import patches as _patches
        before = _patches.param_bytes(dynamics) + _patches.param_bytes(tokenizer)
        _patches.cast_params(dynamics, "bfloat16")
        _patches.cast_params(tokenizer, "bfloat16")
        after = _patches.param_bytes(dynamics) + _patches.param_bytes(tokenizer)
        print(f"bf16_weights   : {before/1e9:.2f} GB -> {after/1e9:.2f} GB resident")

    if "fp8_weights" in args.patches:
        from bench import patches as _patches
        _patches.check_fp8_accuracy()
        _patches.quantize_linear_fp8(dynamics, pkg=MODEL_PKG)
        st = _patches.fp8_stats()
        print(f"fp8_weights    : {st['quantized']} Linear kernels quantized, "
              f"{st['skipped']} left alone   "
              f"{st['params_before']/1e9:.2f} GB -> {st['params_after']/1e9:.2f} GB "
              f"({st.get('compression', 1):.2f}x)")
        _fp8_stats_pending = st

    if "block_attn" in args.patches:
        from bench import patches as _patches
        n_tok = _patches.tag_block_attention(tokenizer)
        n_dyn_tag = _patches.tag_block_attention(
            dynamics, n_latents=N_LATENTS, min_block=args.block_attn_min)
        print(f"block_attn     : tagged {n_tok} tokenizer + {n_dyn_tag} dynamics "
              f"space-attention layers")
        if n_tok + n_dyn_tag == 0:
            raise SystemExit("block_attn: tagged 0 layers -- the walk missed")

    n_dyn, n_dec = param_count(dynamics), param_count(tokenizer.decoder)
    b_dyn, b_dec = param_bytes(dynamics), param_bytes(tokenizer.decoder)
    print(f"dynamics params : {n_dyn/1e6:8.1f} M  -> {b_dyn/1e9:.3f} GB resident")
    print(f"decoder  params : {n_dec/1e6:8.1f} M  -> {b_dec/1e9:.3f} GB resident")
    print(f"encoder  params : {param_count(tokenizer.encoder)/1e6:8.1f} M  (unused during rollout)")

    n_spatial = N_LATENTS // dcfg.packing_factor
    S_dyn = 2 + n_spatial + dcfg.n_register
    n_t_dyn = n_time_layers(dcfg.depth, dcfg.time_every, dcfg.time_layer_offset)
    print(f"dynamics tokens/frame S = {S_dyn}   time layers {n_t_dyn} / {dcfg.depth}")
    print(f"decoder  tokens/frame S = {tcfg.decoder.n_latents + (H//16)*(W//16)}")

    schedule = DenoiseSchedule.init(args.steps, dcfg.k_max)
    gd_dyn, st_dyn = nnx.split(dynamics)
    gd_tok, st_tok = nnx.split(tokenizer)
    compute_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    results = {"device": dev_name, "dtype": args.dtype, "param_dtype": args.param_dtype,
               "steps": args.steps, "params": {"dynamics": n_dyn, "decoder": n_dec},
               "stages": {}}
    if _fp8_stats_pending is not None:
        results["fp8_stats"] = _fp8_stats_pending

    def record(stage, B, ms, p10, p90, comp, flops=None, bytes_=None):
        row = {"batch": B, "ms": ms, "p10": p10, "p90": p90, "compile_s": comp}
        line = f"  {stage:<10} B={B:<4} {ms:8.3f} ms  [{p10:.3f}-{p90:.3f}]  compile {comp:6.1f}s"
        if flops:
            tfs = flops / (ms * 1e-3) / 1e12
            row["tflops_achieved"] = tfs
            line += f"   {tfs:7.1f} TFLOP/s"
            if peak_tf:
                row["mfu"] = tfs / peak_tf
                line += f" ({tfs/peak_tf*100:4.1f}% MFU)"
                # MFU > 100% is impossible; it means the peak is wrong, not that
                # the kernel is magic. Say so loudly rather than let the number
                # propagate into a report.
                if tfs > peak_tf:
                    line += "  <-- >100%: PEAK IS UNDER-MEASURED, rerun gpu_probe"
                    results.setdefault("warnings", []).append(
                        f"{stage} B={B} reports {tfs/peak_tf*100:.0f}% MFU; "
                        f"peak_tflops={peak_tf} is too low")
        if bytes_:
            gbs = bytes_ / (ms * 1e-3) / 1e9
            row["gbps_achieved"] = gbs
            line += f"   {gbs:7.0f} GB/s"
            if peak_bw:
                row["bw_util"] = gbs / peak_bw
                line += f" ({gbs/peak_bw*100:4.1f}%)"
        print(line, flush=True)
        results["stages"].setdefault(stage, []).append(row)

    # ---- jitted stage entry points (rebuilt per batch: latent_shape is static)
    @jax.jit
    def _dyn_fwd(st, actions, step_i, tau_i, latents, caches):
        m = nnx.merge(gd_dyn, st)
        out, (h, new_caches) = m(actions, step_i, tau_i, latents, caches=caches, deterministic=True)
        return out, new_caches

    _dyn_prefill = _dyn_fwd

    @jax.jit
    def _decode(st, z, caches):
        m = nnx.merge(gd_tok, st)
        frames, new_caches = m.decode(z, caches=caches, deterministic=True)
        return frames, new_caches

    profiler_on = False
    if args.trace:
        Path(args.trace).mkdir(parents=True, exist_ok=True)

    for B in args.batch:
        print(f"\n--- batch {B} " + "-" * 66)
        # Dynamics takes UNPACKED encoder tokens (N_LATENTS=512) and packs them
        # 2:1 internally into n_spatial=256. Feeding it n_spatial directly makes
        # it derive S=162 instead of 290 and mismatch the KV cache.
        latent_shape = (B, 1, N_LATENTS, dcfg.d_bottleneck)
        rng = jax.random.PRNGKey(0)

        # Built per batch: `latent_shape` is a static Python tuple closed over here.
        def _make_stage_fns(shape):
            @jax.jit
            def ladder(st, action, rng_, caches):
                m = nnx.merge(gd_dyn, st)
                lat, h, c, rng2, _ = next_latent(
                    m, schedule, action, shape, rng_, caches=caches, task_embedding=None)
                return lat, c, rng2

            @jax.jit
            def frame(st_d, st_t, action, rng_, dyn_c, dec_c):
                md = nnx.merge(gd_dyn, st_d)
                mt = nnx.merge(gd_tok, st_t)
                return next_frame(mt, md, schedule, action, shape, dyn_c, dec_c, rng_)

            return ladder, frame

        _ladder, _frame = _make_stage_fns(latent_shape)

        dyn_caches = dynamics.create_static_caches(
            batch_size=B, n_latents=N_LATENTS,
            window_size=args.ctx, n_agent=0, dtype=compute_dtype)
        # NOTE: window_size here must match what the serving runtime uses. The
        # decoder cache in Tokenizer.create_static_caches defaults to 1024 even
        # though decoder.context_length is 16 -- see the audit notes.
        dec_caches = tokenizer.decoder.create_static_caches(
            batch_size=B, window_size=tcfg.decoder.context_length, dtype=compute_dtype)

        kv_bytes_dyn = sum(c.k.size * c.k.dtype.itemsize + c.v.size * c.v.dtype.itemsize
                           for c in dyn_caches.values())
        print(f"  dynamics KV cache resident: {kv_bytes_dyn/1e9:.3f} GB "
              f"(window {args.ctx}, {n_t_dyn} time layers)")

        act_1 = make_actions(B)                       # time dim squeezed
        act_T = make_actions(B, 1)
        latents_1 = jnp.zeros((B, 1, N_LATENTS, dcfg.d_bottleneck), dtype=compute_dtype)
        step_i = jnp.full((B, 1), schedule.step_idx, dtype=jnp.int32)
        tau_i = jnp.full((B, 1), schedule.k_max, dtype=jnp.int32)
        z_1 = jnp.zeros((B, 1, N_LATENTS, dcfg.d_bottleneck), dtype=compute_dtype)

        f_dyn = dyn_flops(dcfg, B, 1)
        # Derive from what actually fired, not what was requested: if tagging
        # partially failed we would report split FLOPs while executing dense.
        _bs = False
        if "block_attn" in args.patches:
            from bench import patches as _p
            _bs = _p.block_attn_stats().get("shape_mismatch", 0) == 0
        f_dec = dec_flops(tcfg.decoder, B, 1, block_split=_bs)

        if "dyn_fwd" not in args.skip:
            ms, p10, p90, c = bench(_dyn_fwd, st_dyn, act_T, step_i, tau_i, latents_1, dyn_caches,
                                    warmup=args.warmup, iters=args.iters)
            record("dyn_fwd", B, ms, p10, p90, c, flops=f_dyn, bytes_=b_dyn + 3 * kv_bytes_dyn)

        if "prefill" not in args.skip:
            T_ctx = min(args.ctx, 64)
            act_ctx = make_actions(B, T_ctx)
            lat_ctx = jnp.zeros((B, T_ctx, N_LATENTS, dcfg.d_bottleneck), dtype=compute_dtype)
            si = jnp.full((B, T_ctx), schedule.emax, dtype=jnp.int32)
            ti = jnp.full((B, T_ctx), schedule.k_max, dtype=jnp.int32)
            ms, p10, p90, c = bench(_dyn_prefill, st_dyn, act_ctx, si, ti, lat_ctx, dyn_caches,
                                    warmup=args.warmup, iters=max(3, args.iters // 4))
            record(f"prefill{T_ctx}", B, ms, p10, p90, c, flops=dyn_flops(dcfg, B, T_ctx))

        if "ladder" not in args.skip:
            ms, p10, p90, c = bench(_ladder, st_dyn, act_1, rng, dyn_caches,
                                    warmup=args.warmup, iters=args.iters)
            record("ladder", B, ms, p10, p90, c, flops=(args.steps + 1) * f_dyn)

        if "encode" in args.only_add or "encode" not in args.skip:
            T_enc = args.encode_frames
            vid = jnp.zeros((B, T_enc, H, W, 3), dtype=jnp.uint8)

            @jax.jit
            def _encode(st, v):
                m = nnx.merge(gd_tok, st)
                lat, _, _ = m.encode(v, deterministic=True)
                return lat

            try:
                ms, p10, p90, c = bench(_encode, st_tok, vid,
                                        warmup=args.warmup, iters=max(3, args.iters // 4))
                record(f"encode{T_enc}", B, ms, p10, p90, c)
                scores = (tcfg.encoder.n_heads * (N_LATENTS + (H // 16) * (W // 16)) ** 2 * 4)
                print(f"    dense score matrix would be "
                      f"{B * T_enc * scores / 2**30:.2f} GiB "
                      f"({B}x{T_enc}x{tcfg.encoder.n_heads}x"
                      f"{N_LATENTS + (H//16)*(W//16)}^2 fp32)")
            except Exception as exc:
                print(f"  encode{T_enc:<4} B={B:<4} FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:120]}")
                results.setdefault("stages", {}).setdefault(f"encode{T_enc}", []).append(
                    {"batch": B, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

        if "decode" not in args.skip:
            ms, p10, p90, c = bench(_decode, st_tok, z_1, dec_caches,
                                    warmup=args.warmup, iters=args.iters)
            record("decode", B, ms, p10, p90, c, flops=f_dec, bytes_=b_dec)

        if "frame" not in args.skip:
            ms, p10, p90, c = bench(_frame, st_dyn, st_tok, act_1, rng, dyn_caches, dec_caches,
                                    warmup=args.warmup, iters=args.iters)
            record("frame", B, ms, p10, p90, c, flops=(args.steps + 1) * f_dyn + f_dec)
            print(f"  -> {1000/ms:7.1f} fps single-stream, {1000/ms*B:8.1f} fps aggregate")

        if "kv_roll" not in args.skip:
            # MISLEADING IF READ AS "cost of the roll" — this forces the result
            # with k.sum()+v.sum(), so it measures a full materialised read of
            # the KV cache, which XLA otherwise fuses into the attention that
            # consumes it. The ablation measured the roll's true cost at ~0.9 ms
            # per frame at B=1, not the ~5 ms this extrapolation suggests.
            # Treat it as an upper bound on cache-read traffic, nothing more.
            hd = dcfg.d_model // dcfg.n_heads
            cache = KVCache.init(B * S_dyn, args.ctx, dcfg.n_kv_heads, hd, dtype=compute_dtype)
            one_layer_bytes = (cache.k.size + cache.v.size) * cache.k.dtype.itemsize

            @jax.jit
            def _roll(c):
                k, v, m = c.get_ordered_kv(query_len=1)
                return k.sum() + v.sum(), m.sum()

            ms, p10, p90, c_ = bench(_roll, cache, warmup=args.warmup, iters=args.iters)
            print(f"  {'kv_read':<10} B={B:<4} {ms:8.3f} ms/layer materialised  "
                  f"({one_layer_bytes/1e9:.3f} GB/layer)  "
                  f"[upper bound on cache traffic, NOT the roll's cost]")
            results["stages"].setdefault("kv_roll_one_layer", []).append(
                {"batch": B, "ms": ms, "bytes": one_layer_bytes,
                 "extrapolated_ms_per_frame": ms * n_t_dyn * (args.steps + 1)})

        if args.trace and not profiler_on:
            print(f"\n  capturing trace -> {args.trace}")
            jax.profiler.start_trace(args.trace)
            for _ in range(5):
                jax.block_until_ready(_frame(st_dyn, st_tok, act_1, rng, dyn_caches, dec_caches))
            jax.profiler.stop_trace()
            profiler_on = True
            print("  open with https://ui.perfetto.dev")

    if "block_attn" in args.patches:
        from bench import patches as _p
        st = _p.block_attn_stats()
        results["block_attn_stats"] = st
        print(f"\nblock_attn: {st['split']} split, {st['dense']} dense, "
              f"{st.get('shape_mismatch', 0)} tag/shape mismatch, "
              f"{st.get('degenerate_skipped', 0)} degenerate splits skipped")
        if st["split"] == 0:
            print("  FAIL: block_attn was applied but never fired. Every number "
                  "above is the unpatched path.")
            if st["unmatched"]:
                print(f"  masks seen: {st['unmatched']}")
            raise SystemExit(2)
        if st["unmatched"]:
            print(f"  masks not recognised as 2-block: {st['unmatched']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
