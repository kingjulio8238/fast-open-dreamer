"""Warm, steady-state timing of the released rollout with real weights.

Why this exists: the hooked `inference.py` run cannot measure the model.
`latent_rollout` drives generation with `jax.lax.scan`, so `next_latent` is
invoked ONCE — at trace time — regardless of horizon. A hook on it records
tracing, not execution, and reports zero warm calls. Everything else in that
run (32 s checkpoint load, XLA compile) is one-off cost that swamps the
generation it is supposed to measure.

So: load the checkpoint once, then call the released `latent_rollout` several
times and time each. That also exposes something the one-shot CLI hides —
`latent_rollout` is NOT jitted; its `lax.scan` re-traces on every call. For a
CLI that is invisible; for a serving engine it is disqualifying. Call 1 vs 2 vs
3 quantifies it, and `--jit` measures the same rollout wrapped in `jax.jit` for
the ceiling a serving engine could reach without touching the model.

    python bench/warm_rollout.py --checkpoint /ckpt/open-dreamer \
        --context-frames 16 --horizon 64 --num-steps 4 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = os.environ.get("OD_INFERENCE_REPO", "/root/od-inference")
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import gpu_sampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="parent dir of the numbered step dir, e.g. /ckpt/open-dreamer")
    ap.add_argument("--context-frames", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jit", action="store_true",
                    help="also time the rollout wrapped in jax.jit (retrace eliminated)")
    ap.add_argument("--decode", action="store_true",
                    help="also time decoding the generated latents to pixels")
    ap.add_argument("--out", default="/results/warm_rollout.json")
    args = ap.parse_args()

    gpu_sampler.start()

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from pipeline.checkpointing import DynamicsCheckpointBundle
    from pipeline.generation import DenoiseSchedule, latent_rollout
    from pipeline.parallel import build_parallel
    from pipeline.actions import parse_action_dicts, shift_actions

    sys.path.insert(0, REPO)
    import inference as released  # the released module, for its exact IO helpers

    mesh, _, mesh_rules = build_parallel("data")
    ctx_mgr = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else mesh

    results: dict = {"args": vars(args)}

    with ctx_mgr:
        t = time.perf_counter()
        bundle = DynamicsCheckpointBundle.from_pretrained(
            args.checkpoint, mesh_rules=mesh_rules,
            model_names={"tokenizer", "dynamics_ema"})
        # Orbax restore is lazy in places; force the weights onto the device
        # before stopping the clock, or the load time lands in the first rollout.
        jax.block_until_ready(jax.tree.leaves(nnx.state(bundle.dynamics_ema)))
        results["ckpt_load_s"] = round(time.perf_counter() - t, 3)
        print(f"checkpoint loaded in {results['ckpt_load_s']:.1f} s", flush=True)

        tokenizer, dynamics = bundle.tokenizer, bundle.dynamics_ema

        # --- inputs, prepared exactly as inference.py does ---
        import glob
        mp4 = sorted(glob.glob(f"{REPO}/samples/vpt/*.mp4"))[0]
        frames_np, _ = released.read_video(Path(mp4), required_frames=args.context_frames)
        frames = jnp.asarray(frames_np[None])
        raw = released.load_action_dicts(Path(mp4.replace(".mp4", ".jsonl")))
        total = args.context_frames + args.horizon
        actions = released.add_batch_dim(parse_action_dicts(raw[:total]))
        actions = shift_actions(actions, dynamics.cfg.categorical_action_dim)

        t = time.perf_counter()
        latents_ctx = jax.block_until_ready(released.encode_jit(tokenizer, frames))
        results["encode_cold_s"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        jax.block_until_ready(released.encode_jit(tokenizer, frames))
        results["encode_warm_s"] = round(time.perf_counter() - t, 3)

        actions_ctx = actions[:, : args.context_frames]
        actions_future = actions[:, args.context_frames : total]
        schedule = DenoiseSchedule.init(args.num_steps, dynamics.cfg.k_max)

        def run_rollout(rng):
            return latent_rollout(
                dynamics=dynamics, policy=actions_future, schedule=schedule,
                latents_ctx=latents_ctx, actions_ctx=actions_ctx,
                num_steps=args.horizon, rng=rng, deterministic=True,
                use_kv_cache=True)

        # --- as released: unjitted, so lax.scan re-traces every call ---
        print(f"\n=== released path (unjitted), {args.repeats + 1} calls ===", flush=True)
        samples = []
        for i in range(args.repeats + 1):
            rng = jax.random.PRNGKey(i)
            w0 = gpu_sampler.now()
            t = time.perf_counter()
            out = jax.block_until_ready(run_rollout(rng))
            dt = time.perf_counter() - t
            w1 = gpu_sampler.now()
            g = gpu_sampler.summarize(window=(w0, w1))
            tag = "cold" if i == 0 else f"warm{i}"
            print(f"  {tag:>6}: {dt:8.3f} s   {dt / args.horizon * 1e3:7.1f} ms/frame   "
                  f"{args.horizon / dt:6.2f} fps   gpu_util {g.get('gpu_util_mean_pct')}%"
                  f" peak {g.get('gpu_util_peak_pct')}%", flush=True)
            if i > 0:
                samples.append(dt)
            results.setdefault("released_calls", []).append(
                {"call": i, "seconds": round(dt, 4),
                 "ms_per_frame": round(dt / args.horizon * 1e3, 2), "gpu": g})

        if samples:
            med = statistics.median(samples)
            results["released_warm_ms_per_frame"] = round(med / args.horizon * 1e3, 2)
            results["released_warm_fps"] = round(args.horizon / med, 2)
            cold = results["released_calls"][0]["seconds"]
            results["retrace_overhead_s"] = round(cold - med, 3)
            print(f"\n  RELEASED BASELINE (warm median): "
                  f"{med / args.horizon * 1e3:.1f} ms/frame, {args.horizon / med:.2f} fps")
            print(f"  first-call overhead (trace + compile): {cold - med:.1f} s")

        # --- same rollout under jax.jit: no retrace, one compiled graph ---
        if args.jit:
            print(f"\n=== jax.jit(latent_rollout), {args.repeats + 1} calls ===", flush=True)
            jitted = jax.jit(run_rollout)
            jsamples = []
            for i in range(args.repeats + 1):
                rng = jax.random.PRNGKey(100 + i)
                t = time.perf_counter()
                jax.block_until_ready(jitted(rng))
                dt = time.perf_counter() - t
                tag = "compile" if i == 0 else f"warm{i}"
                print(f"  {tag:>7}: {dt:8.3f} s   {dt / args.horizon * 1e3:7.1f} ms/frame   "
                      f"{args.horizon / dt:6.2f} fps", flush=True)
                if i > 0:
                    jsamples.append(dt)
            if jsamples:
                med = statistics.median(jsamples)
                results["jit_warm_ms_per_frame"] = round(med / args.horizon * 1e3, 2)
                results["jit_warm_fps"] = round(args.horizon / med, 2)
                print(f"\n  JITTED: {med / args.horizon * 1e3:.1f} ms/frame, "
                      f"{args.horizon / med:.2f} fps")

        if args.decode:
            print("\n=== decode ===", flush=True)
            lat = jax.block_until_ready(run_rollout(jax.random.PRNGKey(0)))["latents"]
            for i in range(3):
                t = time.perf_counter()
                jax.block_until_ready(released.decode_jit(tokenizer, lat))
                dt = time.perf_counter() - t
                n = lat.shape[1]
                print(f"  call {i}: {dt:7.3f} s for {n} frames "
                      f"({dt / n * 1e3:6.2f} ms/frame)", flush=True)
                if i:
                    results["decode_warm_ms_per_frame"] = round(dt / n * 1e3, 2)

    results["gpu_overall"] = gpu_sampler.summarize()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
