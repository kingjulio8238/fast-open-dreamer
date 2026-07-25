"""Does cutting the tau-ladder cost accuracy? Teacher-forced, fixed-noise test.

A previous version of this script compared free-running rollouts against a
high-step rollout and reported PSNR. That cannot work, and its output proved it:
steps=1, 2 and 4 all scored ~22.1 dB / NMSE ~0.083, indistinguishable. Two
reasons, both fatal:

  1. Each step count consumes a different amount of randomness (the ladder is a
     scan over num_steps, plus a commit-pass noise draw), so the same PRNGKey
     produces different noise from the first frame onward. After 32
     autoregressive steps the two videos are simply different samples from the
     same distribution.
  2. That sampling divergence dominates any step-count error, so the metric
     saturates at "how much do two plausible Minecraft futures differ" and is
     flat across the thing being measured.

The fix is to remove both compounding and RNG divergence:

  - Teacher-force. Prefill the KV cache from GROUND-TRUTH latents up to t, then
    predict only frame t+1. No error accumulation.
  - Fix the noise. Pass the identical rng to every step count, so
    `next_latent`'s initial `jax.random.normal` draw is bit-identical and the
    ONLY difference is how many ODE steps integrate from it.
  - Score against the ground-truth latent, not against another sample.

That isolates exactly the question: starting from the same noise and the same
context, does one big jump land where four small ones do, and near the truth?

Timing is deliberately NOT reported here — `latent_rollout` is unjitted and
re-traces every call, so wall time in this script is dominated by XLA compile.
Use bench_stages / bench_warm for speed; this script is only about accuracy.

    python bench/quality_steps.py --checkpoint /ckpt/open-dreamer \
        --steps 1 2 4 8 --trials 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("OD_INFERENCE_REPO", "/root/od-inference")
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--context-frames", type=int, default=16)
    ap.add_argument("--trials", type=int, default=8,
                    help="teacher-forced predictions, each at a different offset")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--rollout-frames", type=int, default=32,
                    help="also write free-running MP4s this long, for eyeballing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/results/quality")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pipeline.checkpointing import DynamicsCheckpointBundle
    from pipeline.generation import DenoiseSchedule, latent_rollout, next_latent
    from pipeline.parallel import build_parallel
    from pipeline.actions import parse_action_dicts, shift_actions
    from pipeline.utils import normalize_latents
    import inference as released

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, _, mesh_rules = build_parallel("data")
    ctx_mgr = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else mesh

    with ctx_mgr:
        bundle = DynamicsCheckpointBundle.from_pretrained(
            args.checkpoint, mesh_rules=mesh_rules,
            model_names={"tokenizer", "dynamics_ema"})
        tokenizer, dynamics = bundle.tokenizer, bundle.dynamics_ema
        cfg = dynamics.cfg

        import glob
        mp4 = sorted(glob.glob(f"{REPO}/samples/vpt/*.mp4"))[0]
        need = args.context_frames + args.trials + args.rollout_frames + 1
        frames_np, fps = released.read_video(Path(mp4), required_frames=need)
        frames = jnp.asarray(frames_np[None])
        raw = released.load_action_dicts(Path(mp4.replace(".mp4", ".jsonl")))
        actions = released.add_batch_dim(parse_action_dicts(raw[:need]))
        actions = shift_actions(actions, cfg.categorical_action_dim)

        # Ground truth latents for the whole clip.
        latents = released.encode_jit(tokenizer, frames)          # (1, need, 512, 16)
        latents_norm = normalize_latents(latents, cfg.latent_mean, cfg.latent_std)
        n_latents = latents.shape[2]
        latent_shape = (1, 1, n_latents, cfg.d_bottleneck)

        # --------------------------------------------------------------
        # Teacher-forced, fixed-noise one-step accuracy
        # --------------------------------------------------------------
        print(f"teacher-forced next-latent prediction, {args.trials} trials, "
              f"context {args.context_frames}\n", flush=True)

        T_ctx = args.context_frames
        emax = int(np.log2(cfg.k_max))
        errs: dict[int, list[float]] = {n: [] for n in args.steps}
        base_var = []

        for trial in range(args.trials):
            lo = trial
            ctx = latents_norm[:, lo: lo + T_ctx]
            target = latents[:, lo + T_ctx]                       # unnormalised truth
            act_ctx = actions[:, lo: lo + T_ctx]
            act_next = actions[:, lo + T_ctx]

            caches = dynamics.create_static_caches(
                batch_size=1, n_latents=n_latents,
                window_size=min(T_ctx + 1, cfg.context_length),
                n_agent=0, dtype=ctx.dtype)
            _, (_, caches) = dynamics(
                act_ctx,
                jnp.full((1, T_ctx), emax, dtype=jnp.int32),
                jnp.full((1, T_ctx), cfg.k_max, dtype=jnp.int32),
                ctx, caches=caches, deterministic=True)

            # SAME key for every step count -> identical starting noise.
            rng = jax.random.PRNGKey(args.seed + trial)

            for n in args.steps:
                sched = DenoiseSchedule.init(n, cfg.k_max)
                pred, _, _, _, _ = next_latent(
                    dynamics, sched, act_next, latent_shape, rng,
                    caches=caches, task_embedding=None)
                err = float(jnp.mean((pred[:, 0] - target) ** 2))
                errs[n].append(err)

            base_var.append(float(jnp.var(target)))
            print(f"  trial {trial}: " + "  ".join(
                f"steps={n} mse={errs[n][-1]:.5f}" for n in args.steps), flush=True)

        var = float(np.mean(base_var))
        print(f"\n{'steps':>6}{'fwd':>5}{'mean MSE':>12}{'NMSE':>9}"
              f"{'vs best':>9}   (ground-truth latent variance {var:.4f})")
        best = min(float(np.mean(errs[n])) for n in args.steps)
        results = {"latent_variance": var, "trials": args.trials, "runs": []}
        for n in args.steps:
            m = float(np.mean(errs[n]))
            row = {"steps": n, "forwards_per_frame": n + 1,
                   "mean_mse": round(m, 6), "nmse": round(m / var, 4),
                   "ratio_vs_best": round(m / best, 3),
                   "per_trial_mse": [round(e, 6) for e in errs[n]]}
            results["runs"].append(row)
            print(f"{n:>6}{n+1:>5}{m:12.5f}{m/var:9.4f}{m/best:9.3f}x")

        print("\nNMSE near 1.0 would mean the prediction is no better than the "
              "latent mean.\nRatio vs best is the number that matters: how much "
              "accuracy a step cut costs.")

        # --------------------------------------------------------------
        # Free-running MP4s, for judging by eye
        # --------------------------------------------------------------
        if args.rollout_frames > 0:
            print(f"\nwriting free-running {args.rollout_frames}-frame MP4s "
                  f"(visual check; metrics above are the quantitative test)",
                  flush=True)
            lat_ctx = latents[:, :T_ctx]
            act_c = actions[:, :T_ctx]
            act_f = actions[:, T_ctx: T_ctx + args.rollout_frames]
            for n in args.steps:
                sched = DenoiseSchedule.init(n, cfg.k_max)
                out = latent_rollout(
                    dynamics=dynamics, policy=act_f, schedule=sched,
                    latents_ctx=lat_ctx, actions_ctx=act_c,
                    num_steps=args.rollout_frames,
                    rng=jax.random.PRNGKey(args.seed),
                    deterministic=True, use_kv_cache=True)
                px = np.asarray(jnp.clip(
                    released.decode_jit(tokenizer, out["latents"]),
                    0, 255).astype(jnp.uint8))[0]
                released.write_video(out_dir / f"steps{n}.mp4", px, fps=fps)
                print(f"  wrote steps{n}.mp4", flush=True)

    (out_dir / "quality_vs_steps.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/quality_vs_steps.json")


if __name__ == "__main__":
    main()
