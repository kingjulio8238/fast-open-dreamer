"""Long-horizon quality: does a shorter tau-ladder drift?

`quality_steps.py` answered the one-step question — starting from identical
noise and ground-truth context, steps=1 predicts the next latent as accurately
as steps=8 (differences under the n=8 noise floor). That is necessary but not
sufficient. Few-step diffusion samplers characteristically fail by *compounding*:
per-step error too small to measure, visible after 200 autoregressive frames as
drift, over-saturation, or a frozen scene.

This measures the compounding directly:

  drift        per-frame latent MSE against ground truth as the rollout runs.
               Rising is expected (the future is stochastic); the shape matters.
  collapse     generated latent std / ground-truth latent std per frame. Falling
               toward 0 means the model is converging to a fixed point — the
               classic autoregressive death. Rising past ~1.2 means blow-up.
  motion       mean |z_t - z_{t-1}|. Flat-lining means a frozen scene, which
               drift alone will not reveal.
  FVD          Frechet Video Distance on I3D features, using the training
               repo's own implementation and committed weights. Computed as
               FVD(ground-truth-decoded, generated) so it isolates the
               dynamics model rather than re-measuring the tokenizer.

Windows are batched, so B = --windows and the rollouts run concurrently.

    python bench/quality_rollout.py --checkpoint /ckpt/open-dreamer \
        --steps 1 4 --windows 8 --horizon 96
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("OD_INFERENCE_REPO", "/root/od-inference")
FORK = os.environ.get("OD_FORK_REPO", "/root/repo")
sys.path.insert(0, REPO)
sys.path.insert(0, FORK)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--context-frames", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--windows", type=int, default=8,
                    help="context windows sampled from the clip; becomes the batch")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--fvd-chunk", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=16,
                    help="upper bound; auto-reduced to fit the attention score matrix")
    ap.add_argument("--encode-chunk", type=int, default=16,
                    help="upper bound; auto-reduced to fit the attention score matrix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="repeat each step count at these seeds. With one seed a "
                         "step-count difference has no error bar; two or three "
                         "give the seed-to-seed spread it must beat to be real.")
    ap.add_argument("--out-dir", default="/results/quality_rollout")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pipeline.checkpointing import DynamicsCheckpointBundle
    from pipeline.generation import DenoiseSchedule, latent_rollout
    from pipeline.parallel import build_parallel
    from pipeline.actions import parse_action_dicts, shift_actions
    import inference as released

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, _, mesh_rules = build_parallel("data")
    ctx_mgr = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else mesh

    T_ctx, Hz, W = args.context_frames, args.horizon, args.windows
    span = T_ctx + Hz
    need = span * W

    with ctx_mgr:
        bundle = DynamicsCheckpointBundle.from_pretrained(
            args.checkpoint, mesh_rules=mesh_rules,
            model_names={"tokenizer", "dynamics_ema"})
        tokenizer, dynamics = bundle.tokenizer, bundle.dynamics_ema
        cfg = dynamics.cfg

        import glob
        mp4 = sorted(glob.glob(f"{REPO}/samples/vpt/*.mp4"))[0]
        print(f"reading {need} frames for {W} non-overlapping windows ...", flush=True)
        frames_np, fps = released.read_video(Path(mp4), required_frames=need)
        raw = released.load_action_dicts(Path(mp4.replace(".mp4", ".jsonl")))

        # Non-overlapping windows -> batch dimension.
        frames_b = np.stack([frames_np[i * span:(i + 1) * span] for i in range(W)])
        acts = []
        for i in range(W):
            a = parse_action_dicts(raw[i * span:(i + 1) * span])
            acts.append(jax.tree.map(lambda x: None if x is None else jnp.asarray(x), a))
        actions = jax.tree.map(lambda *xs: jnp.stack(xs), *acts)
        actions = shift_actions(actions, cfg.categorical_action_dim)

        # --- chunk sizing, forced by materialised attention -----------------
        # The tokenizer passes an explicit boolean space mask, which sends
        # `jax.nn.dot_product_attention` down the XLA path: it materialises the
        # full (B*T, heads, S, S) score matrix instead of using a fused/flash
        # kernel. With S=1432 that is 197 MB per (batch, frame) slice in the
        # encoder. Encoding 8 windows x 32 frames at once asks for
        #   8*32 * 24 * 1432^2 * 4 = 50,396,135,424 bytes = 46.9 GiB
        # which is exactly the allocation that OOMed an 80 GB H100. Flash
        # attention would make this O(1) in S; until then, chunk to fit.
        S_tok = tokenizer.decoder.n_latents + \
            (tokenizer.decoder.H // tokenizer.decoder.patch_size) * \
            (tokenizer.decoder.W // tokenizer.decoder.patch_size)

        def _chunk(heads, B, budget_bytes=8e9):
            per_bt = heads * S_tok * S_tok * 4          # fp32 softmax scores
            return max(1, int(budget_bytes / (per_bt * max(B, 1))))

        enc_chunk = min(args.encode_chunk, _chunk(tokenizer.encoder.cfg.n_heads, W))
        dec_chunk = min(args.decode_chunk, _chunk(tokenizer.decoder.n_heads, W))
        print(f"attention score matrix S={S_tok}: "
              f"encode chunk {enc_chunk} frames, decode chunk {dec_chunk} frames "
              f"(B={W})", flush=True)

        lat_chunks = [released.encode_jit(tokenizer, jnp.asarray(frames_b[:, s:s + enc_chunk]))
                      for s in range(0, span, enc_chunk)]
        gt_lat = jnp.concatenate(lat_chunks, axis=1)          # (W, span, 512, 16)
        print(f"ground-truth latents {gt_lat.shape}", flush=True)

        def decode_chunked(lat):
            outs = []
            for s in range(0, lat.shape[1], dec_chunk):
                px = released.decode_jit(tokenizer, lat[:, s:s + dec_chunk])
                outs.append(np.asarray(jnp.clip(px, 0, 255).astype(jnp.uint8)))
            return np.concatenate(outs, axis=1)

        gt_px = decode_chunked(gt_lat[:, T_ctx:])             # tokenizer ceiling
        released.write_video(out_dir / "rollout_groundtruth.mp4", gt_px[0], fps=fps)

        gt_tail = gt_lat[:, T_ctx:]
        gt_std = float(jnp.std(gt_tail))
        gt_motion = float(jnp.mean(jnp.abs(gt_tail[:, 1:] - gt_tail[:, :-1])))
        print(f"ground truth: latent std {gt_std:.4f}, motion {gt_motion:.5f}\n", flush=True)

        results = {"windows": W, "horizon": Hz, "context_frames": T_ctx,
                   "gt_latent_std": gt_std, "gt_motion": gt_motion, "runs": []}

        # FVD setup (training repo's implementation + committed I3D weights).
        try:
            from dreamer.fvd import frechet_distance, get_fvd_logits, load_i3d_pretrained
            i3d = load_i3d_pretrained()
            have_fvd = True
        except Exception as exc:
            print(f"FVD unavailable ({exc!r}); reporting drift metrics only", flush=True)
            have_fvd = False

        def fvd_feats(px):
            """px: (B, T, H, W, C) uint8 -> I3D logits over fvd_chunk clips."""
            B, T = px.shape[:2]
            n = (T // args.fvd_chunk) * args.fvd_chunk
            clips = px[:, :n].reshape(B * (n // args.fvd_chunk),
                                      args.fvd_chunk, *px.shape[2:])
            return get_fvd_logits(clips, i3d, bs=8)

        gt_feats = fvd_feats(gt_px) if have_fvd else None

        for n, seed in [(n, s) for n in args.steps for s in args.seeds]:
            sched = DenoiseSchedule.init(n, cfg.k_max)
            out = latent_rollout(
                dynamics=dynamics, policy=actions[:, T_ctx:span],
                schedule=sched, latents_ctx=gt_lat[:, :T_ctx],
                actions_ctx=actions[:, :T_ctx], num_steps=Hz,
                rng=jax.random.PRNGKey(seed),
                deterministic=True, use_kv_cache=True)
            pred = jax.block_until_ready(out["latents"])[:, T_ctx:]   # (W, Hz, 512, 16)

            drift = [float(jnp.mean((pred[:, t] - gt_tail[:, t]) ** 2)) for t in range(Hz)]
            std_r = [float(jnp.std(pred[:, t]) / (jnp.std(gt_tail[:, t]) + 1e-8))
                     for t in range(Hz)]
            motion = [float(jnp.mean(jnp.abs(pred[:, t] - pred[:, t - 1])))
                      for t in range(1, Hz)]

            px = decode_chunked(pred)
            if seed == args.seeds[0]:
                released.write_video(out_dir / f"rollout_steps{n}.mp4", px[0], fps=fps)

            row = {"steps": n, "seed": seed, "forwards_per_frame": n + 1,
                   "drift_first8": round(float(np.mean(drift[:8])), 6),
                   "drift_last8": round(float(np.mean(drift[-8:])), 6),
                   "drift_growth": round(float(np.mean(drift[-8:]) / max(np.mean(drift[:8]), 1e-12)), 2),
                   "std_ratio_first8": round(float(np.mean(std_r[:8])), 4),
                   "std_ratio_last8": round(float(np.mean(std_r[-8:])), 4),
                   "motion_first8": round(float(np.mean(motion[:8])), 6),
                   "motion_last8": round(float(np.mean(motion[-8:])), 6),
                   "motion_vs_gt_last8": round(float(np.mean(motion[-8:]) / max(gt_motion, 1e-12)), 3),
                   "drift_curve": [round(x, 6) for x in drift],
                   "std_ratio_curve": [round(x, 4) for x in std_r]}

            if have_fvd:
                row["fvd_vs_gt_decoded"] = round(float(
                    frechet_distance(fvd_feats(px), gt_feats)), 2)

            results["runs"].append(row)
            print(f"  steps={n} seed={seed} ({n+1} fwd): "
                  f"drift {row['drift_first8']:.5f} -> {row['drift_last8']:.5f} "
                  f"({row['drift_growth']}x)   "
                  f"std ratio {row['std_ratio_first8']:.3f} -> {row['std_ratio_last8']:.3f}   "
                  f"motion vs gt {row['motion_vs_gt_last8']:.2f}"
                  + (f"   FVD {row['fvd_vs_gt_decoded']}" if have_fvd else ""), flush=True)

    # Seed spread is the yardstick: a step-count gap smaller than it is noise.
    if len(args.seeds) > 1 and any("fvd_vs_gt_decoded" in r for r in results["runs"]):
        print("\nFVD by step count (mean [min, max] over seeds):")
        for n in args.steps:
            f = [r["fvd_vs_gt_decoded"] for r in results["runs"]
                 if r["steps"] == n and "fvd_vs_gt_decoded" in r]
            if f:
                print(f"  steps={n}: {sum(f)/len(f):7.1f}  [{min(f):.1f}, {max(f):.1f}]"
                      f"  spread {max(f)-min(f):.1f}")
        print("A step-count difference must exceed the seed spread to count.")

    (out_dir / "quality_rollout.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/quality_rollout.json + rollout_steps*.mp4")
    print("Read: std ratio falling toward 0 = collapse; motion vs gt near 0 = "
          "frozen scene; FVD lower is better.")


if __name__ == "__main__":
    main()
