"""Modal GPU harness for benchmarking Open Dreamer inference.

Benchmarks the model *as released*: the image clones
`reactor-team/open-dreamer` at a pinned commit and imports its `pipeline`
package, so every number here describes the published inference path, not our
fork's copy of it. (The two are byte-identical for models/generation; the
inference repo just adds the policy head.)

RUN IN THIS ORDER — each step gates the next.

    # 1. ~2 min. Does the image build, does JAX see the GPU, and what are this
    #    machine's *measured* ceilings? Everything downstream reports MFU
    #    against these, not against datasheet numbers.
    modal run bench/modal_bench.py::gpu_probe

    # 2. ~10 min, no checkpoint needed. Stage-by-stage breakdown of the
    #    autoregressive frame loop at several batch sizes.
    modal run bench/modal_bench.py::bench_stages

    # 3. Perfetto trace of five generated frames -> results volume.
    modal run bench/modal_bench.py::trace

    # 4. ~7.9 GB from HuggingFace onto the checkpoint volume. Public, ungated.
    modal run bench/modal_bench.py::download_checkpoint

    # 5. The baseline to beat: released inference.py under timing hooks +
    #    a GPU-occupancy sampler, cold vs warm split per stage.
    modal run bench/modal_bench.py::bench_released --horizon 64

    modal volume get open-dreamer-results / ./bench/results

Flip GPU_TYPE to compare hardware. Results land on a volume so they survive
container exit.
"""
from __future__ import annotations

import modal
import re

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
GPU_TYPE = "H100"          # "H100" | "H200" | "A100-80GB" | "B200" | "L40S"
PYTHON_VERSION = "3.11"    # the inference repo pins >=3.11,<3.12

# reactor-team/open-dreamer @ "Rework README into a landing page (#1)".
# Pinned so a benchmark run is reproducible against a known upstream state.
INFERENCE_REPO = "https://github.com/reactor-team/open-dreamer.git"
INFERENCE_SHA = "4f3ab344aaf40fda3ef033b1a4665f92f70bc297"

H = 60 * 60
APP_NAME = "fast-open-dreamer"

RESULTS_VOLUME = modal.Volume.from_name("open-dreamer-results", create_if_missing=True)
CKPT_VOLUME = modal.Volume.from_name("open-dreamer-ckpt", create_if_missing=True)
RESULTS_PATH = "/results"
CKPT_PATH = "/ckpt"

# jax[cuda12] pip wheels bundle the CUDA runtime, so debian_slim is enough —
# no nvidia devel base image needed until we start compiling our own kernels.
IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .env({
        "PYTHONUNBUFFERED": "1",
        # Do not grab the whole GPU at import: the stage benchmarks allocate a
        # ~7 GB model plus KV caches and we want OOM to be visible, not masked.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        # Persistent XLA cache: 30-layer models take ~1 min to compile per shape.
        "JAX_COMPILATION_CACHE_DIR": "/results/xla_cache",
    })
    .apt_install("git", "ffmpeg")
    .pip_install(
        "jax[cuda12]>=0.4.38,<0.5",
        "flax>=0.10.0,<0.11",
        "orbax-checkpoint>=0.8.0,<0.12",
        "optax>=0.2.4,<0.3",
        "omegaconf>=2.3.0,<2.4",
        "hydra-core>=1.3.2,<1.4",
        "einops>=0.8.0,<0.9",
        "numpy<2",
        "imageio[ffmpeg]>=2.31",
        "tqdm>=4.66",
        "huggingface_hub>=0.26",
        "nvidia-ml-py>=12.560",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands(
        f"git clone {INFERENCE_REPO} /root/od-inference",
        f"cd /root/od-inference && git checkout {INFERENCE_SHA}",
    )
    # Our fork supplies the benchmark harness; the released repo supplies the
    # model code under test.
    .add_local_dir(
        ".", "/root/repo",
        ignore=["**/.git", "**/__pycache__", "**/logs", "**/site",
                "**/.venv", "**/*.mp4", "**/*.array_record"],
    )
)

app = modal.App(APP_NAME, image=IMAGE,
                volumes={RESULTS_PATH: RESULTS_VOLUME, CKPT_PATH: CKPT_VOLUME})

# `pipeline` (released inference code) and `bench` (our harness) on the path.
ENV = {
    "PYTHONPATH": "/root/od-inference:/root/repo",
    "OD_MODEL_PKG": "pipeline",
}


def _sh(cmd: str, cwd: str = "/root/repo"):
    import os
    import subprocess
    env = {**os.environ, **ENV}
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)


# ---------------------------------------------------------------------------
# 1. Hardware gate + measured ceilings
# ---------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=20 * 60)
def gpu_probe():
    """Verify JAX sees the GPU and measure this machine's real bf16 GEMM
    throughput and HBM bandwidth. Datasheet peaks overstate what a benchmark
    can actually hit; MFU is only meaningful against a measured ceiling."""
    import json
    import subprocess
    import time

    subprocess.run("nvidia-smi", shell=True, check=True)

    import jax
    import jax.numpy as jnp

    print(f"jax {jax.__version__}  backend {jax.default_backend()}")
    devs = jax.devices()
    print(f"devices: {devs}")
    assert devs[0].platform == "gpu", "no GPU visible to JAX"
    print(f"device_kind: {devs[0].device_kind}")

    # --- bf16 GEMM ceiling ---
    # Time each iteration separately and take the BEST. The obvious loop —
    # enqueue N matmuls, block on the last — under-measures badly: with
    # PREALLOCATE=false, N live 134 MB outputs make the allocator grow the pool
    # inside the timed region, and the GPU has not finished ramping its clocks.
    # That is how an earlier version of this probe reported 407 TFLOP/s on an
    # H100 SXM (true dense bf16 peak ~989), which then made a compute-bound
    # prefill appear to run at 105% MFU.
    n = 8192
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (n, n), jnp.bfloat16)
    b = jax.random.normal(key, (n, n), jnp.bfloat16)
    mm = jax.jit(lambda x, y: x @ y)

    for _ in range(20):                       # warmup: let SM clocks boost
        jax.block_until_ready(mm(a, b))

    best = float("inf")
    for _ in range(30):
        t = time.perf_counter()
        jax.block_until_ready(mm(a, b))
        best = min(best, time.perf_counter() - t)
    tflops = 2 * n ** 3 / best / 1e12
    print(f"measured bf16 GEMM ({n}^3): {tflops:.1f} TFLOP/s "
          f"(best of 30, {best * 1e3:.2f} ms)")

    # --- HBM bandwidth (streaming copy) ---
    elems = 1 << 28                      # 512 MiB of bf16
    src = jax.random.normal(key, (elems,), jnp.bfloat16)
    cp = jax.jit(lambda x: x * 2)
    for _ in range(10):
        jax.block_until_ready(cp(src))
    best_bw = float("inf")
    for _ in range(20):
        t = time.perf_counter()
        jax.block_until_ready(cp(src))
        best_bw = min(best_bw, time.perf_counter() - t)
    dt, iters = best_bw, 1
    gbs = 2 * elems * 2 / dt / 1e9           # read + write
    print(f"measured HBM bandwidth: {gbs:.0f} GB/s")
    print(f"measured ridge point: {tflops * 1e12 / (gbs * 1e9):.0f} FLOP/byte")

    peaks = {"device": devs[0].device_kind, "tflops_bf16": tflops, "gbps": gbs}
    with open(f"{RESULTS_PATH}/measured_peaks.json", "w") as f:
        json.dump(peaks, f, indent=2)
    RESULTS_VOLUME.commit()
    print("\nPass these to the stage benchmark:")
    print(f"  --peak-tflops {tflops:.1f} --peak-bw {gbs:.0f}")


# ---------------------------------------------------------------------------
# 2. Stage-by-stage breakdown (no checkpoint required)
# ---------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=2 * H)
def bench_stages(batch: str = "1 4 16", steps: int = 4, kv_window: int = 192,
                 param_dtype: str = "float32", patches: str = "",
                 attn_impl: str = "", extra: str = ""):
    """Time each unit of the autoregressive frame loop against the released
    `pipeline` package. Random weights: shapes and dtypes fully determine
    performance, so this is a faithful timing of the released architecture.

    `param_dtype` defaults to float32 to match the released configs; run again
    with bfloat16 to measure the weight-traffic half of the gap.

    (The KV window arg is `kv_window`, not `ctx` — Modal's click CLI reserves
    `ctx` and a function taking it fails to launch.)
    """
    import json
    import os

    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"
        print(f"using measured peaks from gpu_probe: {peaks}")
    else:
        print("no measured_peaks.json — run gpu_probe first for MFU numbers")

    tag = f"steps{steps}_ctx{kv_window}_{param_dtype}"
    pf = f"--patches {patches}" if patches else ""
    ai = f"--block-attn-impl {attn_impl}" if attn_impl else ""
    if patches:
        tag += "_" + patches.replace(" ", "+") + (f"_{attn_impl}" if attn_impl else "")
    _sh(f"python bench/profile_inference.py --batch {batch} --steps {steps} "
        f"--ctx {kv_window} --param-dtype {param_dtype} {pf} {ai} {peaks} "
        f"--json {RESULTS_PATH}/stages_{tag}.json {extra}")
    RESULTS_VOLUME.commit()


# ---------------------------------------------------------------------------
# 2b. Ablation ladder — attribute the speedup to individual fixes
# ---------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=4 * H)
def ablate(batch: str = "1 16", steps: int = 4, kv_window: int = 192):
    """Re-run the stage benchmark under a cumulative stack of optimisations so
    each one gets its own number rather than a share of a lump sum.

    `no_roll_kv` and `fast_kv_write` are exact (the harness asserts KV
    equivalence across a ring-buffer wrap before timing anything);
    `bf16_weights` is not, and needs a quality check against real weights
    before it counts as a win.
    """
    import json
    import os

    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"

    stack = [
        ("baseline", [], ""),
        ("+no_remat", ["no_remat"], ""),
        ("+fast_kv_write", ["no_remat", "fast_kv_write"], ""),
        ("+no_roll_kv", ["no_remat", "fast_kv_write", "no_roll_kv"], ""),
        ("+bf16_weights", ["no_remat", "fast_kv_write", "no_roll_kv",
                           "bf16_weights"], ""),
        ("+block_attn(cudnn)", ["no_remat", "fast_kv_write", "no_roll_kv",
                                "bf16_weights", "block_attn"], "cudnn"),
    ]
    for name, patches, impl in stack:
        tag = re.sub(r"[^\w]+", "_", name.strip("+"))
        print("\n" + "#" * 78)
        print(f"# ABLATION: {name}   patches={patches or 'none'}"
              + (f"  sdpa={impl}" if impl else ""))
        print("#" * 78, flush=True)
        pf = f"--patches {' '.join(patches)}" if patches else ""
        ai = f"--block-attn-impl {impl}" if impl else ""
        _sh(f"python bench/profile_inference.py --batch {batch} --steps {steps} "
            f"--ctx {kv_window} {peaks} {pf} {ai} --skip prefill encode "
            f"--json {RESULTS_PATH}/ablate_{tag}.json")
    RESULTS_VOLUME.commit()
    print("\nFetch: modal volume get open-dreamer-results / ./bench/results")


@app.function(gpu=GPU_TYPE, timeout=2 * H)
def bench_attn(batch: str = "1 8", encode_frames: int = 8, param_dtype: str = "bfloat16"):
    """A/B the tokenizer's space attention: dense masked vs block-split.

    Isolates encode + decode, the two stages that carry the (S x S) score
    matrix. The encode stage doubles as the memory proof -- at B=8 with 8
    frames the dense path needs 12.6 GiB of fp32 scores and the split path
    needs none of it.
    """
    import json
    import os
    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"

    # bf16_weights is in both arms so the comparison isolates attention alone.
    # All arms carry bf16_weights so the comparison isolates attention alone.
    #
    # The third arm matters: removing the mask is necessary but not sufficient.
    # `jax.nn.dot_product_attention(implementation=None)` uses the XLA reference
    # path, which materialises the score matrix mask or no mask, so the split
    # alone only buys the 23% of score entries the mask was discarding. cuDNN
    # is what makes it flash attention and drops the matrix entirely.
    for tag, name, pf in (
            ("dense", "dense (as released)", "--patches bf16_weights"),
            ("block_xla", "block_attn, XLA SDPA",
             "--patches bf16_weights block_attn"),
            ("block_cudnn", "block_attn, cuDNN flash SDPA",
             "--patches bf16_weights block_attn --block-attn-impl cudnn")):
        print("\n" + "#" * 78)
        print(f"# TOKENIZER ATTENTION: {name}")
        print("#" * 78, flush=True)
        _sh(f"python bench/profile_inference.py --batch {batch} "
            f"--param-dtype {param_dtype} --encode-frames {encode_frames} {peaks} {pf} "
            f"--skip prefill dyn_fwd ladder frame kv_roll "
            f"--json {RESULTS_PATH}/attn_{tag}.json")
    RESULTS_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=2 * H)
def bench_fp8(batch: str = "1 16", steps: int = 2, cublaslt: bool = False):
    """A/B the dynamics GEMMs: bf16 weights vs e4m3 weights + e4m3 activations.

    Both arms carry the exact patches, so the delta is quantization alone. The
    fp8 arm also dumps a trace, because the whole premise is that XLA folds
    convert+scale+dot into one cublasLt fp8 GEMM -- if it does not, fp8 is
    strictly worse than bf16 (an extra quantize pass in front of the same
    matmul) and the timing would be measuring a regression.
    """
    import json
    import os
    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"

    exact = "no_remat fast_kv_write no_roll_kv block_attn"
    ai = "--block-attn-impl cudnn"
    arms = (("bf16", f"--patches {exact} bf16_weights {ai}", ""),
            ("fp8", f"--patches {exact} bf16_weights fp8_weights {ai}",
             f"--trace {RESULTS_PATH}/trace/fp8"))
    # `--xla_gpu_enable_cublaslt=true` forces cuBLASLt for EVERY GEMM, and the
    # attention einsums hit a shape it rejects:
    #   cublasLtMatmul(...): an unsupported value or parameter was passed
    # which killed the bf16 control arm before fp8 was even exercised. XLA
    # already routes fp8 patterns to cuBLASLt on its own, so the flag is not
    # needed; left behind a default-off switch only for deliberate testing.
    if cublaslt:
        ENV["XLA_FLAGS"] = (ENV.get("XLA_FLAGS", "")
                            + " --xla_gpu_enable_cublaslt=true")

    for tag, pf, tr in arms:
        print("\n" + "#" * 78)
        print(f"# DYNAMICS GEMM: {tag}")
        print("#" * 78, flush=True)
        _sh(f"python bench/profile_inference.py --batch {batch} --steps {steps} "
            f"--param-dtype float32 {pf} {peaks} {tr} "
            f"--skip encode decode kv_roll prefill "
            f"--json {RESULTS_PATH}/fp8_{tag}.json")

    _sh(f"python -c \"import sys; sys.path.insert(0,'/root/repo'); "
        f"from bench import patches; import json; "
        f"print(json.dumps(patches.fp8_gemm_kernels_in_trace("
        f"'{RESULTS_PATH}/trace/fp8'), indent=2))\"")
    ENV.pop("XLA_FLAGS", None)
    RESULTS_VOLUME.commit()


# ---------------------------------------------------------------------------
# 3. Perfetto trace
# ---------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=2 * H)
def trace(batch: int = 1, steps: int = 4, param_dtype: str = "float32",
          patches: str = "", tag: str = ""):
    """Capture a JAX profiler trace of the frame loop.

    Each config writes to its own subdirectory so traces do not overwrite each
    other and baseline vs optimised can be compared side by side.

        modal run bench/modal_bench.py::trace                       # baseline
        modal run bench/modal_bench.py::trace --param-dtype bfloat16 --steps 2

    Then:
        modal volume get open-dreamer-results /trace ./bench/results
        python bench/analyze_trace.py bench/results/trace/<subdir>
    """
    import json
    import os
    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"
    name = tag or f"b{batch}_s{steps}_{param_dtype}" + (
        "_" + patches.replace(" ", "+") if patches else "")
    pf = f"--patches {patches}" if patches else ""
    _sh(f"python bench/profile_inference.py --batch {batch} --steps {steps} "
        f"--param-dtype {param_dtype} {pf} {peaks} "
        f"--iters 5 --skip prefill --trace {RESULTS_PATH}/trace/{name}")
    print(f"\ntrace written to /trace/{name}")
    RESULTS_VOLUME.commit()


# ---------------------------------------------------------------------------
# 4. End-to-end released inference (needs weights)
# ---------------------------------------------------------------------------

HF_REPO = "reactor-team/open-dreamer"
HF_STEP = "250000"          # the only step published; contains dynamics_ema + tokenizer
CKPT_DIR = f"{CKPT_PATH}/open-dreamer"


@app.function(timeout=2 * H)
def download_checkpoint(repo: str = HF_REPO, revision: str = "main"):
    """Pull the published weights onto the checkpoint volume. ~7.9 GB, public
    and ungated, so no HF token is needed.

    Layout on the volume ends up as /ckpt/open-dreamer/250000/{dynamics_ema,
    tokenizer}/... — an Orbax OCDBT checkpoint written by a 4-process job.
    Note there is no `dynamics` (online) item, only `dynamics_ema`, so
    inference must run with --use_ema.
    """
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = snapshot_download(repo_id=repo, revision=revision,
                             local_dir=CKPT_DIR, max_workers=16)
    print(f"downloaded to {path}")
    _sh(f"du -sh {CKPT_DIR} && find {CKPT_DIR} -maxdepth 2 | head -20", cwd="/")
    CKPT_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=4 * H)
def bench_released(context_frames: int = 16, horizon: int = 64, num_steps: int = 4,
                   checkpoint: str = CKPT_DIR,
                   use_ema: bool = True, block: bool = True):
    """Run the released `inference.py` under timing hooks + a GPU sampler.

    This is the number the fork has to beat. The hooks patch `pipeline.*` in
    place, so the profile describes the shipped code path, split into
    ckpt_load / encode / latent_rollout / next_latent / decode with cold vs
    warm separated.

    `block=False` drops the JAX barriers for a truthful end-to-end wall time
    at the cost of per-stage attribution.
    """
    import glob
    import os
    import shutil

    # Orbax's CheckpointManager takes the directory that *contains* numbered
    # step subdirectories and calls latest_step() on it — pointing it at the
    # step dir itself yields "No checkpoint found". Fail here with the fix
    # rather than 40 s later inside the restore.
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Run first:\n"
            f"  modal run bench/modal_bench.py::download_checkpoint")
    steps = [d for d in os.listdir(checkpoint)
             if d.isdigit() and os.path.isdir(os.path.join(checkpoint, d))]
    if not steps:
        raise SystemExit(
            f"{checkpoint} contains no numbered step directory "
            f"(found: {sorted(os.listdir(checkpoint))[:8]}).\n"
            f"Pass the PARENT of the step dir, e.g. {CKPT_DIR} not {CKPT_DIR}/{HF_STEP}.")
    print(f"checkpoint {checkpoint} -> steps {sorted(steps)}", flush=True)

    # Sample VPT clip + actions, fetched by the released downloader.
    _sh("python download_vpt_sample.py --overwrite", cwd="/root/od-inference")
    mp4 = glob.glob("/root/od-inference/samples/vpt/*.mp4")[0]
    jsonl = mp4.replace(".mp4", ".jsonl")

    ema = "--use_ema" if use_ema else ""
    out = f"{RESULTS_PATH}/released_h{horizon}.mp4"
    tag = f"h{horizon}_ctx{context_frames}_s{num_steps}_{GPU_TYPE}"

    env_extra = {
        "BENCH_TIMING_JSONL": f"{RESULTS_PATH}/timing_{tag}.jsonl",
        "BENCH_GPU_CSV": f"{RESULTS_PATH}/gpu_{tag}.csv",
        "BENCH_OUT_DIR": RESULTS_PATH,
        "OD_INFERENCE_REPO": "/root/od-inference",
    }
    if not block:
        env_extra["BENCH_NO_BLOCK"] = "1"
    ENV.update(env_extra)

    # Fresh event log per run, else cold/warm classification is meaningless.
    for p in (env_extra["BENCH_TIMING_JSONL"], env_extra["BENCH_GPU_CSV"]):
        if os.path.exists(p):
            os.remove(p)

    _sh(f"python /root/repo/bench/run_released.py -- "
        f"--checkpoint_path {checkpoint} "
        f"--input_mp4 {mp4} --actions_path {jsonl} --output_mp4 {out} "
        f"--context_frames {context_frames} --horizon {horizon} "
        f"--num_steps {num_steps} {ema}", cwd="/root/repo")

    for name in ("released_profile.json", "released_profile.md"):
        src = f"{RESULTS_PATH}/{name}"
        if os.path.exists(src):
            shutil.copy(src, f"{RESULTS_PATH}/{name.replace('.', f'_{tag}.')}")
    RESULTS_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=4 * H)
def bench_warm(context_frames: int = 16, horizon: int = 64, num_steps: int = 4,
               repeats: int = 1, checkpoint: str = CKPT_DIR,
               jit: bool = True, decode: bool = True):
    """Steady-state rollout timing with the real weights — the honest baseline.

    `bench_released` measures a one-shot CLI: 32 s of checkpoint load and XLA
    compile around a few seconds of generation, and its `next_latent` hook sees
    only the trace-time call because the rollout runs inside `lax.scan`. This
    loads once and then times repeated rollouts, separating:

      - cold call (trace + compile) from warm calls
      - the released unjitted path (which re-traces `lax.scan` every call)
        from the same rollout under `jax.jit`
      - dynamics rollout from tokenizer decode

    GPU utilisation is windowed to each timed call, so the checkpoint load no
    longer drags the mean toward zero.
    """
    import os

    if not os.path.exists(checkpoint):
        raise SystemExit("run download_checkpoint first")

    _sh("python download_vpt_sample.py --overwrite", cwd="/root/od-inference")

    tag = f"h{horizon}_ctx{context_frames}_s{num_steps}_{GPU_TYPE}"
    ENV["BENCH_GPU_CSV"] = f"{RESULTS_PATH}/gpu_warm_{tag}.csv"
    ENV["OD_INFERENCE_REPO"] = "/root/od-inference"

    flags = ("--jit " if jit else "") + ("--decode " if decode else "")
    _sh(f"python bench/warm_rollout.py --checkpoint {checkpoint} "
        f"--context-frames {context_frames} --horizon {horizon} "
        f"--num-steps {num_steps} --repeats {repeats} {flags}"
        f"--out {RESULTS_PATH}/warm_rollout_{tag}.json")
    RESULTS_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=4 * H)
def quality_steps(steps: str = "1 2 4 8", trials: int = 8,
                  rollout_frames: int = 32, context_frames: int = 16,
                  checkpoint: str = CKPT_DIR):
    """The gate on the step-count speedup: same context, same actions, same
    seed, rolled out at each step count and compared against a high-step
    reference. Writes MP4s plus latent NMSE / pixel PSNR.

    Run this before treating 1-step as a real 3.2x — speed without quality is
    not a speedup.
    """
    import os
    if not os.path.exists(checkpoint):
        raise SystemExit("run download_checkpoint first")
    _sh("python download_vpt_sample.py --overwrite", cwd="/root/od-inference")
    ENV["OD_INFERENCE_REPO"] = "/root/od-inference"
    _sh(f"python bench/quality_steps.py --checkpoint {checkpoint} "
        f"--steps {steps} --trials {trials} "
        f"--rollout-frames {rollout_frames} --context-frames {context_frames} "
        f"--out-dir {RESULTS_PATH}/quality")
    RESULTS_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=4 * H)
def quality_rollout(steps: str = "1 2 4", seeds: str = "0 1", windows: int = 8,
                    horizon: int = 96, context_frames: int = 16,
                    checkpoint: str = CKPT_DIR):
    """Long-horizon drift + FVD. The compounding test that one-step accuracy
    cannot cover: few-step samplers fail by accumulating error, not by being
    wrong on any single step.

    Uses the training repo's own FVD implementation and its committed I3D
    weights (dreamer/fvd/i3d_pretrained_400.npz), so no external download.
    """
    import os
    if not os.path.exists(checkpoint):
        raise SystemExit("run download_checkpoint first")
    _sh("python download_vpt_sample.py --overwrite", cwd="/root/od-inference")
    ENV["OD_INFERENCE_REPO"] = "/root/od-inference"
    ENV["OD_FORK_REPO"] = "/root/repo"
    _sh(f"python bench/quality_rollout.py --checkpoint {checkpoint} "
        f"--steps {steps} --seeds {seeds} --windows {windows} --horizon {horizon} "
        f"--context-frames {context_frames} "
        f"--out-dir {RESULTS_PATH}/quality_rollout")
    RESULTS_VOLUME.commit()


@app.function(gpu=GPU_TYPE, timeout=2 * H)
def dump_hlo(batch: int = 1, steps: int = 2, param_dtype: str = "bfloat16",
             patches: str = "no_remat fast_kv_write no_roll_kv block_attn"):
    """Dump post-optimization HLO alongside a trace, so opaque fusion kernel
    names can be resolved to the ops and source lines they came from.

    The profiler's event args carry only occupancy and correlation ids -- no
    HLO link -- so the mapping has to come from XLA itself.
    """
    import json
    import os
    peaks = ""
    p = f"{RESULTS_PATH}/measured_peaks.json"
    if os.path.exists(p):
        d = json.load(open(p))
        peaks = f"--peak-tflops {d['tflops_bf16']:.1f} --peak-bw {d['gbps']:.0f}"
    out = f"{RESULTS_PATH}/hlo"
    _sh(f"rm -rf {out} && mkdir -p {out}", cwd="/")
    ENV["XLA_FLAGS"] = (f"--xla_dump_to={out} "
                        "--xla_dump_hlo_as_text "
                        "--xla_dump_hlo_module_re=.*frame.*")
    pf = f"--patches {patches}" if patches else ""
    _sh(f"python bench/profile_inference.py --batch {batch} --steps {steps} "
        f"--param-dtype {param_dtype} {pf} {peaks} --iters 3 "
        f"--skip prefill dyn_fwd decode kv_roll encode "
        f"--trace {RESULTS_PATH}/trace/hlo_run")
    ENV.pop("XLA_FLAGS", None)
    _sh(f"ls -la {out} | head -30 && du -sh {out}", cwd="/")
    RESULTS_VOLUME.commit()


# ---------------------------------------------------------------------------
# generic runner, for the edit -> run loop
# ---------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=4 * H)
def run(cmd: str, cwd: str = "/root/repo"):
    """Any shell command in the image. `modal run ... ::run --cmd "..."`."""
    _sh(cmd, cwd=cwd)
    RESULTS_VOLUME.commit()


@app.function(timeout=H)
def ls_ckpt():
    """Show what is on the checkpoint volume."""
    _sh(f"ls -laR {CKPT_PATH} | head -100", cwd="/")
