"""Per-stage timing hooks for the *released* Open Dreamer inference path.

Import this before `inference.py` runs and it monkeypatches the real functions
in `pipeline.*`, so the numbers describe the shipped code rather than a
reimplementation of it. Same trick as the DreamDojo bench harness: patch first,
then let the user script import the already-patched names out of sys.modules.

JAX-specific caveat, and the reason this file is not a copy of the torch one:
**JAX dispatch is asynchronous.** Wrapping a call in `time.perf_counter()` with
no barrier measures how long it took to *enqueue* work, not to do it. Every
hook here therefore calls `jax.block_until_ready` on the result before stopping
its timer. That serialises the pipeline, so per-stage times sum to slightly
more than an unhooked end-to-end run — which is the correct trade for phase
attribution. Set `BENCH_NO_BLOCK=1` to measure the unhooked wall time instead.

Stages emitted to BENCH_TIMING_JSONL (one JSON object per line):
  python_imports     importing jax + the pipeline stack
  ckpt_load          DynamicsCheckpointBundle.from_pretrained (host + device)
  encode             tokenizer encode of the context clip
  prefill            the one dynamics forward over T_ctx context frames
  next_latent        ONE generated frame's tau-ladder (num_steps + 1 forwards).
                     CAVEAT: `latent_rollout` drives generation with
                     `jax.lax.scan`, so this fires ONCE, at trace time,
                     whatever the horizon. Treat its duration as tracing cost,
                     not per-frame cost, and use bench/warm_rollout.py or
                     bench/profile_inference.py for the real per-frame number.
  latent_rollout     the whole autoregressive rollout
  decode             each tokenizer-decoder call

Best-effort: a hook whose target has moved emits `hook_install_error` and the
run continues with that stage missing.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time

JSONL_PATH = os.environ.get("BENCH_TIMING_JSONL", "/results/bench_timing.jsonl")
BLOCK = os.environ.get("BENCH_NO_BLOCK", "") != "1"
_lock = threading.Lock()
_t_start = time.perf_counter()
_counters: dict[str, int] = {}

INSTALLED: list[str] = []
FAILED: list[str] = []


def _emit(stage: str, phase: str, **extra) -> None:
    rec = {
        "stage": stage,
        "phase": phase,
        "wall_seconds_since_start": round(time.perf_counter() - _t_start, 6),
        "pid": os.getpid(),
        **extra,
    }
    with _lock:
        try:
            d = os.path.dirname(JSONL_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # telemetry must never break inference


def _block(result):
    if not BLOCK:
        return result
    try:
        import jax
        jax.block_until_ready(result)
    except Exception:
        pass
    return result


def _timed(fn, stage: str):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        n = _counters.get(stage, 0)
        _counters[stage] = n + 1
        _emit(stage, "start", call_index=n)
        t = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            _block(out)
        except Exception as exc:
            _emit(stage, "error", call_index=n,
                  duration_seconds=round(time.perf_counter() - t, 6),
                  error=repr(exc)[:300])
            raise
        _emit(stage, "end", call_index=n,
              duration_seconds=round(time.perf_counter() - t, 6),
              # first call carries compile + autotune; everything after is warm
              cold=(n == 0))
        return out
    return wrapper


def _patch_attr(mod, name: str, stage: str):
    try:
        target = getattr(mod, name)
        setattr(mod, name, _timed(target, stage))
        INSTALLED.append(f"{mod.__name__}.{name}")
    except Exception as exc:
        FAILED.append(f"{mod.__name__}.{name}: {exc!r}")
        _emit("hook_install_error", "error", target=f"{name}", error=repr(exc)[:300])


def _patch_classmethod(cls, name: str, stage: str):
    try:
        target = getattr(cls, name)
        raw = target.__func__ if hasattr(target, "__func__") else target
        setattr(cls, name, classmethod(_timed(raw, stage)))
        INSTALLED.append(f"{cls.__name__}.{name}")
    except Exception as exc:
        FAILED.append(f"{cls.__name__}.{name}: {exc!r}")
        _emit("hook_install_error", "error", target=name, error=repr(exc)[:300])


def install(pkg: str = "pipeline") -> None:
    """Patch the inference path. Call before importing `inference`."""
    t_imp = time.perf_counter()
    _emit("python_imports", "start")

    import importlib
    generation = importlib.import_module(f"{pkg}.generation")
    checkpointing = importlib.import_module(f"{pkg}.checkpointing")
    try:
        sampler = importlib.import_module(f"{pkg}.sampler")
    except Exception:
        sampler = None

    _emit("python_imports", "end",
          duration_seconds=round(time.perf_counter() - t_imp, 6), cold=True)

    # The tau-ladder for one frame. This is the unit that has to get faster:
    # num_steps denoising passes + one cache-commit pass through the 1.6B model.
    _patch_attr(generation, "next_latent", "next_latent")
    _patch_attr(generation, "latent_rollout", "latent_rollout")
    if hasattr(generation, "next_frame"):
        _patch_attr(generation, "next_frame", "next_frame")

    _patch_classmethod(checkpointing.DynamicsCheckpointBundle,
                       "from_pretrained", "ckpt_load")

    if sampler is not None:
        for fname in ("encode_jit", "decode_jit"):
            if hasattr(sampler, fname):
                _patch_attr(sampler, fname, fname.replace("_jit", ""))

    # inference.py defines its own encode_jit/decode_jit at module scope, so
    # patch those too once it is importable.
    try:
        import sys
        if "inference" in sys.modules:
            inf = sys.modules["inference"]
            for fname in ("encode_jit", "decode_jit"):
                if hasattr(inf, fname):
                    _patch_attr(inf, fname, fname.replace("_jit", ""))
    except Exception:
        pass

    _emit("hooks_installed", "info", installed=INSTALLED, failed=FAILED)
    print(f"[timing_hooks] installed {len(INSTALLED)}, failed {len(FAILED)}"
          f"{' -> ' + str(FAILED) if FAILED else ''}", flush=True)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------

def summarize(jsonl_path: str | None = None) -> dict:
    """Fold the event stream into per-stage totals, split cold vs warm."""
    path = jsonl_path or JSONL_PATH
    events = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except FileNotFoundError:
        return {}

    stages: dict[str, dict] = {}
    for e in events:
        if e.get("phase") != "end":
            continue
        s = stages.setdefault(e["stage"], {"cold": [], "warm": []})
        s["cold" if e.get("cold") else "warm"].append(e["duration_seconds"])

    def stat(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {
            "count": len(xs),
            "total_s": round(sum(xs), 4),
            "mean_ms": round(sum(xs) / len(xs) * 1e3, 3),
            "p50_ms": round(xs[len(xs) // 2] * 1e3, 3),
            "p99_ms": round(xs[min(len(xs) - 1, int(len(xs) * 0.99))] * 1e3, 3),
            "min_ms": round(xs[0] * 1e3, 3),
            "max_ms": round(xs[-1] * 1e3, 3),
        }

    return {
        "events": len(events),
        "blocking": BLOCK,
        "stages": {k: {"cold": stat(v["cold"]), "warm": stat(v["warm"])}
                   for k, v in stages.items()},
    }


def report(jsonl_path: str | None = None, gpu_csv: str | None = None) -> str:
    """Markdown summary: per-stage cold/warm table + GPU occupancy."""
    s = summarize(jsonl_path)
    if not s:
        return "no timing events recorded"
    lines = ["# Open Dreamer released-inference profile", "",
             f"blocking barriers: {s['blocking']} "
             f"({'per-stage attribution' if s['blocking'] else 'raw wall time'})", "",
             "| stage | n(warm) | warm p50 ms | warm p99 ms | cold ms |",
             "|---|---:|---:|---:|---:|"]
    for name, v in sorted(s["stages"].items()):
        w, c = v["warm"], v["cold"]
        lines.append(
            f"| {name} | {w['count'] if w else 0} | "
            f"{w['p50_ms'] if w else '-'} | {w['p99_ms'] if w else '-'} | "
            f"{c['mean_ms'] if c else '-'} |")

    nl = s["stages"].get("next_latent", {}).get("warm")
    if nl:
        lines += ["", f"**Per generated frame (dynamics only): {nl['p50_ms']:.1f} ms p50 "
                      f"-> {1000 / nl['p50_ms']:.1f} fps ceiling before decode.**"]

    try:
        from bench import gpu_sampler
        g = gpu_sampler.summarize(gpu_csv)
        if g:
            lines += ["", "## GPU", "",
                      f"- peak memory: {g['memory_peak_mb']} / {g['memory_total_mb']} MB",
                      f"- util mean / peak: {g['gpu_util_mean_pct']}% / {g['gpu_util_peak_pct']}%",
                      f"- busy seconds (util>5%): {g['busy_seconds']}"]
            if g.get("launch_bound_hint"):
                lines.append("- **peak util < 90% -> launch-bound, not compute-bound. "
                             "CUDA graphs / XLA command buffers are on the table.**")
    except Exception:
        pass
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(report(*(sys.argv[1:3] or [None, None])[:2]))
