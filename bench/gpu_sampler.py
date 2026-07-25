"""1 Hz nvidia-smi poller. Writes CSV to BENCH_GPU_CSV.

Adapted from the DreamDojo bench harness. The reason this matters for Open
Dreamer specifically: the frame loop runs ~1000+ tiny kernels per frame at
batch 1, so the failure mode is a GPU that is *busy but idle-ish* — high wall
time with low `utilization.gpu`. That gap is invisible to any timer that only
measures end-to-end latency, and it is the signature of launch-bound execution
(the case for CUDA graphs / XLA command buffers).

Daemon thread — exits silently when the process exits. Flushes after every
sample so partial data survives a crash or OOM kill.
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time

CSV_PATH = os.environ.get("BENCH_GPU_CSV", "/results/bench_gpu.csv")
INTERVAL_SECONDS = float(os.environ.get("BENCH_GPU_SAMPLE_S", "0.25"))

_started = False
_started_lock = threading.Lock()
_T0 = time.perf_counter()   # shared origin for the CSV clock and `now()`


_nvml_handle = None


def _init_nvml():
    """Prefer NVML over shelling out to nvidia-smi.

    Not a micro-optimisation: `subprocess` forks, and forking a multithreaded
    JAX process is a documented deadlock risk — JAX itself warns about it
    ("os.fork() is incompatible with multithreaded code"). A sampler that hangs
    the benchmark it is measuring is worse than no sampler. NVML is an in-process
    library call, so nothing forks.
    """
    global _nvml_handle
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception:
        _nvml_handle = None
        return False


def _sample() -> tuple[int | None, int | None, int | None, int | None]:
    """(memory_used_mb, memory_total_mb, gpu_util_pct, sm_clock_mhz)."""
    if _nvml_handle is not None:
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            clk = pynvml.nvmlDeviceGetClockInfo(_nvml_handle, pynvml.NVML_CLOCK_SM)
            return mem.used // (1 << 20), mem.total // (1 << 20), util.gpu, clk
        except Exception:
            return None, None, None, None

    # Fallback only when NVML is unavailable. Accepts the fork risk because the
    # alternative is no data at all.
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if not out:
            return None, None, None, None
        return tuple(int(x.strip()) for x in out.splitlines()[0].split(","))  # type: ignore
    except Exception:
        return None, None, None, None


def _loop() -> None:
    _init_nvml()
    try:
        d = os.path.dirname(CSV_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        f = open(CSV_PATH, "w", newline="")
    except Exception:
        return
    w = csv.writer(f)
    w.writerow(["wall_seconds", "memory_used_mb", "memory_total_mb",
                "gpu_util_pct", "sm_clock_mhz"])
    f.flush()
    try:
        while True:
            used, total, util, clk = _sample()
            if used is not None:
                w.writerow([round(time.perf_counter() - _T0, 3), used, total, util, clk])
                f.flush()
            time.sleep(INTERVAL_SECONDS)
    except Exception:
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass


def start() -> None:
    """Idempotent — calling more than once is a no-op."""
    global _started
    with _started_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, daemon=True, name="gpu-sampler").start()


def now() -> float:
    """Sampler-clock timestamp, for windowing `summarize`."""
    return time.perf_counter() - _T0


def summarize(csv_path: str | None = None,
              window: tuple[float, float] | None = None) -> dict:
    """Reduce the CSV to the numbers worth putting in a report.

    `window` restricts to (start, end) seconds on the sampler clock. Use it —
    an unwindowed mean over a run that spent 32 s loading a checkpoint reports
    ~5% utilisation and tells you nothing about the model.
    """
    path = csv_path or CSV_PATH
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({k: int(v) if v and v.isdigit() else float(v or 0)
                             for k, v in r.items()})
    except Exception:
        return {}
    if window is not None:
        lo, hi = window
        rows = [r for r in rows if lo <= r["wall_seconds"] <= hi]
    if not rows:
        return {}
    utils = [r["gpu_util_pct"] for r in rows]
    mems = [r["memory_used_mb"] for r in rows]
    busy = [r for r in rows if r["gpu_util_pct"] > 5]
    return {
        "samples": len(rows),
        "gpu_util_mean_pct": round(sum(utils) / len(utils), 1),
        "gpu_util_peak_pct": max(utils),
        "memory_peak_mb": max(mems),
        "memory_total_mb": rows[0]["memory_total_mb"],
        "busy_seconds": round(len(busy) * INTERVAL_SECONDS, 2),
        "window": list(window) if window else None,
        # <100% while the benchmark is clearly running means launch-bound, not
        # compute-bound: the GPU is finishing each kernel faster than the host
        # can enqueue the next.
        "launch_bound_hint": max(utils) < 90,
    }


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(summarize(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
