"""Answer one question from a JAX profiler trace: launch-bound or kernel-bound?

At batch 1 the dynamics forward runs ~30 layers x ~8 kernels x 5 passes per
frame while reaching only ~26% of HBM bandwidth. Two very different causes look
identical from the outside:

  launch-bound   kernels are fine, but the GPU sits idle between them waiting
                 for the host to enqueue the next one. Fix: CUDA graphs / XLA
                 command buffers. Symptom: large summed gaps, short kernels.

  kernel-bound   no idle time; the kernels themselves move more bytes or use
                 the tensor cores worse than they should. Fix: reshape the work
                 (fused attention, better GEMM shapes, fp8). Symptom: near-zero
                 gaps, time concentrated in a few named kernels.

This measures GPU-stream occupancy directly: total kernel time vs the wall span
they cover, plus where the time goes by kernel name.

    python bench/analyze_trace.py bench/results/trace
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def find_trace(root: Path) -> Path:
    """JAX writes plugins/profile/<run>/<host>.trace.json.gz under the log dir.

    Picks the newest by the profile-run directory name (JAX stamps it
    YYYY_MM_DD_HH_MM_SS), NOT the largest -- once several configs have been
    traced under one root, "largest" silently returns the slowest run, which
    is usually the baseline you were trying to compare against.
    """
    cands = list(root.rglob("*.trace.json.gz")) + list(root.rglob("*.trace.json"))
    if not cands:
        raise SystemExit(f"no *.trace.json[.gz] under {root}")
    if len(cands) > 1:
        print(f"{len(cands)} traces under {root}; pass a subdirectory to pick one:")
        for c in sorted(cands, key=lambda p: p.parent.name):
            print(f"    {c.parent.name}  {c.stat().st_size / 1e6:5.2f} MB  "
                  f"{c.relative_to(root)}")
        print()
    return max(cands, key=lambda p: p.parent.name)


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/trace")
    path = find_trace(root)
    print(f"trace: {path}  ({path.stat().st_size / 1e6:.1f} MB)\n")
    data = load(path)
    events = data.get("traceEvents", [])

    # Track metadata: map (pid, tid) -> readable names so GPU streams can be
    # told apart from host-side Python/XLA threads.
    pid_name, tid_name = {}, {}
    for e in events:
        if e.get("ph") == "M":
            if e.get("name") == "process_name":
                pid_name[e["pid"]] = e["args"].get("name", "")
            elif e.get("name") == "thread_name":
                tid_name[(e["pid"], e["tid"])] = e["args"].get("name", "")

    gpu_tracks = {k for k in tid_name
                  if "gpu" in pid_name.get(k[0], "").lower()
                  and "stream" in tid_name[k].lower()}
    if not gpu_tracks:  # fall back: any track under a GPU process
        gpu_tracks = {k for k in tid_name if "gpu" in pid_name.get(k[0], "").lower()}

    print(f"GPU tracks: {len(gpu_tracks)}")
    for k in sorted(gpu_tracks)[:8]:
        print(f"  {pid_name.get(k[0], '?')} / {tid_name[k]}")

    kern = defaultdict(list)      # name -> [durations us]
    spans = []                    # (start, end) us on GPU streams
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if (e.get("pid"), e.get("tid")) not in gpu_tracks:
            continue
        kern[e["name"]].append(e["dur"])
        spans.append((e["ts"], e["ts"] + e["dur"]))

    if not spans:
        raise SystemExit("no GPU kernel events found; is this a GPU trace?")

    total_kernel_us = sum(d for ds in kern.values() for d in ds)
    n_kernels = sum(len(ds) for ds in kern.values())

    # Union of kernel intervals -> busy time; wall span minus busy = idle gaps.
    spans.sort()
    busy, cur_s, cur_e = 0.0, *spans[0]
    gaps = []
    for s, e in spans[1:]:
        if s > cur_e:
            gaps.append(s - cur_e)
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = spans[-1][1] - spans[0][0]

    print(f"\n{'=' * 74}\nOCCUPANCY\n{'=' * 74}")
    print(f"  wall span covered      : {wall / 1e3:9.2f} ms")
    print(f"  GPU busy (union)       : {busy / 1e3:9.2f} ms  ({busy / wall * 100:5.1f}%)")
    print(f"  idle gaps              : {(wall - busy) / 1e3:9.2f} ms  "
          f"({(wall - busy) / wall * 100:5.1f}%)")
    print(f"  kernels launched       : {n_kernels}")
    print(f"  mean kernel duration   : {total_kernel_us / n_kernels:9.2f} us")
    if gaps:
        gaps.sort()
        print(f"  gaps > 1us             : {len(gaps)}  "
              f"median {gaps[len(gaps) // 2]:.2f} us  max {gaps[-1]:.2f} us")

    # Total idle is the WRONG statistic. A trace of N benchmark iterations has
    # a multi-millisecond host gap between each one; those few gaps can dominate
    # the total while every kernel inside an iteration runs back to back.
    # Launch overhead is per-kernel, so only sub-10us gaps measure it.
    idle_frac = (wall - busy) / wall
    launch_idle = sum(g for g in gaps if g < 10)
    launch_frac = launch_idle / wall
    # In-iteration occupancy: remove ONLY the multi-millisecond host gaps
    # between benchmark iterations, keep everything else. An earlier version
    # reported `1 - launch_frac` as "occupancy", which silently discarded all
    # idle in the 10us-1ms band while keeping the big gaps in the denominator.
    # For the fp32 baseline that turned a real 80.4% into a claimed 98.1%.
    inter_iter = sum(g for g in gaps if g >= 1000)
    in_iter_wall = wall - inter_iter
    in_iter_occ = busy / in_iter_wall if in_iter_wall > 0 else float("nan")
    print(f"\n{'=' * 74}\nIDLE BY GAP SIZE\n{'=' * 74}")
    for lo, hi, lab in ((0, 1, "<1us"), (1, 10, "1-10us"), (10, 100, "10-100us"),
                        (100, 1000, "0.1-1ms"), (1000, float("inf"), ">1ms")):
        b = [g for g in gaps if lo <= g < hi]
        pct = sum(b) / max(wall - busy, 1e-9) * 100
        print(f"  {lab:>10}{len(b):8d} gaps{sum(b) / 1e3:10.2f} ms{pct:8.1f}% of idle")

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    print(f"  raw idle                       {idle_frac * 100:5.1f}%")
    print(f"  in-iteration occupancy         {in_iter_occ * 100:5.1f}%  "
          f"(busy / wall minus the {sum(1 for g in gaps if g >= 1000)} gaps >1ms)")
    print(f"  in-iteration idle              {(1 - in_iter_occ) * 100:5.1f}%  "
          f"<- the headroom actually available")
    print(f"  of which per-kernel (<10us)    {launch_frac * 100:5.1f}%  "
          f"<- what launch overhead can explain")
    if launch_frac > 0.15:
        print("\n  LAUNCH-BOUND: the GPU waits on the host between kernels.")
        print("  CUDA graphs / XLA command buffers are the lever.")
    else:
        print("\n  NOT LAUNCH-BOUND: sub-10us idle is small, so per-kernel launch")
        print("  overhead cannot explain the gap. Note this does NOT mean the GPU is")
        print(f"  saturated -- {(1 - in_iter_occ) * 100:.1f}% of in-iteration time is still idle,")
        print("  mostly in 10us-1ms gaps (sync points, host-side work between")
        print("  dispatches). Reshaping the work is the lever, not chasing launches.")

    print(f"\n{'=' * 74}\nTOP KERNELS BY TOTAL TIME\n{'=' * 74}")
    ranked = sorted(kern.items(), key=lambda kv: -sum(kv[1]))
    print(f"  {'total ms':>9}{'%':>7}{'calls':>8}{'mean us':>10}  name")
    for name, ds in ranked[:18]:
        tot = sum(ds)
        print(f"  {tot / 1e3:9.2f}{tot / total_kernel_us * 100:6.1f}%{len(ds):8d}"
              f"{tot / len(ds):10.1f}  {name[:64]}")

    # Group by coarse op class, since XLA fusion names are noisy.
    groups = defaultdict(float)
    for name, ds in kern.items():
        n = name.lower()
        # `nvjet_*` is cuBLAS 12.x's GEMM kernel family -- it carries no
        # "gemm"/"cutlass" substring, so naive matching files ~30 ms of pure
        # matmul under "fusion/other" and makes the model look like it is
        # doing something mysterious when it is doing arithmetic.
        if ("gemm" in n or "matmul" in n or "cutlass" in n or "dot" in n
                or n.startswith("nvjet") or "_gemv" in n
                or any(n.startswith(a) for a in ("sm90_", "sm80_", "ampere_",
                                                 "turing_", "volta_"))):
            g = "GEMM"
        elif "attention" in n or "softmax" in n or "fmha" in n or "flash" in n:
            g = "attention/softmax"
        elif "copy" in n or "memcpy" in n or "transpose" in n or "bitcast" in n:
            g = "copy/transpose"
        elif "convert" in n or "cast" in n:
            g = "dtype convert"
        elif "reduce" in n or "norm" in n:
            g = "reduce/norm"
        else:
            g = "fusion/other"
        groups[g] += sum(ds)
    print(f"\n{'=' * 74}\nBY CLASS\n{'=' * 74}")
    for g, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:<20}{t / 1e3:9.2f} ms{t / total_kernel_us * 100:7.1f}%")


if __name__ == "__main__":
    main()
