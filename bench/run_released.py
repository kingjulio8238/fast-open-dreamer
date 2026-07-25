"""Run the released `inference.py` under the timing hooks and GPU sampler.

Hooks must be installed before `inference` imports the names it calls, so this
wrapper does the patching first and only then hands control to the released
entry point — the same ordering trick DreamDojo's bench runner uses.

    python bench/run_released.py -- --checkpoint_path /ckpt/... --horizon 64 ...

Everything after `--` is forwarded verbatim to the released argument parser, so
the command under test is the documented one with nothing rewritten.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

REPO = os.environ.get("OD_INFERENCE_REPO", "/root/od-inference")
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import gpu_sampler, timing_hooks  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    out_dir = os.environ.get("BENCH_OUT_DIR", "/results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    gpu_sampler.start()
    timing_hooks.install(pkg="pipeline")

    t0 = time.perf_counter()
    sys.argv = ["inference.py"] + argv
    os.chdir(REPO)
    try:
        runpy.run_path(str(Path(REPO) / "inference.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    wall = time.perf_counter() - t0

    md = timing_hooks.report()
    summary = {
        "wall_seconds": round(wall, 3),
        "argv": argv,
        "timing": timing_hooks.summarize(),
        "gpu": gpu_sampler.summarize(),
    }
    Path(out_dir, "released_profile.json").write_text(json.dumps(summary, indent=2))
    Path(out_dir, "released_profile.md").write_text(md)
    print("\n" + "=" * 78)
    print(md)
    print("=" * 78)
    print(f"total wall: {wall:.2f} s")
    print(f"wrote {out_dir}/released_profile.{{json,md}}")


if __name__ == "__main__":
    main()
