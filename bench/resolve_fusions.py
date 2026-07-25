"""Map opaque XLA fusion kernel names back to the model code that produced them.

The profiler names fused kernels `fusion_1758`, `input_concatenate_fusion`,
`loop_pad_fusion_124` and so on. Those names say nothing, and in the optimized
trace that bucket is 26.9% of GPU time -- the largest unattributed cost left.
The trace's `args` carry only occupancy and correlation ids, no HLO link, so
the mapping has to come from an HLO dump (`XLA_FLAGS=--xla_dump_to=...`).

In the optimized HLO, each fusion appears as

    %fusion.1758 = bf16[...] fusion(...), kind=kLoop, calls=%fused_computation.1758,
        metadata={op_name="jit(_frame)/.../mul" source_file="models.py" source_line=322}

and the called computation lists the ops that were fused together. This script
joins the two: for each fusion named in a trace, it reports the ops inside it,
the output shape, and the source lines they came from.

    python bench/resolve_fusions.py bench/results/hlo bench/results/trace/<run>
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Real HLO is messier than it looks: instructions may carry a `ROOT ` prefix,
# shapes carry layout suffixes like `bf16[1,290,1920]{2,1,0}`, operand lists
# contain nested brackets and parens, and a computation header states its full
# signature before the opening brace. Match loosely and pull fields out
# individually rather than trying to write one exact grammar.
# XLA's *dumped* HLO drops the leading `%` that the textual spec shows, and
# computation headers are just `fused_reduce.114 {` with no signature. Match the
# dump, not the spec.
FUSION_INSTR = re.compile(r"^\s*([\w.\-]+)\s*=\s*(.+?)\s+fusion\(")
CALLS = re.compile(r"calls=%?([\w.\-]+)")
KIND = re.compile(r"kind=(\w+)")
COMP_HEADER = re.compile(r"^([A-Za-z_][\w.\-]*)\s*(?:\(.*\))?\s*(?:->.*)?\{\s*$")
INSTR = re.compile(r"^\s*[\w.\-]+\s*=\s*(\S+?)\s+([a-z][\w\-]*)\(")
META = re.compile(r'op_name="([^"]*)"')
SRC = re.compile(r'source_file="([^"]*)"\s+source_line=(\d+)')


def load_trace_kernels(trace_dir: Path) -> dict[str, tuple[int, float]]:
    cands = list(trace_dir.rglob("*.trace.json.gz"))
    if not cands:
        return {}
    ev = json.loads(gzip.open(max(cands, key=lambda p: p.parent.name), "rt").read())
    agg: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for e in ev.get("traceEvents", []):
        if e.get("ph") == "X" and "dur" in e:
            a = agg[e["name"]]
            a[0] += 1
            a[1] += e["dur"]
    return {k: (v[0], v[1] / 1e3) for k, v in agg.items()}


def parse_hlo(text: str):
    """-> ({computation: [(shape, opcode, op_name, src)]}, {fusion: info})."""
    comps: dict[str, list] = {}
    cur, body = None, []
    for line in text.splitlines():
        if cur is None:
            h = COMP_HEADER.match(line)
            if h:
                cur, body = h.group(1), []
            continue
        if line.strip() == "}":
            comps[cur] = body
            cur = None
            continue
        m = INSTR.search(line)
        if m:
            shape, opcode = m.group(1), m.group(2)
            nm, sm = META.search(line), SRC.search(line)
            body.append((shape, opcode,
                         nm.group(1) if nm else "",
                         f"{Path(sm.group(1)).name}:{sm.group(2)}" if sm else ""))
    fusions = {}
    for line in text.splitlines():
        m = FUSION_INSTR.search(line)
        if not m:
            continue
        name, shape = m.group(1), m.group(2).split(",")[0][:48]
        c, k = CALLS.search(line), KIND.search(line)
        nm, sm = META.search(line), SRC.search(line)
        fusions[name] = {
            "shape": shape, "kind": k.group(1) if k else "?",
            "calls": c.group(1) if c else "",
            "op_name": nm.group(1) if nm else "",
            "src": f"{Path(sm.group(1)).name}:{sm.group(2)}" if sm else "",
        }
    return comps, fusions


def norm(n: str) -> str:
    """`fusion.1758` in HLO vs `fusion_1758` in the profiler."""
    return n.replace(".", "_")


def main() -> None:
    hlo_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/hlo")
    trace_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    files = sorted(hlo_dir.rglob("*after_optimizations*.txt")) or \
        sorted(hlo_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no HLO dump under {hlo_dir}; run the dump_hlo step first")
    print(f"HLO files: {len(files)}  (largest: {max(files, key=lambda p: p.stat().st_size).name})\n")

    comps, fusions = {}, {}
    for f in files:
        c, fu = parse_hlo(f.read_text(errors="ignore"))
        comps.update(c)
        fusions.update(fu)
    print(f"parsed {len(fusions)} fusions across {len(comps)} computations")

    by_norm = {norm(k): (k, v) for k, v in fusions.items()}
    kernels = load_trace_kernels(trace_dir) if trace_dir else {}

    if kernels:
        ranked = sorted(((n, c, ms) for n, (c, ms) in kernels.items()
                         if "fusion" in n.lower()), key=lambda t: -t[2])
        print(f"\nresolving the top fusion kernels by GPU time:\n")
        for name, calls, ms in ranked[:12]:
            hit = by_norm.get(norm(name))
            print(f"{'=' * 74}\n{name}   {ms:.2f} ms over {calls} calls "
                  f"({ms / max(calls, 1) * 1e3:.1f} us each)")
            if not hit:
                print("  no matching fusion in the HLO dump "
                      "(name may be from a different module)")
                continue
            hlo_name, info = hit
            print(f"  shape {info['shape']}  kind={info['kind']}")
            if info["op_name"]:
                print(f"  op_name  {info['op_name']}")
            if info["src"]:
                print(f"  source   {info['src']}")
            body = comps.get(info["calls"], [])
            if body:
                counts: dict[str, int] = defaultdict(int)
                for _shape, opcode, *_ in body:
                    counts[opcode] += 1
                print(f"  fused ops ({len(body)}): " + ", ".join(
                    f"{k}x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
                srcs = {row[3] for row in body if row[3]}
                if srcs:
                    print(f"  from     {', '.join(sorted(srcs)[:6])}")
        # The per-kernel view answers "what is fusion_1758"; this answers the
        # question that actually drives work -- which line of model code is
        # buying the GPU time, with fusions collapsed across name churn.
        by_src: dict[str, list] = defaultdict(lambda: [0.0, 0, set()])
        unresolved = 0.0
        for name, (calls, ms) in kernels.items():
            hit = by_norm.get(norm(name))
            if not hit:
                if "fusion" in name.lower():
                    unresolved += ms
                continue
            info = hit[1]
            key = f"{info['src'] or '?'}  {info['op_name'].split('/')[-1] or '?'}"
            e = by_src[key]
            e[0] += ms
            e[1] += calls
            e[2].add(info["kind"])
        total = sum(v[0] for v in by_src.values())
        print(f"\n{'=' * 74}\nFUSION TIME BY SOURCE LINE\n{'=' * 74}")
        print(f"  {'ms':>8}{'%':>7}{'calls':>8}  kind     source / op")
        for k, (ms, calls, kinds) in sorted(by_src.items(), key=lambda kv: -kv[1][0])[:18]:
            print(f"  {ms:8.2f}{ms / max(total, 1e-9) * 100:6.1f}%{calls:8d}  "
                  f"{','.join(sorted(kinds)):8} {k}")
        if unresolved:
            print(f"  {unresolved:8.2f} unresolved (fusion kernels with no HLO match)")

    else:
        print("\n(no trace given; listing the largest fusions by output shape)")
        for k, v in list(fusions.items())[:15]:
            print(f"  {k:28} {v['shape'][:40]:42} {v['src']}")


if __name__ == "__main__":
    main()
