#!/usr/bin/env python3
"""Compare qwen3 layer prover op timings against the pre-fusion baseline.

Usage:
  cargo run --release -p qwen3-layer-prover --bin prove_trace_layer -- ... \
    2>&1 | qwen3-layer-prover/scripts/compare_layer_timings.py

  qwen3-layer-prover/scripts/compare_layer_timings.py timing.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Baseline before SHOUT-backed round fusion into matmul-like ops.
# Measured on qwen3-awy/traces/fox_eos_full_awy, layer0, seq=197.
BASELINE_SECONDS = {
    "softmax": 2.725,
    "silu": 1.313,
    "q_norm": 0.755,
    "qk_score": 0.606,
    "silu_up": 0.571,
    "rms_norm_mlp": 0.396,
    "k_norm": 0.393,
    "rms_norm_atten": 0.381,
    "gate_proj": 0.361,
    "up_proj": 0.345,
    "q_rope": 0.301,
    "q_proj": 0.202,
    "down_proj": 0.192,
    "pv_matmul": 0.159,
    "k_rope": 0.155,
    "o_proj": 0.150,
    "k_proj": 0.114,
    "v_proj": 0.111,
    "residual_add_mlp": 0.045,
    "residual_add_attn": 0.044,
}

TIMING_RE = re.compile(r"^timing: prove_layer\.([A-Za-z0-9_]+)\s+([0-9.]+)s$")


def read_text(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text()


def parse_timings(text: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for line in text.splitlines():
        match = TIMING_RE.match(line.strip())
        if match:
            timings[match.group(1)] = float(match.group(2))
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="prove_trace_layer output log")
    parser.add_argument(
        "--sort",
        choices=("baseline", "current", "saved", "speedup", "name"),
        default="baseline",
        help="row order; default: baseline",
    )
    args = parser.parse_args()

    current = parse_timings(read_text(args.log))
    if not current:
        print("No `timing: prove_layer.<op> ...s` lines found.", file=sys.stderr)
        return 1

    rows = []
    for op, before in BASELINE_SECONDS.items():
        after = current.get(op)
        if after is None:
            rows.append((op, before, None, None, None))
            continue
        saved = before - after
        speedup = before / after if after > 0 else float("inf")
        rows.append((op, before, after, saved, speedup))

    def sort_key(row: tuple[str, float, float | None, float | None, float | None]):
        op, before, after, saved, speedup = row
        if args.sort == "current":
            return (-(after or 0.0), op)
        if args.sort == "saved":
            return (-(saved or 0.0), op)
        if args.sort == "speedup":
            return (-(speedup or 0.0), op)
        if args.sort == "name":
            return (op,)
        return (-before, op)

    rows.sort(key=sort_key)

    total_before = sum(BASELINE_SECONDS.values())
    total_after = sum(current.get(op, 0.0) for op in BASELINE_SECONDS)
    total_saved = total_before - total_after
    total_speedup = total_before / total_after if total_after > 0 else float("inf")

    print("| op | before | current | saved | speedup |")
    print("|---|---:|---:|---:|---:|")
    for op, before, after, saved, speedup in rows:
        if after is None:
            print(f"| {op} | {before:.3f}s | missing | - | - |")
        else:
            print(f"| {op} | {before:.3f}s | {after:.3f}s | {saved:+.3f}s | {speedup:.2f}x |")
    print(f"| **total** | **{total_before:.3f}s** | **{total_after:.3f}s** | **{total_saved:+.3f}s** | **{total_speedup:.2f}x** |")

    missing = sorted(set(BASELINE_SECONDS) - set(current))
    extra = sorted(set(current) - set(BASELINE_SECONDS))
    if missing:
        print("\nMissing baseline ops in current log: " + ", ".join(missing), file=sys.stderr)
    if extra:
        print("\nCurrent log has extra ops: " + ", ".join(extra), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
