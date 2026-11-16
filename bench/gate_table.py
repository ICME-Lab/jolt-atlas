#!/usr/bin/env python3
"""Print the table for bench/gate.sh: per-stage medians per commit and the change against the
first commit. Exits 1 if a commit is outside the thresholds (TIME_PCT, MEM_PCT)."""
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spans import aggregate, secs  # noqa: E402

# (label, innermost span); "prove" and "verify" come from the example's summary lines.
STAGES = [
    ("commit", "commit_witness_polynomials"),
    ("iop", "iop"),
    ("openings", "prove_reduced_openings"),
]
GATED = {"prove", "commit", "iop", "openings"}


def summary_secs(summary, key):
    for line in summary:
        m = re.match(rf"^{key}: ([0-9.]+(?:ns|µs|us|ms|s))", line)
        if m:
            return secs(m.group(1))
    return float("nan")


def peak_gb(summary):
    for line in summary:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", line)
        if m:
            return int(m.group(1)) / 1e6
        m = re.search(r"(\d+)\s+maximum resident set size", line)
        if m:
            return int(m.group(1)) / 1e9
    return float("nan")


def main():
    tag, runs, commits = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    time_pct = float(os.environ.get("TIME_PCT", 3))
    mem_pct = float(os.environ.get("MEM_PCT", 5))
    rows = ["prove"] + [label for label, _ in STAGES] + ["verify", "peak GB"]
    med = {}
    for sha in commits:
        samples = {r: [] for r in rows}
        for i in range(1, runs + 1):
            summary, busy, _ = aggregate(f"bench/runs/gate/{sha}-{tag}-rep{i}.log")
            samples["prove"].append(summary_secs(summary, "prove"))
            samples["verify"].append(summary_secs(summary, "verify"))
            samples["peak GB"].append(peak_gb(summary))
            for label, span in STAGES:
                samples[label].append(busy.get(span, float("nan")))
        med[sha] = {r: statistics.median(v) for r, v in samples.items()}
        med[sha]["spread"] = max(samples["prove"]) - min(samples["prove"])

    base = commits[0]
    print(f"\n{tag}, median of {runs} runs; change is against {base}\n")
    header = f"{'':<10}" + "".join(f"{sha:>12}{'change':>9}" if sha != base else f"{sha:>12}" for sha in commits)
    print(header)
    failed = False
    for r in rows:
        line = f"{r:<10}{med[base][r]:>12.3f}"
        for sha in commits[1:]:
            delta = 100 * (med[sha][r] / med[base][r] - 1)
            limit = mem_pct if r == "peak GB" else time_pct if r in GATED else None
            flag = ""
            if limit is not None and delta > limit:
                flag = " !"
                failed = True
            line += f"{med[sha][r]:>12.3f}{delta:>+7.1f}%{flag}"
        print(line)
    print("\nprove spread (max - min): " + ", ".join(f"{sha} {med[sha]['spread']:.2f} s" for sha in commits))
    print(f"gate: {'FAIL' if failed else 'pass'} (time +{time_pct:g}%, memory +{mem_pct:g}%)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
