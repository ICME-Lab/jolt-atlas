#!/usr/bin/env python3
"""Aggregate `--trace-terminal` span close events: total busy time and count per span.

Usage: bench/spans.py <log> [--top N] [--filter SUBSTR]
Prints the innermost span name (with its fields), count, total busy, and the
summary lines (`prove:`, `verify:`, peak RSS) from the same log.
"""
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def secs(tok):
    m = re.fullmatch(r"([0-9.]+)(ns|µs|us|ms|s)", tok)
    return float(m.group(1)) * UNITS[m.group(2)]


def main():
    path = sys.argv[1]
    top = 40
    flt = None
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if "--filter" in sys.argv:
        flt = sys.argv[sys.argv.index("--filter") + 1]
    busy = defaultdict(float)
    count = defaultdict(int)
    summary = []
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            if re.match(r"^(prover setup|prove|verify|proof size|imported|model )", line) or (
                "maximum resident" in line or line.strip().endswith(" sys") and " real " in line
            ):
                summary.append(line.rstrip())
                continue
            if " close " not in line:
                continue
            m = re.search(r"INFO (.*?): [a-z_]+(?:::[a-z_0-9]+)*: close time\.busy=(\S+)", line)
            if not m:
                continue
            spans, b = m.group(1), m.group(2)
            # innermost span is the last `name{fields}` element of the path
            name = spans.split(":")[-1] if "{" not in spans.split(":")[-1] else spans[spans.rfind(":", 0, spans.rfind("{")) + 1 :]
            name = re.sub(r"\{.*", lambda mm: mm.group(0), name)
            if flt and flt not in name:
                continue
            busy[name] += secs(b)
            count[name] += 1
    for s in summary:
        print(s)
    print()
    print(f"{'span':<90} {'n':>8} {'busy s':>10}")
    for name, t in sorted(busy.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{name[:90]:<90} {count[name]:>8} {t:>10.3f}")


if __name__ == "__main__":
    main()
