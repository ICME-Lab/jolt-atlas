#!/usr/bin/env python3
"""Aggregate a Chrome trace written by `--trace` (tracing_chrome) by span name.

Usage: python3 scripts/agg_trace.py trace-<ts>.json [top_n]
Prints self time (summed over threads, so parallel sections count CPU-like) and
inclusive time per span name; top-level single-threaded spans give wall time.
"""import json, sys, collections
# Aggregates a tracing_chrome trace: inclusive and self time per span name.
path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
data = json.load(open(path))
events = data if isinstance(data, list) else data["traceEvents"]
incl = collections.Counter(); selft = collections.Counter(); count = collections.Counter()
stacks = collections.defaultdict(list)  # (pid,tid) -> [(name, start, child_time)]
wall_start = None; wall_end = 0
for e in sorted(events, key=lambda e: (e.get("ts", 0), 0 if e.get("ph") == "E" else 1)):
    ph = e.get("ph"); ts = e.get("ts", 0)
    key = (e.get("pid"), e.get("tid"))
    if ph == "B":
        stacks[key].append([e.get("name"), ts, 0.0])
        wall_start = ts if wall_start is None else min(wall_start, ts)
    elif ph == "E":
        st = stacks[key]
        if not st: continue
        name, start, child = st.pop()
        dur = ts - start
        incl[name] += dur; selft[name] += dur - child; count[name] += 1
        if st: st[-1][2] += dur
        wall_end = max(wall_end, ts)
    elif ph == "X":
        name = e.get("name"); dur = e.get("dur", 0)
        incl[name] += dur; selft[name] += dur; count[name] += 1
wall = (wall_end - (wall_start or 0)) / 1e6
print(f"wall (first B to last E): {wall:.2f} s")
print(f"\n{'span (by self time, summed over threads)':<70} {'self s':>9} {'incl s':>9} {'calls':>8}")
for name, t in selft.most_common(top):
    print(f"{name[:70]:<70} {t/1e6:>9.2f} {incl[name]/1e6:>9.2f} {count[name]:>8}")
print(f"\n{'span (by inclusive time)':<70} {'incl s':>9} {'calls':>8}")
for name, t in incl.most_common(top):
    print(f"{name[:70]:<70} {t/1e6:>9.2f} {count[name]:>8}")
