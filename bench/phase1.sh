#!/usr/bin/env bash
# Phase 1 of docs/measurement-plan.md: baseline runs on one saved binary.
# Usage: bench/phase1.sh <path-to-prove-binary>
set -u
BIN="$1"
OUT=bench/runs
mkdir -p "$OUT"

run() {
    local name="$1"; shift
    echo "=== $name: $*"
    /usr/bin/time -l "$BIN" "$@" --trace-terminal > "$OUT/$name.log" 2>&1
    echo "exit $?"; grep -E "^(prove|verify|proof size|prover setup):|maximum resident|real " "$OUT/$name.log"
}

# Smoke tests
run gpt2-tiny        --model models/gpt2-tiny
run qwen2-tiny       --model models/qwen2-tiny
run bge-tiny         --model models/bge-tiny

# Reference number, with stats
run gpt2-16-stats    --model models/gpt2 --seq-len 16 --stats

# Context scaling; stops at the first failure (memory)
for n in 32 64 128; do
    run gpt2-$n      --model models/gpt2 --seq-len $n || break
    grep -q "Proof verified" "$OUT/gpt2-$n.log" || { echo "gpt2 --seq-len $n failed; stopping scaling runs"; break; }
done

# Second real model
run bge-small        --model models/bge-small-en-v1.5

# Variance: three repeats of the reference run
for i in 1 2 3; do
    run gpt2-16-rep$i --model models/gpt2 --seq-len 16
done
