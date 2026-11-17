#!/usr/bin/env bash
# A performance gate between commits: build the release `prove` example at each one, run them
# interleaved on one model, and print the per-stage medians with each commit's change against
# the first (the baseline).
#
#   bench/gate.sh <base> <commit>... [--runs N] [--model DIR] [--seq-len N]
#
# Run it where nothing else is using the machine (`scripts/remote.sh run 'bench/gate.sh ...'`
# for the benchmark box). Binaries are kept in bench/bin/prove-<sha> and reused; logs go to
# bench/runs/gate/. Thresholds: stage times within TIME_PCT (default 3), peak memory within
# MEM_PCT (default 5); a commit outside them makes the script exit 1.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
RUNS=3
MODEL=models/gpt2
SEQ=16
COMMITS=()
while [ $# -gt 0 ]; do
    case "$1" in
    --runs) RUNS=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --seq-len) SEQ=$2; shift 2 ;;
    *) COMMITS+=("$(git rev-parse --short=9 "$1")"); shift ;;
    esac
done
[ ${#COMMITS[@]} -ge 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ -d "$MODEL" ] || { echo "model directory $MODEL is missing" >&2; exit 1; }

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}
mkdir -p bench/bin bench/runs/gate
for sha in "${COMMITS[@]}"; do
    [ -x "bench/bin/prove-$sha" ] && continue
    wt=$(mktemp -d)
    git worktree add -q --detach "$wt" "$sha"
    echo "building $sha"
    (cd "$wt" && cargo build --release -p jolt-atlas-core --example prove 2>&1 | tail -1)
    cp "$CARGO_TARGET_DIR/release/examples/prove" "bench/bin/prove-$sha"
    git worktree remove --force "$wt"
done

# A commit from before the BitZ scheme proves with Akita, whose binary reads schedule files at
# run time: take them from the commit. Empty for a commit that has none.
schedules() {
    local dir=bench/bin/schedules-$1
    if [ ! -d "$dir" ] && git cat-file -e "$1:atlas-pcs/schedules" 2>/dev/null; then
        mkdir -p "$dir"
        git archive "$1" atlas-pcs/schedules | tar -x -C "$dir" --strip-components=2
    fi
    [ -d "$dir" ] && echo "$ROOT/$dir"
    return 0
}

if /usr/bin/time -v true 2>/dev/null; then TIME=(/usr/bin/time -v); else TIME=(/usr/bin/time -l); fi
for i in $(seq 1 "$RUNS"); do
    for sha in "${COMMITS[@]}"; do
        log=bench/runs/gate/$sha-$(basename "$MODEL")-$SEQ-rep$i.log
        JOLT_AKITA_SCHEDULE_DIR=$(schedules "$sha") \
            "${TIME[@]}" "bench/bin/prove-$sha" --model "$MODEL" --seq-len "$SEQ" --trace-terminal > "$log" 2>&1
        grep -q "Proof verified" "$log" || { echo "$sha run $i did not verify: $log" >&2; exit 1; }
        echo "ran $sha rep $i: $(grep -E '^prove:' "$log")"
    done
done

python3 bench/gate_table.py "$(basename "$MODEL")-$SEQ" "$RUNS" "${COMMITS[@]}"
