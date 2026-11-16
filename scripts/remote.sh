#!/usr/bin/env bash
# Run builds, tests and benchmarks on a remote machine instead of this one.
#
# The remote is an SSH alias (default `atlas-bench`, override with ATLAS_REMOTE) that logs in
# without a prompt. The working tree, including uncommitted changes and `.git`, is copied with
# rsync, so nothing has to be pushed and the remote needs no GitHub access. Gitignored paths
# (`target/`, `bench/bin/`, the downloaded models) are neither copied nor deleted by `sync`,
# and files over 20 MB are skipped.
#
#   scripts/remote.sh up                 start the machine (runs `atlas-bench-up` if installed)
#   scripts/remote.sh down               stop it (`atlas-bench-down`)
#   scripts/remote.sh setup              one-time: compiler, rustup, cargo-nextest
#   scripts/remote.sh sync               copy the working tree
#   scripts/remote.sh models [dir...]    copy downloaded model directories (default models/gpt2)
#   scripts/remote.sh run '<command>'    sync, then run the command in the repo and wait
#   scripts/remote.sh bg <name> '<cmd>'  sync, then run detached; output in remote-logs/<name>.log
#   scripts/remote.sh tail <name>        follow a detached run's log
#   scripts/remote.sh fetch <path>...    copy files back (e.g. bench/runs)
set -euo pipefail

HOST=${ATLAS_REMOTE:-atlas-bench}
DIR=${ATLAS_REMOTE_DIR:-jolt-atlas}
ROOT=$(git rev-parse --show-toplevel)
ENV='source "$HOME/.cargo/env" 2>/dev/null; export CARGO_TARGET_DIR="$HOME/target"'

hook() {
    local name=atlas-bench-$1
    if command -v "$name" >/dev/null; then
        "$name"
    elif [ -x "$HOME/.local/bin/$name" ]; then
        "$HOME/.local/bin/$name"
    else
        echo "no \`$name\` command on this machine: start or stop \`$HOST\` by hand" >&2
        exit 1
    fi
}

# Copies `.git` and every file git tracks or would track (so `.gitignore` decides, negations
# included), skipping files over 20 MB, then removes remote files that are gone locally.
sync_tree() {
    local list
    list=$(mktemp)
    (cd "$ROOT" && git ls-files -co --exclude-standard -z |
        perl -0ne 'chomp; if (-f $_) { if (-s _ <= 20 * 1024 * 1024) { print "$_\0" } else { print STDERR "sync: skipping $_ (over 20 MB)\n" } }') > "$list"
    ssh "$HOST" "mkdir -p $DIR/.git"
    rsync -az --delete "$ROOT/.git/" "$HOST:$DIR/.git/"
    rsync -az --from0 --files-from="$list" "$ROOT/" "$HOST:$DIR/"
    rsync -az "$list" "$HOST:$DIR/.git/remote-sync-list"
    ssh "$HOST" "cd $DIR && git ls-files -co --exclude-standard -z | sort -z > .git/remote-have &&
        sort -z .git/remote-sync-list | comm -z -23 .git/remote-have - | xargs -0 -r rm -f"
    rm -f "$list"
}

cmd=${1:-}
shift || true
case "$cmd" in
up) hook up ;;
down) hook down ;;
setup)
    ssh "$HOST" 'set -e
        sudo dnf install -y gcc gcc-c++ clang git rsync python3 time >/dev/null
        command -v rustup >/dev/null || [ -x "$HOME/.cargo/bin/rustup" ] ||
            curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
        source "$HOME/.cargo/env"
        case "$(uname -m)" in aarch64) arch=linux-arm ;; *) arch=linux ;; esac
        command -v cargo-nextest >/dev/null ||
            curl -LsSf "https://get.nexte.st/latest/$arch" | tar zxf - -C "$HOME/.cargo/bin"
        echo "setup done: $(rustup --version 2>/dev/null | head -1), $(cargo-nextest nextest --version | head -1)"'
    ;;
sync) sync_tree ;;
models)
    [ $# -gt 0 ] || set -- models/gpt2
    for m in "$@"; do
        ssh "$HOST" "mkdir -p $DIR/$m"
        rsync -az "$ROOT/$m/" "$HOST:$DIR/$m/"
    done
    ;;
run)
    sync_tree
    ssh "$HOST" "cd $DIR && $ENV && $*"
    ;;
bg)
    name=$1
    shift
    sync_tree
    ssh -a "$HOST" "cd $DIR && mkdir -p remote-logs && setsid nohup bash -c $(printf '%q' "$ENV && $*") > remote-logs/$name.log 2>&1 < /dev/null & echo started: remote-logs/$name.log"
    ;;
tail) ssh "$HOST" "tail -n 40 -f $DIR/remote-logs/$1.log" ;;
fetch)
    for p in "$@"; do
        rsync -az "$HOST:$DIR/$p" "$ROOT/$(dirname "$p")/"
    done
    ;;
*)
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
