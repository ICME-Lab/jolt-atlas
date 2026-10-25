# Contributing to jolt-atlas

## Pull requests

PRs are squash-merged into `main`. Two things follow from that:

- **The PR title becomes the commit message on `main`.** Write it as the one-line summary of
  the net change.
- **The commit history only exists for review.** It must be easy to review, and the net diff
  must be clean. The rules are in [`docs/pull-requests.md`](docs/pull-requests.md); run
  `/clean-branch` in Claude Code to audit a branch against them before pushing.

Keep each PR to one purpose. Unrelated work, even useful, goes in its own PR.

## CI

CI (`.github/workflows/rust.yml`) runs on every PR with `RUSTFLAGS=-D warnings`:

| Job | Command |
|-----|---------|
| Format check | `cargo fmt --all --check` |
| Clippy & Test | `cargo clippy --workspace --all-targets` then `cargo nextest run --workspace` |
| Model zoo gates & goldens | `cargo nextest run -p atlas-onnx-import --run-ignored ignored-only` |
| GPT-2 prove & verify | `scripts/download_gpt2.py`, then `cargo nextest run --workspace --run-ignored ignored-only -E 'test(test_gpt2_dory)'` |
| BGE prove & verify | `scripts/download_bge_small_en_v1_5.py`, then `cargo nextest run --workspace --run-ignored ignored-only -E 'test(test_bge_small_en_v1_5)'` |

Run the first two rows locally before pushing. The model jobs need the ONNX files downloaded
first (see [Getting Started](README.md#getting-started)) and are slow; run the one your change
touches.

## Development setup

- Rust toolchain pinned by `rust-toolchain.toml` (installed automatically by `rustup`).
- [cargo-nextest](https://nexte.st/) for tests. Plain `cargo test` also works but is not what CI runs.
- Python 3.11 with a virtual environment at `.venv` for the model download scripts in `scripts/`.

Run tests in the dev profile. A `--release` test build recompiles the workspace for minutes
after every edit; reserve release builds for the examples and benchmarks in the README.

## Agent documentation

`CLAUDE.md` and `AGENTS.md` are identical copies of the project guide for coding agents
(commands, crate map, invariants). Edit both when changing one. Project skills live in
`.claude/skills/`:

| Skill | Description |
|-------|-------------|
| `/clean-branch` | Audit the current branch against `docs/pull-requests.md`; report only, never rewrites history |
