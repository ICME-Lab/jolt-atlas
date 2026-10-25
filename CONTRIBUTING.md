# Contributing to jolt-atlas

## Pull requests

PRs are squash-merged into `main`. Two things follow from that:

- **The PR title becomes the commit message on `main`.** Write it as the one-line summary of
  the net change.
- **The commit history only exists for review.** It must be easy to review, and the net diff
  must be clean. The rules are in [`docs/pull-requests.md`](docs/pull-requests.md); run
  `/clean-branch` in Claude Code to audit a branch against them before pushing.

Keep each PR to one purpose. Unrelated work, even useful, goes in its own PR.

- **Bug fixes, small improvements, documentation**: open a PR directly. No spec needed.
- **Major features, protocol changes, large refactors**: start with a spec (next section).

The rule of thumb: if the implementation will be hard to review (500+ lines of non-trivial
changes), write a spec first. The spec is small and reviewable; the implementation follows from it.

## Spec-driven workflow

Large PRs are expensive to review. Code generation is cheap; human review time is the scarce
resource. We front-load review onto small, readable specs, and a single PR carries a feature from
spec to implementation.

1. **Write the spec** with `/new-spec <feature-name>` in Claude Code, or copy
   [`specs/TEMPLATE.md`](specs/TEMPLATE.md) by hand to `specs/<feature-name>.md`.
2. **Tighten it** with `/analyze-spec`, which scores the spec's ambiguity and interviews you until
   it is clear enough to implement without clarifying questions.
3. **Open a PR** containing only the spec, and put the PR number in the spec's header table.
4. **Spec review**: maintainers review and approve the spec. Design discussion happens here, on a
   document of a few hundred lines, not on the implementation diff.
5. **Implement** on the same branch with `/implement-spec`, or by hand. The skill runs the CI
   checks, validates the acceptance criteria, and runs `/clean-branch` before handoff.
6. **Merge**: maintainers review the implementation against the spec.

A spec has these sections (see the template): Summary; Intent (Goal, Invariants, Non-Goals);
Evaluation (Acceptance Criteria, Testing Strategy, Performance); Design (Architecture,
Alternatives Considered); Documentation; optional Execution; References. Focus on intent and
evaluation. Execution is downstream: the implementer should be able to derive most of it from
what you want and how you will verify it.

Specs are frozen at merge. `Status` goes from `proposed` to `implemented` and then stays. The code
is the source of truth for current behavior; old specs record what was decided and why at the
time, and nobody updates them when the code moves on.

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
| `/new-spec <name>` | Create `specs/<name>.md` from the template through an interview |
| `/analyze-spec` | Interactive ambiguity scoring and Socratic review of a spec |
| `/implement-spec` | Implement an approved spec: plan, build, run CI checks, validate, audit |
| `/clean-branch` | Audit the current branch against `docs/pull-requests.md`; report only, never rewrites history |
