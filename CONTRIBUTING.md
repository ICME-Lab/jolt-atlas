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
   it is clear enough to implement without clarifying questions. This step is required before
   opening the PR, and the PR description says it ran: the analyzer reads the repo, so it catches
   buildability problems (coherence rules, dependency cycles, feature leaks) locally in one pass
   that would otherwise surface as serial review rounds on GitHub.
3. **Open a PR** containing only the spec; put the PR number in the spec's header table in the
   same PR or at merge. If the spec names other specs in `Depends on`/`Supersedes` (or in its
   body), grep those documents and update their cross-references in the same PR — coupled specs
   drift apart one edit at a time, and dangling references to deleted sections are a review
   finding, not a nitpick.
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

| Job | When | Command |
|-----|------|---------|
| Format check | every PR | `cargo fmt --all --check` |
| Clippy | every PR | `cargo clippy --workspace --all-targets` |
| Test | every PR | `cargo nextest run --workspace --profile ci` (unit tests, the op-surface table, the soundness sweeps, the checked-in zoo end to end, golden hashes), then the full-zoo import gates `cargo nextest run -p atlas-onnx-import --profile ci --run-ignored ignored-only -E 'not test(golden_zoo_big)'` |
| Real model (gpt2 / bge / qwen) | main, nightly, `workflow_dispatch`, PRs labelled `ci:heavy` | `scripts/download_<model>.py` (cached), then `cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::<name>$/)'` and `golden_zoo_big` |
| Exhaustive soundness sweeps | nightly, `workflow_dispatch` | `JOLT_SOUNDNESS_EXHAUSTIVE=1 cargo nextest run -p jolt-atlas-core --profile heavy -E 'test(soundness)'` |

Run the first three rows locally before pushing (about a minute of tests). The real-model job
needs the ONNX files downloaded first (see [Getting Started](README.md#getting-started)); run
the model your change touches, or add the `ci:heavy` label to the PR. The tiny stand-ins
(`models/*-tiny`) exercise the same importer passes and prover kernels as the real models, so a
green PR run already covers the GPT-2, Qwen2 and BERT graph shapes.

## Development setup

- Rust toolchain pinned by `rust-toolchain.toml` (installed automatically by `rustup`).
- [cargo-nextest](https://nexte.st/) for tests. Plain `cargo test` also works but is not what CI runs.
- Python 3.11 with a virtual environment at `.venv` for the model download scripts in `scripts/`.

Run tests in the dev profile. A `--release` test build recompiles the workspace for minutes
after every edit; reserve release builds for the `prove` example in the README.

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
