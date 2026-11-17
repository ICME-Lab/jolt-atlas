# Spec: The verifier's build is the trust perimeter

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | — (independent of `specs/relations-as-values.md` and `specs/verification-schedule.md`; may land before or after them) |
| Supersedes  | `docs/upstream-inventory.md`'s "the protocol crates ship both sides … and no feature gate" decision; the prover entries of `jolt-atlas-verifier/clippy.toml` |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`jolt-atlas-verifier` has no prover code, and three mechanisms hold that. The crates under it
are a different matter. `atlas-sumcheck`, `atlas-pcs`, `atlas-algebra`,
`atlas-lookup-tables` and `atlas-trace` ship both sides in one build, and most of the code a
verdict depends on is in them, not in the verifier crate. `docs/verifier.md` scopes the
perimeter in prose ("`engine` …, not the `prover` submodules"), and the tools behind that
prose are weak: a `clippy.toml` list of prover items that must be kept by hand, and a path
grep for `::prover` that the glob `pub use atlas_sumcheck::relation::*` already walks around.
Prover code also sits outside `prover` modules: `mles_product_sum.rs`, the fold and state
types in `opening_reduction/mod.rs`, the witness builders in `shout/mod.rs`, the commit side
of the Akita adapter, the whole of `poly/split_eq.rs` and `poly/ra.rs`, the op kernels and the
execution trace in `atlas-trace`. An auditor cannot tell from the build which files they must
read.

This spec puts the prover side of every first-party crate behind a `prover` cargo feature
that the verifier's build leaves off. The audit surface then has a mechanical definition, the
source files `cargo check -p jolt-atlas-verifier` compiles, and CI keeps a checked-in list of
them, so growth of the perimeter is a reviewed diff.

## Intent

### Goal

Gate every prover-only item of the first-party crates under `jolt-atlas-verifier` behind a
`prover` feature, off in the verifier's own build, and make the list of source files that
build compiles a checked-in, CI-verified artifact.

Introduced or modified:

- `atlas-sumcheck`: feature `prover` (enables `atlas-algebra/prover`,
  `atlas-lookup-tables/prover`, `atlas-transcript/prover`). Every `prover` submodule is
  `#[cfg(feature = "prover")]`. Prover code in shared files moves into the module's `prover`
  submodule: `mles_product_sum.rs` as a whole; `opening_reduction`'s `fold_high_to_low`,
  `SharedDensePolynomial`, `EqAddressState`, `EqCycleState` and `GroupH`;
  `evaluation_reduction`'s `EvalReductionWitness`, `EvalReductionInstance::prove` and
  `compute_h`; `ps_shout`'s `CycleWeight::evals`, `CyclePoly`, `phase_schedule` and
  `OperandRaf`; `shout`'s `compute_instruction_h_indices` and `compute_ra_evals`.
- `atlas-algebra`: feature `prover`. Gated: `poly::{split_eq, ra, one_hot, rlc, lagrange}`,
  the `PolynomialBinding` trait and its impls, the bind, fold and sumcheck-evaluation methods
  of `dense`, `compact` and `multilinear`, `prefix_suffix`'s decomposition and registry (its
  three traits stay), `univariate`'s interpolation and compression constructors,
  `utils::{expanding_table, profiling}`, `utils::thread::drop_in_background_thread`.
- `atlas-pcs`: feature `prover` (enables `atlas-algebra/prover`). Gated, in the trait and in
  the Akita impl: `CommitmentScheme::{setup_prover, commit, batch_commit, batch_commit_all,
  prove, prove_rlc}`, `StreamingCommitmentScheme`, `BatchCommitOutput`,
  `akita::backend::trace_onehot`, the commit and open functions of `backend/scheme.rs` and
  `backend/native_batching.rs`, and the prover helpers of `akita/mod.rs`. `setup_verifier`
  takes a `VerifierSetup` source that exists without the feature.
- `atlas-lookup-tables`: feature `prover`; `Prefixes::update_checkpoints` is gated.
- `atlas-transcript`: the existing `prover` feature also gates `mod prover`,
  `ProverTranscript`, `ProvingTranscript` and `setup::prover_transcript`.
- `atlas-trace`: feature `exec`, in `default`. Gated: `mod exec` (`Model::trace`,
  `Model::forward`), `Trace`, `LayerData`, `NodeAux`, `FusedIntermediates`, `SoftmaxWitness`,
  `RsqrtAdvice`, `TeleportAdvice`, `OpSpec::eval` and the kernels it calls.
  `ModelExecutionIO` moves from `trace.rs` to its own ungated module `io.rs`, re-exported at
  the old path under `exec`.
- `common`: feature `logging` gates `utils::logging` and its two tracing dependencies.
- `jolt-atlas-verifier`: depends on `atlas-trace` with `default-features = false`.
  `onnx_proof/relation.rs` re-exports the relation core by name, not by glob.
- `jolt-atlas-core`: enables `prover` on the four protocol crates and `logging` on `common`.
- `scripts/perimeter.sh`: prints the first-party source files the verifier's build compiles,
  from the dep-info of `cargo check -p jolt-atlas-verifier`.
- `docs/verifier-perimeter.txt`: that list, checked in.
- `.github/workflows/rust.yml`, job `verifier-isolation`: runs clippy on the verifier alone,
  checks each gated crate without `prover`, and diffs `scripts/perimeter.sh` against the
  checked-in list.

### Invariants

Relied on, not changed:

1. **The `atlas-trace` behavioral contract**, all of it. Gating `eval` moves no semantics.
   *Mechanism:* the tracer's golden and property tests, which run with `exec` on.
2. **The protocol is unchanged.** *Mechanism:* `testing/digests.rs` is not edited by this PR.
3. **The verifier crate's own rules.** No panic, Trace-free. *Mechanism:* the crate-root
   lints; the `disallowed-types` entries for the trace types are removed because the types
   no longer exist in the verifier's build (invariant A).

New:

- **A. No prover item is nameable from the verifier's build.** *Mechanism:* the compiler, in
  the CI step `cargo check -p jolt-atlas-verifier`: a gated item does not exist there. This
  replaces the prover entries of `clippy.toml` and the `::prover` path grep. It holds only
  for that command; a `--workspace` build unifies features and turns `prover` on for every
  crate, so a workspace build proves nothing about this invariant.
- **B. Each gated crate builds on both sides of the gate.** *Mechanism:* CI runs
  `cargo check -p <crate>` and `cargo check -p <crate> --features prover` for
  `atlas-algebra`, `atlas-lookup-tables`, `atlas-sumcheck` and `atlas-pcs`,
  `cargo check -p atlas-trace --no-default-features`, and
  `cargo clippy -p jolt-atlas-verifier` under `-D warnings`.
- **C. The perimeter is a reviewed list.** *Mechanism:* CI fails when
  `scripts/perimeter.sh` differs from `docs/verifier-perimeter.txt`; a pull request that
  grows or shrinks the perimeter shows the file list changing.
- **D. No prover file is in the perimeter.** *Mechanism:* CI fails if a line of
  `docs/verifier-perimeter.txt` matches `/prover\.rs$`, `/prover/`, `/exec\.rs$`,
  `/trace\.rs$`, `/trace_onehot/`, `/split_eq\.rs$`, `/ra\.rs$` or
  `/mles_product_sum\.rs$`.
- **E. A shared file holds no prover item.** In a file on the perimeter list, no item exists
  only for the prover. *Mechanism:* the compiler for unused private items (the dead-code
  lint under `-D warnings` in the verifier-alone clippy step); review-held for public items.
- **F. Tests still run.** Tests that use prover items compile only with the feature.
  *Mechanism:* each gated crate lists itself as a dev-dependency with `prover` on, as
  `atlas-transcript` does today, so `cargo nextest run --workspace` and `-p <crate>` run the
  same tests as before; the test count of the workspace run does not drop.

### Non-Goals

- **LayerZero's `akita-*` crates.** `akita-pcs` depends on `akita-prover` and
  `akita-cpu-backend` unconditionally and has no prover feature, so the verifier's build
  compiles them. They stay in the perimeter document as an external dependency at a pinned
  revision. Removing them needs an upstream change or a different PCS.
- **rayon.** The verifier's evaluation paths (`MultilinearPolynomial::evaluate`,
  `evaluate_mle`, `EqPolynomial::evals`) use it; removing it would cost verify time. It is
  not a prover-only dependency.
- **The unreduced-product ladder of `JoltField`.** Its accumulators are associated types of
  the trait and cannot be gated without splitting the trait.
- Moving prover code out of the protocol crates into `jolt-atlas-core`, or splitting any
  crate in two.
- Shrinking what the verifier needs. This spec draws the line; it does not move it.
- Any change to a proof, the transcript, or a public API with the feature on.

## Evaluation

### Acceptance Criteria

- [ ] `testing/digests.rs` is not modified, and its four tests pass.
- [ ] `cargo check -p jolt-atlas-verifier` and `cargo clippy -p jolt-atlas-verifier` pass
      under `RUSTFLAGS="-D warnings"`.
- [ ] `cargo tree -p jolt-atlas-verifier -e features` shows no `prover` feature on
      `atlas-algebra`, `atlas-lookup-tables`, `atlas-sumcheck`, `atlas-pcs` or
      `atlas-transcript`, no `exec` on `atlas-trace` and no `logging` on `common`.
- [ ] Invariant B: the checks listed there pass and are steps of `verifier-isolation`.
- [ ] Invariant C: `docs/verifier-perimeter.txt` exists and equals the output of
      `scripts/perimeter.sh`; the CI step fails when a line is added to one and not the
      other.
- [ ] Invariant D: the CI pattern check passes.
- [ ] In `atlas-sumcheck/src`, every `mod prover` declaration carries
      `#[cfg(feature = "prover")]`, and `mles_product_sum.rs` is not on the perimeter list.
- [ ] `jolt-atlas-verifier/clippy.toml` has no `disallowed-methods` entry and no
      prover or trace entry in `disallowed-types`; the `::prover`/`::witness` path grep and
      the `Trace` grep are removed from the workflow.
- [ ] `jolt-atlas-verifier/src` contains no `pub use atlas_sumcheck::relation::*`.
- [ ] `atlas_trace::trace::ModelExecutionIO` resolves with `exec` on, and
      `atlas_trace::io::ModelExecutionIO` without it.
- [ ] Invariant F: the workspace run reports at least as many tests as on the parent commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] `docs/verifier.md`'s perimeter table names files by reference to
      `docs/verifier-perimeter.txt` and keeps one row per crate for what the crate
      contributes.

### Testing Strategy

Existing tests that must keep passing unchanged: all of them. This spec changes which build
contains an item, not what any item does. The digests cover the prover and verifier paths end
to end; the tracer's `kernel_goldens`, `golden_tables` and `properties` cover `atlas-trace`
with `exec` on; the importer's golden hashes cover the compile path.

New tests: none in Rust. The new checks are CI steps (invariants B, C and D).

CI coverage: the `verifier-isolation` job, extended; the workspace job. Not protocol-level,
so the heavy job is not required. No proof format change, so no new `mutate::Site`.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16`, against this spec's parent commit, on the
benchmark machine.

Expected: no stage moves; with `prover` on the compiled code is the same. Bounds: the gate's
defaults; proof size equal to the byte. The verifier-alone build compiles fewer files, which
is not gated.

## Design

### Architecture

```text
jolt-atlas-core ──(prover)──▶ atlas-sumcheck ──▶ atlas-algebra, atlas-lookup-tables, atlas-transcript
                ──(prover)──▶ atlas-pcs
                ──(exec, default)──▶ atlas-trace
jolt-atlas-verifier ──(no feature)──▶ the same crates, verifier side only
```

A feature is additive, so the workspace build is the union and behaves as today. The
verifier-alone build is the only place the gate can be observed, which is why every
mechanism above is a `-p` command.

Within a crate the rule is the one `atlas-sumcheck` already follows by convention: prover
items live in a module's `prover` submodule, and now the submodule is behind the feature. A
trait with methods on both sides (`CommitmentScheme`) gates the prover methods in the trait
and in each impl. A trait whose whole purpose is the prover (`PolynomialBinding`,
`StreamingCommitmentScheme`, `ProvingTranscript`) is gated as a whole. In `atlas-trace`,
`OpSpec` keeps its model-facing methods (shapes, padding, bounds) and gates `eval`; the
kernels in `op/*.rs` that only `eval` calls are gated with it, and the trace's witness types
with them.

`scripts/perimeter.sh` runs `cargo check -p jolt-atlas-verifier`, reads the `.d` dep-info
files rustc writes for the first-party crates, and prints the sorted, repository-relative
source paths. A module behind a disabled `cfg` is never loaded, so its file is absent. The
script prints paths only; line counts would make every edit a perimeter diff.

### Alternatives Considered

- **Keep the lists.** Extend `clippy.toml` and fix the grep. Rejected: a deny list fails
  open, each new prover item must be remembered, and the workspace clippy run is the only one
  that reads the list, with `prover` on.
- **Split each protocol crate into a verifier crate and a prover crate.** The strongest
  boundary. Rejected for now: it doubles the crate count, forces public visibility on
  internals the two halves share, and `specs/own-sumcheck-engine.md` and its siblings chose
  to keep upstream's two-sided crate shape. The feature gate gets the mechanical perimeter
  without that; a later split would start from the same file moves.
- **A `verifier` feature that selects the small side**, with the prover as the default.
  Rejected: features are additive, so any crate in a build that forgot the flag would switch
  the prover back on; the absence of a feature is the only state a dependent cannot undo.
- **Gate rayon and give the verifier serial evaluation.** Rejected as a non-goal above: a
  verify-time cost for a dependency that is not prover code.
- **Fork `akita-pcs` to make its prover optional.** Rejected: the program moved away from
  carrying patched upstream crates, and the PCS choice is open.

## Documentation

- `docs/verifier.md`: "Trust perimeter" is rewritten around the checked-in list and the
  command that produces it; the three-mechanisms paragraph becomes the feature gate; the
  "outside the lints' reach" paragraph stays.
- `docs/upstream-inventory.md`: the "ship both sides" paragraph records the gate.
- `CLAUDE.md`: the crate-map entries for `atlas-sumcheck` ("which the verifier may not name
  (CI path check)") and `jolt-atlas-verifier` (the `clippy.toml` and grep sentences); the
  essential commands gain `scripts/perimeter.sh`.
- `CONTRIBUTING.md`: the verifier-alone commands, and "a changed
  `docs/verifier-perimeter.txt` is reviewed".
- `docs/pull-requests.md`: new prover code goes in a `prover` submodule.
- Each gated crate's `Cargo.toml` comment and crate-level doc.

## Execution

One crate per commit, leaves first, each leaving the workspace build and the verifier-alone
build green.

1. `atlas-transcript`: finish the existing gate.
2. `atlas-algebra`, then `atlas-lookup-tables`.
3. `atlas-sumcheck`: move the prover code out of shared files, then gate.
4. `atlas-pcs`: the trait, then the Akita adapter.
5. `atlas-trace`: `io.rs`, then `exec`.
6. `common`'s `logging`; the verifier's named re-exports.
7. `scripts/perimeter.sh`, the checked-in list, the CI steps; delete the `clippy.toml`
   entries and the greps; docs.

## References

- `jolt-atlas-verifier/clippy.toml`, `.github/workflows/rust.yml` (`verifier-isolation`),
  `docs/verifier.md`, `docs/upstream-inventory.md`.
- `atlas-transcript/Cargo.toml` (the existing `prover` feature and its self dev-dependency).
- `atlas-sumcheck/src/{mles_product_sum.rs, opening_reduction/mod.rs,
  evaluation_reduction/mod.rs, ps_shout/mod.rs, shout/mod.rs}`.
- `atlas-pcs/src/{commitment_scheme.rs, akita/mod.rs, akita/backend/}`.
- `atlas-algebra/src/poly/`, `atlas-trace/src/{trace.rs, exec.rs, op/mod.rs}`.
- `specs/verifier-audit-simplification.md` (the Trace-free crate and its three mechanisms).
