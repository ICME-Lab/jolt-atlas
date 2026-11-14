# Spec: own-pcs-adapter — the Akita adapter, openings and commitment traits in atlas-pcs

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/atlas-claims.md` (phase 1; `jolt-akita` already carries its first source delta there) |
| Supersedes  | `third_party/PINS.md` sections `jolt-akita`, `jolt-claims and its dependency cone` (the `jolt-openings`/`jolt-crypto` rows) |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`atlas-pcs::akita` is a 1,100-line adapter over `jolt-akita`, which is itself a 9,100-line
adapter over LayerZero's `akita-*` crates, through `jolt-openings`' `CommitmentScheme` trait
and `jolt-crypto`'s `Commitment` trait. Two adapter layers for one PCS is the duplication this
program removes: the schedule catalog, trace one-hot encoding, native batching and prefix
packing live in the vendored crate, and our layer re-derives class setups and row layouts on
top. `jolt-crypto` also drags 5,400 lines of BN254 elliptic-curve code that no build enables.
This phase moves the adapter in-house: `jolt-akita`'s modules become `atlas-pcs::akita::*`,
the three `jolt-openings` items we use become `atlas-pcs` types, the `Commitment` trait is
ours, and `atlas-pcs` depends on `akita-*` directly.

## Intent

### Goal

Move `jolt-akita`, the used parts of `jolt-openings` (`CommitmentScheme`, `GroupOpeningClaim`,
`PrefixPackedClaims`/`PrefixPackedLayout`, `TransparentObjectSetup`, `OpeningsError`, the group
metadata traits) and `jolt-crypto`'s `Commitment` trait into `atlas-pcs`, delete the three
vendored crates, and keep every commitment, opening proof and transcript byte identical.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: the digest tests in the Test job;
  the Akita class openings and commitments are in the proof bytes, so any change to packing,
  schedule selection or the one-hot encoding changes a digest.
- **One scheme interface.** After the phase `atlas-pcs::commitment_scheme::CommitmentScheme` is
  the only PCS trait in the workspace; `jolt-openings`' trait is not re-created under a new
  name. The Akita adapter implements ours directly. Mechanism: grep in CI's verifier-isolation
  job for `UpstreamScheme`/`jolt_openings` (fails if present); review of the trait's method
  list against today's (`setup_prover`, `commit`, `batch_commit*`, `prove`, `prove_rlc`,
  `verify*`, `combine_hints`), which `jolt-atlas-verifier/clippy.toml` already enumerates and
  must keep enumerating with the same prover-item split.
- **Verifier isolation.** `clippy.toml`'s `disallowed-methods` for the prover side of the
  scheme are updated to the new paths in the same PR; the CI path grep still holds. Mechanism:
  CI job.
- **`akita-*` pin unchanged** (`dfdbd7674cab5cfbe95243e947d1b10a982f232b`); the features
  `jolt-akita` enables (`parallel`, `transcript-blake2b` on `akita-pcs`) move to `atlas-pcs`'s
  manifest verbatim. Mechanism: `Cargo.lock` diff reviewed; the schedule catalog test below.
- **The packaged schedule catalog still loads and matches the admitted keys.** Mechanism:
  `jolt-akita`'s `schedule_artifacts` tests and the phase-1 admitted-width equality test move
  with the code and run under `cargo nextest run -p atlas-pcs`.
- **No BN254 in the graph** (unchanged); after this phase the `bn254` features of the deleted
  crates no longer exist, and the CI check is kept for `jolt-field`'s. Mechanism: CI.
- **Contract untouched.** No trace or op change. Mechanism: `atlas-trace/` and
  `jolt-atlas-core/src/onnx_proof/ops/` absent from the diff (review-held).

### Non-Goals

- Forking `akita-*`. It stays a git dependency at the pinned rev (`docs/roadmap.md`, "Open
  decisions").
- Changing the Akita class layout (`ONE_HOT_COLUMNS = 64`, K=16, dense cap `2^28`), the
  schedule profiles, or the hint/proof wire format.
- Reducing the adapter's two-layer structure in the same PR. The move is mechanical; a
  follow-up may merge `atlas-pcs::akita` and the moved `scheme.rs`, gated on digests.
- `jolt-crypto`'s `VectorCommitment` and the `committed` feature of `jolt-sumcheck`: deleted
  with the crate; `jolt-sumcheck`'s `committed` feature stays declared empty until phase 3.

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-akita`, `third_party/jolt-openings`, `third_party/jolt-crypto` are
      deleted; the root `Cargo.toml` has no `exclude` list and no `jolt-openings`/`jolt-akita`
      workspace dependency; `akita-*` dependencies appear in `atlas-pcs/Cargo.toml` only.
- [ ] No source under `atlas-*`/`jolt-atlas-*` names `jolt_akita`, `jolt_openings` or
      `jolt_crypto` (grep, in the CI isolation job).
- [ ] `jolt-akita`'s two binaries (`gen_jolt_schedules`, `check_grouped_schedules`) are
      either moved as `atlas-pcs` examples or deleted with a stated reason in the PR.
- [ ] `cargo nextest run -p atlas-pcs` passes with the moved tests (trace one-hot, schedule
      artifacts, adapters, native batching).
- [ ] Digest pins unchanged; CI green; heavy job green on `main`.
- [ ] `docs/verifier.md` perimeter row for `third_party/` lists only
      `jolt-sumcheck`, `jolt-poly`, `jolt-field`, `jolt-transcript`, `jolt-utils` and the
      `akita-*` crates; `PINS.md`, inventory, crate maps updated.

### Testing Strategy

Existing: the workspace Test job (digests, zoo round-trips), isolation job, heavy job. Moved:
every in-source test of `jolt-akita` (about 1,000 lines under `trace_onehot/tests*`,
`schedule_artifacts/tests.rs`) runs under `atlas-pcs`. New: the grep step. No `mutate::Site`
changes (proof format unchanged). No `op_surface.rs` changes.

### Performance

Prover and verifier time unchanged by construction. Sanity check: the `prove` example on
`models/gpt2-tiny` with `--trace-terminal`, witness + commitment and opening-proof stages
within noise of the previous run, quoted in the PR. Build time: expected small decrease
(`jolt-crypto`'s curve code was compiled but dead); recorded with `cargo build --timings` as in
phase 1.

## Design

### Architecture

`atlas-pcs/src/akita/` gains `jolt-akita`'s modules (`scheme`, `adapters`, `native_batching`,
`trace_onehot/*`, `schedule_registry`, `schedule_artifacts`, `schedules`, `configs`,
`one_hot_family`, `shape_guard`, `family_contract`); today's `atlas-pcs/src/akita.rs` becomes
`akita/mod.rs`. The schedule fixtures under `third_party/jolt-akita/schedules/` move to
`atlas-pcs/schedules/` and the default-directory lookup is updated. `jolt-openings`' used
items go to `atlas-pcs/src/openings.rs` (prefix packing, group claims, error);
`jolt-crypto::Commitment` becomes a trait in `atlas-pcs::commitment_scheme`. The `akita-*`
manifest rows move to `atlas-pcs/Cargo.toml` with the same rev and features. The root
`[patch."https://github.com/a16z/jolt"]` stays (the `akita-*` crates still bind `jolt-field`
and `jolt-poly`).

### Alternatives Considered

- **Keep `jolt-openings` as the scheme trait and own only `jolt-akita`.** Rejected: two
  scheme traits would remain and the `Upstream`/`UpstreamScheme` aliasing in `atlas-pcs` is
  the duplication being removed.
- **Fork `akita-*` now.** Rejected for this phase (size; we do not modify it). Recorded as an
  open decision in the roadmap.
- **Merge the two adapter layers in the same PR.** Rejected: a mechanical move is reviewable
  against digests; a re-architecture is not.

## Documentation

`PINS.md` (three sections removed), `docs/upstream-inventory.md` (provenance rows for the
moved modules), `docs/verifier.md` (perimeter), `CLAUDE.md`/`AGENTS.md` (crate map: `atlas-pcs`
owns the Akita adapter; `akita-*` by git), `docs/roadmap.md` (phase 2 landed).

## Execution

1. Move `jolt-akita` sources and fixtures under `atlas-pcs::akita`, keeping its imports of
   `jolt_openings`/`jolt_crypto`; gate `cargo nextest run -p atlas-pcs` and digests.
2. Replace the `jolt-openings` and `jolt-crypto` imports with `atlas-pcs` items; delete the
   three crates, manifests and lock entries; `clippy.toml` paths.
3. CI grep step, docs, spec status.

## References

`third_party/PINS.md`; `docs/roadmap.md`; `specs/adoption-program.md` stage 7 (Akita adoption);
`a16z/jolt` `crates/jolt-akita`, `crates/jolt-openings`, `crates/jolt-crypto` at `f3265b5a`;
LayerZero `akita` at `dfdbd767`.
