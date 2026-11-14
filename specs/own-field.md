# Spec: own-field — one field crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-poly.md` (phase 5; every other vendored crate is gone, so `jolt-field`'s only consumers are `atlas-algebra`, the `jolt-poly` compatibility package and `akita-*`) |
| Supersedes  | `third_party/PINS.md` sections `jolt-field` and `The arkworks fork pin` (the field part); `specs/adoption-program.md` stage 2 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`atlas-algebra::field` is a trait on a trait: our `JoltField` extends upstream
`jolt_field::JoltField`, and `Fp128` is a newtype over upstream's `Prime128OffsetA7F7` because
orphan rules stop us implementing `CanonicalSerialize`, `Div`, `UniformRand` and `Allocative`
on a foreign type. The Solinas arithmetic that dominates every sumcheck round (fp128 kernels,
unreduced products, SIMD packing, parallel folds, about 5,000 lines under
`jolt-field/src/solinas/`) is upstream's, and next to it sit BN254 and binary-field modules
no build enables. This phase makes the field ours: one crate, one trait, the Solinas kernels
in it, the newtype gone, and `jolt-field` kept only as the package name `akita-*` binds.

## Intent

### Goal

Move `jolt-field`'s `solinas/*`, `limbs`, `unreduced`, `algebra`, `packed`, `schedules`,
`fp128_accumulators`, `signed` and `ops` modules into `atlas-algebra::field`, define `Fp128`
directly on the moved type (no newtype), collapse `atlas_algebra::field::JoltField` and
`jolt_field::JoltField`/`Field`/`Ring` into one trait hierarchy owned here, keep an owned
`jolt-field` package exposing the `akita-*` surface, and delete `third_party/jolt-field`,
with proofs byte identical.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job; the field is under
  every byte of the proof.
- **The kernels are the kernels.** `solinas/fp128.rs`, `solinas/unreduced.rs`,
  `solinas/word.rs`, `solinas/packed/*`, `solinas/parallel.rs`, `fp128/add_sub.rs` and the
  `asm/` sources and `build.rs` are moved unchanged in the first commit; the PR diff for
  those files is a rename. Mechanism: `git diff -M` in the PR shows 100% renames for them
  (review-held); the upstream in-source tests (the Solinas test modules, the `unreduced`
  ladder tests) run under `cargo nextest run -p atlas-algebra`.
- **The ladder is unchanged.** Our unreduced-product ladder (`mul_u64_unreduced`,
  `mul_to_product_accum`, `reduce_*`, `fmadd`) keeps its semantics and its `probe_assert!`
  self-checks. Mechanism: the existing ladder tests in `atlas-algebra`; digests.
- **The `akita-*` surface is exact.** The owned `jolt-field` package exports `Prime`,
  `Field`, `Ring`, `ExtField`, `FpExt*`, `CanonicalEncoding`, `Unreduced`, `Fold`,
  `MulBaseUnreduced`, `PseudoMersenne`, `WithCommitAccumulator`, `AdditiveGroup`, `Zero`,
  `One`, `Fp128x8i32` and `solinas::parallel::*` with the signatures `akita-*` calls, as
  re-exports of the moved items. Mechanism: the compiler; Akita digests.
- **No BN254, no binary field.** `bn254/` and `binary/` are not copied; the `bn254` feature
  and `ark-bn254` dependency disappear from the workspace. Mechanism: the CI no-BN254 check
  (now trivially true) and `Cargo.lock` review.
- **`asm` stays on** for the kernels that use it (`features = ["solinas", "asm"]` today).
  Mechanism: digests do not see this; the performance gate does.
- **`JoltField` is one trait.** `atlas_algebra::field::JoltField` no longer has an upstream
  supertrait; `every_field_satisfies_the_upstream_bound` is replaced by a test that the
  `akita-*` bound (`jolt_field::Field` from the compatibility package) is satisfied by
  `Fp128`. Mechanism: the compiler and that test.

### Non-Goals

- Changing the prime, the reduction algorithm, the ladder, or the challenge width.
- Removing `ark-ff`/`ark-serialize`: phase 7 (`specs/drop-arkworks.md`). This phase still
  uses `BigInt` and `CanonicalSerialize` from the fork pin.
- The extension field (`solinas/ext.rs`, `extension.rs`): moved because `akita-*` uses it;
  atlas does not and the compatibility package is its only exporter.
- Forking `akita-*`.

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-field` is deleted; `third_party/` is empty and removed; `PINS.md` is
      deleted and its remaining policy text (the `akita-*` pin, the `[patch]` rationale) moved
      to `docs/upstream-inventory.md`.
- [ ] The owned `jolt-field` package exists at the root, depends only on `atlas-algebra`, and
      `akita-*` builds against it; the `[patch."https://github.com/a16z/jolt"]` entries point
      at the two owned packages.
- [ ] `atlas_algebra::field::Fp128` is the moved Solinas type (no `pub Inner` newtype);
      `JoltField` has no `jolt_field` supertrait; no source under `atlas-*`/`jolt-atlas-*`
      names `jolt_field`.
- [ ] Kernel files land as renames; upstream's field tests pass under `atlas-algebra`.
- [ ] GPT-2 witness + commitment and IOP stages within 3% of baseline (same machine, same
      day); heavy job green on `main`.
- [ ] Digest pins unchanged; CI green; `docs/verifier.md` perimeter row for `third_party/`
      removed; crate maps and roadmap updated.

### Testing Strategy

Workspace Test job (digests, zoo, soundness, op-surface), isolation job, heavy job; upstream's
in-source field tests ported. No proof-format change. The `prove` example is the benchmark.

### Performance

Field multiplication is the floor of prover time. Benchmark: the `prove` example on
`models/gpt2`, `--trace-terminal`, all three prover stages, three runs, baseline on `dev`.
Gate: each stage within 3%. Expected: no change (same kernels, same `asm`); removing the
newtype's forwarding may improve inlining marginally, not a target. The `asm` feature must be
shown enabled in the release build (`cargo build --release -vv` output or a `cfg` probe test).

## Design

### Architecture

`atlas-algebra/src/field/` gains `solinas/` (kernels, packed SIMD, parallel folds, ext),
`limbs.rs`, `unreduced.rs`, `algebra.rs` (the `Ring`/`Field`/`Prime` traits, which become the
lower layer of our `JoltField`), `packed.rs`, `schedules.rs`, `fp128_accumulators.rs`,
`signed.rs`, `ops.rs`; `asm/` and `build.rs` move to `atlas-algebra`. `fp128.rs` becomes the
`JoltField` impl on the moved type plus the ladder; `folded_accum.rs` stays. The root-level
`jolt-field/` compatibility package re-exports the `akita-*` surface. The arkworks
`BigInt`/`CanonicalSerialize` uses stay until phase 7.

### Alternatives Considered

- **Keep `jolt-field` vendored and only delete the newtype.** Impossible: the newtype exists
  because of orphan rules on a foreign type.
- **Fork `akita-*` so no compatibility package is needed.** Deferred (roadmap).
- **Rename the owned package `atlas-field`.** Blocked by the `akita-*` binding; revisited with
  the fork decision.

## Documentation

`PINS.md` deleted; `docs/upstream-inventory.md` (`field/` row: owned; provenance of the
kernels; the `akita-*` pin and `[patch]` rationale), `docs/verifier.md`, `CLAUDE.md`/`AGENTS.md`
(the `third_party/` paragraph removed; `atlas-algebra` described as owning the field),
`docs/roadmap.md`.

## Execution

1. Move kernels and traits as renames under `atlas-algebra::field`, keep the newtype,
   compile; port tests. 2. Collapse the newtype and the trait layering; re-point
   `atlas-poly`'s and the engine's bounds. 3. Compatibility package, `[patch]`, delete the
   crate and `third_party/`; docs; GPT-2 runs recorded.

## References

`a16z/jolt` `crates/jolt-field` at `f3265b5a` (and PR #1676, the fp128 origin recorded in
`atlas-algebra/src/field/fp128.rs`); `third_party/PINS.md`; `docs/roadmap.md`; the
`akita-*` manifests at `dfdbd767` (`jolt-field` features they enable).
