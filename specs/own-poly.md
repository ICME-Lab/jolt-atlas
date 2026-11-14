# Spec: own-poly — one polynomial library

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-transcript.md` (phase 4); `specs/own-sumcheck-engine.md` (the engine is the main `UnivariatePoly` consumer) |
| Supersedes  | `third_party/PINS.md` sections `jolt-poly`, `jolt-utils`; `specs/adoption-program.md` stage 4b's "operand polynomials stay ours" split |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The workspace has two polynomial libraries. `atlas-algebra::poly` (8,067 lines) holds the
operand polynomials every op relation binds: `MultilinearPolynomial`, `DensePolynomial`,
`EqPolynomial`, `SplitEqUnreducedFold`, `UniPoly`, Lagrange, one-hot, prefix-suffix, `ra`,
identity and range-mask polynomials, bound with our 125-bit challenges. `jolt-poly`
(6,365 lines) holds upstream's: `UnivariatePoly` and `CompressedPoly` (every round message),
`GruenSplitEqPolynomial` (every Gruen round), `EqPolynomial`, `Polynomial`, `OneHotPolynomial`,
`Point`, `OmittedConstantPoly`, and a second dense, eq, split-eq, Lagrange, lt and identity
implementation. Stage 4b measured upstream's full-field binding slower than ours and kept
both. The result is two `EqPolynomial`s, two split-eq types and two univariate types in the
same inner loop. This phase makes one library, ours, and keeps the faster implementation of
each pair, measured.

## Intent

### Goal

Merge `jolt-poly` into `atlas-algebra::poly` as one owned polynomial library (one type per
concept, the faster implementation kept per the measurement below), fold `jolt-utils`' one
used helper into `atlas-algebra::utils`, keep an owned crate under the package name
`jolt-poly` exposing only the items `akita-*` binds, and delete `third_party/jolt-poly` and
`third_party/jolt-utils`, with proofs byte identical.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job. Polynomial
  evaluation and binding are deterministic field arithmetic; a wrong merge changes a round
  message, so a digest.
- **One type per concept.** After the phase there is one eq polynomial, one split-eq (Gruen)
  type, one univariate round-message type with its compressed form, one dense multilinear
  type and one opening point type in the workspace. Mechanism: review of
  `atlas-algebra/src/poly/mod.rs` exports; a CI grep that `jolt_poly::` appears nowhere in
  `atlas-*`/`jolt-atlas-*`.
- **No slower inner loop.** For each merged pair the kept implementation is the one that
  measures faster on the real GPT-2 IOP stage, or ours when within noise. Mechanism: the
  measurements recorded in the PR (per the benchmark below); the `+3%` gate on the IOP stage.
- **The `akita-*` surface is exact.** The `jolt-poly` package our root `[patch]` points at
  exports `UnivariatePoly`, `OmittedConstantPoly`, `MultilinearPoly`, `OneHotPolynomial`,
  `OneHotIndexOrder`, `Point`, `Polynomial`, `HIGH_TO_LOW` with the signatures `akita-*`
  calls, implemented over the merged library's types (type aliases or thin newtypes).
  Mechanism: the compiler (`akita-*` builds against it) and the Akita digests.
- **`evaluate_mle` discipline kept** (`CLAUDE.md`: prefer `x.evaluate_mle(&r)`). Mechanism:
  review-held.
- **Contract untouched.** No trace change. Mechanism: `atlas-trace/` absent from the diff.

### Non-Goals

- Changing any binding order, challenge width or fold algorithm. Where the two
  implementations differ in algorithm, the kept one is the one atlas runs today for that
  call site, unless the measurement says otherwise.
- Prefix-suffix, `ra`, identity, range-mask and multiquadratic polynomials: atlas-only, not
  touched beyond module moves.
- Deleting the `jolt-poly` package name: it stays as long as `akita-*` is external
  (`docs/roadmap.md`, open decisions).

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-poly` and `third_party/jolt-utils` are deleted; the owned `jolt-poly`
      compatibility package exists at the root, depends only on `atlas-algebra`, and is what
      the root `[patch."https://github.com/a16z/jolt"]` names; `akita-*` builds.
- [ ] No source under `atlas-*`/`jolt-atlas-*` names `jolt_poly` or `jolt_utils`; the merged
      library exports one type per concept (list in the PR).
- [ ] Per-pair measurement table in the PR (dense, eq, split-eq, univariate; GPT-2 IOP stage,
      three runs each) and the kept implementation named with its number.
- [ ] GPT-2 IOP stage within 3% of the `dev` baseline (same machine, same day); the heavy job
      green on `main`.
- [ ] Digest pins unchanged; `cargo nextest run -p atlas-algebra` passes with upstream's
      ported in-source tests; CI green.
- [ ] `PINS.md`, inventory (`poly/` rows), `docs/verifier.md`, crate maps, roadmap updated.

### Testing Strategy

Workspace Test job (digests, zoo, soundness, op-surface), isolation job, heavy job; upstream's
in-source poly tests ported into `atlas-algebra`. No proof-format change. The `prove` example
is the benchmark driver.

### Performance

This is the one phase that can move prover time either way. Benchmark: the `prove` example on
`models/gpt2` (real download), `--trace-terminal`, IOP proving stage and opening reduction
stage; baseline on `dev` before the branch, three runs each. Gate: IOP stage within 3% (worse)
of baseline; anything faster is recorded. Memory: peak within 5% (the README machine has
16 GB and GPT-2 peaks at 6.2 GB).

## Design

### Architecture

`atlas-algebra/src/poly/` absorbs `jolt-poly`'s modules, pair by pair:
`univariate.rs`+`compressed_univariate.rs` with `unipoly.rs`; `split_eq.rs` with
`split_eq_poly.rs`; `eq.rs` with `eq_poly.rs`; `dense.rs`+`multilinear.rs` with
`dense_mlpoly.rs`+`multilinear_polynomial.rs`; `point.rs` with `opening_point.rs`;
`lagrange.rs`, `lt.rs`, `identity.rs`, `eq_plus_one.rs`, `one_hot.rs` likewise; `mle.rs`,
`omitted_constant.rs`, `evaluation_claim.rs`, `binding.rs` moved. A root-level `jolt-poly/`
package (owned, about 100 lines) re-exports the `akita-*` surface. `jolt-utils`'
`unsafe_allocate_zero_vec` joins `atlas-algebra::utils`. Consumers: `atlas-sumcheck::engine`
and the relation provers (`UnivariatePoly`, `GruenSplitEqPolynomial`), `atlas-pcs`
(`EqPolynomial`, `Polynomial`), the op provers in `jolt-atlas-core`.

### Alternatives Considered

- **Keep both and only own `jolt-poly`.** Rejected: the duplication is the problem.
- **Adopt upstream's types wholesale and delete ours.** Rejected by stage 4b's measurement
  (our binding was faster); this phase measures again rather than assuming either way.
- **Fork `akita-*` to drop the compatibility package.** Deferred (roadmap).

## Documentation

`PINS.md`, `docs/upstream-inventory.md` (every `poly/` row collapses to "owned, merged"),
`docs/verifier.md`, `CLAUDE.md`/`AGENTS.md` (the "Fields and polynomials share a crate"
sentence stays true), `docs/roadmap.md`.

## Execution

One PR per pair is acceptable if the measurement for a pair is contentious; otherwise one PR
with one commit per pair, each digest-green, the measurement table in the last commit. The
compatibility package lands first so `akita-*` never breaks. `jolt-utils` goes in the last
commit.

## References

`a16z/jolt` `crates/jolt-poly`, `crates/jolt-utils` at `f3265b5a`; `docs/upstream-inventory.md`
`poly/` rows and the stage-4b measurement; `docs/roadmap.md`; memory of the stage-4b split
(`specs/adoption-program.md`).
