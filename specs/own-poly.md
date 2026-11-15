# Spec: own-poly — one polynomial library

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | implemented                    |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-transcript.md` (phase 4); `specs/own-sumcheck-engine.md` (the engine is the main `UnivariatePoly` consumer) |
| Supersedes  | `third_party/PINS.md` sections `jolt-poly`, `jolt-utils`; `specs/adoption-program.md` stage 4b's "operand polynomials stay ours" split |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The workspace carries two polynomial libraries. `atlas-algebra::poly` (8,067 lines) holds the
operand polynomials every op relation binds (`MultilinearPolynomial`, `DensePolynomial`,
`EqPolynomial`, `UniPoly`, Lagrange, one-hot, prefix-suffix, `ra`, identity and range-mask),
bound with our 125-bit challenges. `third_party/jolt-poly` (6,365 lines) is upstream's. In
practice the overlap is narrower than two parallel libraries:

- **Split-eq is already one type.** Every Gruen round runs on upstream's
  `GruenSplitEqPolynomial`; `poly/split_eq_poly.rs` is only atlas's `SplitEqUnreducedFold`
  extension trait over it.
- **Most of upstream's second implementations are dead in atlas.** Upstream's dense, Lagrange,
  lt, identity and eq-plus-one have no atlas caller. Its `EqPolynomial` and `Polynomial` are used
  once, on the commit side (`atlas-pcs/src/akita/mod.rs`).
- **Three pairs are live duplicates:**
  - `UniPoly`/`CompressedUniPoly` (the proof wire form: `SumcheckInstanceProof`, evaluation
    reduction's `h`) against `UnivariatePoly`/`CompressedPoly` (every round message, 25 files);
  - `OpeningPoint` against `Point` (`atlas-pcs/src/openings/prefix.rs`);
  - the operand `MultilinearPolynomial`/`OneHotPolynomial` against upstream's
    `Polynomial`/`MultilinearPoly`/`OneHotPolynomial`/`OneHotIndexOrder`, which only the Akita
    adapter in `atlas-pcs/src/akita/backend/` uses.

This phase makes one owned library: upstream's live types move into `atlas-algebra::poly`,
each live pair collapses to one type carrying whatever upstream does better, and dead code
goes, upstream's and atlas's alike.

## Intent

### Goal

Make `atlas-algebra::poly` the one polynomial library:
- move upstream's live types in;
- collapse each live duplicate pair to one type, adopting upstream's improvements as updates to
  the kept code;
- strip all dead code, pre-existing or moved;
- drop the redundant `_poly`/`_polynomial` suffixes from module names;
- fold `jolt-utils`' one used helper into `atlas-algebra::utils`;
- keep an owned root-level package under the name `jolt-poly` exposing exactly what `akita-*`
  binds;
- delete `third_party/jolt-poly` and `third_party/jolt-utils`, with proofs byte identical.

**Scope rule.** For each live concept there is one type. Where upstream's implementation of a
method is an optimization or improvement (faster kernel, exceptional-point handling, cleaner
API), it replaces ours; otherwise ours stays. Nothing is ported only because upstream has it.

**Univariate decision.** `UnivariatePoly` and `CompressedPoly` (upstream's names, which
`akita-*` binds) are the single types. They gain `CanonicalSerialize`/`CanonicalDeserialize`
impls with exactly `UniPoly`/`CompressedUniPoly`'s byte layout, so the proof wire form is
unchanged. Atlas's methods with live callers (for example `divide_with_remainder`,
`eval_as_univariate`, `check_sum_evals`, `from_linear_times_quadratic_with_hint`, challenge-typed
`evaluate`) move into an `impl<F: JoltField>` block. `UniPoly` and `CompressedUniPoly` are
deleted; `SumcheckInstanceProof` and evaluation reduction's `h` switch type.

**Point decision.** `OpeningPoint` and `Point` collapse to one endianness-tagged point type;
the kept one is whichever serves the most callers with the least conversion, and the other's
live methods join it.

**Dead-code rule.** A `pub` item or method with no caller in any workspace crate's non-test
code (its own unit tests aside) is deleted together with those tests. This applies to
`atlas-algebra::poly` and to the moved upstream code. The PR lists what was removed.

**Module names.** `atlas-algebra/src/poly/` modules drop the redundant suffix:
`eq_poly` → `eq`, `eq_plus_one_poly` → `eq_plus_one`, `lt_poly` → `lt`, `identity_poly` →
`identity`, `signed_identity_poly` → `signed_identity`, `lagrange_poly` → `lagrange`,
`multiquadratic_poly` → `multiquadratic`, `ra_poly` → `ra`, `split_eq_poly` → `split_eq`,
`one_hot_polynomial` → `one_hot`, `range_mask_polynomial` → `range_mask`, `rlc_polynomial` →
`rlc`, `compact_polynomial` → `compact`, `multilinear_polynomial` → `multilinear`,
`dense_mlpoly` → `dense`, `unipoly` → `univariate`, `opening_point` → `point`.
`prefix_suffix` keeps its name. A module whose contents are all dead is deleted rather than
renamed.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job. Polynomial evaluation
  and binding are deterministic field arithmetic; a wrong merge changes a round message, so a
  digest. `CompressedPoly`'s `CanonicalSerialize` reproduces `CompressedUniPoly`'s bytes.
- **One type per concept.** One eq polynomial, one split-eq (Gruen) type, one univariate
  round-message type with its compressed form, one dense multilinear type, one one-hot type and
  one opening-point type. Mechanism: review of `atlas-algebra/src/poly/mod.rs` exports; a CI
  grep that `jolt_poly` and `jolt_utils` appear nowhere in `atlas-*`/`jolt-atlas-*`.
- **No slower inner loop.** Each hot-path implementation swap is measured (see Performance).
  Mechanism: the `gpt2-tiny` runs and the final GPT-2 IOP gate recorded in the PR.
- **The `akita-*` surface is exact.** `akita-*` (at `dfdbd767`) uses `jolt_poly::{UnivariatePoly,
  CompressedPoly, OmittedConstantPoly, UnivariatePolynomial}` and nothing else, with
  `F: jolt_field::Field`. The owned `jolt-poly` package re-exports those four from
  `atlas-algebra`, and they stay bounded by `jolt_field::Field` (not atlas's `JoltField`) until
  phase 6 (`specs/own-field.md`). `akita-serialization` implements its own traits on these
  types, which works through re-exports (the traits are local to it). Mechanism: the compiler
  (`akita-*` builds against it) and the Akita digests.
- **`evaluate_mle` discipline kept** (`CLAUDE.md`: prefer `x.evaluate_mle(&r)`). Mechanism:
  review-held.
- **Contract untouched.** No trace change. Mechanism: `atlas-trace/` absent from the diff.

### Non-Goals

- Changing any binding order, challenge width or fold algorithm, unless an adopted upstream
  improvement does so with an identical result (digests).
- Prefix-suffix, `ra`, identity, range-mask and multiquadratic polynomials: atlas-only, not
  touched beyond the module renames and dead-code removal.
- Deleting the `jolt-poly` package name: it stays as long as `akita-*` is external
  (`docs/roadmap.md`, open decisions).

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-poly` and `third_party/jolt-utils` are deleted; the owned `jolt-poly`
      compatibility package exists at the root, depends only on `atlas-algebra` and
      `jolt-field`, exports exactly the four `akita-*` items, and is what the root
      `[patch."https://github.com/a16z/jolt"]` names; `akita-*` builds.
- [ ] No source under `atlas-*`/`jolt-atlas-*` names `jolt_poly` or `jolt_utils` (CI grep); the
      merged library exports one type per concept (list in the PR).
- [ ] `UniPoly`/`CompressedUniPoly` are gone; `poly/` modules carry the suffix-free names.
- [ ] The PR lists the dead code removed and each upstream improvement adopted.
- [ ] Each hot-path swap has a `gpt2-tiny` `prove` run recorded; real GPT-2 IOP stage within 3%
      of the `dev` baseline (three runs each, same machine, same day); peak memory within 5%.
- [ ] Digest pins unchanged; `cargo nextest run -p atlas-algebra` passes with upstream's
      in-source tests for the moved live code; CI green; the heavy job green on `main`.
- [ ] `PINS.md`, inventory (`poly/` rows), `docs/verifier.md`, crate maps, roadmap updated.

### Testing Strategy

Workspace Test job (digests, zoo, soundness, op-surface), isolation job, heavy job; upstream's
in-source tests for the code that moves live, ported into `atlas-algebra`. No proof-format
change. The `prove` example is the benchmark driver.

### Performance

Evidence scales with what changes:
- **Hot-path implementation swap** (an upstream kernel replacing ours inside a round message
  or bind): one `prove` run on `models/gpt2-tiny` before and after, recorded in the PR.
- **Final gate:** the `prove` example on `models/gpt2` (real download), `--trace-terminal`, IOP
  proving stage and opening reduction stage, three runs each on `dev` and on the branch. Gate:
  IOP stage within 3% (worse) of baseline; anything faster is recorded. Memory: peak within 5%
  (the README machine has 16 GB and GPT-2 peaks at 6.2 GB).
- **Cold-path or API-only changes** (moves, renames, dead-code removal, exceptional-point
  handling): digests only.

## Design

### Architecture

`atlas-algebra/src/poly/` after the phase:
- `univariate.rs`: `UnivariatePoly`, `CompressedPoly`, `UnivariatePolynomial`,
  `OmittedConstantPoly` (from upstream's `univariate.rs`, `compressed_univariate.rs`,
  `omitted_constant.rs`, plus atlas's live `UniPoly` methods and wire serialization);
- `split_eq.rs`: `GruenSplitEqPolynomial` (upstream's) with the `SplitEqUnreducedFold`
  extension;
- `eq.rs`, `dense.rs`, `multilinear.rs`, `one_hot.rs`, `point.rs`: one type each, the live
  upstream items the Akita adapter and `openings/prefix.rs` use (`Polynomial`,
  `MultilinearPoly`, `OneHotIndexOrder`, `EvaluationClaim`, `eq_index_msb`, `HIGH_TO_LOW`)
  merged into or mapped onto the atlas type;
- the renamed atlas-only modules.

Upstream's `lagrange.rs`, `lt.rs`, `identity.rs`, `eq_plus_one.rs`, `mle.rs` and `binding.rs`
are not ported unless a live caller needs an item from them. A root-level `jolt-poly/` package
(owned, a few dozen lines) re-exports the `akita-*` surface. Because `atlas-algebra` depends on
`atlas-trace`, this puts `atlas-trace` into `akita-*`'s dependency graph; that is harmless (no
cycle: `atlas-algebra` depends on no `akita-*` crate) but visible in `cargo tree`.
`jolt-utils`' `unsafe_allocate_zero_vec` joins `atlas-algebra::utils`. Consumers:
`atlas-sumcheck::engine` and the relation provers (`UnivariatePoly`,
`GruenSplitEqPolynomial`), `atlas-pcs` (eq, dense, one-hot, point), the op provers in
`jolt-atlas-core`.

### Alternatives Considered

- **Keep both and only own `jolt-poly`.** Rejected: the duplication is the problem.
- **Port every upstream module and measure every pair.** Rejected at analysis: most upstream
  modules have no atlas caller, and split-eq is already single.
- **Keep `UniPoly` as a wire-only type.** Rejected: two univariate types.
- **Fork `akita-*` to drop the compatibility package.** Deferred (roadmap).

## Documentation

`PINS.md`, `docs/upstream-inventory.md` (every `poly/` row collapses to "owned, merged"),
`docs/verifier.md`, `CLAUDE.md`/`AGENTS.md` (the "Fields and polynomials share a crate"
sentence stays true), `docs/roadmap.md`.

## Execution

One PR, commits in order:
1. The module renames, alone (mechanical).
2. The compatibility package, so `akita-*` never breaks.
3. One commit per live pair (univariate, point, eq/dense/one-hot), each digest-green.
4. Dead-code removal.
5. `jolt-utils` and the vendored-crate deletion, docs, CI, measurements.

## References

`a16z/jolt` `crates/jolt-poly`, `crates/jolt-utils` at `f3265b5a`; `docs/upstream-inventory.md`
`poly/` rows and the stage-4b measurement; `docs/roadmap.md`; memory of the stage-4b split
(`specs/adoption-program.md`).
