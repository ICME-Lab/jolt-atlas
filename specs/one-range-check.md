# Spec: One range check

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/no-scalar-special-cases.md` |
| Supersedes  | `docs/prover-plan.md` §3 item 7 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The verifier proves "this value is small" in four different ways. The fused-rescale remainders
of Einsum, Mul, Square and Cube are packed into remainder buckets and checked by dense base-4
digits (`atlas_sumcheck::digit_range`). The remainders of Div, MeanOfSquares, Rsqrt (two of
them) and the trig teleport are each checked per node by a prefix-suffix read-raf into
`UnsignedLessThanTable` over a 64-bit address, with sixteen committed one-hot chunks per check
and a degree-17 ra-virtualization. Softmax's two remainders use a third mechanism, the identity
range check, with its own one-hot chunks. A scalar MeanOfSquares uses a fourth, an in-the-clear
tie. All of them prove `0 ≤ R < B` for a bound that is a constant or another opening at the
same point. This spec makes that one mechanism: every bounded remainder is a member of a
remainder bucket, checked by the dense digit range check, and the lookup-based range checks,
their table, their one-hot families and their deferred batches are deleted.

## Intent

### Goal

Prove every remainder bound of Div, MeanOfSquares, Rsqrt, Cos, Sin and SoftmaxLastAxis by
membership in the packed remainder buckets (`global_clamp.rs`, `clamp_split.rs`), and delete
`onnx_proof/range_checking/` in both crates, `ops/softmax_last_axis/rc.rs`, and everything that
exists only to serve them.

A bucket member generalises from "a node's `RescaleRemainder` opening" to a **range claim**: a
node, a width `w`, and an affine expression over opening claims that already exist at one point
`r_n`. Two forms cover every case:

- `R` itself, proving `R ∈ [0, 2^w)`;
- the slack `B − 1 − R`, proving `R < B`, where `B` is a constant or an opening at `r_n`.

A bound that is a power of two with `w` a multiple of `log₂ DIGIT_BASE` needs the first form
only; every other bound needs both.

| Check today | `R` | `B` | Range claims |
|---|---|---|---|
| Div | `DivRemainder` | divisor, `NodeOutput(input 1)` | `R`, `B − 1 − R` |
| Rsqrt, division | `DivRemainder` | input `x̂`, `NodeOutput(input 0)` | `R`, `B − 1 − R` |
| Rsqrt, square root | `SqrtRemainder` | `2·v̂ + 1`, `v̂ = NodeOutput(node)` | `R`, `B − 1 − R` |
| MeanOfSquares | `RescaleRemainder` | constant `count·2^scale` | `R`, `B − 1 − R` |
| Cos / Sin teleport | `TeleportRemainder` | constant `TRIG_PERIOD_MODULUS` | `R`, `B − 1 − R` |
| Softmax recip-mult | `SoftmaxRecipMultRemainder` | `2^scale` | `R` |
| Softmax exp | `SoftmaxExpRemainder` | `2^scale` | `R` |

Introduced or modified:

- `jolt-atlas-verifier`: `global_clamp.rs` (range claims as bucket members;
  `ClampBucket::remainder_inputs` evaluates the affine expression), `clamp_split.rs`
  (`remainder_value_params` unchanged in shape), `plan.rs` (`OpRangeCheck`,
  `OpRelation::RangeCheck`, `OpRelation::RangeCheckOneHot` and `OpTie::ScalarRemainder`
  deleted), `deferred_lookups.rs` (`DeferredBatch::RangeCheck` and `RangeCheckRaChecks`
  deleted), `types.rs` (`ProofType::RangeCheck`, `DeferredRangeCheck`,
  `DeferredRangeCheckRaChecks` retired), the plans of `div.rs`, `rsqrt.rs`,
  `mean_of_squares.rs`, `sin.rs`, `cos.rs` and `softmax_last_axis/mod.rs`,
  `clamp_lookups/mod.rs` (`is_scalar`, `recover_small_int` deleted).
- `common`: `CommittedPoly::{DivRangeCheckRaD, SqrtDivRangeCheckRaD, SqrtRangeCheckRaD,
  MeanOfSquaresRangeCheckRaD, TeleportRangeCheckRaD, SoftmaxRemainderRaD,
  SoftmaxExpRemainderRaD}` and `VirtualPoly::{DivRangeCheckRa, SqrtRangeCheckRa,
  MeanOfSquaresRangeCheckRa, TeleportRangeCheckRa, SoftmaxRemainderRa,
  SoftmaxExpRemainderRa}` deleted.
- `jolt-atlas-core`: the prover mirrors, and bucket witness assembly for the new members.
- `atlas-sumcheck`, `atlas-lookup-tables`: `identity_range_check` and `UnsignedLessThanTable`
  are deleted if this leaves them with no caller outside their own tests.

### Invariants

Relied on, not changed:

1. **Remainder semantics.** The Euclidean split (`remainder ∈ [0, D)`) of the `atlas-trace`
   contract, and the division, square-root, teleport and softmax identities each op relation
   proves. Outputs do not move. *Mechanism:* the importer's golden hashes are not re-pinned.
2. **The remainder-bucket argument.** The value sumcheck `Σ_t W(t)·V(t) = Σ_n γ_n·V_n(r_n)`
   with `W` the packed cycle weight, followed by the digit range check on `V(r′)`, as in
   `clamp_split.rs`. The `r_n` are drawn after the digit columns are committed.
   *Mechanism:* review-held; `testing/digests.rs`.
3. **Each op's relation is unchanged.** Only the discharge of its remainder bound moves.
   *Mechanism:* `relation_specs_match_their_dispatched_op_kind`; review on the diff.
4. **Verifier hygiene.** As in every verifier change. *Mechanism:* the crate lints, the CI path
   checks, the mutation sweep with no tolerance entry, the drain check.

New:

- **A. Same predicate.** For each row of the table above, an accepted proof implies
  `0 ≤ R < B` over the integers on every element, the predicate the lookup proves today. Today's
  read-raf address is the interleave of the two operands as unsigned 32-bit values, so a claim
  outside `[0, 2^32)` is already unprovable; the analysis step confirms this reading for Div
  with a negative divisor before implementation starts, and the spec is amended if it is wrong.
  *Mechanism:* `witness_tamper_sweep` sites that set `R = B` and `R = −1` (with the op's own
  relation kept satisfied by adjusting the quotient) for each family; all rejected with `Err`.
- **B. No wrap.** With `R ∈ [0, 2^w)` and `B − 1 − R ∈ [0, 2^w)` as field elements and
  `|B| < 2^32`, both hold over the integers, since `2^w + 2^32` is far below the Fp128 modulus
  for every `w ≤ 64`. The module header of `global_clamp.rs` states this. *Mechanism:*
  review-held.
- **C. Same point.** The claims a slack expression reads (`R` and `B`) are bound at one point
  `r_n`. `remainder_inputs` returns `Err` if their points differ. *Mechanism:* the check, and a
  unit test that stages mismatched points.
- **D. Width rule.** A single-form member is allowed only when the bound is `2^w` with `w` a
  multiple of `log₂ DIGIT_BASE`; a function, not a per-op choice, decides. *Mechanism:* a unit
  test over the seven rows; the tamper sites of invariant A.
- **E. No scalar exception.** A one-element node is a one-cell member. `is_scalar` and
  `OpTie::ScalarRemainder` do not exist, and a scalar Div is range-checked like any other.
  *Mechanism:* the compiler; op-surface cases for scalar Div, Rsqrt and MeanOfSquares.
- **F. Nothing left behind.** No committed polynomial, virtual polynomial, proof type or
  deferred batch of the old mechanisms remains. *Mechanism:* the compiler;
  `proof_types_all_exercised`; the retired tags (4, 21, 22) fail to deserialize.

### Non-Goals

- `ScalarConstDiv`: its protocol is not changed by this spec.
- The clamp buckets and the saturating clamp (`specs/one-clamp.md`).
- Lookups that read a table: ReLU, Clamp, the activations, the trig tables and downscale, the
  softmax exp digits and significance clamp, Gather (`specs/post-loop-batches.md`).
- Replacing Rsqrt's two identities by one relation (`out²·x ≤ S³ < (out+1)²·x`), which
  `docs/prover-plan.md` item 7 also proposes: a protocol redesign of one op, separate.
- The row check, declared widths, or any change to `digit_range` itself.
- Any change to trace semantics, `SemVersion` or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] The analysis of invariant A for Div with a negative divisor is recorded in the PR.
- [ ] `onnx_proof/range_checking/` does not exist in either crate;
      `ops/softmax_last_axis/rc.rs` does not exist in either crate.
- [ ] The seven `CommittedPoly` and six `VirtualPoly` variants listed above do not exist.
- [ ] `OpRangeCheck`, `OpRelation::RangeCheck`, `OpRelation::RangeCheckOneHot`,
      `OpTie::ScalarRemainder`, `DeferredBatch::RangeCheck`, `DeferredBatch::RangeCheckRaChecks`,
      `is_scalar` and `recover_small_int` do not exist.
- [ ] `ProofType` tags 4, 21 and 22 are in the deleted list of `types.rs` and
      `docs/verifier.md`; `proof_types_all_exercised` passes.
- [ ] `identity_range_check` and `UnsignedLessThanTable` are deleted, or the PR names their
      remaining caller.
- [ ] Invariant A: two tamper sites per family (`R = B`, `R = −1`), each rejected with `Err`.
- [ ] Invariants C and D: the named unit tests pass.
- [ ] Invariant E: op-surface cases for a scalar Div, a scalar Rsqrt and a scalar
      MeanOfSquares prove and verify.
- [ ] The three sweeps pass over the probe models with no tolerance entry.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 proves and verifies under the `heavy` profile (`zoo::gpt2`).
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the op-surface rows of Div, Rsqrt,
MeanOfSquares, Sin, Cos and SoftmaxLastAxis; the three sweeps; the zoo with its serialization
round-trip; `relation_specs_match_their_dispatched_op_kind`; the `digit_range_*` unit tests in
`atlas-sumcheck`; `fuzz_regressions_replay`, with its corpus regenerated in the digest commit
if the format no longer parses it.

New tests: the tamper sites of invariant A; the unit tests of invariants C and D; the three
scalar op-surface cases; one `mutate::Site` per proof field whose shape changes. No new
relation is introduced, so no relation-level test is added.

Coverage of the moved families by the sweeps: the PR confirms that each of Div, Rsqrt,
MeanOfSquares, Sin or Cos, and SoftmaxLastAxis has a node in a probe model, and adds one where
it does not.

CI coverage: the workspace job; the heavy job for GPT-2 (25 Rsqrt, 25 MeanOfSquares and 12
softmax nodes move). The `zk` feature is not touched.

### Performance

Benchmark: `gpt2 --seq-len 16` and `bge-small-en-v1.5` with the run table of
`docs/measurement-plan.md` against `bench/bin/prove-c65753bfa`.

These families are under 2% of committed digit rows on GPT-2, so no stage is expected to move
outside noise. Direction, for the record: a 32-bit check goes from sixteen K = 16 one-hot
columns to thirty-two dense base-4 columns, roughly a quarter of the commit cost per element at
the per-digit costs measured in the plan, and loses a `64 + log T`-round read-raf and a
degree-17 ra-virtualization.

Bounds:

- Prove and verify on both models within 3% of baseline, or better.
- Proof size no larger than baseline.
- Peak memory no higher than baseline.
- The number of opened Akita classes on both models does not rise.

## Design

### Architecture

```text
node turn:   op relation  →  claims R(r_n), and B(r_n) where B is an opening
post-loop:   Batch[DeferredRemainderRC]          value sumcheck per bucket:
                 Σ_t W(t)·V(t) = Σ_n γ_n · claim_n,   claim_n ∈ { R(r_n),  B(r_n) − 1 − R(r_n) }
             Batch[DeferredRemainderDigitRange]  digit_range(V) per bucket
```

`remainder_buckets(model)` returns, per node, the range claims of the table: the fused ops' one
claim as today, the moved families' one or two. `pack_buckets` groups by width as it does
today; a node with two claims occupies two segments. The value sumcheck's input claim is the
`γ`-combination of the members' affine expressions, evaluated from the accumulator. The digit
columns are `GlobalRemainderDigit(bucket, j)`, as today.

Each op plan loses its range-check steps and nothing else. `Div` drops
`Relation{RangeCheck}` and its one-hot batch; `Rsqrt` and `MeanOfSquares` drop their four and
two `Defer` steps; `Sin` and `Cos` drop the teleport member of both in-loop batches; softmax
drops the identity range checks from stages 1 and 2 and the remainder one-hot checks from
stages 2 and 3. The `Open` steps that bind the remainder advice stay.

Widths: 32 bits for a tensor-valued bound, 34 for Rsqrt's `2·v̂ + 1`, `⌈log₂ B⌉` rounded up to
the digit base for a constant bound, `scale` for softmax.

### Alternatives Considered

- **Keep the lookups, share their batches.** Rejected: it keeps the table, the 64-bit address,
  sixteen one-hot chunks per check and the degree-17 product.
- **A per-node digit range check instead of bucket membership.** Rejected: one relation
  instance and one set of committed columns per node, where the buckets give one per width.
- **Commit the slack as its own advice column.** Rejected: the slack is affine in claims that
  exist, so it needs no opening of its own.
- **One decomposition with a tight width for constant bounds.** Only sound when the bound is a
  power of the digit base; invariant D is that rule.
- **Do softmax later with `specs/softmax-aux-removal.md`.** Rejected: its two remainders are
  the same statement as the others, and moving them removes `rc.rs` and two one-hot families
  before that spec starts.
- **Upstream `a16z/jolt`.** Range checks there are Shout lookups; the dense-digit form is
  SuperNeo's, already adopted for the buckets.

## Documentation

- `docs/verifier.md`: the perimeter table loses `UnsignedLessThanTable` and the identity range
  check if deleted; the reading order's "Lookup machinery" entry loses `range_checking/`; the
  deleted-tag list gains 4, 21, 22.
- `docs/prover-plan.md`: item 7 marked landed for the range-check half; the Rsqrt relation
  stays open.
- Module docs: `global_clamp.rs` (range claims, invariant B), the six op modules.
- `CLAUDE.md`: no change; it names no range-check mechanism.
- `README.md`: no change unless a benchmark row moves outside noise.

## Execution

Suggested order: (1) range claims in `global_clamp.rs` with the fused ops still the only
members, digests unchanged; (2) MeanOfSquares and the trig teleport (constant bounds); (3) Div
and Rsqrt (opening bounds); (4) softmax; (5) deletions; (6) the digest re-pin.

## References

- `jolt-atlas-verifier/src/onnx_proof/range_checking/` and `ops/softmax_last_axis/rc.rs` (what
  is deleted); `global_clamp.rs`, `clamp_split.rs` (what is extended).
- `atlas-sumcheck/src/digit_range/`.
- `specs/clamp-dense-digit-range-check.md`; `docs/prover-plan.md` §3 items 2 and 7.
- Neo / SuperNeo, ePrint 2026/242.
