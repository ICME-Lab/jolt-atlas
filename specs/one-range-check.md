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

The verifier proves "this remainder is below its bound" in four different ways, and for one op
not at all. The fused-rescale remainders of Einsum, Mul, Square and Cube are packed into
remainder buckets and checked by dense base-4 digits (`atlas_sumcheck::digit_range`). The
remainders of Div, MeanOfSquares, Rsqrt (two of them) and the trig teleport are each checked
per node by a prefix-suffix read-raf into `UnsignedLessThanTable` over a 64-bit address, with
sixteen committed one-hot chunks per check and a degree-17 ra-virtualization. Softmax's two
remainders use a third mechanism, the identity range check, with its own one-hot chunks. A
one-element MeanOfSquares uses a fourth, an in-the-clear tie, and a one-element Div has none.
`ScalarConstDiv` commits its remainder and never bounds it, so its relation does not pin the
quotient to the floor. Every one of these is the statement `0 ≤ R < B` for a bound that is a
constant or another opening at the same point. This spec makes that one mechanism: every
bounded remainder is a member of a remainder bucket, checked by the dense digit range check.
The lookup-based range checks, their table, their one-hot families and their deferred batches
are deleted, and `ScalarConstDiv` gains the check it lacks.

The remainders of Div, Rsqrt, the teleport and `ScalarConstDiv` are recomputed inside the
prover today, in two slightly different ways, where the soundness sweeps cannot reach them.
They move into the trace, beside the fused-rescale remainders that are already there.

## Intent

### Goal

Prove every remainder bound of Div, ScalarConstDiv, MeanOfSquares, Rsqrt, Cos, Sin and
SoftmaxLastAxis by membership in the packed remainder buckets (`global_clamp.rs`,
`clamp_split.rs`), capture those remainders in the trace, and delete
`onnx_proof/range_checking/` in both crates and everything that exists only to serve the
lookup-based checks.

A bucket member generalises from "node `n`'s `RescaleRemainder` opening" to a **range claim**:
a node, a claim kind, a width `w`, and an affine expression over opening claims that already
exist at one point `r_n`. Two forms cover every case:

- `R` itself, proving `R ∈ [0, 2^w)`;
- the slack `B − 1 − R`, proving `R < B`, where `B` is a constant or an opening at `r_n`.

A bound `B = 2^w` with `w` a multiple of `log₂ DIGIT_BASE` needs the first form only, at width
`w`. Every other bound needs both forms, at width 32.

| Check | `R` | `B` | Range claims |
|---|---|---|---|
| Div | `DivRemainder` | divisor, `NodeOutput(input 1)` | `R`, `B − 1 − R` at 32 |
| ScalarConstDiv | `ScalarConstDivRemainder` (new, virtual) | the constant divisor | `R` at `w` if the divisor is `2^w`, `w` even; else both at 32 |
| Rsqrt, division | `DivRemainder` | input `x̂`, `NodeOutput(input 0)` | `R`, `B − 1 − R` at 32 |
| Rsqrt, square root | `SqrtRemainder` | `2·v̂ + 1`, `v̂ = NodeOutput(node)` | `R`, `2·v̂ − R` at 32 |
| MeanOfSquares | `RescaleRemainder` | constant `count·2^scale` | `R`, `B − 1 − R` at 32 |
| Cos / Sin teleport | `TeleportRemainder` | constant `TRIG_PERIOD_MODULUS` | `R`, `B − 1 − R` at 32 |
| Softmax recip-mult | `SoftmaxRecipMultRemainder` | `2^scale` | `R` at `scale` |
| Softmax exp | `SoftmaxExpRemainder` | `2^scale` | `R` at `scale` |

Every opening an expression reads is named by its full `OpeningId`. In particular the bounds
that are node outputs are the claims the op's own relation produced under
`SumcheckId::NodeExecution(node)` at its bound point, not the node's reduced output opening,
which is at a different point.

Introduced or modified:

- `atlas-trace`: `NodeAux` carries the remainder of Div and ScalarConstDiv, the division
  quotient and both remainders of Rsqrt, and the teleport quotient and remainder of Sin and
  Cos, computed by the op's own evaluation (`rem_floor`, the integer square root,
  `rem_euclid`). The behavioral contract's "captured in the trace for the prover" clause
  extends to them.
- `jolt-atlas-verifier`: `global_clamp.rs` (range claims as bucket members, each identified by
  node and claim kind; `ClampBucket::remainder_inputs` evaluates the affine expression and
  checks the common point), `clamp_split.rs` (`remainder_value_params` unchanged in shape),
  `plan.rs` (`OpRangeCheck`, `OpRelation::RangeCheck`, `OpRelation::RangeCheckOneHot` and
  `OpTie::ScalarRemainder` deleted), `deferred_lookups.rs` (`DeferredBatch::RangeCheck` and
  `RangeCheckRaChecks` deleted), `types.rs` (`ProofType::RangeCheck`, `DeferredRangeCheck`,
  `DeferredRangeCheckRaChecks` retired), the plans of `div.rs`, `scalar_const_div.rs`,
  `rsqrt.rs`, `mean_of_squares.rs`, `sin.rs`, `cos.rs` and `softmax_last_axis/mod.rs`,
  `clamp_lookups/mod.rs` (`is_scalar`, `recover_small_int` deleted),
  `node_view.rs` (`ComputationNode::is_scalar` deleted).
- `common`: `CommittedPoly::{DivRangeCheckRaD, SqrtDivRangeCheckRaD, SqrtRangeCheckRaD,
  MeanOfSquaresRangeCheckRaD, TeleportRangeCheckRaD, SoftmaxRemainderRaD,
  SoftmaxExpRemainderRaD, ScalarConstDivNodeRemainder}` and `VirtualPoly::{DivRangeCheckRa,
  SqrtRangeCheckRa, MeanOfSquaresRangeCheckRa, TeleportRangeCheckRa, SoftmaxRemainderRa,
  SoftmaxExpRemainderRa}` deleted; `VirtualPoly::ScalarConstDivRemainder` added.
- `jolt-atlas-core`: the prover mirrors; bucket witness assembly for the new members;
  `range_checking/`, `utils::adjusted_remainder`, `neural_teleport::division::compute_division`
  and the remainder and quotient arms of `witness.rs` deleted in favour of the trace.
- `atlas-sumcheck`, `atlas-lookup-tables`: `identity_range_check`, `ps_shout::binary` and
  `UnsignedLessThanTable` are deleted if this leaves them with no caller outside their own
  tests.
- `ops/softmax_last_axis/rc.rs` is deleted; the exp-hi and exp-lo one-hot encodings it also
  holds move to `exponentiation/`.

### Invariants

Changed:

1. **The trace contract's capture clause.** "The quotient and remainder are captured in the
   trace for the prover" now also covers Div, ScalarConstDiv, Rsqrt, Sin and Cos. Outputs do
   not move and `SemVersion` does not change. *Mechanism:* the importer's golden hashes are not
   re-pinned; unit tests in `atlas-trace` check each captured pair against its defining
   identity and range on random inputs (`a = q·b + r`, `S³ = q·x + r`, `q = v² + r`,
   `x = P·q + r`).

Relied on, not changed:

2. **Remainder semantics.** The Euclidean split of the fused ops, floor division for Div and
   ScalarConstDiv, and the division, square-root, teleport and softmax identities each op
   relation proves. *Mechanism:* unchanged op-surface cases and golden hashes.
3. **The remainder-bucket argument.** The value sumcheck `Σ_t W(t)·V(t) = Σ_m γ_m·claim_m`
   with `W` the packed cycle weight, followed by the digit range check on `V(r′)`, as in
   `clamp_split.rs`. Every `r_n` is drawn after the digit columns are committed.
   *Mechanism:* review-held; `testing/digests.rs`.
4. **Each op's relation is unchanged**, except that ScalarConstDiv's produces a virtual
   remainder opening where it produced a committed one. *Mechanism:*
   `relation_specs_match_their_dispatched_op_kind`; review on the diff.
5. **Verifier hygiene.** Rejects with `Err`, never panics, consumes the proof exactly, stays
   Trace-free. *Mechanism:* the crate lints, the CI path checks, the mutation sweep with no
   tolerance entry, the drain check.

New:

- **A. The predicate, and bounds must be positive.** An accepted proof implies `B ≥ 1` and
  `0 ≤ R < B` over the integers on every element of every member's segment. Today's lookup
  proves `u32(R) < u32(B)` over signed 32-bit values (its raf is a signed operand polynomial),
  which is the same statement when `B ≥ 1`. For `B ≤ 0` the two differ, and the new rule is
  that there is no proof: a Div or ScalarConstDiv by a negative value (floor division puts
  that remainder in `(B, 0]`), an Rsqrt of a non-positive input, and a zero bound, including a
  padded lane of a Div, are unprovable. Of these only the negative Rsqrt input changes
  outcome: today's lookup accepts any remainder for it, although the op's doc says it is
  rejected. The importer emits none of them (it emits no `Div`, and `ScalarConstDiv` only with
  divisor `2^scale`). *Mechanism:* forged-quotient sites in `witness_tamper_sweep` for Div,
  ScalarConstDiv, Rsqrt division, Rsqrt square root and MeanOfSquares: the quotient moved by
  one and the remainder by the bound in the opposite direction, both ways, so the op's
  identity still holds and only the range check can reject. All are rejected with `Err`.
  Teleport and softmax have single-field tamper sites on their captured remainders and the
  bucket-digit sites; a forged-quotient site would not isolate the range check there (a
  teleport remainder of `P` also leaves the trig table, and a softmax remainder cascades
  through later stages).
- **B. No wrap.** With `R ∈ [0, 2^32)` and `B − 1 − R ∈ [0, 2^32)` as field elements and
  `|B| < 2^33`, both hold over the integers, since these magnitudes are far below the Fp128
  modulus. The module header of `global_clamp.rs` states this. *Mechanism:* review-held.
- **C. Same point.** The claims an expression reads are bound at one point `r_n`.
  `remainder_inputs` returns `Err` if their points differ; today's check discards the points.
  *Mechanism:* the check, and a unit test that stages mismatched points.
- **D. Width rule.** One function maps a bound to its range claims: a single claim at width
  `w` exactly when the bound is `2^w` with `w` a multiple of `log₂ DIGIT_BASE`, otherwise the
  two-sided pair at width 32. No op chooses a width. *Mechanism:* a unit test over the eight
  rows, including a ScalarConstDiv by `2^14`, by `2^7` and by 7.
- **E. No one-element exception.** A one-element node is a one-cell member. `is_scalar`, in
  both its definitions, and `OpTie::ScalarRemainder` do not exist, and a one-element Div is
  range-checked like any other. *Mechanism:* the compiler; op-surface cases for a one-element
  Div, Rsqrt, ScalarConstDiv, MeanOfSquares, Sin and softmax.
- **F. ScalarConstDiv is bounded.** Its remainder is a virtual advice opening bound through a
  remainder bucket, its plan returns `Err` for a divisor that is not positive, and the "known
  soundness gap" sentence leaves its relation's doc. *Mechanism:* its forged-quotient site;
  a plan test for divisors 0 and −3.
- **G. One source for each remainder.** The prover reads every bucket member's values from the
  trace and recomputes none. *Mechanism:* the compiler (`adjusted_remainder`,
  `compute_division` and the `witness.rs` arms are deleted).
- **H. Every member is in exactly one bucket.** Each (node, claim kind) of the table appears
  once in `remainder_buckets(model)`. *Mechanism:* a unit test on a model holding one node of
  each family; the bucket-digit tamper sites are placed on every member, not only on the first
  node of a bucket.
- **I. Nothing left behind.** No committed polynomial, virtual polynomial, proof type or
  deferred batch of the old mechanisms remains. *Mechanism:* the compiler;
  `proof_types_all_exercised`; the retired tags (4, 21, 22) fail to deserialize.

### Non-Goals

- Supporting a negative divisor, for either a constant or a tensor.
- The committed quotients (`DivNodeQuotient`, `RsqrtQuotient`, `TeleportNodeQuotient`): they
  stay committed; only their values' source moves to the trace.
- The clamp buckets and the saturating clamp (`specs/one-clamp.md`).
- Lookups that read a table: ReLU, Clamp, the activations, the trig tables and downscale, the
  softmax exp digits and significance clamp, Gather (`specs/post-loop-batches.md`).
- Replacing Rsqrt's two identities by one relation (`out²·x ≤ S³ < (out+1)²·x`), which
  `docs/prover-plan.md` item 7 also proposes: a protocol redesign of one op, separate.
- The row check, declared widths, `pack_buckets`, or any change to `digit_range` itself.
- Any change to model outputs, `SemVersion` or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] Invariant 1: the `atlas-trace` identity-and-range tests pass for Div, ScalarConstDiv,
      Rsqrt, Sin and Cos; the contract comment in `atlas-trace/src/lib.rs` and its copy in
      `CLAUDE.md` name the newly captured advice.
- [ ] `onnx_proof/range_checking/` and `ops/softmax_last_axis/rc.rs` do not exist in either
      crate; the exp-hi and exp-lo encodings live in `exponentiation/`.
- [ ] The eight `CommittedPoly` and six `VirtualPoly` variants listed above do not exist.
- [ ] `OpRangeCheck`, `OpRelation::RangeCheck`, `OpRelation::RangeCheckOneHot`,
      `OpTie::ScalarRemainder`, `DeferredBatch::RangeCheck`, `DeferredBatch::RangeCheckRaChecks`,
      both `is_scalar` definitions, `recover_small_int`, `adjusted_remainder` and
      `compute_division` do not exist.
- [ ] `ProofType` tags 4, 21 and 22 are in the deleted list of `types.rs` and
      `docs/verifier.md`; `proof_types_all_exercised` passes.
- [ ] `identity_range_check`, `ps_shout::binary` and `UnsignedLessThanTable` are deleted, or the
      PR names the remaining caller of each.
- [ ] Invariant A: the forged-quotient sites for the five families, each in both directions,
      are rejected with `Err`; an op-surface case pins that a Div by a negative divisor and an
      Rsqrt of a negative input have no accepted proof.
- [ ] Invariant F: the ScalarConstDiv plan test passes; its relation's doc states the S ⊆ R
      argument with the bucket named as the protocol it relies on.
- [ ] Invariants C, D and H: the named unit tests pass.
- [ ] Invariant E: the six one-element op-surface cases prove and verify.
- [ ] Every moved family has a node in a probe model, and the three sweeps pass over the probe
      models with no tolerance entry.
- [ ] The importer's golden hashes are untouched. The commit that moves the remainders into the
      trace leaves `testing/digests.rs` unchanged; the digests are re-pinned once, in their own
      commit, after the protocol change.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the op-surface rows of Div,
ScalarConstDiv, Rsqrt, MeanOfSquares, Sin, Cos and SoftmaxLastAxis; the three sweeps; the zoo
with its serialization round-trip; `relation_specs_match_their_dispatched_op_kind`; the
`digit_range_*` unit tests in `atlas-sumcheck`; `fuzz_regressions_replay`, with its corpus
regenerated in the digest commit if the format no longer parses it;
`atlas-trace/tests/kernel_goldens.rs`.

New tests:

- `atlas-trace`: the identity-and-range tests of invariant 1.
- `witness_tamper_sweep`: a `WitnessSite` for each captured advice field, and the paired
  forged-quotient form of invariant A. A site that would need a negative remainder commits its
  low 32 bits, which no honest decomposition matches.
- The unit tests of invariants C, D and H; the plan test of invariant F.
- Op-surface: the six one-element cases; a Div by a negative divisor and an Rsqrt of a
  negative input as expected rejections; a ScalarConstDiv by a non-power-of-two (two-sided
  form) beside the existing ÷128.

No new relation is introduced, so no relation-level test is added. The probe models hold a
softmax and a Div (`attention_like`), an Rsqrt, Cos, Sin, MeanOfSquares and a ScalarConstDiv
(`reductions`, `tables_and_layout`) today; the PR confirms this still holds.

CI coverage: the workspace job; the heavy job for GPT-2 and bge-small (25 Rsqrt, 25
MeanOfSquares and 12 softmax nodes move on each, and bge-small holds the one ScalarConstDiv).
The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5` against the
baseline commit `c65753bfa`, on the benchmark machine.

These families are under 2% of committed digit rows on GPT-2, so no stage is expected to move
outside noise. Direction, for the record: on GPT-2 about 1,200 small one-hot polynomials (the
Rsqrt and MeanOfSquares range checks) and 96 softmax one-hot polynomials are replaced by the
digit columns of one new 32-bit bucket group and extra members of the existing 14-bit buckets,
and 75 read-rafs over a 64-bit address with their degree-17 ra-virtualizations disappear.

Bounds:

- Prove and verify on both models within the gate's 3%, or better.
- Proof size no larger than baseline.
- Peak memory within the gate's 5%.
- The PR reports the committed class list before and after; the opening stage stays within 3%.

## Design

### Architecture

```text
trace:       each op's evaluation records its remainder (and recomputed quotient advice)
node turn:   op relation  →  claims R(r_n), and B(r_n) where B is an opening
post-loop:   Batch[DeferredRemainderRC]          value sumcheck per bucket:
                 Σ_t W(t)·V(t) = Σ_m γ_m · claim_m,   claim_m ∈ { R(r_n),  B(r_n) − 1 − R(r_n) }
             Batch[DeferredRemainderDigitRange]  digit_range(V) per bucket
```

A member is `(node, claim kind)`; `BucketNode` gains the kind, so `remainder_inputs`, the
prover's `assemble_bits` and the sweep's sites can tell a node's members apart.
`remainder_buckets(model)` returns, per node, the range claims of the table: the fused ops'
one claim as today, the moved families' one, two or (Rsqrt) four. `pack_buckets` groups by
width as it does today, so the widths in use are `scale` (the fused ops, softmax, a
ScalarConstDiv by `2^scale`), `2·scale` (Cube) and 32 (every two-sided member). The value
sumcheck's input claim is the `γ`-combination of the members' affine expressions, evaluated
from the accumulator with the virtual or committed getter each opening needs. The digit
columns are `GlobalRemainderDigit(bucket, j)`, as today.

Each op plan loses its range-check steps and nothing else. `Div` drops `Relation{RangeCheck}`
and its one-hot batch, with no one-element branch; `Rsqrt` and `MeanOfSquares` drop their four
and two `Defer` steps and MeanOfSquares its in-the-clear tie; `Sin` and `Cos` drop the
teleport member of both in-loop batches, which keep the downscale and table lookups; softmax
drops the identity range checks from stages 1 and 2 and the remainder one-hot checks from
stages 2 and 3, leaving `[recip-mult, exp-sum]`, `[exp product, max indicator]` and
`[exp-hi, exp-lo, significance clamp]`. The `Open` steps that bind remainder advice stay.

`ScalarConstDiv`'s relation keeps its expression `d·out(r₀) = Σ eq·(left − rem)`; `rem` is
opened as `VirtualPoly::ScalarConstDivRemainder` at the bound point, the op commits nothing,
and its plan checks `divisor ≥ 1`.

In `atlas-trace`, the evaluation of each op returns its advice in `NodeAux` as the fused ops
do. For an Rsqrt input that is not positive the output is 0 and the advice is zero; such a
node is unprovable by invariant A.

Prover: `BucketWitnesses` builds each member's values from the trace before the commitment,
the slack as `B − 1 − R` per element from the trace's operand and output tensors.

### Alternatives Considered

- **Keep the lookups, share their batches.** Rejected: it keeps the table, the 64-bit address,
  sixteen one-hot chunks per check and the degree-17 product.
- **A per-node digit range check instead of bucket membership.** Rejected: one relation
  instance and one set of committed columns per node, where the buckets give one per width.
- **Commit the slack as its own advice column.** Rejected: the slack is affine in claims that
  exist, so it needs no opening of its own.
- **Tight widths for constant bounds** (20, 22 or 24 bits for the MeanOfSquares divisor and
  the trig period, 34 for Rsqrt's `2v̂ + 1`). Rejected: in the two-sided form the width only
  has to cover honest values, 32 bits covers every bound here, and each extra width is an
  extra bucket group and commitment class.
- **Support negative divisors.** Rejected at analysis: a tensor divisor's sign varies per
  element, a constant one would add a sign case, and the importer emits neither.
- **A test-only override for the recomputed remainders** instead of moving them into the
  trace. Rejected at analysis: the trace is where the fused remainders already are, it removes
  the two divergent recomputations in the prover, and the existing sweep machinery then
  reaches every remainder.
- **Keep ScalarConstDiv's remainder committed and add only the bound.** Rejected: the bucket
  digits already commit the value, as they do for every other remainder.
- **Leave softmax for `specs/softmax-aux-removal.md`.** Rejected: its two remainders are the
  same statement as the others, and moving them removes two one-hot families before that spec
  starts.
- **Upstream `a16z/jolt`.** Range checks there are Shout lookups; the dense-digit form is
  SuperNeo's, already adopted for the buckets.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the behavioral contract's capture clause.
- `docs/verifier.md`: the perimeter table loses `UnsignedLessThanTable`, the identity range
  check and the binary read-raf if deleted; the reading order's "Lookup machinery" entry loses
  `range_checking/`; the deleted-tag list gains 4, 21, 22.
- `docs/prover-plan.md`: item 7 marked landed for the range-check half; the Rsqrt relation
  stays open.
- Module docs: `global_clamp.rs` (range claims, the positive-bound rule, invariant B), the
  seven op modules; the ScalarConstDiv and Rsqrt relation docs state what the bucket proves
  for them.
- `README.md`: no change unless a benchmark row moves outside noise.

## Execution

Suggested order, each step leaving the tree provable:

1. Trace capture in `atlas-trace`; the prover reads the advice from the trace and the
   recomputation is deleted. Digests unchanged.
2. Range claims and member identity in `global_clamp.rs`, with the fused ops still the only
   members. Digests unchanged.
3. MeanOfSquares and the trig teleport (constant bounds).
4. Div and Rsqrt (opening bounds).
5. Softmax, moving the exp encodings out of `rc.rs`.
6. ScalarConstDiv.
7. Deletions, the sweep's new sites, the digest re-pin.

## References

- `jolt-atlas-verifier/src/onnx_proof/range_checking/` and `ops/softmax_last_axis/rc.rs` (what
  is deleted); `global_clamp.rs`, `clamp_split.rs` (what is extended);
  `ops/scalar_const_div.rs` (the unbounded remainder).
- `atlas-algebra/src/poly/signed_identity.rs` and `atlas-sumcheck/src/ps_shout/binary/` (what
  today's lookup proves).
- `atlas-trace/src/semantics/fixed.rs` (`floor_div`, `rem_floor`, `rebase`),
  `atlas-trace/src/trace.rs` (`NodeAux`).
- `atlas-sumcheck/src/digit_range/`.
- `specs/clamp-dense-digit-range-check.md`, `specs/no-scalar-special-cases.md`;
  `docs/prover-plan.md` §3 items 2 and 7.
- Neo / SuperNeo, ePrint 2026/242.
