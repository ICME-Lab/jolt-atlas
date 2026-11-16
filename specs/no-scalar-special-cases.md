# Spec: No scalar special cases in the clamp and remainder protocol

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | —                              |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

A clamped node whose padded output is a single element (`log_T = 0`) is verified by a different
protocol from every other clamped node. It is left out of the packed clamp and remainder buckets,
and the verifier instead recomputes the saturating clamp and the remainder bound in the clear
from opening claims (`OpTie::ScalarAccumulation`, `OpTie::ScalarClamp`,
`OpTie::ScalarRemainder`). The stated reason is that a one-hot opening degenerates with an empty
cycle dimension. That comment dates from #256, when each clamp committed its own one-hot chunks
under HyperKZG. The buckets now commit dense digit columns, and the split-eq polynomial, the
sumcheck engine, the one-hot opening reduction and the Akita adapter each handle zero variables
explicitly. The scalar path costs three tie variants, a branch in five op plans and in the
fused-rescale skeleton, and a second way of proving the same statement that no probe model
contains. This spec removes it: a scalar node is a one-cell segment of its bucket.

## Intent

### Goal

Make a `log_T = 0` clamped node a member of its packed clamp bucket and, where it has a fused
rescale, of its remainder bucket, verified by the same plan steps as any other size, and delete
the in-the-clear scalar ties for the clamp and for the fused-rescale remainder.

Introduced or modified:

- `jolt-atlas-verifier`: `onnx_proof/global_clamp.rs` (`clamp_buckets` and `has_remainder_rc`
  lose their `!is_scalar` filter), `onnx_proof/plan.rs` (`OpTie::ScalarAccumulation` and
  `OpTie::ScalarClamp` deleted; `fused_rescale_plan` and `clamp_steps` lose their scalar
  branches), `onnx_proof/ops/add.rs` (`accumulation_plan`), `onnx_proof/ops/sum/mod.rs`,
  `onnx_proof/ops/mean_of_squares.rs` (the clamp half), `onnx_proof/clamp_lookups/mod.rs`
  (`verify_scalar_clamp` deleted).
- `jolt-atlas-core`: the `ScalarAccumulation` and `ScalarClamp` arms of the tie prover in
  `ops/mod.rs`; the scalar filters of the census in `examples/prove.rs`; the scalar filter of
  `saturating_probe_saturates` in `testing/clamp_buckets.rs`. The bucket witness code needs no
  change: it already accepts a one-cell segment.

Left in place by this spec and removed by `specs/one-range-check.md`: `OpTie::ScalarRemainder`
for `MeanOfSquares` (its remainder is a lookup range check today, not a bucket member),
`recover_small_int`, `is_scalar` itself, and the scalar branches of `ops/div.rs` and of
`MeanOfSquares`'s committed-polynomial list. The two halves of `MeanOfSquares` are independent:
its clamp reads `ClampAcc` and the output, its remainder tie reads only `RescaleRemainder`.

### Invariants

Relied on, not changed:

1. **Clamp and rescale semantics.** `output = sat_clamp_i32(acc)` and the Euclidean
   quotient/remainder split of the `atlas-trace` contract. Model outputs do not move.
   *Mechanism:* the importer's golden hashes are not re-pinned; the op-surface cases compare
   against the trace.
2. **Bucket soundness argument.** The split, exact-value, digit-range and slack relations of
   `clamp_split.rs` and the remainder value and digit-range relations keep their expressions; a
   one-cell segment is the case `r_n = ()` of the packed cycle weight
   `Σ_n γ_n·eq(prefix_n, hi)·eq(r_n, lo)`. *Mechanism:* review-held on the diff; the sweeps.
3. **Commit before challenge.** Unchanged: bucket columns are committed before any challenge.
   *Mechanism:* `testing/digests.rs` pins the transcript.
4. **Unsaturated model outputs.** A scalar that is a model output with an unsaturated public
   value is tied `acc = out` in its own turn and is a gap in its bucket, like any other size.
   *Mechanism:* the six unsaturated scalar op-surface cases, all of which are model outputs.
5. **Verifier hygiene.** Rejects with `Err`, never panics, consumes the proof exactly, stays
   Trace-free. *Mechanism:* the crate lints, the CI path checks, the mutation sweep with no
   tolerance entry, the drain check.

New:

- **A. One protocol per statement.** For every op in `{Add, Sub, Sum, MeanOfSquares, Einsum,
  Mul, Square, Cube}`, the clamp steps of a node's plan do not depend on whether its padded
  output has one element. *Mechanism:* the compiler for the two deleted tie variants; for the
  rest, a caller list checked in review: after this PR `is_scalar` has exactly three callers
  (`ops/mean_of_squares.rs` for the remainder tie and for its committed-polynomial list, and
  `ops/div.rs`), and `OpTie::ScalarRemainder` has exactly one emitter. `specs/one-range-check.md`
  deletes `is_scalar`, after which the property is compiler-held.
- **B. A bucket may have `log_t = 0`.** A bucket of one cell commits, proves and verifies,
  including the case where every value and digit-range instance of a batch has zero rounds.
  *Mechanism:* the six saturating scalar op-surface cases. Each is a model with one clamped
  node that saturates, so with the filter removed each produces exactly this bucket.
- **C. The scalar path is under the sweeps.** A probe model contains an internal scalar clamped
  node and an internal scalar fused node, so `proof_mutation_sweep`, `witness_tamper_sweep`
  and `opening_forgery_sweep` reach a one-cell segment. The scalar values are chosen so the
  keep-value digit tamper has something to move: `OUT = out + 2^31` and the remainder each
  have a non-zero digit above the least significant one. With all upper digits zero the tamper
  is a no-op and the sweep reports a violation. The tamper sweep places its digit site on the
  first node of each bucket only, so the probe is built so that a scalar is the first node of
  a clamp bucket and of a remainder bucket (a width no larger node shares).
  *Mechanism:* the probe model and the sweeps; a test asserting the probe's bucket layout.
- **D. Saturating scalars.** A scalar that saturates carries one-hot slack chunks with zero
  cycle variables and is accepted; a tampered slack is rejected. *Mechanism:* the cases of
  invariant B for acceptance; the tamper sweep over a saturating scalar in a probe model for
  rejection.
- **E. Layout cost is stated, not avoided.** `pack_buckets` is not changed. A scalar is placed
  first-fit after every larger node, so it may (i) raise a bucket whose fill is exactly a
  power of two below capacity to the next power of two, doubling that bucket's columns,
  (ii) open a new bucket and shift the index of every later bucket of that kind, and (iii) in
  a model whose only clamped nodes are scalars, where capacity is one cell, give one bucket
  per scalar. *Mechanism:* a unit test on `pack_buckets` for each of the three; the PR reports
  the bucket list of every zoo model before and after.

If invariant B cannot be met for a reason found during implementation, the fallback is a
layout rule, not a protocol: `pack_buckets` gives every bucket `log_t ≥ 1`. The scalar ties do
not come back.

### Non-Goals

- The `MeanOfSquares`, `Div`, `Rsqrt` and trig remainder range checks, including their scalar
  branches (`specs/one-range-check.md`).
- Any change to `pack_buckets`, including a rule that keeps a scalar from growing a bucket.
- Any change to the split, exact-value, digit-range or slack relations.
- The exact flag and the unsaturated-output tie (`specs/one-clamp.md`).
- Scalar handling in ops with no scalar branch today (Rsqrt, Cos, Sin, softmax, activations).
- A zero-variable case in the Akita unit tests; the end-to-end cases are the coverage.
- Any change to trace semantics, `SemVersion` or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] `OpTie::ScalarAccumulation`, `OpTie::ScalarClamp` and `verify_scalar_clamp` do not exist.
- [ ] `clamp_buckets` and `has_remainder_rc` have no scalar filter; `fused_rescale_plan`,
      `clamp_steps`, `accumulation_plan` and the `Sum` plan have no scalar branch.
- [ ] Invariant A: `is_scalar` has exactly the three callers named there, and
      `OpTie::ScalarRemainder` exactly one emitter; the PR lists them.
- [ ] Invariant B: the six saturating scalar cases (`add_scalar_sat_pos`, `add_scalar_sat_neg`,
      `sub_scalar_sat_pos`, `sub_scalar_sat_neg`, `sum_scalar_saturating`,
      `mul_scalar_rescale_sat`) pass, each with its proof round-tripped through serialization.
- [ ] New op-surface cases pass for an internal unsaturated scalar `Add`, `Sum` and fused
      `Mul` (a bucket member that is not a model output), and for a scalar `Square`, `Cube`
      and `Einsum` output.
- [ ] Invariant C: a probe model in `testing/models.rs` has an internal scalar clamped node, an
      internal scalar fused node and a saturating scalar, with values meeting the digit
      condition; the three sweeps pass over it with no tolerance entry.
- [ ] Invariant E: the three `pack_buckets` unit tests pass; the PR has the before-and-after
      bucket lists.
- [ ] `proof_types_all_exercised` passes; no `ProofType` is added or removed.
- [ ] The importer's golden hashes are untouched. `digest_op_gauntlet` is re-pinned in its own
      commit (the gauntlet has an internal scalar `Sum`), with the `fuzz_regressions_replay`
      corpus regenerated there. The other three digests are expected unchanged; the PR states
      whether each moved.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 proves and verifies under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the three sweeps; the zoo;
`exact_output_bucket_commits_nothing`; `saturating_probe_saturates`, with its scalar filter
removed; `fuzz_regressions_replay`.

The twelve scalar op-surface cases, all of which have the scalar as the model output:

- Through a one-cell bucket after this change: the six saturating cases of invariant B.
- Through the unsaturated-output tie, never reaching a bucket: `add_tiny_t1`,
  `add_scalar_no_sat`, `sub_tiny_t1`, `sub_scalar_no_sat`, `mean_of_squares_scalar_out`,
  `sum_axis0_1d`. These hold invariant 4; `mean_of_squares_scalar_out` also keeps exercising
  `OpTie::ScalarRemainder`.

New tests:

- The internal-scalar and `Square`/`Cube`/`Einsum` op-surface cases of the acceptance list.
- The probe model of invariant C.
- The three `pack_buckets` unit tests of invariant E: a scalar joining a bucket filled to an
  exact power of two; a scalar opening a new bucket; two scalars with no other clamped node.

CI coverage: the workspace job covers all of it. GPT-2 has no scalar clamped node, so its
proof does not change; the heavy run is the repository's rule for a change to the clamp
protocol's membership, run once.

### Performance

Benchmark: `gpt2 --seq-len 16` with `bench/gate.sh` against the baseline commit `c65753bfa`.
GPT-2, bge-small and qwen2-tiny have no scalar clamped node, so their bucket lists and proofs
are unchanged: prove and verify within the gate's 3%, proof size identical.

A model with scalar clamped nodes pays the layout cost of invariant E. The worst case is one
bucket doubled per width, or one new bucket per width. This is accepted; `op_gauntlet`'s
before-and-after bucket list in the PR is the record of what it costs where it occurs.

## Design

### Architecture

A scalar node enters `clamp_buckets` and `remainder_buckets` like any other node:
`pow2_padded_num_output_elements() = 1`, `log_t = 0`, an offset that is a multiple of 1.
`PackedSegment { prefix: offset, prefix_len: bucket.log_t, r: [] }` selects one cell. The plan
then emits the same steps as for a larger node:

```text
Add/Sub:   Reduce, Open(ClampAcc), [Tie(ExactClampOutput)], Tie(Add/SubAccumulation)
Sum:       Reduce, Open(ClampAcc), [Tie(ExactClampOutput)], Relation(SumAxis)
fused:     Reduce, Open(RescaleRemainder), Open(ClampAcc), [Tie(ExactClampOutput)], Relation(...)
```

and the clamp and remainder are proven in the post-loop bucket batches. A scalar `Add` or
`Sub` gains an `Open(ClampAcc)` it does not have today; `Sum` and the fused ops already open
it for scalars.

Nothing in `atlas-sumcheck`, `atlas-pcs` or `atlas-trace` changes. What the path relies on,
each present today:

- `Math::log_2(1) = 0`, and `pack_buckets` never sees a zero size.
- `Relation::opening_points` returns the incoming point for a zero-round relation; the split
  and value relations then open `OUT`, `S` and `IND` at the empty point.
- The engine skips a zero-round member, and a batch whose members all have zero rounds carries
  an empty proof, which the verifier takes and the drain check accounts for.
- `GruenSplitEqPolynomial::new(&[])` has an explicit empty case; the Booleanity and one-hot
  opening-reduction provers handle `log T = 0`.
- Akita commits a zero-variable dense column in a class padded to its minimum arity, and a
  K = 16 one-hot column below sixteen variables as a dense column.

### Alternatives Considered

- **Keep the scalar ties.** Rejected: a second protocol for the same statement, outside the
  probe models.
- **Keep a scalar from growing a bucket**: place it only where it fits under the bucket's
  current power of two, otherwise in a small bucket shared by such scalars. Rejected at
  analysis: it adds a packing rule to remove a cost no benchmark model pays. Invariant E
  states the cost instead.
- **A general best-fit packing.** Rejected: it moves the layout and the digests of every model.
- **Pad every scalar node to two elements in `pad_graph`.** Rejected: changes node shapes for
  every consumer to remove a branch in the bucket code.
- **Minimum bucket size of two.** Kept as the fallback for invariant B only.
- **A plan-shape test for invariant A**, comparing the step kinds of a scalar and a
  two-element node. Rejected at analysis: `Step` and the relation enums derive neither `Debug`
  nor `PartialEq` and no test builds a plan, so it needs a new public method in the trust
  perimeter to guard a property that becomes compiler-held one spec later.
- **Do this inside `specs/one-range-check.md`.** Rejected: that spec moves five families onto
  the buckets and needs one-cell segments to work first; separating them keeps a failure
  attributable.

## Documentation

- `docs/upstream-inventory.md`: the two lines that describe "non-scalar nodes" and "scalar
  clamp checks".
- Comments and module docs that describe a scalar path: `plan.rs` (`OpOpen::ClampAcc`,
  `clamp_steps`), `ops/add.rs`, `ops/sum/mod.rs`, `fused_rebase.rs`, and the header of
  `clamp_lookups/mod.rs` in both crates.
- `docs/verifier.md` needs no change: it does not mention the scalar ties.
- No `README.md` change: no user-facing behaviour or benchmark moves.

## Execution

Suggested order: (1) remove the two filters and run the six saturating scalar cases, to learn
whether a one-cell bucket works before anything is deleted; (2) the plan branches and tie
variants; (3) the new op-surface cases, the probe model and the `pack_buckets` tests;
(4) the `op_gauntlet` digest re-pin and docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/clamp_lookups/mod.rs` (the scalar rationale),
  `global_clamp.rs` (`pack_buckets`), `plan.rs` (`fused_rescale_plan`, `clamp_steps`).
- `atlas-pcs/src/akita/mod.rs` (`MIN_DENSE_NUM_VARS`, `MIN_ONE_HOT_NUM_VARS`, `ClassKind::of`);
  `atlas-algebra/src/poly/split_eq.rs` (the empty-point case).
- `scripts/gen_test_models.py` (the gauntlet's scalar `ReduceSum`).
- #256 (the change that introduced the scalar path).
- `specs/clamp-dense-digit-range-check.md` (the change that removed the one-hot output chunks).
- `specs/one-range-check.md`, which finishes the removal.
