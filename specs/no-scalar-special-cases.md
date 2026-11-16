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
cycle dimension. That reason dates from when the buckets committed one-hot chunks; they now
commit dense digit columns, and the Akita adapter, the split-eq polynomial and the sumcheck
engine all handle zero variables explicitly. The scalar path costs three tie variants, a branch
in five op plans and in the fused-rescale skeleton, and a second way of proving the same
statement that the sweeps cover only through single-op cases. This spec removes it: a scalar
node is a one-cell segment of its bucket.

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
- `jolt-atlas-core`: the matching arms of `ops::prove_steps`, and `global_clamp.rs` witness
  assembly for one-cell segments.

Left in place by this spec and removed by `specs/one-range-check.md`: `OpTie::ScalarRemainder`
for `MeanOfSquares` (its remainder is a lookup range check today, not a bucket member),
`recover_small_int`, `is_scalar` itself, and the scalar branches of `ops/div.rs` and of
`MeanOfSquares`'s committed-polynomial list.

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
4. **Verifier hygiene.** Rejects with `Err`, never panics, consumes the proof exactly, stays
   Trace-free. *Mechanism:* the crate lints, the CI path checks, the mutation sweep with no
   tolerance entry, the drain check.

New:

- **A. One protocol per statement.** For every op in `{Add, Sub, Sum, MeanOfSquares, Einsum,
  Mul, Square, Cube}`, the plan of a node does not depend on whether its padded output has one
  element, except through sizes. *Mechanism:* the compiler (the two tie variants no longer
  exist) and a test asserting that the plan of a scalar node and of a two-element node of the
  same op have the same step kinds in the same order.
- **B. A bucket may have `log_t = 0`.** A model whose only clamped nodes of some width are
  scalars yields a bucket of one cell; it commits, proves and verifies. *Mechanism:* a new
  op-surface case (see Testing Strategy). This is the property the old comment says cannot
  hold, so it is the first thing the implementation establishes.
- **C. The scalar path is under the sweeps.** A probe model contains a scalar clamped node and
  a scalar fused node, so `proof_mutation_sweep`, `witness_tamper_sweep` and
  `opening_forgery_sweep` reach a one-cell segment. *Mechanism:* the probe model and the sweeps.
- **D. Saturating scalars.** A scalar that saturates carries one-hot slack chunks with zero
  cycle variables and is accepted; a tampered slack is rejected. *Mechanism:* the existing
  `add_scalar_sat_pos`/`_neg`, `sub_scalar_sat_pos`/`_neg`, `sum_scalar_saturating` and
  `mul_scalar_rescale_sat` cases, which now run through the bucket, plus the tamper sweep over
  the probe of invariant C.

If invariant B cannot be met for a reason found during implementation, the fallback is a
layout rule, not a protocol: `pack_buckets` gives every bucket `log_t ≥ 1`. The scalar ties do
not come back.

### Non-Goals

- The `MeanOfSquares`, `Div`, `Rsqrt` and trig remainder range checks, including their scalar
  branches (`specs/one-range-check.md`).
- Any change to the split, exact-value, digit-range or slack relations.
- The exact flag and the unsaturated-output tie (`specs/one-clamp.md`).
- Scalar handling in ops with no scalar branch today (Rsqrt, Cos, Sin, softmax, activations).
- Any change to trace semantics, `SemVersion` or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] `OpTie::ScalarAccumulation`, `OpTie::ScalarClamp` and `verify_scalar_clamp` do not exist.
- [ ] `clamp_buckets` and `has_remainder_rc` have no scalar filter; `fused_rescale_plan`,
      `clamp_steps`, `accumulation_plan` and the `Sum` plan have no scalar branch.
- [ ] `OpTie::ScalarRemainder` has exactly one emitter, `ops/mean_of_squares.rs`.
- [ ] Invariant A: the plan-shape test passes for each of the eight ops.
- [ ] Invariant B: a model whose only clamped node is a scalar proves and verifies, and its
      proof round-trips through serialization.
- [ ] Invariant C: a probe model in `testing/models.rs` has a scalar clamped node and a scalar
      fused node; the three sweeps pass over it with no tolerance entry.
- [ ] The scalar op-surface cases listed under Testing Strategy pass unchanged in their
      expected outcome.
- [ ] `proof_types_all_exercised` passes; no `ProofType` is added or removed.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit only if a zoo model contains a scalar clamped node, and the PR says which.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing, now through the bucket path: `add_tiny_t1`,
`add_scalar_sat_pos`, `add_scalar_sat_neg`, `add_scalar_no_sat`, `sub_tiny_t1`,
`sub_scalar_sat_pos`, `sub_scalar_sat_neg`, `sub_scalar_no_sat`, `mul_scalar_rescale_sat`,
`mean_of_squares_scalar_out`, `sum_axis0_1d`, `sum_scalar_saturating` in
`testing/op_surface.rs`; the three sweeps; the zoo; `exact_output_bucket_commits_nothing` and
`saturating_probe_saturates`; `fuzz_regressions_replay`.

New tests:

- An op-surface case for a model whose only clamped node is a scalar (invariant B), and one
  each for a scalar `Square`, `Cube` and `Einsum` output, which have no case today.
- The plan-shape test of invariant A.
- The probe model of invariant C.

CI coverage: the workspace job covers all of it. No real-model run is required unless the
digests move; if they do, GPT-2 under the `heavy` profile once.

### Performance

No change expected on any stage. The zoo and the real models have at most a handful of scalar
nodes, each adding one cell to an existing bucket. Benchmark: `gpt2 --seq-len 16` against
`bench/bin/prove-c65753bfa`, prove and verify each within the 3% run-to-run noise, proof size
within 1%. A model whose scalar nodes open a new bucket width pays that bucket's fixed cost;
the PR reports the bucket count before and after on the zoo.

## Design

### Architecture

A scalar node enters `clamp_buckets` and `remainder_buckets` like any other node:
`pow2_padded_num_output_elements() = 1`, `log_t = 0`, an offset that is a multiple of 1.
`PackedSegment { prefix: offset, prefix_len: bucket.log_t, r: [] }` selects one cell. The plan
then emits the same steps as for a larger node:

```text
Add/Sub:   Reduce, Open(ClampAcc), Tie(Add/SubAccumulation), [Tie(ExactClampOutput)]
Sum:       Reduce, Open(ClampAcc), [Tie(ExactClampOutput)], Relation(SumAxis)
fused:     Reduce, Open(RescaleRemainder), Open(ClampAcc), [Tie(ExactClampOutput)], Relation(...)
```

and the clamp and remainder are proven in the post-loop bucket batches. Nothing in
`atlas-sumcheck`, `atlas-pcs` or `atlas-trace` changes: `digit_range` has no minimum size,
`GruenSplitEqPolynomial::new` and the engine handle zero variables, and the Akita adapter pads
a short dense class to its minimum arity and commits a short one-hot column in a dense class.

### Alternatives Considered

- **Keep the scalar ties.** Rejected: a second protocol for the same statement, outside the
  probe models.
- **Pad every scalar node to two elements in `pad_graph`.** Rejected: changes node shapes for
  every consumer to remove a branch in the bucket code.
- **Minimum bucket size of two.** Kept as the fallback for invariant B only; a one-cell bucket
  is preferred because it needs no rule.
- **Do this inside `specs/one-range-check.md`.** Rejected: that spec moves five families onto
  the buckets and needs one-cell segments to work first; separating them keeps a failure
  attributable.

## Documentation

- `docs/verifier.md`: the "Plans" section drops the scalar ties.
- Module docs: `clamp_lookups/mod.rs`, `fused_rebase.rs` and `ops/sum/mod.rs` in both crates
  stop describing a scalar path.
- No `README.md` change: no user-facing behaviour or benchmark moves.

## Execution

Suggested order: (1) the op-surface case of invariant B with the filters removed, to learn
whether a one-cell bucket works before anything is deleted; (2) the plan branches and tie
variants; (3) the probe model and the plan-shape test; (4) docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/clamp_lookups/mod.rs` (the scalar rationale),
  `global_clamp.rs` (`pack_buckets`), `plan.rs` (`fused_rescale_plan`, `clamp_steps`).
- `atlas-pcs/src/akita/mod.rs` (`MIN_DENSE_NUM_VARS`, `MIN_ONE_HOT_NUM_VARS`, `ClassKind::of`).
- `specs/clamp-dense-digit-range-check.md` (the change that removed the one-hot output chunks).
- `specs/one-range-check.md`, which finishes the removal.
