# Spec: Deferred rescale

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-18                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-clamp.md`, `specs/one-range-check.md`, `specs/wave-schedule.md` |
| Supersedes  | `docs/prover-plan.md` section B, item 3 (as a plan item) |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every product in a model (einsum, Mul, Square, Cube) rescales its exact accumulation back to
the model's scale at once, and each rescale commits 46 bits per output cell: a 14-bit
remainder and a 32-bit quotient, both range-checked. The range-check columns are over 90% of
what a proof commits, and commit time, opening time and peak memory follow the committed bits.

A rescale is needed only where a value has to be narrow again: before a table lookup, before a
model output, or before a further product would outgrow the field. Between those points the
exact accumulation is not advice. It is a function of operands that are already checked, so it
needs no commitment, only a bound that keeps it from wrapping.

Measured on the lowered graphs at 16 tokens, by share of rescaled output cells:

| Qwen | Share | Rescale needed |
|---|---|---|
| Vocabulary projection (the model output) | 22.5% | yes; its quotient is public already |
| MLP gate projection, read by SiLU | 16.8% | yes, a table index |
| MLP up projection, read by the gate multiply | 16.8% | no |
| Gate multiply, read by the down projection | 16.8% | no |
| RMSNorm gain multiply, read by a projection | 8.5% | no |
| Projections read by a residual Add | 7.4% | deferrable past the Add |
| Rotary products read by Add or Sub | 4.7% | deferrable past the Add or Sub |
| Multiply read by a multiply (inside RMSNorm) | 4.3% | no |

On GPT-2, 42.5% of rescaled cells are projections read by a bias Add, 9.6% a constant multiply
read by an Add, 9.6% a multiply read by a multiply, and 9.2% a constant multiply read by a
projection.

Qwen's MLP is `down(silu(gate(x)) · up(x))`: four rescales, of which only the gate's is needed
before the end. With the other three taken as one at the down projection's output, the two
8,192-wide tensors in the middle commit nothing and the 1,024-wide output carries a 42-bit
remainder in place of a 14-bit one. That chain alone is about 40% of Qwen's committed bits.

This spec makes the rescale an operation of its own, lets a product's exact accumulation be a
node's value, and places rescales where a bound requires them.

**The governing choice.** The lowering computes the model in whatever arrangement is cheapest
to prove and verify. Where a rescale is placed changes how often a value is rounded, so
outputs change; that is accepted, and there is no compatibility with the outputs of the
lowering this replaces.

## Intent

### Goal

Let a node's value be an exact integer at a multiple of the model's scale (a *wide* value),
prove products and sums over wide values with no commitment, and rescale once per chain with
one range-checked remainder and one range-checked quotient.

- **Levels.** A node has a level `j ≥ 1`: its value `v` represents `v / 2^(j·S)`. Level 1 is
  today's `i32` value. A product's level is the sum of its operands' levels. Add, Sub, Sum and
  Neg take operands of one level (a lower one is lifted by `2^((j−i)·S)`, exactly) and keep it.
- **Wide values are node outputs.** `OpSpec::eval` takes and returns a value that is narrow
  (`Tensor<i32>`) or wide; the trace holds each node's output in that form. An op with no wide
  form rejects a wide operand at compile time. Stage 1 has one wide type, `Tensor<i64>`, and
  one wide level, 2; stage 3 adds `i128` and levels above 2.
- **Exact products.** `Einsum`, `Mul`, `Square` and `Cube` gain a *deferred* form whose output
  is the exact accumulation at the product's level. It is a new form: the existing
  `rebase: false` forms wrap in `i32` (`atlas-trace/src/op/fused.rs`) and stay as they are for
  integer-typed products, and `Einsum` has no unrescaled form today.
- **`Rescale`.** A new op taking a level-`j` value to level 1: `v = q·2^((j−1)·S) + r`,
  Euclidean, `q` in `i32`, with `r` captured in the trace as an `i64` tensor (a remainder of
  28 or 42 bits does not fit the `i32` remainder a fused product captures).
- **Magnitude bounds.** A static bound per node, computed at compile time: `2^31` for a
  level-1 value, the largest absolute entry for a constant, the largest absolute entry of its
  table for a table read, the product of operand bounds times the contraction length for a
  product, the sum of operand bounds for a sum. `Model::compile` rejects a graph in which a
  wide value's bound reaches `2^125`, or `2^62` while the wide type is `i64`.
- **Placement, per value.** The importer lowers a product deferred when every consumer of its
  value can take it wide and the bounds allow, and rescales it otherwise. A value is wide for
  all of its consumers or level 1 for all of them: there is one rescale of a value, never one
  per consumer. A graph output is level 1. A wide value is never an operand of an op without a
  wide form.
- **Proof.** A wide node is not digit-backed and has no range claim; its relation is the op's
  arithmetic relation on its reduced claim, the shape the verifier already has for a product
  without a rescale (`fused_rebase::input_expression` with `bits == 0`). `Rescale` is a
  zero-round tie `wide(r) = out(r)·2^((j−1)·S) + rem(r)`, with `rem` in the remainder bucket of
  width `(j−1)·S` and `out` in the output bucket; `trig::trig_period` is the existing tie of
  this shape.

A product that is rescaled at once (today's fused op) is the same thing with the tie folded
into the relation's input claim; it stays as it is.

### Invariants

- **A wide value never wraps.** Every wide node's magnitude bound is below `2^125`, so both
  sides of every relation that reads or produces it are integers of magnitude below half the
  field modulus, and equality in the field is equality over the integers. Each relation's
  "No wrap" note, which today assumes `i32` operands (`ops/einsum/mod.rs`, `ops/mul.rs`,
  `ops/add.rs`, `ops/sum/axis.rs`), is restated in terms of its operands' bounds. Mechanism:
  `Model::compile` computes and checks the bounds; the verifier's `Schedule::compile` calls the
  same function and returns `Err` for a model that violates one; a unit test per op kind at the
  boundary; a schedule test that a model built with an over-wide chain does not compile.
- **A wide value is determined, not chosen.** It has no commitment and no advice: the claim on
  it is produced by its consumers' relations and consumed by its own, whose operands are
  level-1 values (range-checked or public) or wide values to which this holds in turn. A
  malicious prover has no value to set. Mechanism: `witness_tamper_sweep` over a probe model
  with a deferred chain (a tampered wide tensor must be rejected), and
  `relation_kinds_all_exercised`.
- **The rescale's quotient and remainder are unique.** `rem ∈ [0, 2^((j−1)·S))` and
  `out ∈ [−2^31, 2^31)` are range claims, and the wide value's bound makes
  `out·2^((j−1)·S) + rem` an integer identity, so the pair is the Euclidean one. Mechanism: the
  range buckets, which take any even width (`range_bucket::range_claims`); a dedicated test
  that proves from a trace whose remainder is shifted by the divisor and expects rejection.
- **A deferred chain's output is the exact chain value rounded once.** Mechanism: a test that
  computes each chain of the probe models and the tiny zoo with no intermediate rounding, in
  wide integers written out independently of the kernels, and requires the lowered model's
  level-1 value to equal its Euclidean quotient exactly.
- **Execution is exact or absent.** A wide accumulation outside its node's bound, or a
  `Rescale` quotient outside `i32`, is an execution error. Mechanism: tracer unit tests at the
  bounds.
- **A wide operand is never read by a kernel that cannot hold it.** In particular the einsum
  fold's unreduced run sum (`atlas_algebra::poly::compact::run_sum_unreduced`) takes values of
  at most 32 bits and treats a wider one as zero; a wide operand takes a path that multiplies
  by the full value. Mechanism: the fold's wide path has its own test against
  `MultilinearPolynomial::evaluate`, and `relation_prover_equiv_contraction` runs on a
  contraction with a wide operand.
- **The verifier derives everything from the model.** Levels, bounds and rescale placement are
  in the compiled graph; nothing is read from the proof. Mechanism: `Schedule::compile` takes
  the model alone; the schedule pins.
- **The behavioral contract changes in two clauses.** "Quantized values are `i32` at one
  global scale" becomes: a node's value is an integer at `2^(j·S)` for its level `j`, `i32` at
  level 1, and a product's accumulation may be a node's value. The fused-rescale clause
  (quotient and remainder captured, quotient in `i32`) covers `Rescale` as it does a fused
  product. Mechanism: review-held for the text in `atlas-trace/src/lib.rs` and `CLAUDE.md`.

### Non-Goals

- Compatibility with the outputs of the current lowering. The golden hashes and the proof and
  schedule pins are re-pinned; no second lowering is kept.
- A new `SemVersion`. No output of the `semantics` module changes; the graph does.
- Deferral per consumer (a wide value kept for some readers and rescaled for others), and so
  a residual stream that stays wide across blocks: its bound would reach the limit within a
  few blocks.
- Deferral through a table read, a gather index, softmax, Rsqrt or MeanOfSquares: they take
  level-1 operands.
- Narrower output widths (`specs/per-node-output-widths.md`) and the row check on wide rows.
  Both compose with this and are separate.
- Folding a constant gain into the next projection's weights at import time. Deferral removes
  that product's commitment already; removing its sumcheck too is a later choice under the
  same governing rule.
- Committing weights, or any change to the commitment scheme.

## Evaluation

### Acceptance Criteria

Stage 1 (this spec's first PR): one deferred product, wide values in `i64`.

- [ ] `OpSpec::eval` and the trace carry narrow and wide values; an op with no wide form
      rejects a wide operand when the model compiles.
- [ ] Levels and magnitude bounds are inferred by `Model::compile`, which rejects a wide value
      whose bound reaches `2^62`.
- [ ] The deferred forms of `Einsum` and `Mul` over level-1 operands, wide `Add` and `Sub`
      with a lifted level-1 operand, and `Rescale` from level 2 each have a row in
      `op_surface.rs` and prove and verify.
- [ ] The importer lowers Qwen's MLP up projection deferred (rescaled after the gate
      multiply, which takes one wide operand) and GPT-2's projection-plus-bias to one rescale
      after the Add.
- [ ] The exact-chain test passes on the probe models and the tiny zoo.
- [ ] The three soundness sweeps pass on a probe model holding a level-2 chain, and the
      shifted-remainder test rejects.
- [ ] On the benchmark machine, with the stage-1 lowering: Qwen at 16 tokens commits at least
      15% fewer bits than the base commit, prove time is lower, and verify time and proof size
      are within 10%.

Later stages, each its own PR:

- [ ] Stage 2: deferred `Square` and `Cube`, wide `Sum` and `Neg`.
- [ ] Stage 3: `i128` wide values and levels 3 and 4, the bound limit at `2^125`, `Rescale`
      with a 28-bit and a 42-bit remainder; Qwen's MLP lowered to one rescale after the down
      projection and one before SiLU; Qwen at 16 tokens commits at least 30% fewer bits than
      the stage-0 base and proves at least 10% faster.

### Testing Strategy

- `cargo nextest run --workspace` throughout. The importer's golden hashes and the digest and
  schedule pins move in each stage that changes the lowering, re-pinned in their own commit.
- New in `atlas-trace`: level and bound inference; the deferred kernels against sums written
  out in wide integers; `Rescale` at the `i32` boundary of the quotient; compile rejection at
  the bound and for a wide operand of an op without a wide form.
- New in `onnx_proof/testing`: a probe model in `models.rs` with a deferred chain per stage
  (einsum, Add, Rescale; einsum, Mul, einsum, Rescale) under the three sweeps; the
  shifted-remainder test; the exact-chain test; `op_surface.rs` rows for `Rescale` and the
  deferred forms, with saturation stances (a quotient outside `i32` is an expected execution
  error). `OpKind::of` has no wildcard, so the build fails until the rows exist.
- A new relation kind follows `docs/pull-requests.md` section 4: `OpRelKind::Rescale` with
  its arms in `relation_id`, `serves` and the four wildcard-free matches of
  `jolt-atlas-core/src/onnx_proof/ops/mod.rs`, and its S ⊆ R note.
- Real models: GPT-2 and Qwen prove and verify on the benchmark machine at each stage; the
  heavy CI job covers them.
- No new `mutate::Site` kind: `Rescale` appends its remainder as `OpOpen::RescaleRemainder`
  does today.

### Performance

Benchmark: `bench/gate.sh` on GPT-2 and Qwen at 16 tokens, and `prove` at `--seq-len 64`, on
the benchmark machine, each stage against the commit before it in one session.

- Committed bits: about 40% fewer on Qwen from the full MLP chain (stage 3), about half of
  that from the up projection alone (stage 1), and more from the smaller cases; on GPT-2 the
  bias Adds' output columns go at stage 1. These are counts from the graph, not measurements
  of a build.
- Commit, openings and peak memory follow the committed bits. The one measured point is that
  removing 14% of Qwen's bits took 6.7% off prove time and 7% off peak memory at 64 tokens;
  by that ratio the expectation at stage 3 is 15 to 20% of prove time and about a third of
  peak memory.
- Sumchecks: about level. The relations of a chain are today's minus their remainder opens;
  a chain adds waves (a wide node waits for its consumer), so a few more batched sumchecks.
- Trace: a deferred einsum over level-1 operands is today's `i64` accumulation without its
  rescale pass. A product with a wide operand needs wider arithmetic; the budget is no more
  than 10% on trace time, measured at stage 3.
- Verify and proof size: within 10%.

## Design

### Architecture

```
importer:  product ─deferred─> wide value ──> product / Add ──> ... ──> Rescale ──> level 1
                                    │                                      │
tracer:                 wide tensor, bound-checked            q (i32), r (i64) captured
                                    │                                      │
proof:           no commitment, no range claim       tie: wide(r) = q(r)·2^(k·S) + r(r)
                 claim comes from the consumer        r in the width-k·S bucket, q in the
                 relation, goes to the op relation    output bucket (unless public)
```

- **`atlas-trace`.** `Eval.output`, the operands of `OpSpec::eval`, `Trace::node_outputs` and
  `LayerData` hold a value that is narrow or wide; today all are `Tensor<i32>`. `Tensor`
  gains an `i128` element at stage 3. Levels and bounds live on the compiled graph beside
  each node, inferred in one pass after padding; a deferred product is zero-preserving under
  padding as its fused form is. The bound of a constant is over its actual entries, which is
  what makes a three-product chain fit: with 16-bit weights and an 18-bit table output, the
  down projection's accumulation in Qwen's MLP is below `2^102`.
- **`jolt-atlas-core`.** Every reader of a node's output goes through `TraceView`
  (`trace_view.rs`); it gains the wide accessors. A wide operand's witness is
  `MultilinearPolynomial::I64Scalars` or `I128Scalars`, which exist. The einsum layouts' folds
  take a wide operand by multiplying by its full value. The bucket witnesses take the
  `Rescale` remainder as `i64`.
- **`jolt-atlas-verifier`.** `fused_rebase::op_rebase_bits` reads a product's form; a
  deferred product has `bits == 0`, so `fused_rescale_plan` already emits no remainder open
  and `input_expression` is the reduced output alone. `checks_output` is false for it, so it
  is not digit-backed and takes its claim by `Reduce` after its consumers; the wave
  compiler's readiness rule orders that. `Rescale` is `OpRelKind::Rescale`, a tie built like
  `trig::trig_period`, with its remainder in `node_remainders` as
  `Bound::Const(1 << ((j−1)·S))`.
- **`atlas-onnx-import`.** Placement is a pass in `passes::run_all`, after
  `matmul_gemm_to_einsum` has made the einsums and before `lower::lower`: mark each product
  deferred when every consumer has a wide form and the bound allows, and insert `Rescale`
  after the last wide node of each chain. It reports, per model, the cells rescaled before
  and after.

### Alternatives Considered

- **Level 2 only, for good.** Fits `i64` and needs no `i128`, but leaves the gate multiply's
  output, half of the MLP saving. It is stage 1 here, behind the interface the rest needs.
- **Fused chain ops** (one op for `down(act(gate)·up)`, one for projection plus bias). Keeps
  every node at `i32`, but each pattern is a new op with its own kernel, relation and tests,
  and the next model's pattern is another. Levels are one mechanism for all of them.
- **Deferral per consumer.** Reaches a residual stream kept wide across blocks, but the bound
  grows along the stream and forces a rescale within a few blocks, and a value held in two
  forms needs two sets of claims.
- **A wide value as committed advice with a range check.** That is today's cost in another
  place.
- **Worst-case bounds from `i32` operands alone.** Three products of `i32` values reach
  `2^93` before any contraction length and a fourth passes the field; the public constants'
  actual magnitudes are what make useful chains fit.
- **Keeping the current lowering beside the new one.** Two sets of pins and goldens for an
  arrangement nobody wants to prove.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the behavioral contract's scale clause and the
  fused-rescale clause.
- `docs/verifier.md`: the `Rescale` tie, the bound as part of the trust perimeter (the
  verifier recomputes it from the model), and that a wide value has no claim of its own.
- `docs/prover-plan.md`: section B, item 3 points here; the measured rows at each stage.
- `README.md`: the rounding note (a chain rounds once).

## Execution

1. **Stage 1.** `atlas-trace`: the narrow-or-wide value through `eval`, the trace and
   `LayerData`, with every op unchanged in behavior (a mechanical commit); levels, bounds, the
   deferred `Einsum` and `Mul`, wide `Add` and `Sub`, `Rescale`. Proof: wide operands in the
   product and sum relations, the `Rescale` tie, the probe model and sweeps. Importer: the
   placement pass. Re-pin, gate, measure.
2. **Stage 2.** The remaining deferred forms.
3. **Stage 3.** `i128`, levels 3 and 4, the wider remainders, the full MLP chain. Re-pin,
   gate, measure.

Decisions taken in review (2026-11-18): wide values are real node outputs, staged from
`i64`; deferral is per value; the lowering is free to change outputs for the sake of proving
and verifying cost, with correctness held by the exact-chain test and no compatibility with
the previous lowering.

## References

- `jolt-atlas-verifier/src/onnx_proof/fused_rebase.rs`: the fused rescale this generalises.
- `jolt-atlas-verifier/src/onnx_proof/ops/trig.rs`, `trig_period`: the existing tie of the
  form `P·q + rem`.
- `specs/one-clamp.md`: exact-or-absent semantics. `specs/one-range-check.md`: the buckets.
- `docs/prover-plan.md` section B: committed volume.
- The measurement: the lowered node lists of GPT-2 and Qwen at 16 tokens, classified by the
  consumers of each rescaled node (2026-11-18).
- Thaler, "Time-Optimal Interactive Proofs for Circuit Evaluation" (2013): layered evaluation
  with uncommitted intermediate layers.
