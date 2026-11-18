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

## Intent

### Goal

Let a node's value be an exact integer at a multiple of the model's scale (a *wide* value),
prove products and sums over wide values with no commitment, and rescale once per chain with
one range-checked remainder and one range-checked quotient.

- `atlas-trace`: a node has a *level* `j ≥ 1`: its value `v` represents `v / 2^(j·S)`. Level 1
  is today's `i32` value. A level above 1 is wide, held in the trace as an `i128` tensor. A
  product's level is the sum of its operands' levels; Add, Sub, Sum and Neg take operands of
  one level (a lower one is lifted by `2^((j−i)·S)`, exactly) and keep it. A new op `Rescale`
  takes a level-`j` value to level 1: `v = q·2^((j−1)·S) + r`, Euclidean, `q` in `i32`.
- `atlas-trace`: a static *magnitude bound* per node, computed at compile time from the
  graph: `2^31` for a level-1 value, the largest absolute entry for a constant, the largest
  absolute entry of its table for a table read, the product of operand bounds times the
  contraction length for a product, the sum of operand bounds for a sum. `Model::compile`
  rejects a graph in which a wide value's bound reaches `2^125`.
- `atlas-onnx-import`: a pass that lowers each product raw (no rescale) and inserts `Rescale`
  where a consumer needs level 1 (a table read, a gather index, a model output, an op with no
  wide form) or where the bound of the next product would reach `2^125`.
- `jolt-atlas-verifier` and `jolt-atlas-core`: a wide node is not digit-backed and has no
  range claim; its relation is the op's arithmetic relation on its reduced claim, as a raw
  `Mul` is today. `Rescale` is a zero-round tie `wide(r) = out(r)·2^((j−1)·S) + rem(r)` with
  `rem` in the remainder bucket of width `(j−1)·S` and `out` in the output bucket.

A product that is rescaled at once (today's fused op) is the same thing with the tie folded
into the relation's input claim; it stays as it is.

### Invariants

- **A wide value never wraps.** Every wide node's magnitude bound is below `2^125`, so both
  sides of every relation that reads or produces it are integers of magnitude below half the
  field modulus, and equality in the field is equality over the integers. Mechanism:
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
  range buckets (a width per `(j−1)·S`, as Cube's 28 bits is today); a dedicated test that
  proves from a trace whose remainder is shifted by the divisor and expects rejection.
- **Execution is exact or absent.** A wide accumulation outside its node's bound, or a
  `Rescale` quotient outside `i32`, is an execution error. The bound makes the first
  impossible for level-1 operands in `i32`; the tracer still checks it, in `i128`. Mechanism:
  tracer unit tests at the bounds.
- **The verifier derives everything from the model.** Levels, bounds and rescale placement are
  in the compiled graph; nothing is read from the proof. Mechanism: `Schedule::compile` takes
  the model alone; the schedule pins.
- **The behavioral contract changes in one clause.** "Quantized values are `i32` at one global
  scale" becomes: a node's value is an integer at `2^(j·S)` for its level `j`, `i32` at level
  1, and a product's accumulation may be a node's value. The fused-rescale clause (quotient
  and remainder captured, quotient in `i32`) holds for `Rescale` as it does for a fused
  product. Mechanism: review-held for the text in `atlas-trace/src/lib.rs` and `CLAUDE.md`.
- **Outputs change, deliberately, and only by rounding.** A chain rounds once where it
  rounded at every product. Mechanism: the importer's golden hashes are re-pinned in their own
  commit; a test compares the deferred lowering's outputs with the eager lowering's on the zoo
  and bounds the difference (a few units in the last place per chain; the bound is fixed when
  measured).

### Non-Goals

- Changing the scale, the tables, or any op's arithmetic. Only where rounding happens moves.
- Deferral through a table read, a gather index, softmax, Rsqrt or MeanOfSquares: they take
  level-1 operands.
- Narrower output widths (`specs/per-node-output-widths.md`) and the row check on wide rows.
  Both compose with this and are separate.
- Folding a constant gain into the next projection's weights at import time. It removes the
  same rescale by changing the model's constants; deferral removes it without.
- Committing weights, or any change to the commitment scheme.

## Evaluation

### Acceptance Criteria

- [ ] `atlas-trace` has levels, the `Rescale` op, wide tensors in the trace, and the
      magnitude bound, with `Model::compile` rejecting a wide value whose bound reaches
      `2^125`.
- [ ] Raw `Einsum`, `Mul`, `Square` and `Cube` and wide `Add`, `Sub`, `Sum` and `Neg` execute
      exactly on wide operands, and each has a row in `op_surface.rs`.
- [ ] `Rescale` proves and verifies at levels 2, 3 and 4, and the three soundness sweeps pass
      on a probe model holding a chain of each length.
- [ ] The importer's deferral pass lowers Qwen's MLP to one rescale after the down projection
      and one before SiLU, and GPT-2's projection-plus-bias to one rescale after the Add.
- [ ] With deferral off, every digest pin, schedule pin and golden hash passes unchanged.
- [ ] With deferral on, the zoo proves and verifies, and the deferred outputs differ from the
      eager ones by at most the stated rounding bound.
- [ ] On the benchmark machine, Qwen at 16 tokens commits at least 30% fewer bits with
      deferral on, and prove time is at least 10% lower. Verify time and proof size are
      within 10% of the eager lowering's.

### Testing Strategy

- Existing, unchanged with deferral off: `cargo nextest run --workspace`, the importer's zoo
  gates and golden hashes.
- New in `atlas-trace`: level and bound inference; the wide kernels against sums written out
  in `i128`; `Rescale` at the `i32` boundary of the quotient; compile rejection at the bound.
- New in `onnx_proof/testing`: a probe model in `models.rs` with a level-2, a level-3 and a
  level-4 chain (einsum, Mul, einsum, with an Add in one) under the three sweeps; the
  shifted-remainder test; `op_surface.rs` rows for `Rescale` and the wide forms, with
  saturation stances (a quotient outside `i32` is an expected execution error).
- Schedule pins and digest pins for the zoo under the deferred lowering, added beside the
  eager ones while both lowerings exist.
- Real models: GPT-2 and Qwen prove and verify with deferral on, on the benchmark machine;
  the heavy CI job runs the default lowering.
- No new `mutate::Site` kind: `Rescale` appends its remainder as `OpOpen::RescaleRemainder`
  does today.

### Performance

Benchmark: `bench/gate.sh` on GPT-2 and Qwen at 16 tokens, and `prove` at `--seq-len 64`, on
the benchmark machine, deferral on against off in one session.

- Committed bits: about 40% fewer on Qwen from the MLP chain, and more from the smaller
  cases; on GPT-2 the bias Adds' output columns go. These are counts from the graph, not
  measurements of a build.
- Commit, openings and peak memory follow the committed bits. The one measured point is that
  removing 14% of Qwen's bits took 6.7% off prove time and 7% off peak memory at 64 tokens;
  by that ratio the expectation is 15 to 20% of prove time at 64 tokens and about a third of
  peak memory. At 16 tokens the share of commit and openings is the same, so the expectation
  is similar.
- Sumchecks: about level. The relations of a chain are today's minus their remainder opens;
  a chain adds waves (a wide node waits for its consumer), so a few more batched sumchecks.
- Trace: wide accumulation in `i128` for the deferred products. The einsum kernel sums `i32`
  products in `i64`; a raw einsum over level-1 operands still does, and only a product with a
  wide operand needs wider arithmetic. To be measured; the budget is no more than 10% on
  trace time.
- Verify and proof size: within 10%.

## Design

### Architecture

```
importer:  product ──raw──> wide value ──> product / Add ──> ... ──> Rescale ──> level 1
                                 │                                      │
tracer:              i128 tensor, bound-checked            q (i32), r captured
                                 │                                      │
proof:        no commitment, no range claim       tie: wide(r) = q(r)·2^(k·S) + r(r)
              claim comes from the consumer        r in the width-k·S bucket, q in the
              relation, goes to the op relation    output bucket (unless public)
```

- **Levels and bounds** live on the compiled graph beside each node, inferred in one pass
  after padding. The bound of a constant is over its actual entries, which is what makes a
  three-product chain fit: with 16-bit weights and an 18-bit table output, the down
  projection's accumulation in Qwen's MLP is below `2^102`.
- **The trace** holds a wide node's tensor in a side map by node (`Trace::wide`), leaving
  `node_outputs` as the `i32` tensors every level-1 consumer reads; a wide node's entry there
  is empty. The prover's witness for a wide operand is its field image.
- **The relations** exist: a raw `Mul` (`rebase: false`) and an einsum without a rescale are
  proven against their reduced output opening today. They gain wide operands, which changes
  their witnesses and nothing in their expressions. `Rescale` is a new `OpRelKind`, a tie.
- **The schedule.** A wide node is not digit-backed (`checks_its_output` is false), so it
  takes its claim by `Reduce` after its consumers, as any uncommitted node does. The readiness
  rule of the wave compiler already orders this.
- **Placement** is the importer's: lower raw, then walk the graph in order and insert
  `Rescale` before a consumer that needs level 1 and before a product whose bound would
  reach the limit, choosing the point with the fewest cells when a chain can be cut in more
  than one place. The pass reports, per model, the cells rescaled before and after.

### Alternatives Considered

- **Fused chain ops** (one op for `down(act(gate)·up)`, one for projection plus bias). Keeps
  every node at `i32` and needs no wide tensors, but each pattern is a new op with its own
  kernel, relation and tests, and the next model's pattern is another. Levels are one
  mechanism for all of them.
- **A wide value as committed advice with a range check.** That is today's cost in another
  place.
- **Folding gains and biases into weights at import.** Removes some rescales with no proof
  change, but alters the constants and cannot reach the gate multiply, which is the largest
  case.
- **Worst-case bounds from `i32` operands alone.** Three products of `i32` values reach
  `2^93` before any contraction length and a fourth passes the field; the public constants'
  actual magnitudes are what make useful chains fit.
- **A lower scale or narrower activations.** Changes what the model computes; this does not.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the behavioral contract's scale clause and the
  fused-rescale clause.
- `docs/verifier.md`: the `Rescale` tie, the bound as part of the trust perimeter (the
  verifier recomputes it from the model), and that a wide value has no claim of its own.
- `docs/prover-plan.md`: section B, item 3 points here; the measured rows when built.
- `README.md`: the importer option and the rounding note.

## Execution

Stages, each mergeable with deferral off by default:

1. `atlas-trace`: levels, bounds, wide tensors, `Rescale`, the wide kernels. No importer
   change; tests build graphs by hand.
2. Proof: wide operands in the raw relations, the `Rescale` tie, the probe model and sweeps.
3. Importer: the placement pass behind an option, with its report; golden hashes and pins for
   the deferred lowering beside the eager ones.
4. Measure on GPT-2 and Qwen. If the criteria hold, make deferral the default in its own
   commit, re-pin, and remove the eager pins.

Open questions for the author before implementation:

- Whether the changed outputs need a new `SemVersion` or are a property of the lowering (the
  kernels of existing ops do not change; the graph does). The proposal is the lowering.
- Whether the first stage should stop at level 2 (one deferred product: no `i128`, since a
  level-2 accumulation of `i32` operands with public bounds fits `i64`), which reaches the up
  projection and the bias Add but not the gate multiply's output.
- The rounding bound to assert between the two lowerings.

## References

- `jolt-atlas-verifier/src/onnx_proof/fused_rebase.rs`: the fused rescale this generalises.
- `specs/one-clamp.md`: exact-or-absent semantics. `specs/one-range-check.md`: the buckets.
- `docs/prover-plan.md` section B: committed volume.
- The measurement: the lowered node lists of GPT-2 and Qwen at 16 tokens, classified by the
  consumers of each rescaled node (2026-11-18).
- Thaler, "Time-Optimal Interactive Proofs for Circuit Evaluation" (2013): layered evaluation
  with uncommitted intermediate layers.
