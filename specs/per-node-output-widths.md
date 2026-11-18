# Spec: Per-node output widths

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-18                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/digits-as-trace.md` (stage 1), `specs/one-range-check.md` |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every exact-arithmetic node whose output is not public commits that output as 32 bits per
element (`out + 2^31`, 16 base-4 digits), because the trace contract bounds it by `i32`. Those
columns are 62% of the committed rows of Qwen and of GPT-2 at 64 tokens, and commit time,
opening time and peak memory all grow with the committed bits. The values are far narrower
than 32 bits: on both models over 98% of range-checked elements fit in 20 bits and none needs
more than 29. This spec lets a compiled model state a width below 32 for a node's output. The
tracer treats a value outside the width as an execution error, as it treats a value outside
`i32` today, and the prover commits and range-checks the output at that width.

Measured with `prove --census` at 64 tokens (synthetic token ids, real weights): committing each
output at the width its node's largest value needs cuts the output digit rows by 1.59x on Qwen
and 1.54x on GPT-2, which is about 23% of all committed rows on Qwen.

## Intent

### Goal

Give each node of a compiled `atlas_trace::Model` an output width `w ≤ 32`, enforce it in the
trace engine, and range-check a digit-backed node's output as `out + 2^(w−1) ∈ [0, 2^w)` in a
range bucket of width `w`.

Types and boundaries:

- `atlas-trace`: the model records a width per node (32 unless stated). `Model::compile` takes
  the widths through `CompileOptions`. The trace engine rejects an output element outside
  `[−2^(w−1), 2^(w−1))` with an execution error.
- `jolt-atlas-verifier::onnx_proof::range_bucket`: `OUTPUT_WIDTH` stops being a constant. The
  width of a `Remainder::Output` claim is the node's width, read from the model.
- `jolt-atlas-core`: the bucket witness packs an output with the offset of its node's width.
- `jolt-atlas-core/examples/prove.rs`: `--census` can write the widths it measures to a file,
  and the example reads that file when it compiles the model.

Widths are restricted to the classes `{16, 20, 24, 28, 32}`. A class is even, so its base-4
digits bound the value exactly, and a small set keeps the number of buckets small.

### Invariants

- **The verifier derives every width from the model.** The widths are part of the compiled
  model the verifier's preprocessing holds; nothing about them is read from the proof.
  Mechanism: `Schedule::compile` takes the model alone, and the schedule pins
  (`testing/schedules.rs`) change when a width changes.
- **A narrower width is a stronger statement, never a weaker one.** `[−2^(w−1), 2^(w−1))` is
  inside `i32` for every allowed `w`, so every argument that today relies on an output being an
  `i32` still holds: the uniqueness of a fused rescale's quotient and remainder, and the bounds
  a consumer assumes on its operands. Mechanism: `Model::compile` rejects a width outside the
  classes; a unit test covers the rejection.
- **Execution and proof agree on the bound.** The tracer fails on exactly the values the range
  claim excludes: an honest trace always has a proof, and a trace that executed has no element
  outside its node's width. Mechanism: a tracer unit test at the boundary values
  (`2^(w−1) − 1` passes, `2^(w−1)` and `−2^(w−1) − 1` fail), and the zoo end-to-end tests run
  on a model with stated widths.
- **A malicious prover cannot use a value outside the width.** The output of a digit-backed
  node is its bucket's committed digits, `w/2` of them, each in `[0, 4)` by its commitment, so
  the recomposed value is in `[0, 2^w)`. Mechanism: `witness_tamper_sweep` and
  `proof_mutation_sweep` over a probe model with a narrowed node; a dedicated test proves from
  a trace whose narrowed output is set to `2^(w−1)` and requires the verifier to reject.
- **The contract clause "arithmetic is exact or absent" changes its bound, not its rule.** A
  sum, difference, axis sum, negation or rescaled quotient outside its node's width is an
  execution error, never a clamped value. The clause in `atlas-trace/src/lib.rs` is reworded to
  say "outside the node's output width (`i32` unless the model states less)". Mechanism:
  review-held for the wording; the tracer test above for the behavior.
- **A model without stated widths is unchanged.** Every node has width 32 and the schedule,
  proof bytes and golden hashes are what they are today. Mechanism: the existing digest and
  schedule pins and the importer's golden hashes pass without re-pinning.
- **Public outputs are untouched.** A node whose output is public
  (`range_bucket::is_public_output`) has no output claim at any width; a width stated for it is
  enforced by the tracer only.

### Non-Goals

- Deriving widths by static bound analysis. Worst-case bounds through a matrix product are far
  looser than the measured values.
- Per-element or per-channel widths, or any encoding that varies inside a node.
- Narrowing the remainder columns (their width is the scale) or the softmax gap columns.
- Dropping the output check of Add, Sub or Sum.
- Changing the digit base, the bucket packing or the value sumcheck.
- Choosing widths for the checked-in real models as part of the repository. The mechanism and
  the tool land here; a model's widths are the operator's statement about the inputs they
  intend to prove.

## Evaluation

### Acceptance Criteria

- [ ] `Model::compile` accepts a per-node width table, rejects a width outside
      `{16, 20, 24, 28, 32}` and a table naming a node the model lacks, and defaults every node
      to 32.
- [ ] The tracer returns an execution error for an output element outside its node's width, at
      both boundaries, for each op kind that checks its output (Add, Sub, Sum, Einsum,
      MeanOfSquares, and the fused Mul, Square and Cube).
- [ ] A digit-backed node of width `w` is a member of a width-`w` bucket with `w/2` digit
      columns; `prove --stats` shows the buckets by width.
- [ ] `prove --census --write-widths <file>` writes, per range-checked node, the smallest class
      holding its largest observed value plus a margin (default 2 bits), and `prove --widths
      <file>` compiles the model with them.
- [ ] `qwen2-tiny` and `gpt2-tiny` prove and verify with widths written by the census at
      sequence length 16, and the same proofs are rejected after the dedicated out-of-width
      tamper.
- [ ] With no widths stated, every existing digest pin, schedule pin and importer golden hash
      passes unchanged.
- [ ] On the benchmark machine, Qwen at 64 tokens with census widths commits at least 15%
      fewer bits than without, and `bench/gate.sh` at the default 16 tokens with no widths
      stated passes against the base commit.

### Testing Strategy

- Existing, unchanged: `cargo nextest run --workspace` (op-surface table, soundness sweeps,
  zoo, digest and schedule pins), the `atlas-onnx-import` zoo gates and golden hashes.
- New in `atlas-trace`: compile-time validation of the table; boundary tests of the width
  check per op kind.
- New in `jolt-atlas-core/src/onnx_proof/testing/`:
  - a probe model in `models.rs` with one narrowed node per width class, so the three
    soundness sweeps cover narrowed outputs;
  - a schedule pin for `qwen2-tiny` compiled with a fixed, checked-in width table;
  - the out-of-width tamper test (`prove_from_trace` on a trace whose narrowed output is
    `2^(w−1)`), expecting `Err` from `verify`.
- No new `mutate::Site` kind: a narrowed bucket's digit columns are read as today's are.
- Real models: one run each of Qwen and GPT-2 with census widths at 64 tokens on the benchmark
  machine (prove and verify), recorded in `docs/prover-plan.md`. The heavy CI job runs the
  models without widths and is unchanged.

### Performance

Benchmark: `prove` on `models/qwen` and `models/gpt2` at `--seq-len 64` on the benchmark
machine, with and without `--widths`, same session.

- Expected to move, with widths: commit, openings and peak memory, each roughly in proportion
  to the committed bits. At exact per-node widths the census gives 23% fewer committed rows on
  Qwen; with classes of four bits and a two-bit margin the expectation is 15–18%.
- Expected not to move: the sumcheck stage beyond the bucket value sumchecks, verify time
  (within 5%), proof size (within 5%; more buckets add digit openings, fewer digits per bucket
  remove them).
- Without widths: no change. `bench/gate.sh <base> HEAD --runs 3` at 16 tokens passes its
  +3% time and +5% memory bounds on both models.
- To watch: the bucket count. Buckets are split by width and by capacity, each is padded to a
  power of two and runs its own value sumcheck. If five classes cost more in padding than they
  save in digits on either model, the classes shrink to `{20, 24, 32}`; the census reports the
  padded total for both choices before the choice is fixed.

## Design

### Architecture

```
census (one trace) ──> widths file ──> CompileOptions ──> Model { width per node }
                                                            │
                         trace engine: element outside width = execution error
                                                            │
        range_bucket::node_range_claims: (Remainder::Output, width of node)
                                                            │
                  buckets by width ──> value sumcheck over w/2 digit columns
```

- **`atlas-trace`.** The width lives beside the node (`Node` or a table on `Model`, whichever
  keeps `Graph` serialization stable for models without widths). The i32 narrowing each
  checked op already performs becomes a narrowing to the node's width, in the one place the
  engine applies it. `SemVersion` does not change: the tables and op semantics are the same,
  and a model with widths is a different model, as a model with different weights is.
- **`jolt-atlas-verifier`.** `node_range_claims` reads the node's width in place of
  `OUTPUT_WIDTH`. The offset of an output claim, `2^31` today wherever the output is recomposed
  from its digits (`range_bucket.rs`, the fused-rescale input claim in `schedule.rs`), becomes
  `2^(w−1)`. Buckets already group claims by width.
- **`jolt-atlas-core`.** `BucketWitnesses::build` packs `out + 2^(w−1)` into `w/2` digits. The
  commitment layout already commits a digit column at two bits a coefficient, so a narrowed
  bucket commits `w` bits per row with no change to `atlas-pcs`.
- **The widths file.** JSON, node index to width, with the model's node count and the sequence
  length it was measured at, so a file applied to a different compilation is rejected. It is an
  input to compilation, like the sequence length.

### Alternatives Considered

- **One narrower global width.** GPT-2's residual sums and mean-of-squares reach 29 bits, so a
  global width could drop at most two bits. Rejected: 6% of the output digits against 35–40%.
- **Static interval analysis.** Sound for every input with no calibration, but the bound on a
  matrix product is the product of operand bounds times the fan-in, which exceeds 32 bits
  after one layer. Rejected as the source of widths; it remains possible later for Add and Sub
  alone.
- **A zero-friendly encoding at 32 bits** (sign and magnitude, so high digits are zero). The
  commitment's cost follows the number of committed bits, not the number of set bits; only the
  grand product skips empty columns. Rejected: it does not reduce commit time or memory.
- **Widths in the prover only**, with the verifier still checking 32 bits. Not possible: the
  width decides how many digit columns exist, and the verifier reads the commitments at the
  widths its schedule states.
- **Clamping instead of failing.** Contradicts the contract (exact or absent) and would need a
  clamp relation per node. Rejected.

## Documentation

- `atlas-trace/src/lib.rs`: the "exact or absent" clause, reworded as above.
- `CLAUDE.md` / `AGENTS.md`: the same clause in the behavioral contract section; the `prove`
  flags.
- `docs/verifier.md`: the range-claim paragraph states the width as the node's.
- `docs/prover-plan.md`: the measured rows at 64 tokens, with and without widths.
- `README.md`: one paragraph on widths as an operator choice, and what happens when an input
  exceeds one (execution fails before proving; recompile without widths or with wider ones).

## Execution

Suggested commit sequence:

1. `atlas-trace`: the width table on the model, compile validation, default 32. Mechanical;
   no behavior change.
2. `atlas-trace`: the engine narrows to the node's width. Boundary tests.
3. Verifier and prover: the output claim's width and offset come from the node. With no widths
   stated every pin passes unchanged; that is the check for this commit.
4. Probe model, tamper test, soundness coverage.
5. `prove --census --write-widths` and `prove --widths`; the padded-total report for the two
   class sets.
6. Measurements, docs, the checked-in width table for `qwen2-tiny` and its schedule pin.

Open questions for the author before implementation:

- Whether an input that exceeds a calibrated width failing to execute is acceptable to
  operators, or whether the example should fall back to compiling without widths.
- The default margin (2 bits proposed) and whether the class set is fixed here or left to the
  measurement in step 5.
- Whether widths measured at one sequence length may be applied at another. The proposal ties
  a file to its sequence length.

## References

- `specs/one-range-check.md`, `specs/digits-as-trace.md`, `specs/clamp-dense-digit-range-check.md`:
  the bucket and digit design this builds on.
- `jolt-atlas-core/examples/prove.rs`, `census`: the measurement behind the numbers above.
- DeepProve (Lagrange Labs): requantization as 27% of proving cost, the same pressure from a
  different protocol.
