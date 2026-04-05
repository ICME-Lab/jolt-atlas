# Qwen Hotspots Notes (2026-04-03)

Current high-level hotspots after selector optimization:

- `BatchedSumcheck::prove`: about 39s
- `ONNXProof::prove_reduced_openings`: about 27s
- `SoftmaxAxes::prove`: about 23s
- `ONNXProof::commit_witness_polynomials`: about 17s
- `tensor::ops::einsum`: about 17s
- `ProverOpeningAccumulator::prove_batch_opening_sumcheck`: about 15s

## Hotspot Table

Using:

- `prove`: `106.30s`
- `total`: `121.28s`

Note:
- These spans are inclusive and overlap.
- In particular, `BatchedSumcheck::prove` is inside several larger proof paths.
- The percentages below should be read as impact indicators, not additive shares.

| Hotspot | Time | % of `prove` | % of `total` |
|---|---:|---:|---:|
| `BatchedSumcheck::prove` | `39.15s` | `36.8%` | `32.3%` |
| `ONNXProof::prove_reduced_openings` | `27.11s` | `25.5%` | `22.4%` |
| `SoftmaxAxes::prove` | `23.26s` | `21.9%` | `19.2%` |
| `ONNXProof::commit_witness_polynomials` | `17.49s` | `16.5%` | `14.4%` |
| `tensor::ops::einsum` | `17.48s` | `16.4%` | `14.4%` |
| `ProverOpeningAccumulator::prove_batch_opening_sumcheck` | `14.92s` | `14.0%` | `12.3%` |

## What Each Hotspot Does

### `BatchedSumcheck::prove`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/subprotocols/sumcheck.rs`

Role:
- Shared batching layer for many sumcheck instances.
- For each round it:
  - calls each instance's `compute_message`
  - linearly combines the resulting univariate polynomials
  - appends prover messages to the transcript
  - derives and ingests challenges
  - caches opening claims at the end

Why it is heavy:
- It is a common bus for many expensive proof paths.
- In Qwen it is used heavily by:
  - `SoftmaxAxes`
  - activation LUT proofs
  - `Gather`
  - reduced opening proofs
- The cost mostly reflects the total work done by the child instances' `compute_message`.

Detailed metrics from `cargo run --release --package jolt-atlas-core --example qwen -- --use-cache`:

- `calls`: `225`
- `instances_total`: `70889`
- `max_instances`: `27302`
- `rounds_total`: `6938`
- `max_rounds`: `81`
- `total`: `37.96s`

Breakdown:

| Sub-step | Time | % of `BatchedSumcheck::prove` |
|---|---:|---:|
| `compute_messages` | `21.39s` | `56.4%` |
| `ingest_challenges` | `14.61s` | `38.5%` |
| `cache_openings` | `1.40s` | `3.7%` |
| `update_individual_claims` | `393.60ms` | `1.0%` |
| `combine_univariate_polys` | `80.44ms` | `0.2%` |
| `append_input_claims` | `30.61ms` | `<0.1%` |
| `batching_coeffs` | `23.85ms` | `<0.1%` |
| `transcript_and_challenge` | `17.90ms` | `<0.1%` |
| `initialize_individual_claims` | `3.88ms` | `<0.1%` |
| `compress` | `399.98µs` | `<0.1%` |
| `finalize_instances` | `93.19µs` | `<0.1%` |

Interpretation:

- The hotspot is overwhelmingly in child-instance work, not batching mechanics.
- The two dominant costs are:
  - `compute_messages`
  - `ingest_challenges`
- Transcript work, polynomial compression, and linear combination are negligible by comparison.
- So the next drill-down should focus on which prover families dominate:
  - `compute_message`
  - `ingest_challenge`
  within the batched instances.

### `compute_message` breakdown

Additional `Top compute_message spans` from the same run:

| Prover | Time |
|---|---:|
| `OneHotPolynomialProverOpening::compute_message` | `7.83s` |
| `BooleanitySumcheckProver::compute_message` | `3.47s` |
| `RaSumcheckProver::compute_message` | `2.48s` |
| `DensePolynomialProverOpening::compute_message` | `1.07s` |
| `HammingWeightSumcheckProver::compute_message` | `621.20ms` |
| `InstructionReadRafSumcheckProver::compute_message` | `523.38ms` |
| `ReshapeSumcheckProver::compute_message` | `118.78ms` |
| `SliceSumcheckProver::compute_message` | `80.53ms` |
| `ConcatSumcheckProver::compute_message` | `39.70ms` |

Current interpretation:

- The largest instrumented share comes from opening-related provers:
  - `OneHotPolynomialProverOpening`
  - `DensePolynomialProverOpening`
- The next largest share comes from lookup/constraint machinery:
  - `BooleanitySumcheckProver`
  - `RaSumcheckProver`
- Selector-based reorder ops are now negligible in `compute_message`.
- The instrumented list does not sum to the full `compute_messages` total, which means some relevant `compute_message` implementations are still uninstrumented. The likely next gap is operator-specific sumchecks, especially inside softmax-related proofs.

Updated run after instrumenting previously unmeasured `compute_message` implementations:

| Prover | Time |
|---|---:|
| `OpeningProofReductionSumcheckProver::compute_message` | `8.51s` |
| `OneHotPolynomialProverOpening::compute_message` | `7.40s` |
| `SoftmaxExponentiationSumcheckProver::compute_message` | `3.60s` |
| `BooleanitySumcheckProver::compute_message` | `3.50s` |
| `RaSumcheckProver::compute_message` | `2.49s` |
| `DensePolynomialProverOpening::compute_message` | `1.02s` |
| `SoftmaxMaxSumcheckProver::compute_message` | `714.32ms` |
| `SoftmaxScalarDivSumcheckProver::compute_message` | `704.97ms` |
| `HammingWeightSumcheckProver::compute_message` | `631.88ms` |
| `SoftmaxSumSumcheckProver::compute_message` | `511.47ms` |
| `InstructionReadRafSumcheckProver::compute_message` | `508.93ms` |
| `ScalarConstDivSumcheckProver::compute_message` | `480.96ms` |
| `MulSumcheckProver::compute_message` | `401.37ms` |
| `AddSumcheckProver::compute_message` | `140.32ms` |
| `ReshapeSumcheckProver::compute_message` | `119.31ms` |
| `MkKnMnProver::compute_message` | `83.36ms` |
| `SliceSumcheckProver::compute_message` | `81.98ms` |
| `ConcatSumcheckProver::compute_message` | `40.08ms` |
| `RbmkRbnkBmnProver::compute_message` | `38.01ms` |
| `TeleportDivisionSumcheckProver::compute_message` | `37.18ms` |
| `IffSumcheckProver::compute_message` | `30.98ms` |
| `NegSumcheckProver::compute_message` | `26.57ms` |
| `SigmoidSumcheckProver::compute_message` | `17.68ms` |
| `SumAxisSumcheckProver::compute_message` | `14.84ms` |
| `RsqrtSumcheckProver::compute_message` | `9.51ms` |
| `GatherSumcheckProver::compute_message` | `9.39ms` |
| `SinSumcheckProver::compute_message` | `720.96µs` |
| `CosSumcheckProver::compute_message` | `604.58µs` |
| `HammingBooleanitySumcheckProver::compute_message` | `173.08µs` |

Important note:

- These `compute_message` spans are also inclusive.
- In particular, `OpeningProofReductionSumcheckProver::compute_message` is a wrapper over:
  - `OneHotPolynomialProverOpening::compute_message`
  - `DensePolynomialProverOpening::compute_message`
  so those times overlap and should not be added directly.

Current interpretation:

- The biggest `compute_message` family is opening reduction, dominated by one-hot openings.
- The next biggest operator-specific family is softmax:
  - exponentiation
  - max
  - scalar_div
  - sum
- `Booleanity` and `RaSumcheck` remain major generic lookup/constraint costs.
- Reorder ops (`reshape / slice / concat`) and gather execution are now small.

### `ONNXProof::prove_reduced_openings`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/jolt-atlas-core/src/onnx_proof/prover.rs`

Role:
- Final reduction of many opening claims into one joint opening proof.
- Main stages:
  - `prepare_for_sumcheck`
  - `prove_batch_opening_sumcheck`
  - `finalize_batch_opening_sumcheck`
  - `build_materialized_rlc`
  - `PCS::prove`

Why it is heavy:
- It aggregates the entire proof's committed polynomial openings.
- Its cost scales with the total number of claims, not one operator.
- It combines:
  - opening reduction sumcheck
  - RLC materialization
  - PCS opening proof

### `ProverOpeningAccumulator::prove_batch_opening_sumcheck`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/poly/opening_proof.rs`

Role:
- Core reduction sumcheck for batched openings.
- Takes all accumulated opening-sumcheck instances and feeds them into `BatchedSumcheck::prove`.

Why it is heavy:
- Dense openings and one-hot openings both accumulate here.
- Cost scales with the number of opening instances.
- This is a large fraction of `prove_reduced_openings`.

### `SoftmaxAxes::prove`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/jolt-atlas-core/src/onnx_proof/ops/softmax_axes/mod.rs`

Role:
- Proves the entire softmax operator.
- Main internal stages:
  - `generate_trace_cache`
  - `prove_div_sum_max`
  - `prove_exponentiation`
  - `prove_operand_claims`

Why it is heavy:
- One softmax node expands into multiple proof stages.
- It works feature-chunk by feature-chunk.
- `prove_exponentiation` includes both read/raf and RA one-hot proof work.
- `generate_trace_cache` reruns `softmax_fixed_128::<true>` to rebuild proof-side trace data.

### `ONNXProof::commit_witness_polynomials`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/jolt-atlas-core/src/onnx_proof/prover.rs`

Role:
- Builds witness polynomials and commits to them.
- Main stages:
  - `polynomial_map`
  - `commit_to_polynomials`
  - transcript append of commitments

Why it is heavy:
- Many witness polynomials are generated in Qwen.
- Many PCS commitments are produced.
- Cost is a mix of:
  - witness generation
  - memory traffic/materialization
  - commitment/MSM work

### `tensor::ops::einsum`

File:
- `/Users/clankpan/Develop/ZKP/jolt-atlas/atlas-onnx-tracer/src/tensor/ops.rs`

Role:
- Trace-generation cost, not proof-system cost.
- Called while executing the model to build the trace before proving.

Why it is heavy:
- General einsum implementation does:
  - shape and index analysis
  - summation-coordinate setup
  - precomputation of index helpers
  - nested contraction loops over output and sum coordinates
- Qwen has many einsum nodes, so this adds up.

## Current Interpretation

- `BatchedSumcheck::prove` is a shared proof-path hotspot.
- `prove_reduced_openings` and `prove_batch_opening_sumcheck` are global opening-reduction hotspots.
- `SoftmaxAxes::prove` is the largest obvious operator-specific hotspot.
- `commit_witness_polynomials` is witness-generation plus commit overhead.
- `tensor::ops::einsum` is trace-generation runtime.

## Suggested Next Investigation Order

1. Split `prove_reduced_openings` into:
   - `prove_batch_opening_sumcheck`
   - `build_materialized_rlc`
   - `PCS::prove`
2. Split `SoftmaxAxes::prove` into:
   - `generate_trace_cache`
   - `prove_div_sum_max`
   - `prove_exponentiation`
   - `prove_operand_claims`
3. Split `commit_witness_polynomials` into:
   - `polynomial_map`
   - `commit_to_polynomials`

## Exclusive Prove Breakdown

Latest run with exclusive phase accounting:

- `prove`: `115.26s`
- `summed_top_level`: `115.23s`

This is intentionally different from the inclusive span view. These phases are
measured so that their sum matches the full prove time.

### Top-level phases

| Phase | Time |
|---|---:|
| `trace` | `18.80s` |
| `trace_io` | `690.75µs` |
| `prover_new` | `1.81s` |
| `commit_witness_polynomials` | `17.17s` |
| `output_claim` | `6.41ms` |
| `iop` | `52.39s` |
| `prove_reduced_openings` | `24.77s` |
| `finalize_proof` | `282.81ms` |

### Reduced openings subphases

| Phase | Time |
|---|---:|
| `prepare_for_sumcheck` | `328.34ms` |
| `prove_batch_opening_sumcheck` | `13.30s` |
| `finalize_batch_opening_sumcheck` | `10.22ms` |
| `build_materialized_rlc` | `6.49s` |
| `pcs_prove` | `4.64s` |
| `summed_reduced_openings` | `24.77s` |

### IOP per-op totals

These are also exclusive and sum to `iop = 52.39s`.

| Operator | Total | Count |
|---|---:|---:|
| `SoftmaxAxes` | `22.66s` | `24` |
| `Constant` | `15.10s` | `322` |
| `Sigmoid` | `4.12s` | `24` |
| `Einsum` | `3.03s` | `218` |
| `ScalarConstDiv` | `2.15s` | `607` |
| `Rsqrt` | `1.75s` | `49` |
| `GatherLarge` | `824.24ms` | `1` |
| `Add` | `665.24ms` | `241` |
| `Mul` | `609.20ms` | `339` |
| `Reshape` | `479.05ms` | `147` |
| `Slice` | `231.67ms` | `96` |
| `MoveAxis` | `202.36ms` | `48` |
| `Concat` | `190.02ms` | `49` |

### IOP top nodes

Largest exclusive node times:

- `node 0 Constant`: `3.39s`
- `node 2848 Constant`: `3.30s`
- then many `SoftmaxAxes` nodes around `0.94s` to `0.96s`

Interpretation:

- The largest exclusive prove phase is now `iop`, followed by `prove_reduced_openings`.
- Within `iop`, `SoftmaxAxes` is the dominant operator family.
- `Constant` is surprisingly large in aggregate and now deserves direct investigation.
- `trace` is also a significant exclusive cost, but it is separate from proof generation logic.

## Constant Breakdown

The first `Constant` aggregate was inflated by a metrics bug: the code was using
`format!("{op:?}")` to derive the operator name, which forced debug-formatting of
large constant tensors. This was fixed by adding `Operator::variant_name()` and
using that instead.

After fixing the metrics labeling overhead, the latest run is:

- `prove`: `102.94s`
- `iop`: `38.12s`
- `prove_reduced_openings`: `26.53s`

### Corrected IOP per-op totals

| Operator | Total | Count |
|---|---:|---:|
| `SoftmaxAxes` | `22.54s` | `24` |
| `Sigmoid` | `4.28s` | `24` |
| `Einsum` | `3.17s` | `218` |
| `ScalarConstDiv` | `2.15s` | `607` |
| `Rsqrt` | `1.76s` | `49` |
| `GatherLarge` | `821.79ms` | `1` |
| `Add` | `675.50ms` | `241` |
| `Mul` | `609.53ms` | `339` |
| `Constant` | `594.83ms` | `322` |

### Constant per-phase totals

| Phase | Total | Count |
|---|---:|---:|
| `Constant/eval_reduction` | `594.42ms` | `322` |
| `Constant/execution` | `216.92µs` | `322` |

### Eval reduction spans

Across all nodes:

| Span | Total | Count |
|---|---:|---:|
| `NodeEvalReduction::prove` | `2.71s` | `2851` |
| `NodeEvalReduction::prove/EvalReductionProtocol::prove` | `2.17s` | `2851` |
| `EvalReductionProtocol::prove` | `2.17s` | `2851` |
| `NodeEvalReduction::prove/output_mle` | `536.49ms` | `2851` |

Interpretation:

- `Constant` itself is not a real hotspot after fixing the metrics bug.
- Almost all `Constant` time is in shared `eval_reduction`, not in `Constant::prove`.
- The `execution` path for constants is effectively free.
- The real remaining hotspots are still `SoftmaxAxes`, `reduced_openings`, and shared sumcheck/opening reduction.
