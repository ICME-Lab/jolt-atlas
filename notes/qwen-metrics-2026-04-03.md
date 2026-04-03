# Qwen Metrics (2026-04-03)

Command:

```bash
cargo run --release --package jolt-atlas-core --example qwen -- --use-cache
```

## Phase Metrics

- `cache_hit`: `true`
- `cache_read`: `3.10s`
- `prover_preprocess`: `4.03s`
- `prove`: `156.98s`
- `verify`: `59.95s`
- `total`: `225.20s`

## Top Spans

- `ONNXProof::prove`: `156.98s`
- `ONNXProof::iop`: `90.83s`
- `ONNXProof::verify`: `59.95s`
- `ONNXProof::verify_iop`: `57.64s`
- `Sumcheck::verify`: `53.75s`
- `BatchedSumcheck::prove`: `38.40s`
- `Reshape::verify`: `28.80s`
- `Reshape::prove`: `28.40s`
- `ONNXProof::prove_reduced_openings`: `26.38s`
- `SoftmaxAxes::prove`: `23.26s`
- `SoftmaxAxesProver::prove`: `23.26s`
- `Model::trace`: `20.14s`
- `Model::execute_graph`: `20.14s`
- `prove_exponentiation`: `19.56s`
- `tensor::ops::einsum`: `17.77s`
- `ONNXProof::commit_witness_polynomials`: `17.71s`
- `msm`: `14.68s`
- `ProverOpeningAccumulator::prove_batch_opening_sumcheck`: `14.32s`
- `Concat::verify`: `12.49s`
- `Slice::verify`: `12.35s`
- `Concat::prove`: `12.25s`
- `Slice::prove`: `12.17s`
- `commit_to_polynomials`: `8.97s`
- `HyperKZG::commit`: `8.96s`

## Top Op Spans

- `Sumcheck::verify`: `53.75s`
- `BatchedSumcheck::prove`: `38.40s`
- `Reshape::verify`: `28.80s`
- `Reshape::prove`: `28.40s`
- `SoftmaxAxes::prove`: `23.26s`
- `ProverOpeningAccumulator::prove_batch_opening_sumcheck`: `14.32s`
- `Concat::verify`: `12.49s`
- `Slice::verify`: `12.35s`
- `Concat::prove`: `12.25s`
- `Slice::prove`: `12.17s`
- `HyperKZG::prove`: `5.05s`
- `BooleanitySumcheckProver::compute_message`: `3.47s`
- `Einsum::prove`: `3.16s`
- `MkKnMnProver::initialize`: `2.90s`
- `Constant::verify`: `2.74s`
- `RaSumcheckProver::compute_message`: `2.47s`
- `Sumcheck::prove`: `2.45s`
- `BatchedSumcheck::verify`: `1.50s`
- `ScalarConstDiv::prove`: `964.93ms`

## Immediate Optimization Targets

1. `reshape / concat / slice`
   - These are a large aggregate cost on both prove and verify.
   - They now use selector-based proofs, so selector construction and repeated sumcheck work are likely the first place to optimize.

2. Reduced openings and commitments
   - `ONNXProof::prove_reduced_openings`
   - `ProverOpeningAccumulator::prove_batch_opening_sumcheck`
   - `ONNXProof::commit_witness_polynomials`
   - `HyperKZG::commit`

3. Softmax proving path
   - `SoftmaxAxes::prove`
   - `prove_exponentiation`

4. Sumcheck verification cost
   - `Sumcheck::verify` is currently the single largest named bucket in the verifier path.

## Notes

- The current Qwen example prints these metrics directly from `jolt-atlas-core/examples/qwen.rs`.
- These numbers were collected with shared preprocessing cache enabled.
- `Model::trace` and `Model::execute_graph` appear in the span list because they are instrumented upstream, even though the top-level phase summary used a cache hit.
