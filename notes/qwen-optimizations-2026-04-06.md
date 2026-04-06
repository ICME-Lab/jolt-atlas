# Qwen Optimizations From `exp/qwen-metrics`

This note records the optimizations that worked on branch `exp/qwen-metrics`.

The branch is experimental and is not expected to be merged directly. The goal of
this note is to make it easy to re-apply the successful changes on other branches.

## Summary

The large speedup did **not** come from BN254 `Fr` backend changes.

The two changes that mattered were:

1. `build_materialized_rlc` was rewritten to accumulate in cache-friendly chunks.
2. `BatchedSumcheck::prove` was changed to parallelize per-instance
   `compute_message` and `ingest_challenge`.

These two changes moved Qwen from roughly:

- `prove ~97.57s`
- `total ~112.32s`

to:

- `prove ~69.02s`
- `total ~84.09s`

## Files To Look At

- [/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/poly/rlc_polynomial.rs](/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/poly/rlc_polynomial.rs)
- [/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/subprotocols/sumcheck.rs](/Users/clankpan/Develop/ZKP/jolt-atlas/joltworks/src/subprotocols/sumcheck.rs)

## Optimization 1: `build_materialized_rlc`

### What the old code did

The old implementation built the joint RLC polynomial coefficient-by-coefficient:

- for each `joint[i]`
- iterate over every dense polynomial
- read coefficient `i`
- accumulate `sum_j coeff_j * poly_j[i]`

This is mathematically fine, but memory behavior is poor:

- it re-reads every polynomial many times
- locality is bad
- cache reuse is poor

### What changed

The new implementation accumulates by chunks:

- split `joint_coeffs` into fixed-size chunks
- for one chunk, walk each polynomial once
- add its overlapping slice into that chunk

This keeps reads and writes much more contiguous.

### Why it helped

This optimization mostly reduced memory traffic and improved locality. It did not
change the algebra, only the order of accumulation.

### Measured effect

Observed on Qwen:

- `build_materialized_rlc: ~7.45s -> ~0.36s`
- `prove_reduced_openings: ~19.50s -> ~7.44s` after combining with the sumcheck change

## Optimization 2: `BatchedSumcheck::prove`

### What the old code did

`BatchedSumcheck::prove` handled the batch of sumcheck instances serially:

- compute all per-instance messages one-by-one
- ingest all per-instance challenges one-by-one

This is wasteful because the instances are mostly independent at these steps.

### What changed

The implementation now:

- precomputes `input_claims`
- precomputes `num_rounds` per instance
- uses parallel iteration for per-instance `compute_message`
- uses parallel iteration for per-instance `ingest_challenge`

The transcript and batched polynomial combination remain serialized where required.

### Why it helped

This removed a large serial bottleneck. The workload already existed; it was just
being processed one instance at a time instead of across cores.

### Measured effect

Observed on Qwen:

- `BatchedSumcheck::prove: ~37.68s -> ~8.81s`
- `compute_messages: ~20.72s -> ~3.82s`
- `ingest_challenges: ~14.93s -> ~2.90s`

This also dropped:

- `SoftmaxAxes: ~23.20s -> ~7.74s`
- `prove_batch_opening_sumcheck: ~13.62s -> ~1.46s`

## What Did Not Matter Much

BN254 `Fr` backend experiments with `s2n-bignum` on AArch64 were not the source of
the large win.

The best `s2n-bignum` variant was close to baseline but not meaningfully better.
The large improvement came from algorithm/dataflow changes, not field backend changes.

Relevant experimental files for that work:

- [/Users/clankpan/Develop/ZKP/jolt-atlas/third_party/arkworks-algebra/ff/src/fields/models/fp/montgomery_backend.rs](/Users/clankpan/Develop/ZKP/jolt-atlas/third_party/arkworks-algebra/ff/src/fields/models/fp/montgomery_backend.rs)
- [/Users/clankpan/Develop/ZKP/jolt-atlas/third_party/arkworks-algebra/ff/src/fields/models/fp/aarch64_s2n_bn254.rs](/Users/clankpan/Develop/ZKP/jolt-atlas/third_party/arkworks-algebra/ff/src/fields/models/fp/aarch64_s2n_bn254.rs)

These are optional experiments, not part of the core optimization story.

## Current Interpretation

The real bottlenecks were:

- bad memory access in RLC materialization
- unnecessary serial processing inside batched sumcheck

The speedup came from:

- better locality
- better core utilization

not from reducing BN254 multiplication latency directly.

## Suggested Re-apply Order On Other Branches

1. Port the `build_materialized_rlc` chunked accumulation change.
2. Port the `BatchedSumcheck::prove` per-instance parallelization.
3. Re-run Qwen.
4. Only after that, evaluate whether field-backend experiments are still worth it.
