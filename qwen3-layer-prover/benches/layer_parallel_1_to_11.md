# Layer Parallel Proving Benchmark

This benchmark measures `prove_trace_decoder` with `--layers 1..11` on the
`qwen3-awy/traces/fox_eos_full_awy` trace.

The trace has `seq = 197`. Timing was collected with
`QWEN3_LAYER_PROVER_OP_TIMING=1`, so each worker also reports its own
`prove_layer` time.

## Command

```bash
for n in 1 2 3 4 5 6 7 8 9 10 11; do
  QWEN3_LAYER_PROVER_OP_TIMING=1 cargo run --release -p qwen3-layer-prover --bin prove_trace_decoder -- \
    --trace qwen3-awy/traces/fox_eos_full_awy \
    --layers "$n"
done
```

## Results

| layers | prove_layers_parallel wall | decoder total | prove_layer avg | prove_layer max | prove_layer sum |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.401s | 14.455s | 13.401s | 13.401s | 13.401s |
| 2 | 13.831s | 14.912s | 13.714s | 13.831s | 27.428s |
| 3 | 14.177s | 15.332s | 14.083s | 14.177s | 42.250s |
| 4 | 14.405s | 15.573s | 14.298s | 14.405s | 57.191s |
| 5 | 14.580s | 15.775s | 14.439s | 14.580s | 72.197s |
| 6 | 16.128s | 17.505s | 15.990s | 16.128s | 95.941s |
| 7 | 17.191s | 18.639s | 17.026s | 17.190s | 119.184s |
| 8 | 18.518s | 20.035s | 18.356s | 18.518s | 146.849s |
| 9 | 19.430s | 21.011s | 19.273s | 19.430s | 173.458s |
| 10 | 20.435s | 22.126s | 20.288s | 20.435s | 202.885s |
| 11 | 21.540s | 23.311s | 21.352s | 21.540s | 234.875s |

Generated artifacts:

- `qwen3-layer-prover/benches/layer_parallel_1_to_11.csv`
- `qwen3-layer-prover/benches/layer_parallel_1_to_11.png`
- `qwen3-layer-prover/benches/layer_parallel_work_sum_1_to_11.png`

## Interpretation

Up to 5 layers, parallel proving scales reasonably well. The wall time only
increases from 13.401s to 14.580s, while the total amount of layer work grows
from 13.401s to 72.197s.

Starting around 6 layers, each individual `prove_layer` becomes slower. The
average per-layer time grows from 13.401s at 1 layer to 21.352s at 11 layers.
This means the slowdown is not just queueing or scheduling overhead. The layer
workers themselves are running more slowly under concurrency.

The likely bottleneck is shared resource pressure, especially memory bandwidth,
cache pressure, or allocation pressure. This matches the earlier observation
that large parallel layer proving can become memory-sensitive even when the
trace already stores accumulator tensors and `build_witness` is no longer the
dominant cost.

For future optimization, the important target is reducing per-layer memory
traffic and temporary allocation inside the proving path, not only increasing
the number of Rayon workers.

## Timing Lock Check

The first run used `QWEN3_LAYER_PROVER_OP_TIMING=1`, which prints many timing
lines from inside each op. Since `stderr` is internally locked, this could have
introduced artificial contention.

To check that, layer-level timing was split into
`QWEN3_LAYER_PROVER_LAYER_TIMING=1`. The benchmark was rerun with only
layer-level timing enabled, so op-level `eprintln!` calls were disabled.

| layers | prove_layers_parallel wall | decoder total | prove_layer avg | prove_layer max | prove_layer sum |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.372s | 14.415s | 13.372s | 13.372s | 13.372s |
| 2 | 13.818s | 14.893s | 13.712s | 13.818s | 27.423s |
| 3 | 14.239s | 15.366s | 14.143s | 14.233s | 42.430s |
| 4 | 14.280s | 15.431s | 14.161s | 14.280s | 56.643s |
| 5 | 14.619s | 15.823s | 14.501s | 14.619s | 72.504s |
| 6 | 15.987s | 17.341s | 15.831s | 15.987s | 94.983s |
| 7 | 17.457s | 18.909s | 17.214s | 17.456s | 120.500s |
| 8 | 18.417s | 19.942s | 18.234s | 18.417s | 145.875s |
| 9 | 19.488s | 21.158s | 19.284s | 19.488s | 173.557s |
| 10 | 20.560s | 22.264s | 20.323s | 20.560s | 203.229s |
| 11 | 21.525s | 23.314s | 21.363s | 21.525s | 234.991s |

The results are essentially the same as the op-timing run. Therefore, the
slowdown is not primarily caused by `stderr` timing-output lock contention.

The remaining likely causes are memory bandwidth pressure, cache pressure, or
allocator contention inside the proving path.

## Proof Retention Check

Two additional benchmark-only variants were tested and then removed because
they did not materially change the timing.

1. `drop-proofs`: run normal `prove_layer`, but drop each `LayerProof` before
   collecting the parallel result.
2. `drop-op-proofs`: additionally avoid retaining op proofs until the end of
   `prove_layer`; each op proof is dropped as soon as its claims are extracted.

| mode | layers | prove wall | decoder total | prove_layer sum | prove_layer max |
|---|---:|---:|---:|---:|---:|
| normal | 1 | 13.404s | 14.417s | 13.404s | 13.404s |
| normal | 5 | 14.627s | 15.796s | 72.446s | 14.626s |
| normal | 11 | 21.590s | 23.358s | 234.515s | 21.590s |
| drop-proofs | 1 | 13.420s | 13.554s | 13.420s | 13.420s |
| drop-proofs | 5 | 14.738s | 14.956s | 72.723s | 14.738s |
| drop-proofs | 11 | 21.646s | 22.051s | 235.111s | 21.645s |
| drop-op-proofs | 1 | 13.530s | 13.661s | 13.530s | 13.530s |
| drop-op-proofs | 5 | 14.726s | 14.952s | 72.799s | 14.726s |
| drop-op-proofs | 11 | 21.568s | 21.963s | 234.486s | 21.568s |

The per-layer prove time is almost unchanged. Therefore, retaining proof objects
is not the cause of the parallel slowdown. The total decoder time is lower in
the drop variants because verification is skipped, but the core proving wall
time remains the same.

## Cause Isolation

`N=1` and `N=11` were rerun with op-level timing to identify whether one op is
responsible or whether the slowdown is broad.

| op | N=1 avg | N=11 avg | slowdown |
|---|---:|---:|---:|
| softmax | 3.919s | 6.217s | 1.59x |
| silu | 2.193s | 3.785s | 1.73x |
| q_norm | 1.031s | 1.805s | 1.75x |
| qk_score | 1.013s | 1.564s | 1.54x |
| silu_up | 0.837s | 1.395s | 1.67x |
| gate_proj | 0.452s | 0.737s | 1.63x |
| up_proj | 0.452s | 0.720s | 1.59x |
| rms_norm_mlp | 0.516s | 0.789s | 1.53x |
| k_norm | 0.517s | 0.778s | 1.50x |
| q_rope | 0.428s | 0.662s | 1.55x |

The slowdown is broad, not isolated to a single op.

The detailed `softmax` and `silu` phase timings show the same pattern:

| phase | N=1 avg | N=11 avg | slowdown |
|---|---:|---:|---:|
| softmax.lookup.sumcheck | 1.314s | 2.082s | 1.58x |
| softmax.acc_sumcheck | 0.540s | 0.819s | 1.52x |
| softmax.output_round | 0.467s | 0.776s | 1.66x |
| softmax.floor | 0.477s | 0.757s | 1.59x |
| softmax.exp_round | 0.476s | 0.748s | 1.57x |
| softmax.lookup.shout | 0.268s | 0.439s | 1.64x |
| softmax.lookup.remainder_shout | 0.271s | 0.438s | 1.62x |
| silu.relation_sumcheck | 0.897s | 1.413s | 1.58x |
| silu.base_shout | 0.204s | 0.406s | 1.99x |
| silu.slope_shout | 0.204s | 0.420s | 2.06x |
| silu.round_shout | 0.285s | 0.478s | 1.68x |

Setup/allocation-looking phases are small:

| phase | N=1 avg | N=11 avg |
|---|---:|---:|
| silu.setup_validate | 0.017s | 0.035s |
| silu.witness_polys | 0.023s | 0.034s |
| silu.eq_polys | 0.033s | 0.049s |
| softmax.acc_polys | 0.026s | 0.042s |
| softmax.lookup.witness_polys | 0.017s | 0.028s |
| softmax.lookup.eq_polys | 0.025s | 0.042s |

This makes allocator churn an unlikely primary cause. Allocation may still
contribute inside lower-level polynomial routines, but the dominant measured
slowdown is in sumcheck/shout phases that repeatedly bind and scan large
polynomial buffers.

Current conclusion: the parallel slowdown is primarily memory/cache pressure
from concurrent large-polynomial sumcheck/shout work, not proof retention,
timing-output locks, or a single pathological op.

## Follow-Up Negative Checks

Several additional implementation experiments were tried and then reverted
because they did not improve `prove_layers_parallel` wall time.

### RMSNorm Row Expansion Virtualization

`rms_norm` expands row-wise values such as `row_eq` and `inv_rms` across the
hidden dimension. A virtual row-expanded polynomial was tested to avoid
materializing those full tensors.

Correctness tests passed, but runtime did not improve:

| mode | layers | prove_layers_parallel wall | note |
|---|---:|---:|---|
| before | 11 | ~21.5s | layer-level timing baseline |
| row-expanded virtual RMSNorm | 11 | 22.108s | no improvement |

This suggests that those full row-expanded tensors are not the dominant source
of the parallel slowdown.

### Softmax / SiLU Heavy-Op Concurrency Guard

A temporary guard was tested that kept layer proving parallel but limited
concurrent `softmax` and `silu` proving sections via
`QWEN3_LAYER_PROVER_HEAVY_OP_PERMITS`.

With `HEAVY_OP_PERMITS=5`, `--layers 11` became slower:

| mode | layers | prove_layers_parallel wall |
|---|---:|---:|
| baseline | 11 | ~21.5s |
| heavy op permits = 5 | 11 | 27.988s |

This confirms that simply queueing the two heaviest ops is not a good fix.

### Polynomial Bind Allocation Path

`prove_layers_parallel` disables inner Rayon parallelism with
`ParallelFlagGuard::disabled()`, but the op code still calls
`bind_parallel()`. A temporary fallback was tested so that `DensePolynomial`
and `CompactPolynomial` use ordinary in-place `bind()` when inner parallelism
is disabled.

Correctness tests passed, but runtime remained essentially unchanged:

| mode | layers | prove_layers_parallel wall |
|---|---:|---:|
| baseline | 1 | ~13.37s |
| bind fallback | 1 | 13.361s |
| baseline | 11 | ~21.5s |
| bind fallback | 11 | 21.729s |

So the slowdown is not primarily caused by `bind_parallel()` allocating a new
buffer while inner parallelism is disabled.

## Sumcheck Phase Breakdown

To separate common sumcheck overhead from op-specific behavior,
`JOLTWORKS_SUMCHECK_TIMING=1` was added around the shared
`Sumcheck::prove` loop. This measures each sumcheck's common phases:

- `compute_message`: prover-specific polynomial scan / relation evaluation.
- `ingest`: binding the challenge into the prover's multilinear polynomials.
- `transcript`, `challenge`, `compress`, `evaluate`, `cache_openings`: common
  protocol bookkeeping.

The same `seq = 197` trace was measured with `--layers 1` and `--layers 11`.

| layers | sumchecks | prove wall | sumcheck total | compute_message | ingest | other phases |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 13.341s | 5.974s | 5.200s | 0.772s | 0.002s |
| 11 | 847 | 21.941s | 101.459s | 89.123s | 12.292s | 0.040s |

Per-sumcheck averages:

| layers | total / sumcheck | compute_message / sumcheck | ingest / sumcheck |
|---:|---:|---:|---:|
| 1 | 0.0776s | 0.0675s | 0.0100s |
| 11 | 0.1198s | 0.1052s | 0.0145s |

Slowdown from 1 to 11 layers:

| phase | slowdown |
|---|---:|
| compute_message | 1.56x |
| ingest | 1.45x |
| transcript/challenge/compress/evaluate/cache_openings combined | negligible |

Conclusion: the parallel slowdown is not caused by transcript locks, Fiat-Shamir
challenge generation, proof compression, or opening accumulation. The shared
problem is in the heavy polynomial work:

1. `compute_message`, which scans bound polynomial buffers and evaluates each
   relation.
2. `ingest`, which binds challenges into multilinear polynomial buffers.

This matches a memory/cache-bandwidth bottleneck much more closely than an
op-specific algorithmic bug.

## Build Optimization Level Check

The same `--layers 11` benchmark was run with different release profile
optimization levels. For the `opt-level` comparison, LTO was disabled to isolate
the effect of the optimizer level itself:

```bash
CARGO_TARGET_DIR=target/optbench-o2 \
CARGO_PROFILE_RELEASE_OPT_LEVEL=2 \
CARGO_PROFILE_RELEASE_LTO=off \
QWEN3_LAYER_PROVER_LAYER_TIMING=1 \
cargo run --release -p qwen3-layer-prover --bin prove_trace_decoder -- \
  --trace qwen3-awy/traces/fox_eos_full_awy \
  --layers 11
```

Results:

| profile | LTO | prove_layers_parallel wall | decoder total |
|---|---|---:|---:|
| release opt-level=1 | off | 30.175s | 32.698s |
| release opt-level=2 | off | 27.826s | 30.001s |
| release opt-level=3 | off | 27.323s | 29.288s |
| repo default release | fat | 23.688s | 25.619s |

Notes:

- `opt-level=1` is clearly slower.
- `opt-level=2` and `opt-level=3` are close for this workload.
- The repo default `release` profile is much faster mainly because it uses
  `lto = "fat"` in addition to release's default `opt-level=3`.
- Build time is not included in the table; only runtime timing emitted by
  `prove_trace_decoder` is shown.
