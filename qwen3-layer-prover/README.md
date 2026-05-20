# qwen3-layer-prover

## Layer Timing Comparison

Use `scripts/compare_layer_timings.py` to compare a fresh `prove_trace_layer`
run against the pre-fusion baseline.

```bash
cargo run --release -p qwen3-layer-prover --bin prove_trace_layer -- \
  --trace qwen3-awy/traces/fox_eos_full_awy \
  --model qwen3-awy/models/qwen3-0.6b/model.safetensors \
  --layer 0 \
  2>&1 | qwen3-layer-prover/scripts/compare_layer_timings.py
```

Sort by saved time:

```bash
cargo run --release -p qwen3-layer-prover --bin prove_trace_layer -- \
  --trace qwen3-awy/traces/fox_eos_full_awy \
  --model qwen3-awy/models/qwen3-0.6b/model.safetensors \
  --layer 0 \
  2>&1 | qwen3-layer-prover/scripts/compare_layer_timings.py --sort saved
```

The embedded baseline is the layer0, seq=197 timing table before SHOUT-backed
round fusion into matmul-like ops.
