# qwen2-awy

Standalone Qwen2 float implementation for AWY pseudo-quantization experiments.

The crate intentionally does not depend on `qwen2-prover`. It implements the
full Qwen2 forward pass in f32 and can insert quantize/dequantize steps around
each MatMul:

- `A`: MatMul left input, quantized row-wise.
- `W`: MatMul right input, quantized per output channel.
- `Y`: MatMul output, quantized row-wise.

This is meant for PPL and error-ablation reports, not for proving or production
inference.

## Run

Float baseline with a 128-token context:

```bash
cargo run --release -p qwen2-awy -- --seq-len 128 --a-bits none --w-bits none --y-bits none --full "hello world this is a test"
```

W8A8Y8 for every MatMul:

```bash
cargo run --release -p qwen2-awy -- --seq-len 128 --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

Only MLP MatMuls:

```bash
cargo run --release -p qwen2-awy -- --matmuls mlp --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

Only one projection:

```bash
cargo run --release -p qwen2-awy -- --matmuls gate --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

`--matmuls` accepts `all`, `attention`, `mlp`, or a comma-separated list of
`q,k,v,o,gate,up,down,lm_head`.

`--seq-len` controls the causal context length. Input tokens are truncated to
that length and padded with EOS if shorter. PPL is computed over non-EOS
next-token targets within that context, and the output reports `ppl_targets` so
short and long runs can be compared honestly.
