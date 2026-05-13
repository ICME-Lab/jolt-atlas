# qwen3-awy

Standalone Qwen3-0.6B float implementation for AWY pseudo-quantization experiments.

The crate intentionally does not depend on `qwen3-prover`. It implements the
full Qwen3 forward pass in f32 and can insert quantize/dequantize steps around
each MatMul:

- `A`: MatMul left input, quantized row-wise.
- `W`: MatMul right input, quantized per output channel.
- `Y`: MatMul output, quantized row-wise.

This is meant for PPL and error-ablation reports, not for proving or production
inference.

## Model

Download the model files into this crate-local ignored directory:

```bash
python scripts/download_qwen3_0_6b_safetensors.py
```

The default runtime paths are:

- `qwen3-awy/models/qwen3-0.6b/model.safetensors`
- `qwen3-awy/models/qwen3-0.6b/tokenizer.json`

The model directory is ignored by git.

`generation_config.json` for Qwen3-0.6B uses:

- `do_sample: true`
- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`
- `eos_token_id: [151645, 151643]`
- `pad_token_id: 151643`

Those sampling defaults are used when `--sample` or `--seed` is enabled. Greedy
decoding is still available by omitting sampling flags, but Qwen's docs warn
against greedy decoding for thinking mode because it can degrade quality and
lead to repetitions.

Qwen3-0.6B differs from the Qwen2-0.5B implementation in a few important
places:

- hidden size 1024, intermediate size 3072, 28 layers.
- 16 attention heads, 8 KV heads, explicit head dim 128.
- `q_proj` outputs 2048 channels, while `k_proj` and `v_proj` output 1024.
- Q/K per-head RMSNorm is applied before RoPE.
- attention projections have no bias.

## Run

Float baseline with a 128-token context:

```bash
cargo run --release -p qwen3-awy -- --seq-len 128 --a-bits none --w-bits none --y-bits none --full "hello world this is a test"
```

W8A8Y8 for every MatMul:

```bash
cargo run --release -p qwen3-awy -- --seq-len 128 --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

Only MLP MatMuls:

```bash
cargo run --release -p qwen3-awy -- --matmuls mlp --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

Only one projection:

```bash
cargo run --release -p qwen3-awy -- --matmuls gate --a-bits 8 --w-bits 8 --y-bits 8 --full "hello world this is a test"
```

`--matmuls` accepts `all`, `attention`, `mlp`, or a comma-separated list of
`q,k,v,o,gate,up,down,lm_head`.

`--seq-len` controls the causal context length. Input tokens are truncated to
that length and padded with EOS if shorter. PPL is computed over non-EOS
next-token targets within that context, and the output reports `ppl_targets` so
short and long runs can be compared honestly.
