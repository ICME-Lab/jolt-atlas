# qwen3-awy

Standalone Qwen3-0.6B fixed-point implementation for AWY experiments.

The crate intentionally does not depend on `qwen3-prover`. It implements the
full Qwen3 forward pass with block-internal tensors kept in fixed point:

- `A`: MatMul left input, quantized row-wise.
- `W`: MatMul right input, quantized per output channel.
- `Y`: MatMul output, quantized row-wise.

This is meant for PPL and fixed-point behavior experiments, not for proving or
production inference.

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

Those sampling defaults are enabled by default. Greedy decoding is still
available with `--greedy`, but Qwen's docs warn against greedy decoding for
thinking mode because it can degrade quality and lead to repetitions.

Input text is wrapped in Qwen3's no-think chat prompt by default:

```text
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Qwen3-0.6B differs from the Qwen2-0.5B implementation in a few important
places:

- hidden size 1024, intermediate size 3072, 28 layers.
- 16 attention heads, 8 KV heads, explicit head dim 128.
- `q_proj` outputs 2048 channels, while `k_proj` and `v_proj` output 1024.
- Q/K per-head RMSNorm is applied before RoPE.
- attention projections have no bias.

## Run

Default QX.8 fixed-point run with a 128-token context:

```bash
cargo run --release -p qwen3-awy -- --seq-len 128 --full "hello world this is a test"
```

Generation:

```bash
cargo run --release -p qwen3-awy -- --seq-len 128 --generate 64 "Tell a fairy tale about a quiet fox helping a lost rabbit home. Once upon a time, in a forest, a quiet fox"
```

Longer fixed-point story sample:

```bash
cargo run --release -p qwen3-awy -- --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
```

With the default QX.8 runtime, nearest MatMul rebase, nearest sigmoid input
rounding, and the QX.8 lookup tables for rounding, sigmoid, and coarse exp, this
currently produces:

```text
Once upon a time, in a small village nestled between a forest and a river, there lived a quiet fox named Lila. She was known for her gentle heart and her ability to see things from another's perspective. One day, a lost rabbit named Timmy wandered into the village, his fur brown and his eyes wide with worry. He had never heard of the village, and he felt lost and alone.

Lila noticed him and decided to help him. She knelt beside him and said, "Timmy, I know you're lost, and I'm here for you. Please don't worry. I'm here to help." Timmy looked up and saw her, and he felt safe. After a while, Lila led him home, and together they made a new home for Timmy. The village was grateful for the kindness of Lila and Timmy.

In the end, Lila and Timmy became friends, and Lila helped others when they were lost. Timmy, who had once been alone, now lived happily in the village with Lila. The story taught the value of kindness and the importance of helping others, even when they feel lost.
```

The block runtime is fixed point by default. It keeps block-internal tensors in
QX.8 by default, uses integer MatMul accumulation, uses nearest MatMul
accumulator rebases and sigmoid input indices for fixed-point SiLU, fixed-point
RoPE, fixed-point attention score/value products, and leaves only the rsqrt
advice, the coarse exp advice, `lm_head`, and decode-time sampling/logit softmax
in float.

```bash
cargo run --release -p qwen3-awy -- --fixed-frac 9 --seq-len 128 --generate 64 "Tell a fairy tale about a quiet fox helping a lost rabbit home. Once upon a time, in a forest, a quiet fox"
```

`--seq-len` controls the causal context length. Input tokens are truncated to
that length and padded with EOS if shorter. PPL is computed over non-EOS
next-token targets within that context, and the output reports `ppl_targets` so
short and long runs can be compared honestly.

Use `--fixed-frac N` to run the fixed path as QX.N instead. The current range is
`0..=12`; higher values would risk overflowing the current `i32` quantized
storage and `i64` MatMul accumulators.

Rounding ablations:

```bash
cargo run --release -p qwen3-awy -- --matmul-rebase-rounding round --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
cargo run --release -p qwen3-awy -- --sigmoid-input-rounding floor --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
cargo run --release -p qwen3-awy -- --sigmoid-input-rounding ceil --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
```

`--matmul-rebase-rounding` controls the MatMul accumulator rebase from
`QX.(a_frac + w_frac)` back to `QX.fixed_frac`. `--sigmoid-input-rounding`
controls how the fixed-point SwiGLU gate value is reduced to the integer index
used by the sigmoid approximation. Both accept `round`, `floor`, or `ceil`,
where `round` is nearest rounding based on `f = x - floor(x)`. The default is
`matmul-rebase-rounding=round` and `sigmoid-input-rounding=round`.

Per-MatMul float comparison reports are disabled by default because they run an
extra reference MatMul for every quantized MatMul and make generation much
slower. Add `--report` only when collecting error metrics.
