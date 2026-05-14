# qwen3-awy

Standalone Qwen3-0.6B fixed-point implementation.

The crate intentionally does not depend on `qwen3-prover`. It implements the
full Qwen3 forward pass with block-internal tensors kept in fixed point:

- `A`: MatMul left input, quantized row-wise.
- `W`: MatMul right input, quantized per output channel.
- `Y`: MatMul output, quantized row-wise.

This is a small execution path for the current QX.8 fixed-point runtime, not a
proving implementation or a general experiment harness.

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

Default QX.8 fixed-point generation:

```bash
cargo run --release -p qwen3-awy -- --seq-len 128 --generate 64 "Tell a fairy tale about a quiet fox helping a lost rabbit home. Once upon a time, in a forest, a quiet fox"
```

Longer fixed-point story sample:

```bash
cargo run --release -p qwen3-awy -- --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
```

With the default QX.8 runtime, nearest MatMul rebase, nearest sigmoid input
rounding, fixed `lm_head`, fixed decode-time softmax weights, and the QX.8
lookup tables for rounding, sigmoid, and coarse exp, this currently produces:

```text
**Title: The Whispering Fox and the Lost Rabbit**  

In a quiet village nestled between rolling hills and ancient trees, there lived a fox named Lira. She was known for her gentle heart and quiet kindness. One day, a rabbit named Luna was lost and wandering alone, lost in the forest.  

Luna had been following her mother, who had vanished one day, and had been lost for days. She had no place to go, and no one to trust.  

Lira noticed Luna's sadness and decided to help. She stepped forward and said, "Luna, I heard you were lost. I will help
```

The block runtime is fixed QX.8. It uses integer MatMul accumulation, nearest
MatMul accumulator rebases, fixed-point SiLU, fixed-point RoPE, fixed-point
attention score/value products, fixed-point `lm_head`, and fixed-point
decode-time softmax weights. The remaining float use is `rsqrt` advice plus
one-time conversion of sampling controls such as temperature/top-p.

On first run, the executable writes a fixed-weight cache next to the safetensors
model, using the same path with a `.q8.bin` extension. The cache stores QX.8
i32 weights after quantization and transposition, including layer weights,
`lm_head`, and final norm. Later runs reuse that file and skip the BF16/F16/F32
to i32 conversion path. The cache is a generated model artifact and should not
be committed.

`--timing` prints decode timing broken down by embedding, layer norms,
attention, QKV projection, RoPE, attention value product, output projection,
MLP gate/up/down, SiLU, residual adds, `lm_head`, and token choice.

`--dump-final-awy PATH` records the final generated token only. Generation runs
normally first, then the executable rebuilds the KV cache for the prefix and
replays the last token with tracing enabled. The output directory contains a
`manifest.jsonl` plus raw little-endian `i32` tensor files. MatMul, RMSNorm,
RoPE, residual adds, attention score/value products, softmax probabilities,
SwiGLU/SiLU, final norm, and `lm_head` are recorded. `A` and `Y` tensors are
stored as files; `W` tensors are referenced by model weight name or LUT name so
the fixed-weight cache is not duplicated into the trace.

```bash
cargo run --release -p qwen3-awy -- \
  --seq-len 128 \
  --generate 64 \
  --dump-final-awy /private/tmp/qwen3-awy-final-awy \
  "Tell a fairy tale about a quiet fox helping a lost rabbit home."
```

LUT clipping note:

`sigmoid` and softmax `exp` use small LUTs over clipped integer inputs. For a
ZK circuit, the clipped integer part can be handled by separating the same
number of low bits as the fractional extraction uses. The circuit checks that
the high part plus those low bits reconstructs the original value. A boolean
flag selects whether the LUT index is the clipped boundary or the reconstructed
unclipped integer. When the flag is `1`, the clipped boundary is used; when it
is `0`, the original integer input is used. The flag must be constrained so the
prover cannot choose an arbitrary LUT entry.

Rounding ablations:

```bash
cargo run --release -p qwen3-awy -- --matmul-rebase-rounding round --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
cargo run --release -p qwen3-awy -- --sigmoid-input-rounding floor --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
cargo run --release -p qwen3-awy -- --sigmoid-input-rounding ceil --seq-len 384 --generate 320 --seed 1 "Tell a fairy tale about a quiet fox helping a lost rabbit home."
```

`--matmul-rebase-rounding` controls the MatMul accumulator rebase from QX.16
back to QX.8. `--sigmoid-input-rounding` controls how the fixed-point SwiGLU
gate value is reduced to the integer index used by the sigmoid approximation.
Both accept `round`, `floor`, or `ceil`, where `round` is nearest rounding based
on `f = x - floor(x)`. The default is `matmul-rebase-rounding=round` and
`sigmoid-input-rounding=round`.
