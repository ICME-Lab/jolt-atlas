# qwen2-prover

This crate contains direct Qwen2-0.5B experiments using the local
`atlas-onnx-tracer/models/qwen/model.safetensors` weights.

## Float Reference

The float implementation in `src/float.rs` mirrors the Qwen2 model structure:

- token embedding
- 24 decoder layers
- RMSNorm
- grouped-query attention
- RoPE
- causal softmax
- MLP with `silu(gate) * up`
- final RMSNorm
- tied LM head

Run:

```bash
cargo run --release -p qwen2-prover --bin float_ppl -- --full "hello world this is a test"
```

The float path was checked against Hugging Face `AutoModelForCausalLM` loaded
from the same local Qwen directory. For `hello world this is a test`, the
results matched closely:

| Metric | Rust float | HF reference |
| --- | ---: | ---: |
| PPL, first 3 targets | 104.0064 | 103.9934 |
| PPL, full text targets | 48.4647 | 48.4578 |
| first target rank | 100 | 100 |
| first target score | 7.51718 | 7.51720 |
| top token | 198 | 198 |
| top score | 15.39798 | 15.39803 |

This indicates the float model structure is correct to normal f32 tolerance.

## Fixed-Point Scale Sweep

The fixed-point CPU path can be swept over different fractional bit counts
with:

```bash
cargo run --release -p qwen2-prover --bin fixed_sweep -- --full --scales 10,12,14,15,16 "The quick brown fox jumps over the lazy dog."
```

Here `S` means `ONE = 1 << S`. Larger `S` gives more fractional precision, but
can also increase risk from large intermediate values.

Observed results:

### `hello world this is a test`

Float PPL: `48.4647`

| S | Fixed PPL |
| ---: | ---: |
| 8 | 330.7558 |
| 12 | 42.4518 |
| 14 | 144.0963 |
| 15 | 74.9019 |
| 16 | 90.2417 |

`S=12` is lower than the float PPL on this short sample. This should not be
interpreted as more correct. With only a few target tokens, quantization error
can accidentally raise the correct token logits and lower PPL.

### `The quick brown fox jumps over the lazy dog.`

Float PPL: `4.3798`

| S | Fixed PPL |
| ---: | ---: |
| 10 | 11.9783 |
| 12 | 5.4570 |
| 14 | 4.3534 |
| 15 | 4.3714 |
| 16 | 4.7667 |

### `Once upon a time, there was a small village near the sea.`

Float PPL: `3.9601`

| S | Fixed PPL |
| ---: | ---: |
| 10 | 15.1044 |
| 12 | 7.5888 |
| 14 | 3.7295 |
| 15 | 3.9738 |
| 16 | 4.5136 |

## Current Takeaway

`S=8` is clearly too low for this Qwen2 path. `S=12` is much better, but can be
unstable across prompts. On the short natural-language samples above, `S=14`
and `S=15` are closest to the float reference overall.

Use PPL lower than the float reference as a warning sign, not as proof of
better quality. For small samples it usually means quantization error happened
to favor the target token. The better criterion is consistent closeness to the
float logits, ranks, and NLL across many texts.
