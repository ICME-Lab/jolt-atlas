# qwen3-layer-prover

This crate proves and verifies one independent Qwen3 decoder-layer transition.

Start here:

1. `src/layer/mod.rs`
   - module guide and public exports.
2. `src/layer/prover.rs`
   - complete proving boundary: witness, commitments, IOP, openings.
3. `src/layer/verifier.rs`
   - verification boundary matching `prover.rs`.
4. `src/layer/iop.rs`
   - hand-written layer equations and reverse claim flow. No PCS code here.
5. `src/layer/commitments.rs`
   - committed polynomial construction and transcript binding.
6. `src/layer/openings.rs`
   - adapter into `jolt_atlas_core::opening_reduction`.
7. `src/layer/witness.rs` and `src/layer/tensors.rs`
   - witness views and op parameter wiring details.

Experiment records are kept under `benches/`. They are historical data, not the
current code path.

## Command

```bash
cargo run --release -p qwen3-layer-prover --bin prove_trace_layer -- \
  --trace qwen3-awy/traces/fox_eos_full_awy \
  --model qwen3-awy/models/qwen3-0.6b/model.safetensors \
  --layer 0
```

## Layer Flow

```mermaid
flowchart LR
  hidden_in["hidden_in"] --> rms1["RMSNorm 1"]

  subgraph qkv["QKV projections"]
    direction TB
    q_proj["q_proj"]
    k_proj["k_proj"]
    v_proj["v_proj"]
  end

  rms1 --> q_proj
  rms1 --> k_proj
  rms1 --> v_proj

  subgraph attn["Attention"]
    direction LR

    q_proj --> q_norm["q_norm"]
    k_proj --> k_norm["k_norm"]

    q_norm --> q_rope_a["RoPE q claim 0"]
    q_norm --> q_rope_b["RoPE q claim 1"]

    k_norm --> k_rope_a["RoPE k claim 0"]
    k_norm --> k_rope_b["RoPE k claim 1"]

    q_rope_a --> qk_score["QK score"]
    q_rope_b --> qk_score
    k_rope_a --> qk_score
    k_rope_b --> qk_score

    qk_score --> softmax["softmax"]
    softmax --> pv["PV matmul"]
    v_proj --> pv

    pv --> o_proj["o_proj"]
  end

  hidden_in --> residual_attn["residual add 1"]
  o_proj --> residual_attn

  residual_attn --> rms2["RMSNorm 2"]

  subgraph mlp["MLP"]
    direction LR

    rms2 --> gate_proj["gate_proj"]
    rms2 --> up_proj["up_proj"]

    gate_proj --> silu["SiLU approx"]
    silu --> silu_mul["Hadamard mul"]
    up_proj --> silu_mul

    silu_mul --> down_proj["down_proj"]
  end

  residual_attn --> residual_mlp["residual add 2"]
  down_proj --> residual_mlp

  residual_mlp --> hidden_out["hidden_out"]
```
