# JOLT Atlas

JOLT Atlas is a zero-knowledge machine learning (zkML) framework that extends the [JOLT](https://github.com/a16z/jolt) proving system to support ML inference verification from ONNX models. 

Made with ❤️ by [ICME Labs](https://blog.icme.io/).

<img width="983" height="394" alt="icme_labs" src="https://github.com/user-attachments/assets/ffc334ed-c301-4ce6-8ca3-a565328904fe" />

## Overview

JOLT Atlas enables practical zero-knowledge machine learning by leveraging Just One Lookup Table (JOLT) technology. Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups eliminate the need for circuit representation entirely.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: no quotient polynomials, no byte decomposition, no grand products, no permutation checks, and most importantly — no complicated circuits.

Our core ethos is to reduce commitment costs via sumcheck while committing only to small-value polynomials.

## Examples

Examples live in `jolt-atlas-core/examples/` and demonstrate end-to-end prove → verify flows for various ONNX models.

### GPT-2

GPT-2 proof and verification flow.

```bash
cargo run --release --package jolt-atlas-core --example gpt2
```

### nanoGPT

A ~0.25M-parameter GPT model (4 transformer layers). Loads the ONNX graph, generates a SNARK proof of inference, and verifies it.

```bash
cargo run --release --package jolt-atlas-core --example nanoGPT
```

### Transformer (self-attention)

Single self-attention block proof.

```bash
cargo run --release --package jolt-atlas-core --example transformer
```

### MiniGPT / MicroGPT

Smaller GPT variants useful for quick iteration and debugging.

```bash
cargo run --release --package jolt-atlas-core --example minigpt
cargo run --release --package jolt-atlas-core --example microgpt
```

## Benchmarks

**System specs:** MacBook Pro M3, 16GB RAM

### GPT-2 (125M params)

GPT-2 is a 125-million-parameter transformer model from OpenAI.

**JOLT Atlas**

| Stage | Wall clock |
| ----- | ---------- |
| Proving/verifying key generation (`setup_prover`) | 1.003 s |
| Witness + commitment phase (`ONNXProof::commit_witness_polynomials`) | 0.762 s |
| IOP proving (`ONNXProof::iop`) | 5.997 s |
| Reduction opening proof (excluding `HyperKZG::prove`) | 1.899 s |
| HyperKZG prove (`HyperKZG::prove`) | 2.392 s |
| Proof time (`ONNXProof::prove`) | 14.889 s |
| Verify time (`ONNXProof::verify`) | 1.038 s |
| **End-to-end total (`setup_prover` + prove + verify)** | **16.930 s** |

### nanoGPT (~0.25M params, 4 transformer layers)

nanoGPT is the standard workload we use for cross-project comparison. It is a ~250k-parameter GPT model with 4 transformer layers.

**JOLT Atlas**:

| Stage | Wall clock |
| ----- | ---------- |
| Verifying key generation (`setup_verifier`) | <0.001 s |
| Proving key generation (`setup_prover`) | 0.263 s |
| Proof time (`ONNXProof::prove`) | 2.288 s |
| Verify time (`ONNXProof::verify`) | 0.127 s |
| **End-to-end total (`setup_prover` + prove + verify)** | **2.678 s** |

**ezkl** on the same model ([source](https://blog.ezkl.xyz/post/nanogpt/)):

| Stage | Wall clock |
| ----- | ---------- |
| Verifying key generation | 192 s |
| Proving key generation | 212 s |
| Proof time | 237 s |
| Verify time | 0.34 s |

JOLT Atlas produces a proof for nanoGPT in **~2.29 s** versus ezkl's **~237 s proof time** (not counting their 400+ s of key generation). That is roughly a **104× speed-up** on proof generation alone.

### How to reproduce locally

```bash
# from repo root
cargo run --release --package jolt-atlas-core --example gpt2
```

Add `-- --trace` for Chrome Tracing JSON output (view in `chrome://tracing`), or `-- --trace-terminal` for timing printed to the terminal.

## Getting Started

### GPT-2 (first run)

GPT-2 uses a Hugging Face–hosted ONNX model that is **not** checked into the
repo. A helper script downloads and prepares it automatically.

1. Clone the repository.
2. Install Rust and Cargo.
3. Download the model:

```bash
# Create a virtual environment (one-time)
python3 -m venv .venv
source .venv/bin/activate

# Run the download script
python scripts/download_gpt2.py
```

This exports GPT-2 via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
into `atlas-onnx-tracer/models/gpt2/` and copies `model.onnx` → `network.onnx`.

4. Test the model (trace only, no proof):

```bash
cargo run --release --package atlas-onnx-tracer --example gpt2
```

You should see the model graph printed and an output shape like
`[1, 16, 65536]` (vocab size 50257 padded to the next power of two).

5. Prove & verify GPT-2:

```bash
cargo run --release --package jolt-atlas-core --example gpt2
```

A successful run prints `Proof verified successfully!`.

### Qwen3 0.6B trace and layer proving

Qwen3 0.6B uses Hugging Face `safetensors` weights that are not checked into
the repo. The Qwen3 flow has three stages:

1. Download the model and build the fixed-point `model.q8.bin` cache.
2. Run fixed-point generation and dump a trace.
3. Prove one or more transformer layers from that trace.

Run all commands from the repository root.

#### 1. Download the model

```bash
python3 scripts/download_qwen3_0_6b_safetensors.py
```

This downloads Qwen3 0.6B into:

```text
models/qwen3-0.6b/
```

By default it also builds:

```text
models/qwen3-0.6b/model.q8.bin
```

That q8 cache is used by generation and proving. To download only, without
building the q8 cache, pass `--no-q8-cache`.

#### 2. Generate tokens without a trace

Use this first as a quick smoke test:

```bash
cargo run --release -p qwen3-tracer --bin qwen3_generate -- \
  --seq-len 256 \
  --no-think \
  "Explain why zero-knowledge proofs are useful for scalable blockchains in three short paragraphs."
```

Generated text is streamed to stdout by default. Metadata and timings are
hidden by default. Add `--summary --timing` if you want the token counts and
timing breakdown.

#### 3. Generate a trace

Add `--trace PATH` to dump all layer trace tensors after generation:

```bash
cargo run --release -p qwen3-tracer --bin qwen3_generate -- \
  --seq-len 256 \
  --no-think \
  --trace traces/qwen3-0.6b/zkp_scalable_blockchains \
  "Explain why zero-knowledge proofs are useful for scalable blockchains in three short paragraphs."
```

The trace directory contains `manifest.jsonl` plus tensor files for each layer.
Trace output can be large. The default ignored location is:

```text
traces/qwen3-0.6b/
```

#### 4. Prove from the trace

Start with one layer:

```bash
cargo run --release -p qwen3-prover --bin qwen3_prove_trace -- \
  --trace traces/qwen3-0.6b/zkp_scalable_blockchains \
  --layer 0
```

Prove multiple layers in parallel:

```bash
cargo run --release -p qwen3-prover --bin qwen3_prove_trace -- \
  --trace traces/qwen3-0.6b/zkp_scalable_blockchains \
  --layers 0,1,2
```

Prove all layers:

```bash
cargo run --release -p qwen3-prover --bin qwen3_prove_trace -- \
  --trace traces/qwen3-0.6b/zkp_scalable_blockchains \
  --layers all
```

By default Rayon chooses the worker thread count, usually the number of logical
CPUs unless `RAYON_NUM_THREADS` is set. To pin it explicitly:

```bash
cargo run --release -p qwen3-prover --bin qwen3_prove_trace -- \
  --trace traces/qwen3-0.6b/zkp_scalable_blockchains \
  --layers 0,1,2 \
  --jobs 3
```

The prover currently reports success and timing to stdout. It does not yet
write proof objects to disk.

## Acknowledgments

Thanks to the Jolt team for their foundational work. We are standing on the shoulders of giants.
