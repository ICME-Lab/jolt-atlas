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

### nanoGPT (~0.25M params, 4 transformer layers)

nanoGPT is the standard workload we use for cross-project comparison. It is a ~250k-parameter GPT model with 4 transformer layers.

**JOLT Atlas** end-to-end proving breakdown:

| Stage | Wall clock |
| ----- | ---------- |
| Verifying key generation | 0.246 s |
| Proving key generation | 0.246 s |
| Proof time | 14 s |
| Verify time | 0.517 s |

**ezkl** on the same model ([source](https://blog.ezkl.xyz/post/nanogpt/)):

| Stage | Wall clock |
| ----- | ---------- |
| Verifying key generation | 192 s |
| Proving key generation | 212 s |
| Proof time | 237 s |
| Verify time | 0.34 s |

JOLT Atlas produces a proof for nanoGPT in **~14 s** versus ezkl's **~237 s proof time** (not counting their 400+ s of key generation). That is roughly a **17× speed-up** on proof generation alone.

### GPT-2 (125M params)

GPT-2 is a 125-million-parameter transformer model from OpenAI.

**JOLT Atlas** end-to-end proving breakdown:

| Stage | Wall clock |
| ----- | ---------- |
| Proving/verifying key generation | 0.872 s |
| Witness generation | ~7.5 s |
| Commitment time | ~3.5 s |
| Sum-check proving | ~16 s |
| Reduction opening proof | ~7 s |
| HyperKZG prove | ~3 s |
| **End-to-end total** | **~38 s** |

### How to reproduce locally

```bash
# from repo root
cargo run --release --package jolt-atlas-core --example nanoGPT
```

Add `-- --trace` for Chrome Tracing JSON output (view in `chrome://tracing`), or `-- --trace-terminal` for timing printed to the terminal.

## Getting Started

1. Clone the repository
2. Install Rust and Cargo
3. Run an example:
   ```bash
   cargo run --release --package jolt-atlas-core --example nanoGPT
   ```

### GPT-2

GPT-2 uses a Hugging Face–hosted ONNX model that is **not** checked into the
repo. A helper script downloads and prepares it automatically.

#### 1. Download the model

```bash
# Create a virtual environment (one-time)
python3 -m venv .venv
source .venv/bin/activate

# Run the download script
python scripts/download_gpt2.py
```

This exports GPT-2 via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
into `atlas-onnx-tracer/models/gpt2/` and copies `model.onnx` → `network.onnx`.

#### 2. Test the model (trace only, no proof)

```bash
cargo run --release --package atlas-onnx-tracer --example gpt2
```

You should see the model graph printed and an output shape like
`[1, 16, 65536]` (vocab size 50257 padded to the next power of two).

#### 3. Prove & verify GPT-2

```bash
cargo run --release --package jolt-atlas-core --example gpt2
```

A successful run prints `Proof verified successfully!`.

## Acknowledgments

Thanks to the Jolt team for their foundational work. We are standing on the shoulders of giants.