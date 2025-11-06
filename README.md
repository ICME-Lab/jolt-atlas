# JOLT Atlas

JOLT Atlas is a zero-knowledge machine learning (zkML) framework that extends the [JOLT](https://github.com/a16z/jolt) proving system to support ML inference verification from ONNX models. 

Made with ❤️ by [ICME Labs](https://blog.icme.io/).

<img width="983" height="394" alt="icme_labs" src="https://github.com/user-attachments/assets/ffc334ed-c301-4ce6-8ca3-a565328904fe" />

## Overview

JOLT Atlas enables practical zero-knowledge machine learning by leveraging Just One Lookup Table (JOLT) technology. Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups eliminate the need for circuit representation entirely.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: no quotient polynomials, no byte decomposition, no grand products, no permutation checks, and most importantly — no complicated circuits.

## Examples

The `examples/` directory contains practical demonstrations of zkML models:

### Article Classification

A text classification model that categorizes articles into business, tech, sport, entertainment, and politics.

```bash
cargo run --release --example article_classification
```

This example:
- Tests model accuracy on sample texts
- Generates a SNARK proof for one classification
- Verifies the proof cryptographically

### Transaction Authorization

A financial transaction authorization model that decides whether to approve or deny transactions based on features like budget, trust score, amount, etc.

```bash
cargo run --release --example authorization
```

This example:
- Tests the model on various transaction scenarios
- Shows authorization decisions with confidence scores
- Generates and verifies a SNARK proof for one transaction

## Benchmarks

### Transformer (self-attention) profile

Latest run (`cargo run -r -- profile --name self-attention --format default`):

| Stage  | Wall clock |
| ------ | ----------- |
| Prove  | 20.8 s |
| Verify | 143 ms |
| End-to-end CLI run | 25.8 s |

The prover hit a peak allocated footprint of roughly 5.6 GB during sumcheck round 10, which matches what we have seen in the integration test harness. Numbers were collected from this workstation; expect ±10% variance depending on CPU, memory bandwidth.

### Cross-project snapshot

Article-classification workload comparison

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | ~0.7s   | in-tree article-classification bench |
| mina-zkml  | ~2.0s   |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | missing gather primitive     |
| zk-torch   | N/A     | missing reduceSum primitive  |

Perceptron MLP baseline (easy sanity workload):

| Project    | Latency | Notes                |
| ---------- | ------- | -------------------- |
| zkml-jolt  | ~800ms  |                      |
| deep-prove | ~200ms  | lacks MCC            |

### How to reproduce locally

```bash
# from repo root
cd zkml-jolt-core

cargo run -r -- profile --name article-classification --format default
cargo run -r -- profile --name self-attention --format default
cargo run -r -- profile --name mlp --format default
```

Add `--format chrome` if you want a tracing JSON for Chrome's `chrome://tracing` viewer instead of plain-text timings.

## Getting Started

1. Clone the repository
2. Install Rust and Cargo
3. Run the examples:
   ```bash
   cargo run --example article_classification
   cargo run --example authorization
   ```

## Acknowledgments

Thanks to the Jolt team for their foundational work. We are standing on the shoulders of giants.