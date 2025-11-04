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

We benchmarked a multi-classification model across different zkML projects:

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | ~0.7s   |                              |
| mina-zkml  | ~2.0s   |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | doesn't support gather op    |
| zk-torch   | N/A     | doesn't support reduceSum op |

We also benchmarked an MLP model:

| Project    | Latency | Notes                |
| ---------- | ------- | -------------------- |
| zkml-jolt  | ~800ms  |                      |
| deep-prove | ~200ms  | lacks MCC            |

### Running benchmarks

```bash
# enter zkml-jolt-core
cd zkml-jolt-core

# multi-class benchmark
cargo run -r -- profile --name multi-class --format chrome

# sentiment benchmark
cargo run -r -- profile --name sentiment --format chrome

# mlp benchmark
cargo run -r -- profile --name mlp --format chrome
```

When using `--format chrome`, the benchmark generates trace files (trace-<timestamp>.json) viewable in Chrome's tracing tool:
1. Open Chrome and go to `chrome://tracing`.
2. Load the generated trace file to visualize performance.

Alternatively, use `--format default` to view performance times directly in the terminal.

Both models (preprocessing, proving, and verifying) take ~600ms–800ms.

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