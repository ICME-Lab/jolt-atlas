# JOLT Atla## Benchmarks

We benchmarked a multi-classification model across different zkML projects:

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | \~0.7s  |                              |
| mina-zkml  | \~2.0s  |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | doesn't support gather op    |
| zk-torch   | N/A     | doesn't support reduceSum op |

We also benchmarked an MLP model:

| Project    | Latency | Notes                |
| ---------- | ------- | -------------------- |
| zkml-jolt  | \~800ms |                      |
| deep-prove | \~200ms | lacks MCC            |

## Background

Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups, on the other hand, eliminate the need for circuit representation entirely. Just One Lookup Table (JOLT) was designed from first principles to use only lookup arguments.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: ‘no quotient polynomials, no byte decomposition, no grand products, no permutation checks’, and most importantly — no complicated circuits.


## Benchmarks

We benchmarked a multi-classification model across different zkML projects:

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | \~0.7s  |                              |
| mina-zkml  | \~2.0s  |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | doesn’t support gather op    |
| zk-torch   | N/A     | doesn’t support reduceSum op |

### Running benchmarks

```
# enter zkml-jolt-core
cd zkml-jolt-core

# multi-class benchmark
cargo run -r -- profile --name multi-class --format chrome

# sentiment benchmark
cargo run -r -- profile --name sentiment --format chrome

# mlp benchmark
cargo run -r -- profile --name mlp --format chrome
```

When using `--format chrome`, the benchmark generates trace files (trace-<timestamp>.json) viewable in Chrome’s tracing tool:
1. Open Chrome and go to `chrome://tracing`.
2. Load the generated trace file to visualize performance.

Alternatively, use `--format default` to view performance times directly in the terminal.

Both models (preprocessing, proving, and verifying) take `~600ms–800ms`.


## Acknowledgments

Thanks to the Jolt team for their work. We are standing on the shoulders of giants.
