# JOLT Atlas

JOLT Atlas is a zero-knowledge machine learning (zkML) framework that extends the [JOLT](https://github.com/a16z/jolt) proving system to support ML inference verification from ONNX models. 

Made with ❤️ by [ICME Labs](https://blog.icme.io/).

<img width="983" height="394" alt="icme_labs" src="https://github.com/user-attachments/assets/ffc334ed-c301-4ce6-8ca3-a565328904fe" />

## Overview

JOLT Atlas enables practical zero-knowledge machine learning by leveraging Just One Lookup Table (JOLT) technology. Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups eliminate the need for circuit representation entirely.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: no quotient polynomials, no byte decomposition, no grand products, no permutation checks, and most importantly — no complicated circuits.

Our core ethos is to reduce commitment costs via sumcheck while committing only to small-value polynomials.

## Examples

One example, `prove`, proves and verifies any model in the zoo and prints
timings and proof size:

```bash
# Checked-in stand-ins (no download needed)
cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen2-tiny
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2-tiny --pcs dory

# Real models (see "Getting Started" for the downloads)
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory
cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen --pcs dory \
    --input "The quick brown fox jumps over the lazy dog" --use-cache
```

Options: `--pcs hyperkzg|dory` (default HyperKZG), `--seq-len N` (default 16),
`--input "<text>"` (tokenizes with the model's `tokenizer.json`), `--srs-vars N`
(over-provision the SRS; Dory), `--use-cache` (reuse the shared preprocessing
across runs), `--stats` (clamp-width histogram and packed buckets),
`--trace` (Chrome Tracing JSON, view in `chrome://tracing`) or
`--trace-terminal` (timings on the terminal).

### The model zoo (`models/`)

| Model | Source | In CI |
| ----- | ------ | ----- |
| `op_gauntlet` | `scripts/gen_test_models.py`: one synthetic block covering the lowerable op surface | every run |
| `gpt2-tiny`, `qwen2-tiny`, `bge-tiny` | `scripts/gen_tiny_models.py`: random-init GPT-2 / Qwen2 / BERT with the production architectures (GQA ratio 7, RoPE, native LayerNormalization, fused QKV split, Erf GELU, …), exported with the same `optimum` flags as the real models, under 1 MB each | every run |
| `reject_slice`, `reject_concat` | `scripts/gen_test_models.py`: fixtures the importer must reject | every run |
| `gpt2`, `bge-small-en-v1.5`, `qwen` | `scripts/download_*.py` (Hugging Face Optimum exports; gitignored) | main, nightly, PRs labelled `ci:heavy` |

Regenerating any fixture changes its weights (and, with a different exporter,
possibly its graph), so the pinned forward-pass hashes in
`atlas-onnx-import/tests/golden_zoo.rs` must be re-pinned afterwards. Use the
pinned toolchain:

```bash
python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements-models.txt
.venv/bin/python scripts/gen_tiny_models.py          # all three, or name one
cargo test -p atlas-onnx-import --test golden_zoo    # prints the new hashes on mismatch
```

## Benchmarks

**System specs:** MacBook Pro M3, 16GB RAM

### GPT-2 (125M params)

GPT-2 is a 125-million-parameter transformer model from OpenAI.

**JOLT Atlas** (Dory PCS)

| Stage | Wall clock |
| ----- | ---------- |
| Proving/verifying key generation (`DoryScheme::setup_prover`, cached URS) | 0.843 s |
| Witness + commitment phase (`ONNXProof::commit_witness_polynomials`) | 10.0 s |
| IOP proving (`ONNXProof::iop`) | 8.47 s |
| Opening reduction sumcheck + Dory opening (`ONNXProof::prove_reduced_openings`) | 7.83 s |
| Proof time (`ONNXProof::prove`) | 29.3 s |
| Verify time (`ONNXProof::verify`) | 3.43 s |
| **End-to-end total (`setup_prover` + prove + verify)** | **33.6 s** |

Stage rows come from one `--trace-terminal` run, which adds tracing overhead. Without
tracing, the median of 3 runs is 27.6 s to prove and 3.4 s to verify. Dory is the
benchmark backend: the per-stage performance gate in `specs/adoption-program.md` measures
this command.

### How to reproduce locally

```bash
# from repo root
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory
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

# Run the download script (installs the pinned export toolchain)
python scripts/download_gpt2.py
```

This exports GPT-2 via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
into `models/gpt2/` and renames `model.onnx` → `network.onnx`. The same
pattern gets `scripts/download_bge_small_en_v1_5.py` and
`scripts/download_qwen.py` (2.5 GB).

4. Inspect the imported model (no proof):

```bash
cargo run --release --package atlas-onnx-import --example dump_lowered_ops -- \
    models/gpt2/network.onnx batch_size=1 sequence_length=16 past_sequence_length=0
```

You should see the lowered op inventory and the canonical einsum equations.

5. Prove & verify GPT-2:

```bash
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2
```

A successful run prints `Proof verified successfully!`.

## Testing

```bash
cargo nextest run --workspace                       # everything CI runs on a PR (~2 min)
cargo nextest run -p jolt-atlas-core -E 'test(op_surface)'   # the exhaustive single-op table
cargo nextest run -p jolt-atlas-core -E 'test(soundness)'    # mutation / tamper / forgery sweeps
cargo nextest run -p jolt-atlas-core -E 'test(zoo)'          # end-to-end models
cargo nextest run -p jolt-atlas-core --run-ignored ignored-only -E 'test(known_gap)'  # pinned gaps
```

`jolt-atlas-core/src/onnx_proof/testing/` holds the whole suite:

* `op_surface.rs` is a table with one case per `atlas_trace::Op` variant (plus
  saturation, tiny-tensor, non-power-of-two, padded and rejection cases).
  Adding an op variant fails to compile until the table declares a stance for it.
* `soundness.rs` attacks the proof system generically: every field of a
  finished proof is perturbed (`proof_mutation_sweep`), the execution trace is
  tampered before proving (`witness_tamper_sweep`), and every opening claim is
  forged at the accumulator (`opening_forgery_sweep`). Set
  `JOLT_SOUNDNESS_EXHAUSTIVE=1` for every site instead of the sampled default
  (CI runs that nightly).
* `zoo.rs` proves the checked-in models on every run and the real downloads
  behind `#[ignore]` (the heavy CI job).
* Tests named `known_gap_*` are `#[ignore]`d pins of open issues; each fails
  today for the reason in its `#[ignore]` message and turns green when fixed.

## Acknowledgments

Thanks to the Jolt team for their foundational work. We are standing on the shoulders of giants.