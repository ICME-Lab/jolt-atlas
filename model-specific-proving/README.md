# Proving Transformer inference

A Transformer model can be proven directly from its structure. Each layer
combines a small set of operations, including attention, an MLP, and
normalization. This crate implements proofs for those operations and defines
how to combine them for each supported model. The goal is to reduce the amount
of code needed to implement and maintain Transformer proofs.

## What the ONNX graph adds

ONNX represents many kinds of computation. An export can break a Transformer
block into numerous general-purpose tensor operations and include bookkeeping
for tensor shapes, batching, and layouts. A graph can calculate shapes with
ONNX operators and use `Concat` and `Slice` to assemble or select parts of
batched tensors. Much of this bookkeeping follows from public dimensions.
Removing it from a general graph requires the importer to trace how those
values are computed and used. Any of these operations that make it into the
proof flow require operator support that has to be written and maintained.

Rotary position embeddings (RoPE) illustrate the cost. The ONNX proof path
supports general `Sin` and `Cos` operations. For RoPE, their values are fixed
by the model configuration, position, and channel. The prover and verifier
can precompute a public table, index it directly, and check the rotation with
those coefficients. Recognizing this pattern in a general ONNX graph would
require a separate transformation.

Token embeddings have a similar cost. The ONNX proof path handles embedding
reads with the general `Gather` relation and commits to one-hot read
addresses. When both parties have the public embedding table and the token
IDs are public, they can read the same rows directly. This removes the
`Gather` proof and its read-address commitments.

## How the model is proven directly

This crate implements proofs for the following model components:

- **Attention** computes query, key, and value projections, applies the model's
  attention mask, and combines the values using the attention weights.
- **MLP** applies linear projections and an activation. Gated MLPs also
  multiply the activated values by a second projection.
- **Normalization** supports LayerNorm and RMSNorm.
- **Softmax** converts attention scores into weights.
- **Activation functions** include GELU and SiLU.
- **Residual addition** adds a block's output to its input.

Each supported model specifies in Rust how these components are connected and
repeated. Public tensor shapes follow from the model configuration and input
sizes. The crate reads model weights from safetensors.

Those components share three kinds of checks:

- **Sumcheck for arithmetic.** It checks matrix and elementwise products,
  normalization, and the arithmetic of fixed-point rescaling.
- **Public table lookups.** They check activation corrections for GELU and SiLU
  and the table values used by Softmax.
- **Bit and range checks.** They enforce Boolean and one-hot constraints and
  bound integer values such as fixed-point remainders.

The model's structure fixes where evaluation claims go. Each component returns
its claims as Rust values, and the model passes them directly to the proofs
that consume them. Rust checks the types and arity at each call. The ONNX path
routes claims through a runtime accumulator and checks for missing or unused
claims at runtime. The direct proof chains Sumchecks through their evaluation
claims. For each four-layer window, the prover batches the PCS opening
requests for witness data such as remainders and one-hot vectors.

## Implementation size

The crate has about 5,500 lines of Rust source excluding tests and vendored dependencies, including
model execution, model definitions, and proof generation and verification.
The existing Atlas codebase has about 84,000 lines of
Rust across its broader ONNX pipeline and supporting code. The direct approach
concentrates Transformer proof maintenance in the shared components. Adding a
model built from those components mainly requires defining how its layers fit
together.

## Proving with BitZ

This crate requires Apple Silicon and the Xcode Metal toolchain. It uses Metal
for the first two BitZ GKR leaf rounds on large groups. The
`safetensors-weights` feature reads `config.json` and a single
`model.safetensors` file containing F32, F16, or BF16 weights.

```sh
cargo run --release --manifest-path model-specific-proving/Cargo.toml \
  --features safetensors-weights --bin prove-model -- \
  models/gpt2-tiny 14 17 42
```

The CLI proves and returns the Top-8 candidates for up to the last four
positions of a decoder input. For an encoder, it returns the output tensor.
`Model::prove_top_k_to` and `Model::prove_to` write their proofs to an empty,
seekable file. Use `Model::verify_top_k_from` or `Model::verify_from` with the
same inputs and the public output. The prover stores large layer traces and
packed witness columns in temporary files. Both sides handle the BitZ proofs
one group at a time. Provision disk space for the temporary data and the proof.
