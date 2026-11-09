# The verifier

`jolt-atlas-verifier` is the trust perimeter of jolt-atlas: a proof is sound iff this crate,
with the verifier items of the protocol crates under it, rejects every false statement. This
document says what an audit reads, in what order, and what a verdict depends on.

## Trust perimeter

A verdict of `jolt_atlas_verifier::onnx_proof::verify` depends on:

| Crate | Items a verdict depends on |
|-------|----------------------------|
| `jolt-atlas-verifier` | all of it |
| `atlas-sumcheck` | `relation` (`RelationVerifier`, `verify_relation{,_batch}`, `resolve`, `relation_input_claim`), `opening_proof` (`VerifierOpeningAccumulator`, the opening-reduction relation), `evaluation_reduction` (verify side), the one-hot, booleanity, shout and prefix-suffix relations and their parameters. Not the `prover` submodules. |
| `atlas-pcs` | `CommitmentScheme::{setup_verifier, verify, verify_rlc}` and the `akita` implementation of them |
| `atlas-transcript` | `VerifyingTranscript` and the sponge under it |
| `atlas-algebra` | `field::fp128`, multilinear evaluation, `EqPolynomial`, opening points |
| `atlas-lookup-tables` | table evaluation (`evaluate_mle`) and the prefix-suffix decompositions |
| `atlas-trace` | `Model` and its ops, `ModelExecutionIO`, `Tensor`, and `semantics` (the table generators). Not `Trace` or the executor. |
| `third_party/` | `jolt-sumcheck` (`verify_compressed`), `jolt-claims` (`SymbolicSumcheck`, `Expr`), `jolt-poly`, `jolt-field`, `jolt-transcript`, and LayerZero's `akita-*` crates by git |

The crate has no prover code. Three mechanisms hold that:

- `jolt-atlas-verifier/clippy.toml` disallows the prover items of the vendored and atlas
  protocol crates, and the execution trace's types (`atlas_trace::trace::{Trace,
  FusedIntermediates, NodeAux, SoftmaxWitness}`): the crate is **Trace-free**.
- The `verifier-isolation` CI job builds the crate alone, rejects `jolt-atlas-core` in its
  dependency tree, and greps its sources for `::prover` / `::witness` paths and for `Trace`.
- Prover methods on shared types are extension traits in `jolt-atlas-core`
  (`RelationWitnesses`, `TieClaims`, `LookupWitness`, `RangeCheckWitness`, `EinsumFold`,
  `OpLookupProve`, `RangeCheckProve`, `NodeEvalReductionProve`).

## Reading order

1. **Entry point**: `onnx_proof/mod.rs`. `verify` binds the public inputs into the transcript
   (`append_inputs_to_transcript`), loads the proof's claims and commitments, checks the output
   claim against the public outputs, runs the IOP over the nodes in reverse topological order,
   then verifies the reduced opening proof.
2. **Interpreter**: `onnx_proof/verifier.rs` (the `Verifier` state and the four steps above),
   `onnx_proof/ops/mod.rs` (dispatch on `atlas_trace::Op`, `OpLayout`, `VerifyOp`) and
   `onnx_proof/deferred_lookups.rs` (the checks batched after the node loop).
3. **Relation core**: `atlas_sumcheck::relation` and `onnx_proof/relation.rs`. Every sumcheck
   is a symbolic relation; `RelationVerifier` is derived from it, and `verify_tie` checks a
   zero-round one. Each op relation documents its S ⊆ R argument beside its definition in
   `onnx_proof/ops/<op>.rs`.
4. **Lookup machinery**: `onnx_proof/op_lookups/`, `range_checking/`, `clamp_lookups/`,
   `clamp_split.rs`, `global_clamp.rs`, `fused_rebase.rs`, `neural_teleport/`: the relations
   and model-only layouts of the lookup, clamp and range-check subprotocols.
5. **Proof object**: `proof.rs`, `types.rs` (`ProofId`, `ProofType`, `Claims`) and
   `proof_serialization.rs` (the wire format).

## What moved out (the Trace-free inventory)

Everything in the crate that took an `atlas_trace::Trace`, with its destination in
`jolt-atlas-core`:

| Item | Destination |
|------|-------------|
| `EngineRelation::witnesses` | `relation::RelationWitnesses`, implemented in `ops/{reshape, slice, concat, iff, scalar_const_div, mul, rsqrt, div}.rs` and `ops/sum/axis.rs` |
| `TieRelation::claims` | `relation::TieClaims`, implemented in `ops/{add, neg, is_nan, moveaxis, broadcast, constant, div}.rs` |
| `LookupOperandsTrait::{witness, lookup_bits}` | `op_lookups::LookupWitness`, implemented beside each helper's prover (`op_lookups`, `clamp_lookups`, `ops/clamp.rs`, `ops/activation_clamped`, `neural_teleport/trig_downscale.rs`, `ops/softmax_last_axis/significance_clamp.rs`) |
| `RangeCheckingOperandsTrait::build_lookup_operands`, `RangeCheckOperands::{build_lookup_operands, compute_lookup_indices}` | `range_checking::range_check_operands::{RangeCheckWitness, range_check_lookup_indices}` |
| `clamp_lookups::{clamp_intermediate, clamp_lookup_bits}` | `clamp_lookups` |
| `clamp_split::{NodeSplit, BucketSplit, value_chunk_indices, exact_output_prover}` | `clamp_split` |
| `ClampBucket::assemble_bits` | `global_clamp::assemble_bits` |
| `EinsumLayout::fold`, `Folded` | `ops/einsum/dot.rs` (`EinsumFold`, `Folded`), implemented in the three layout modules |
| `neural_teleport::division::compute_division` | `neural_teleport/division.rs` |
| `neural_teleport::utils::compute_ra_evals_{nbits_2comp, from_usize_indices}` | `neural_teleport/utils.rs` |
| `utils::{adjusted_remainder, compute_lookup_indices_from_operands}` | `utils` |
| `onnx_proof/trace_view.rs` (`TraceView`) | `onnx_proof/trace_view.rs` |
| `probe.rs`, `probe_assert!`, `probe_assert_eq!` | `onnx_proof/testing/probe.rs` |

Deleted with the move, having no reader left: the `precomputed` field of
`SaturatingAccClampOperands`, the `z` field of `SoftmaxSignificanceClampOperands`, the `scale`
fields of `RiRangeCheckOperands` / `RsRangeCheckOperands`, the `node_idx` field of
`MeanOfSquaresRangeCheckOperands`, and `utils::dims::transpose_flat_matrix`. The `rayon` and
`num` dependencies went with the moved code; the verifier is single-threaded.
