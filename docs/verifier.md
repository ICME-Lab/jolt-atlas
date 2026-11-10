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
2. **Interpreter**: `onnx_proof/verifier.rs` (the `Verifier` state, `run_plan` and
   `run_deferred`), `onnx_proof/plan.rs` (`Plan`, `Step`, the closed relation enums and every
   op's `OpPlan`), `onnx_proof/ops/mod.rs` (dispatch on `atlas_trace::Op`, `OpLayout`) and
   `onnx_proof/deferred_lookups.rs` (the post-loop plan and the deferred batches).
3. **Relation core**: `atlas_sumcheck::relation` and `onnx_proof/relation.rs`. Every sumcheck
   is a symbolic relation; `RelationVerifier` is derived from it, and `verify_tie` checks a
   zero-round one. Each op relation documents its S ⊆ R argument beside its definition in
   `onnx_proof/ops/<op>.rs`.
4. **Lookup machinery**: `onnx_proof/op_lookups/`, `range_checking/`, `clamp_lookups/`,
   `clamp_split.rs`, `global_clamp.rs`, `fused_rebase.rs`, `neural_teleport/`: the relations
   and model-only layouts of the lookup, clamp and range-check subprotocols.
5. **Proof object**: `proof.rs`, `types.rs` (`ProofId`, `ProofType`, `Claims`) and
   `proof_serialization.rs` (the wire format).

## Plans

Every op's verification is a `Plan` (`onnx_proof/plan.rs`): an ordered list of `Step`s over
relations, built by `OpPlan::plan(node, model, io)` before the transcript runs. `Reduce` is the
node's evaluation reduction; `Open` appends advice openings at a stated point (`OpOpen`);
`Relation` is a sumcheck stored under a `ProofType`; `Batch` several relations proven as one
batched sumcheck; `Tie` a zero-round identity or an in-the-clear check (`OpTie`); `Defer`
queues a relation for one of the post-loop batches (`DeferredBatch`, in
`deferred_lookups.rs`). Steps hold relations (the closed enums `OpRelation`, `OpTie`, `OpOpen`),
never closures; whatever depends on earlier claims or on challenges is read and drawn in the
relation's `setup` when the interpreter reaches the step (a one-hot batch's relations are set up
when the batch runs, since they read the `ra` claim the parent batch produces;
`DeferredBatch::setup` says which). `Verifier::run_plan` in `onnx_proof/verifier.rs` is the
interpreter; the prover's `ops::prove_plan` walks the same plan with the relations' witnesses,
so an op's transcript order is its plan's step order on both sides.

After the node loop, `Verifier::run_deferred` runs the post-loop plan
(`deferred_lookups::plan`, built from the model and the public IO: the packed clamp buckets'
exact flags, their split or exact value sumchecks, the digit range checks of their dense `OUT`
digits (`atlas_sumcheck::digit_range`) and the slack chunk checks of the buckets not declared
exact, then the remainder buckets' value sumchecks and the digit range checks of their dense
digits; a model output node whose
public output is unsaturated ties `acc = out` in its own turn and is a gap in its bucket, and a
clamp bucket made only of such nodes is not committed at all), then drains the one deferral
queue, batch by batch in `DeferredBatch` order. Nothing is registered during the loop and
nothing reconciled afterwards. A plan batch whose relations set up no sumcheck instance (the
slack batch when every bucket is exact) carries no proof entry.

Softmax is the stress case: its stage driver is a plan (aux vectors as `Open`, stages 1–2 as
`Batch`, stage 3 and its one-hot checks as `Defer`); the relations of a stage read the aux
vectors back from their openings in their setup (`SoftmaxShape::aux`).

## Proof-shape rules

- **No panic.** The crate denies `clippy::{unwrap_used, expect_used, panic, unreachable,
  unimplemented, todo, indexing_slicing}` at its root, with no `#[allow]` outside tests. The
  accumulator getters and appends, `SetupRelation::setup`, and `Relation::{opening_points,
  derive, table}` return `Result`; an expression is evaluated only after every opening and
  derived value it names has been resolved (`atlas_sumcheck::relation::evaluate_expression`).
  Prover code `expect`s the same calls: a missing claim there is a prover bug.
- **Checked inputs.** `verify` first checks that every node's operands are earlier nodes
  (`node_view::check_model_shape`) and that the public IO names the model's inputs and
  outputs with one tensor of the node's dims each (`verifier::check_io_shape`). The list of
  committed polynomials is a function of the model and the public IO
  (`get_models_committed_polynomials`).
- **Exact consumption.** Sumcheck proofs and evaluation-reduction proofs are read through
  `ProofLedger::take`, which fails on a missing entry and on a second take.
  `Verifier::check_consumed` ends `verify`: an entry never taken, or an opening claim no append
  bound to a point, is an `Err`.
- **No dead fields.** A node with a single claim on its output has no evaluation-reduction
  entry (the claim is the reduced instance). Every `ProofType` is emitted for some checked-in
  model; deleted tags (1, 7, 8, 11, 13, 17) do not deserialize.
- **What holds it.** The mutation sweep requires an `Err` at every site (no tolerance), with
  orphan sites for claims, sumcheck proofs and reduction proofs and count/shape sites for the
  IO; `proof_types_all_exercised`; `fuzz_regressions_replay` and the cargo-fuzz target in
  `jolt-atlas-core/fuzz/`.

Outside the lints' reach, and model- or code-driven rather than proof-driven: `assert!`s on
model constants, `Math::log_2`, `Model::node` indexing in `atlas-trace`, and slicing inside
`atlas-sumcheck`'s subprotocols.

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
| `clamp_split::{NodeSplit, BucketSplit, value_chunk_indices, exact_output_prover}` | `clamp_split` (`BucketSplit` built once per proof in `global_clamp::BucketWitnesses`) |
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
