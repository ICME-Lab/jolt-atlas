# The verifier

`jolt-atlas-verifier` is the trust perimeter of jolt-atlas: a proof is sound iff this crate,
with the verifier items of the protocol crates under it, rejects every false statement. This
document says what an audit reads, in what order, and what a verdict depends on.

## Trust perimeter

A verdict of `jolt_atlas_verifier::onnx_proof::verify` depends on:

| Crate | Items a verdict depends on |
|-------|----------------------------|
| `jolt-atlas-verifier` | all of it |
| `atlas-sumcheck` | `engine` (`verify_compressed`, `absorb_input_claims`, `BatchPrelude`), `relation` (`RelationVerifier`, `verify_relation{,_batch}`, `resolve`, `relation_input_claim`), `opening_proof` (`VerifierOpeningAccumulator`, the opening-reduction relation), `evaluation_reduction` (verify side), the one-hot, booleanity, shout and prefix-suffix relations and their parameters. Not the `prover` submodules. |
| `atlas-claims` | all of it: `SymbolicSumcheck`, the `Expr` algebra and its evaluation, the claim-data traits |
| `atlas-pcs` | `CommitmentScheme::{setup_verifier, verify, verify_rlc}` and the `akita` implementation of them, including `akita::backend` (the adapter over `akita-*`) and `openings` |
| `atlas-transcript` | `VerifyingTranscript`, the sponge and the spongefish verifier surface under it (`VerifierTranscript`, session setup, the proof-of-work check, the challenge sponges), and `words` |
| `atlas-algebra` | `field::fp128`, multilinear evaluation, `EqPolynomial`, opening points, `univariate` (`CompressedPoly` round messages, evaluation reduction's `UnivariatePoly`) |
| `atlas-lookup-tables` | table evaluation (`evaluate_mle`) and the prefix-suffix decompositions |
| `atlas-trace` | `Model` and its ops, `ModelExecutionIO`, `Tensor`, and `semantics` (the table generators). Not `Trace` or the executor. |
| `common::serialize` (`atlas_algebra::serialize`) | `Canonical` and its decoders: the proof's byte format, decoded with validation and bounded allocation |
| `atlas-algebra::field` | the Solinas field (`Fp128`, its kernels and assembly), the `Ring`/`Field` spine, unreduced products |
| `jolt-field`, `jolt-poly` (root) | the `atlas-algebra` re-exports `akita-*` binds; no code |
| LayerZero's `akita-*` crates | by git at `dfdbd767` (`docs/upstream-inventory.md`) |

The crate has no prover code. Three mechanisms hold that:

- `jolt-atlas-verifier/clippy.toml` disallows the prover items of the vendored and atlas
  protocol crates, and the execution trace's types (`atlas_trace::trace::{Trace,
  FusedIntermediates, NodeAux, SoftmaxWitness}`): the crate is **Trace-free**.
- The `verifier-isolation` CI job builds the crate alone, rejects `jolt-atlas-core` in its
  dependency tree, and greps its sources for `::prover` / `::witness` paths and for `Trace`.
- Prover methods on shared types are extension traits in `jolt-atlas-core`
  (`RelationWitnesses`, `TieClaims`, `LookupWitness`, `EinsumFold`, `OpLookupProve`,
  `NodeEvalReductionProve`).

## Reading order

1. **Entry point**: `onnx_proof/mod.rs`. `verify` binds the public inputs into the transcript
   (`append_inputs_to_transcript`), loads the proof's claims and commitments, checks the output
   claim against the public outputs, runs the IOP over the nodes in reverse topological order,
   then verifies the reduced opening proof.
2. **Interpreter**: `onnx_proof/verifier.rs` (the `Verifier` state, `run_plan` and
   `run_deferred`), `onnx_proof/plan.rs` (`Plan`, `Step`, the closed relation enums and every
   op's `OpPlan`), `onnx_proof/ops/mod.rs` (dispatch on `atlas_trace::Op`, `OpLayout`) and
   `onnx_proof/deferred_lookups.rs` (the post-loop plan and the deferred batches).
3. **Relation core**: `atlas_sumcheck::relation` and `onnx_proof/relation.rs`, then
   `atlas-claims` (the `SymbolicSumcheck` trait and the `Expr` algebra they evaluate) and
   `atlas_sumcheck::engine` (`verify_compressed`: the round checks, the grind and the
   transcript encoding every sumcheck runs through). Every
   sumcheck is a symbolic relation; `RelationVerifier` is derived from it, and `verify_tie` checks a
   zero-round one. Each op relation documents its S ⊆ R argument beside its definition in
   `onnx_proof/ops/<op>.rs`.
4. **Lookup machinery**: `onnx_proof/op_lookups/`, `range_check.rs`,
   `range_bucket.rs`, `fused_rebase.rs`, `neural_teleport/`: the relations and model-only
   layouts of the lookup and range-check subprotocols. Every remainder bound (the fused
   rescales, Div, ScalarConstDiv, Rsqrt, MeanOfSquares, the trig teleport, softmax) is a range
   claim in a packed range bucket (`range_bucket::RangeClaim`): the remainder, or its slack
   below the bound, checked by the dense digit range check. A bound that is not positive has
   no proof. The output of an exact-arithmetic node (Add, Sub, Sum, Einsum, MeanOfSquares and
   the fused Mul, Square and Cube) that is not a model output is a range claim in the same
   buckets: `out + 2^31 ∈ [0, 2^32)`, read from the node's reduced output opening. A model
   output is a public `i32` tensor and has no claim.
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
relation's `setup` when the interpreter reaches the step (a relation queued for `OneHot` is set
up when that batch runs, since it reads the `ra` claim a lookup produces; `DeferredBatch::setup`
says which). `Verifier::run_plan` in `onnx_proof/verifier.rs` is the
interpreter; the prover's `ops::prove_plan` walks the same plan with the relations' witnesses,
so an op's transcript order is its plan's step order on both sides.

After the node loop, `Verifier::run_deferred` runs the post-loop plan
(`deferred_lookups::plan`, built from the model alone: the packed range buckets' value
sumchecks as one batched sumcheck, each opening its bucket's packed value at a fresh point,
then the digit range checks of their dense digits (`atlas_sumcheck::digit_range`) as a second),
then drains the one deferral queue in two batches: `Lookups`, every table read whose sumcheck
produces only its own read-address claim (set up in its node's turn), then `OneHot`, every
one-hot check of a read address. Gather's dictionary read stays in its node's turn, since it
produces a claim on its dictionary operand; its one-hot checks are in `OneHot`. A model with
no range claim has no bucket entries, and a deferred batch with no sumcheck instance (a model
with no lookup) carries no proof entry.

The layout ops are ties, not sumchecks: MoveAxis, Broadcast, Identity, and also Reshape, Slice
and Concat, whose ties re-point the node's reduced output claim at the operands (Reshape at
the same point; Slice at the point with the slice's block bits fixed on its axis; Concat as
the sum of each operand's block indicator times the operand at the remaining coordinates).
Those three hold only on power-of-two dims with aligned blocks, which compiling with padding
establishes (`atlas_trace`'s `pad_graph`) and `node_view::check_model_shape` checks again on
the model the verifier is handed. A Slice's tie opens its operand at a point whose sliced-axis
coordinates are fixed bits, so the cells a Slice discards, when nothing else reads them, are
not checked: they are no part of the statement about the public output.

Softmax is a plan like any other: the row gaps as an `Open`, one in-loop sumcheck that leaves
its three table reads at one point, the row shift and the table address as an `Open`, a table
read and the clamp lookup as `Defer`s to `Lookups`, the address link as a `Tie`, and the two
one-hot checks as `Defer`s to `OneHot`. Nothing is sent per row and the verifier does no
per-row work; the tables are generated once per process.

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
  committed polynomials is a function of the model alone
  (`get_models_committed_polynomials`).
- **Exact consumption.** Sumcheck proofs and evaluation-reduction proofs are read through
  `ProofLedger::take`, which fails on a missing entry and on a second take.
  `Verifier::check_consumed` ends `verify`: an entry never taken, or an opening claim no append
  bound to a point, is an `Err`.
- **No dead fields.** A node with a single claim on its output has no evaluation-reduction
  entry (the claim is the reduced instance). Every `ProofType` is emitted for some checked-in
  model; deleted tags (1 to 8, 11, 13 to 15, 17 to 26) do not deserialize.
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
| `EngineRelation::witnesses` | `relation::RelationWitnesses`, implemented in `ops/{iff, scalar_const_div, mul, rsqrt, div}.rs` and `ops/sum/axis.rs` |
| `TieRelation::claims` | `relation::TieClaims`, implemented in `ops/{add, neg, is_nan, moveaxis, broadcast, constant, div}.rs` |
| `LookupOperandsTrait::{witness, lookup_bits}` | `op_lookups::LookupWitness`, implemented beside each helper's prover (`op_lookups`, `ops/clamp.rs`, `ops/activation_clamped`, `neural_teleport/trig_downscale.rs`, `ops/softmax_last_axis/significance_clamp.rs`) |
| `RangeBucket::assemble_bits` | `range_bucket::assemble_bits` |
| `EinsumLayout::fold`, `Folded` | `ops/einsum/dot.rs` (`EinsumFold`, `Folded`), implemented in the three layout modules |
| `neural_teleport::utils::compute_ra_evals_{nbits_2comp, from_usize_indices}` | `neural_teleport/utils.rs` |
| `utils::compute_lookup_indices` | `utils` |
| `onnx_proof/trace_view.rs` (`TraceView`) | `onnx_proof/trace_view.rs` |
| `probe.rs`, `probe_assert!`, `probe_assert_eq!` | `onnx_proof/testing/probe.rs` |

Deleted with the move, having no reader left: the `z` field of
`SoftmaxSignificanceClampOperands` and `utils::dims::transpose_flat_matrix`. The `rayon` and
`num` dependencies went with the moved code; the verifier is single-threaded.
