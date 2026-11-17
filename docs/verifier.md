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
| `atlas-pcs` | `CommitmentScheme::{setup_verifier, verify_rlc}` and the `bitz` implementation: `bitz::scheme` (`Layout`, `opening_weights`, `verify_rlc`), `bitz::verifier`, `gkr::verify`, `ring_switch::reduce`, `ligerito::verifier`, `integers` (the prime draw), `params` (the soundness parameters), `merkle::verify_row`, `rs::Code`, `gf128`; BLAKE3 for the Merkle tree |
| `atlas-transcript` | `VerifyingTranscript` and its reads (`read_scalar`, `read_scalars`, `read_serializable`: the only way a value leaves the proof), the sponge and the spongefish verifier surface under it (`VerifierTranscript`, session setup, the proof-of-work check, the challenge sponges), and `words` |
| `atlas-algebra` | `field::fp128`, multilinear evaluation, `EqPolynomial`, opening points, `univariate` (`CompressedPoly` round messages, evaluation reduction's `UnivariatePoly`) |
| `atlas-lookup-tables` | table evaluation (`evaluate_mle`) and the prefix-suffix decompositions |
| `atlas-trace` | `Model` and its ops, `ModelExecutionIO`, `Tensor`, and `semantics` (the table generators). Not `Trace` or the executor. |
| `common::serialize` (`atlas_algebra::serialize`) | `Canonical` and its decoders: the encoding of a field element and of the framed values a proof carries (the commitments, the joint opening proof), decoded with validation and bounded allocation |
| `atlas-algebra::field` | the Solinas field (`Fp128`, its kernels and assembly), the `Ring`/`Field` spine, unreduced products |

The crate has no prover code. Three mechanisms hold that:

- `jolt-atlas-verifier/clippy.toml` disallows the prover items of the vendored and atlas
  protocol crates, and the execution trace's types (`atlas_trace::trace::{Trace,
  FusedIntermediates, NodeAux, SoftmaxWitness}`): the crate is **Trace-free**.
- The `verifier-isolation` CI job builds the crate alone, rejects `jolt-atlas-core` in its
  dependency tree, and greps its sources for `::prover` / `::witness` paths and for `Trace`.
- The prover side of a shared value is a function in `jolt-atlas-core`, matching on
  `OpRelKind` with no wildcard arm where it serves the op relations
  (`ops::{op_rel_witnesses, op_rel_tie_claims, op_rel_prover, op_rel_setup_claims}`,
  `lookup::{table_read_operand, table_read_prover}`), or an extension trait there
  (`EinsumFold`, `NodeEvalReductionProve`).

## Reading order

1. **Entry point**: `onnx_proof/mod.rs`. `verify` hands the proof's bytes to its transcript
   and reads them there, front to back: it binds the public inputs into the transcript
   (`append_inputs_to_transcript`), reads the commitments (`verifier::read_commitments`: the
   batch commitment, then one per polynomial the schedule opens), checks the output claim
   against the public outputs, runs the model's schedule (each node in reverse topological
   order, then the model-wide sumchecks), reduces the committed openings to one point and
   verifies the joint opening proof it reads there (`verifier::verify_reduced_openings`), and
   requires the bytes to end (`VerifyingTranscript::finish`).
2. **The schedule and its interpreter**: `onnx_proof/schedule.rs` (`Schedule`, `Step`,
   `ScheduleBuilder` and its checks, the closed relation enum `OpRelation` and the `OpPlan`
   trait each op implements), `onnx_proof/verifier.rs` (the `Verifier` state and
   `Verifier::run`, the one loop over the schedule's steps), `onnx_proof/preprocessing.rs`
   (where the schedule is compiled) and `onnx_proof/ops/mod.rs` (dispatch on
   `atlas_trace::Op`).
3. **Relation core**: `atlas_sumcheck::relation` and `onnx_proof/relation.rs`, then
   `atlas-claims` (the `SymbolicSumcheck` trait and the `Expr` algebra they evaluate) and
   `atlas_sumcheck::engine` (`verify_compressed`: the round checks, the grind and the
   transcript encoding every sumcheck runs through). Every sumcheck is a symbolic relation
   and `RelationVerifier` is derived from it. An op relation is a value, `relation::OpRel`:
   its kind (`OpRelKind`), node, rounds, degree, input and output expressions (`OpExpr`,
   integer coefficients read in the field), produced openings, where its incoming point
   comes from, the challenges its setup draws, the advice openings its setup appends, and
   `Custom`, four optional `fn` hooks (`points`, `derive`, `finish_setup`, `table`) for what
   is not the default. `op_rel_verifier` runs the setup and appends the setup opens;
   `verify_tie` checks a zero-round relation. Each op builds its relations with constructor
   functions in `onnx_proof/ops/<op>.rs` (`ops::mul::product`, `ops::add::accumulation`),
   and a constructor's doc states the S ⊆ R argument. `OpRelKind` is the prover's dispatch
   key; `OpRelKind::serves` declares the ops a kind may be built for.
4. **Lookup machinery**: `onnx_proof/lookup.rs`, `range_check.rs`, `range_bucket.rs`,
   `fused_rebase.rs`: the relations and model-only layouts of the lookup and range-check
   subprotocols. `lookup.rs` holds the two descriptions of a table read and the encoding of
   its addresses: `SmallTableDef`, a materialized small table read by the `small_table`
   relation (the clamped activations, the smooth ReLU correction, the trig ops);
   `TableRead`, a read of a prefix-suffix table of `atlas-lookup-tables`
   (`PrefixSuffixTable`), verified by `table_read_verifier`; and `RaEncoding`, the committed
   one-hot chunks and the openings the one-hot checks of a read start from. Every remainder
   bound (the fused rescales, ScalarConstDiv, Rsqrt, MeanOfSquares, the trig period
   reduction, softmax) is a range
   claim in a packed range bucket (`range_bucket::RangeClaim`): the remainder, or its slack
   below the bound, checked by the dense digit range check. A bound that is not positive has
   no proof. The output of an exact-arithmetic node (Add, Sub, Sum, Einsum, MeanOfSquares and
   the fused Mul, Square and Cube) that is not a model output is a range claim in the same
   buckets: `out + 2^31 ∈ [0, 2^32)`, read from the node's reduced output opening. A model
   output is a public `i32` tensor and has no claim.
5. **The proof**: `proof.rs`. `ONNXProof` is the NARG byte string of the prover's transcript,
   with no field, key or format of its own; its doc lists the order of the stream. Under it,
   `atlas_transcript::sponge` (`SpongeVerifier`'s reads) and
   `atlas_sumcheck::opening_proof` (`VerifierOpeningAccumulator`, where a claim is read when an
   append binds it).

## The schedule

What the verifier checks for a model, and in what order, is one `Schedule`
(`onnx_proof/schedule.rs`): a table of relations and a list of `Step`s over them. It is a
function of the model alone (`Schedule::compile(model)`), compiled once when the prover's or
the verifier's preprocessing is constructed and held there; `verify` does no planning.
`cargo run --release -p jolt-atlas-core --example prove -- --model <dir> --schedule` prints a
model's schedule, one line per step, and `testing/schedules.rs` pins a hash of that text for
the checked-in models.

The steps: `Reduce` is a node's evaluation reduction; `Open` appends advice openings at a
stated point (`OpOpen`); `Setup` runs a relation's setup (its draws and appends) and leaves it
to a later sumcheck; `Sumcheck` sets up each member not yet set up, in order, then verifies
the members as one sumcheck (one relation's sumcheck, or a batch, which draws a coefficient
per instance), reading its round messages and then its members' produced claims from the
transcript; `Tie` checks a zero-round relation, an `OpRel`,
with `verify_tie`. Steps hold values, never closures: a relation is an `OpRelation` (a closed
enum with one variant per kind of verifier construction: an `OpRel`, a `TableRead`, the
one-hot checks of an `RaEncoding`, GatherSmall's two one-hot stages, softmax's three-table
read, a range bucket's value and digit relations), an opening an `OpOpen`. Whatever depends on
earlier claims or on challenges is read and drawn in the relation's `setup`, when the
interpreter sets the relation up.

The schedule is each node's steps in reverse topological order, written by its op's
`OpPlan::plan(node, model, builder)` to a `ScheduleBuilder`, followed by the model-wide steps
`ScheduleBuilder::finish` adds: the packed range buckets' value sumchecks as one batched
sumcheck, each opening its bucket's packed value at a fresh point; the digit range checks of
their dense digits (`atlas_sumcheck::digit_range`) as a second; the lookups sumcheck, every
table read whose sumcheck produces only its own read-address claim; and the one-hot sumcheck,
every one-hot check of a read address. A table read is set up in its node's turn (`lookup`
emits a `Setup` there), where it draws its challenges and appends its operand claim; a
one-hot check reads the read-address claim a read produces, so it has no step in its node's
turn (`one_hot`) and is set up when the one-hot sumcheck runs. Gather's dictionary read is a
sumcheck in its node's turn, since it produces a claim on its dictionary operand; its one-hot
checks are in the one-hot sumcheck. `finish` returns `Err` unless every relation is set up at
most once, before its sumcheck, and verified exactly once, and no sumcheck is empty: a model
with no range claim has no bucket step, a model with no table read no lookups or one-hot
step, and its proof carries nothing for either.

`Verifier::run` in `onnx_proof/verifier.rs` is the interpreter: one loop over the steps,
keeping the verifiers of a relation set up by a `Setup` step until its `Sumcheck`. The
prover's `ops::prove_schedule` is the same loop with the relations' witnesses, so the
transcript order is the schedule's step order on both sides. One step reads the public IO: an
`Input` node's tie records the input's position, and the interpreter fills in its public
tensor from the IO it holds (the prover's walk from its trace) before checking the tie. The
polynomials a proof commits are the ones the schedule opens (`Schedule::committed`: every
`Open` step's and every relation's committed openings), so no committed polynomial goes
unopened.

The layout ops are ties, not sumchecks: MoveAxis, Broadcast, Identity, and also Reshape, Slice
and Concat, whose ties re-point the node's reduced output claim at the operands (Reshape at
the same point; Slice at the point with the slice's block bits fixed on its axis; Concat as
the sum of each operand's block indicator times the operand at the remaining coordinates).
Those three hold only on power-of-two dims with aligned blocks, which compiling with padding
establishes (`atlas_trace`'s `pad_graph`) and `node_view::check_model_shape` checks again on
the model the verifier is handed. A Slice's tie opens its operand at a point whose sliced-axis
coordinates are fixed bits, so the cells a Slice discards, when nothing else reads them, are
not checked: they are no part of the statement about the public output.

Softmax is a plan like any other: the row gaps as an `Open`, one sumcheck in its node's turn
that leaves its three table reads at one point, the row shift and the table address as an
`Open`, a table read and the clamp lookup set up there and proven in the lookups sumcheck, the
address link as a `Tie`, and the two one-hot checks in the one-hot sumcheck. Nothing is sent per row and the verifier does no
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
  committed polynomials is a function of the model alone: the ones its schedule opens
  (`Schedule::committed`), in sorted order.
- **One stream, read once.** The proof is the bytes the prover's transcript wrote, and the
  verifier has no access to them but its transcript's sequential reads: the commitments, then
  what each step of the schedule reads (an `Open`'s claims; a `Reduce`'s restriction
  polynomial when a node's output has several claims; a `Sumcheck`'s round messages, each
  followed by its proof-of-work nonce, then its members' produced claims; a `Tie`'s produced
  claims), then the opening reduction's sumcheck and reduced claims and the joint opening
  proof. There is no key to look a value up by, so a value cannot be missing, unread, read
  twice or stored where nothing asks: a missing value is a short read, a surplus one is
  trailing bytes, and `VerifyingTranscript::finish` ends `verify`. A model that opens no
  committed polynomial has no opening reduction and no opening proof in its stream.
- **Every read is absorbed.** A read is a spongefish prover message: the sponge absorbs it in
  the read itself, so no challenge drawn afterwards is independent of it. No `read_*` returns
  a value it did not absorb.
- **Sizes are the verifier's.** The number of commitments, of claims, of rounds, of
  coefficients in a round message (the relation's degree) and in an evaluation reduction's
  polynomial (its degree bound, `evaluation_reduction::h_coefficients`) come from the
  schedule. The one length the bytes state is that of a framed value (a commitment, the joint
  opening proof), checked against the bytes left before anything is copied. A field element
  is read only from its canonical encoding, and a frame must hold exactly one value.
- **A claim exists from its bind.** `VerifierOpeningAccumulator` holds a claim only from the
  append that reads it, at the point that append states. An id bound again at its point (an
  op whose operands are one node names one opening twice) reads nothing, since the stream
  carries each claim once; an id bound at another point is an `Err`.
- **The joint opening has its own transcript.** A commitment scheme draws its challenges
  while it builds its proof, so it cannot send the finished proof before them on one
  transcript. `verify_reduced_openings` seeds a second transcript from the proof's
  (`seed_opening_transcript`: two challenges drawn there, binding everything read so far),
  reads the joint opening proof as the last value of the stream, and runs
  `PCS::verify_rlc` on the seeded transcript. The scheme sends nothing through a transcript:
  what it sends is in its proof.
- **What holds it.** The mutation sweep takes its sites from the log of a recording
  transcript (`testing/recording.rs`), so every value read is a site, and requires an `Err`
  at each (no tolerance), at the proof cut at every read boundary and at the proof with a
  byte appended, with count and shape sites for the IO; `relation_kinds_all_exercised`;
  `fuzz_regressions_replay` and the cargo-fuzz target in `jolt-atlas-core/fuzz/`.

Outside the lints' reach, and model- or code-driven rather than proof-driven: `assert!`s on
model constants, `Math::log_2`, `Model::node` indexing in `atlas-trace`, and slicing inside
`atlas-sumcheck`'s subprotocols.

## The Trace-free inventory

What reads an `atlas_trace::Trace` for a verifier-crate item, and where it is in
`jolt-atlas-core`:

| Verifier-crate item | Its prover side in `jolt-atlas-core` |
|------|-------------|
| `relation::OpRel`, a sumcheck | `ops::op_rel_witnesses` (the engine's witnesses), `ops::op_rel_prover` (the specialized provers), `ops::op_rel_setup_claims`, each a match on `OpRelKind` calling into the op's module |
| `relation::OpRel`, a tie | `ops::op_rel_tie_claims`, a match on `OpRelKind` calling into the op's module; `relation::prove_tie` |
| `lookup::TableRead` | `lookup::{table_read_operand, table_read_prover}` |
| `lookup::small_table` | `lookup::{SmallTableProver, small_table_addresses, small_table_setup_claims}` |
| `range_bucket::RangeBucket` | `range_bucket::assemble_bits` |
| `EinsumLayout` | `ops/einsum/dot.rs` (`EinsumFold`, `Folded`), implemented in the three layout modules |
| One-hot address evaluations | `lookup::compute_ra_evals_from_usize_indices` |
| Lookup indices | `utils::compute_lookup_indices` |
| Trace access by node | `onnx_proof/trace_view.rs` (`TraceView`) |
| Prover self-checks | `onnx_proof/testing/probe.rs` (`probe_assert!`, `probe_assert_eq!`) |

The verifier crate depends on neither `rayon` nor `num`: it is single-threaded.
