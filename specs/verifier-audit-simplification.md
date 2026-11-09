# Spec: Verifier Audit Simplification — Trace-Free Crate, Exact Proof Shape, One Verification Plan

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-28                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/adoption-program.md` (stages 1, 5, 6, 8, 9 and 10 as landed on `dev`: the verifier crate, every sumcheck a relation, the legacy drivers retired) |
| Supersedes  | — |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`jolt-atlas-verifier` is the trust perimeter: a proof is sound iff this crate (with the verifier
items of the protocol crates under it) rejects every false statement. An auditor cannot yet read
it as one thing. First, it carries the prover's witness construction for every relation and
lookup — everything that takes an `atlas_trace::Trace` — so the crate boundary that the adoption
program made physical does not tell prover code from verifier code: a reader of the clamp split
or a lookup module meets the one-hot chunk assembly before the identity it checks. Second, the
control flow has four shapes: relation ops bound by macro, hand-written `VerifyOp` impls under
two reduction flows, three deferral queues reconciled by a hand-ordered post-loop, and softmax's
stage driver; every op module reaches into the transcript, the accumulator and the proof map
itself, so "what does the verifier check for this op, in what order" is answered by reading each
op. Third, a malformed proof can crash the verifier instead of being rejected: a missing claim or
proof entry or an IO tensor of the wrong shape panics inside the accumulator or an index, the
mutation sweep tolerates those panics, and the proof format carries fields the verifier never
reads and tags no prover emits. This spec removes all three in three stages: prover code leaves
the crate; the verifier rejects rather than panics and consumes the proof exactly; every op's
verification is a plan — data the one interpreter runs and the prover follows — so the audit
reads one interpreter, the relations, and per-op plans the size of their relation lists.

## Intent

### Goal

Make `jolt-atlas-verifier` auditable as a single artifact: a crate with no prover code (no
`Trace` in scope, enforced by lint), one interpreter in `onnx_proof/verifier.rs` that is the
only code touching the proof, the transcript and the opening accumulator, per-op `Plan`s
(ordered `Step`s over relations, in `onnx_proof/plan.rs`) that both the verifier and
`jolt-atlas-core`'s prover interpret, a verifier that returns `Err` on every malformed proof or
IO and never panics, and a proof format the verifier consumes exactly — every entry read, no
field dead.

### Invariants

Each invariant names the mechanism that holds it.

- **The verifier crate is Trace-free.** No item in `jolt-atlas-verifier` names
  `atlas_trace::Trace`, `FusedIntermediates`, `NodeAux` or `SoftmaxWitness`; witness
  construction lives in `jolt-atlas-core`. Mechanism: `jolt-atlas-verifier/clippy.toml`
  `disallowed-types` lists the four types (the workspace clippy job runs with `-D warnings`),
  and the `verifier-isolation` CI job's path grep gains `trace::Trace` as the backstop. From
  stage 1 on.
- **Verification happens in one place.** `Verifier`'s accumulator, transcript, proof map and
  deferral queue are private to `onnx_proof/verifier.rs`; op modules return `Plan`s and cannot
  read a proof entry, draw a challenge or append an opening. Mechanism: Rust privacy — the
  fields are private and the module exposes no mutable accessor (compiler). From stage 3 on.
- **Prover and verifier run the same plan.** An op has one `OpPlan::plan(node, model, io)`;
  core's prover walks its steps with the witness traits, the verifier walks them with the
  relation verifiers. No hand-written `ProveOp`/`VerifyOp` sequence exists (the traits are
  deleted — compiler). The transcript order of an op is therefore the plan's step order on
  both sides by construction. Mechanism: compiler plus the digest harness. From stage 3 on.
- **The verifier never panics on any proof or IO.** Mechanisms, all from stage 2 on:
  `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unreachable,
  clippy::unimplemented, clippy::todo, clippy::indexing_slicing)]` at the crate root with no
  `#[allow]` of them outside `#[cfg(test)]` (CI clippy); the shared `OpeningAccumulator`
  getters and the verifier accumulator's appends return `Result`, so a missing claim is an
  `Err` not a panic; the mutation sweep's `Expect` enum is deleted so every site must reject
  and a verifier panic anywhere in any sweep fails the test; a cargo-fuzz target that mutates
  serialized honest proofs and IO, whose regression corpus is replayed by a unit test.
- **The proof is consumed exactly.** `verify` returns `Ok` only if every sumcheck proof entry
  was taken exactly once, every evaluation-reduction proof was taken, and every opening claim
  was bound to a point; a proof entry, reduction proof or claim nothing consumed is
  `Err`. Mechanism: the ledger drain check at the end of `verify` (stage 2), the sweep sites
  `ClaimOrphan`, `ProofOrphan` and `EvalRedOrphan` (must reject), and the fuzz target.
- **No dead proof fields or tags.** Every `ProofType` variant occurs in the proof of some
  checked-in model, and every proof field is read by the verifier (a field not read is an
  orphan under the invariant above). Mechanism: test `proof_types_all_exercised` over the zoo
  and probe models (stage 2); the ledger.
- **Relations are untouched.** No relation's identity, input claim, opening points or append
  order changes; S ⊆ R notes carry over unchanged; the lookup subprotocols in `atlas-sumcheck`
  are not modified. Mechanism: the digest harness (`onnx_proof/testing/digests.rs`) passes
  unchanged in stages 1 and 3 and re-pins once in stage 2, in its own commit naming the cause
  (single-consumer evaluation-reduction entries dropped); review on the `docs/pull-requests.md`
  §4 checklist.
- **Sweep coverage never shrinks.** The mutation, tamper and forgery sweeps' site counts per
  class are non-decreasing across every stage PR and no tolerance is introduced. Mechanism: the
  sweep report printed by each test, compared in the PR body; `Expect` deleted (compiler) so a
  tolerance cannot be re-added without a new enum.
- **The trace behavioral contract is untouched** (`atlas-trace/src/lib.rs`: scale, fused
  rescale split, table generation, padding). Mechanism: importer goldens and the zoo.

### Non-Goals

- Changing the softmax protocol (#218, the auxiliary vectors). Stage 3 fits softmax into the
  plan as it is; the aux vectors become `Open` steps.
- Changing any relation's identity, any lookup subprotocol's internals, or the transcript
  order, except stage 2's declared drop of single-consumer evaluation-reduction entries.
- A wire-format redesign or proof versioning. Dead tags are deleted with their numbers kept;
  old proofs need not verify (pre-production, as `specs/adoption-program.md` states).
- Minimizing the verifier's dependency cone (no-std, splitting `atlas-trace` into a model crate
  and a trace crate). Stage 1 makes the split possible; it is a follow-up.
- Verifier or prover performance work beyond the no-regression gate.
- Committing to model inputs (#201) or any change to what the public IO is.
- Cleaning `jolt-atlas-core`'s prover beyond what following the plan requires: specialized
  witnesses (einsum dot engine, mean-of-squares, deferred batch provers) keep their code.

## Evaluation

### Acceptance Criteria

The main list is **stage 1, prover code out of the verifier crate**, ticked by the stage 1
PR. It is code motion: behavior-preserving, digests unchanged.

- [x] `jolt-atlas-verifier/clippy.toml` `disallowed-types` lists `atlas_trace::trace::Trace`,
  `atlas_trace::trace::FusedIntermediates`, `atlas_trace::trace::NodeAux` and
  `atlas_trace::trace::SoftmaxWitness`; the `verifier-isolation` job's grep rejects
  `trace::Trace` in the crate's sources; both are green.
- [x] `EngineRelation::witnesses` and `TieRelation::claims` no longer exist in the verifier
  crate. Core defines the extension traits `RelationWitnesses` (the engine relation's witness
  polynomials from the trace) and `TieClaims` (a tie's produced claims from the trace), on the
  `OpLookupProve` / `RangeCheckProve` precedent, implements them for every relation, and
  `RelationProver::initialize` and `prove_tie` bound on them.
- [x] Every function or method in the verifier crate that took a `&Trace` is in core: the
  split and bucket witnesses (`NodeSplit::from_trace`, `BucketSplit::from_trace`,
  `witness_polys`, `chunk_indices`, `saturated`, `empty_witness_polys`,
  `value_chunk_indices`, `exact_output_prover`), the clamp intermediates
  (`clamp_intermediate`, `clamp_lookup_bits`), the range-check and op-lookup index builders, the activation, trig-downscale, softmax significance-clamp,
  sum-axis and einsum witness helpers, and `trace_view.rs` whole. The PR body lists each item
  with its destination; `docs/verifier.md` records the list.
- [x] `probe.rs`, `probe_assert!` and `probe_assert_eq!` live in `jolt-atlas-core`; the
  verifier's `test-util` feature provides only the `&mut` proof accessors.
- [x] The verifier's `Cargo.toml` lists no dependency only the moved code used; `rayon` is
  gone unless the PR names a verifier-side use that stays.
- [x] `docs/verifier.md` exists: the audit reading order (entry point, interpreter, relation
  core, lookup machinery, proof object), the trust-perimeter statement (which crates and which
  of their items a verdict depends on), and the stage 1 inventory. `CLAUDE.md`'s crate-map
  paragraph for `jolt-atlas-verifier` says the crate is Trace-free and points at the doc.
- [ ] `cargo nextest run --workspace` green with every sweep's site counts unchanged;
  `onnx_proof/testing/digests.rs` passes unchanged; the GPT-2 Akita heavy job passes once on
  the branch; the Performance gate holds.

### Later Stages

Each box is ticked by its own stage PR; the invariants' gates apply to all.

- [ ] **Stage 2 — exact proof shape, no panics** (one PR; protocol-breaking, one re-pin
  commit). The crate-root `deny` lints above are on with no `#[allow]` outside `#[cfg(test)]`,
  which also removes `VerifyOp`'s panicking default bodies (its methods become required).
  The shared `OpeningAccumulator` getters return `Result<_, ProofVerifyError>`; `resolve`,
  `relation_input_claim`, `DynRelationVerifier::{input_claim, output_claim}` and `verify_tie`
  return `Result`, resolving every opening an expression references before evaluating it; the
  verifier accumulator's `append_*` return `Err` on a key the proof did not carry (prover-side
  call sites treat `Err` as a prover bug). `Verifier` gains `take_proof(ProofId)` and
  `take_eval_reduction(node)` that record consumption, and `verify` ends with the drain check
  (new `ProofVerifyError` variants `UnconsumedProof`, `UnconsumedEvalReduction`,
  `UnboundClaim`, `MissingClaim`). The prover emits no evaluation-reduction proof for a
  single-consumer node and the verifier requires its absence (the dead `h` field disappears;
  digests re-pinned). `ProofType` loses the variants no prover emits, numbering kept and
  `TryFrom<u8>` rejecting the freed tags; test `proof_types_all_exercised` holds over the zoo
  and probe models. `testing/mutate.rs`: `Expect` is deleted, `panics_tolerated` is deleted,
  `ClaimOrphan` must reject, new sites `ProofOrphan`, `EvalRedOrphan`, `IoOutputCount`,
  `IoOutputShape`, `IoInputCount` must reject. A cargo-fuzz target `verify` under
  `jolt-atlas-core/fuzz` (seeded from serialized probe-model proofs, mutating proof bytes and
  IO; any panic is a finding) with its regression corpus replayed by a `#[test]`; run per the
  importer's `fuzz/README.md` precedent, not in CI.
- [ ] **Stage 3 — one verification plan** (one PR per op family, each behavior-preserving;
  the last deletes the old traits). `onnx_proof/plan.rs` defines `Plan` (an ordered
  `Vec<Step>`) and `Step`: `Reduce` (the node's evaluation reduction), `Open` (openings
  appended at a stated point, including softmax's aux vectors), `Relation { proof: ProofType,
  relation }`, `Batch { proof, relations }`, `Tie { relation }`, `Defer { batch: DeferredBatch,
  step }`. Steps hold relations (data), never closures; a relation whose parameters depend on
  earlier claims reads them in its `SetupRelation::setup`, which already receives the
  accumulator read-only. One trait `OpPlan { fn plan(&self, node, model, io) -> Plan }`
  replaces `OpLayout::reduction_flow`, `ReductionFlow`, `VerifyOp` and `ProveOp`
  (`OpLayout::get_committed_polynomials` stays). The interpreter in `onnx_proof/verifier.rs`
  runs plans and owns the one deferral queue (`BTreeMap<DeferredBatch, Vec<Step>>`), run in
  `DeferredBatch` order after the node loop, each batch's one-hot checks right after it;
  `VerifierLookupJob`, `DeferredOneHot`, `deferred_onehots` and `deferred_batches` are deleted,
  and the clamp and remainder post-loop derives its buckets from model and IO with no
  registration set to reconcile. `Verifier`'s accumulator, transcript, proof map and queue are
  private. Core's `prove_iop` walks the same plans: `Relation`/`Batch` prove through the
  stage 1 witness traits (specialized witnesses attach there), `Defer` batches keep
  `prove_relation_batch_parallel`. Softmax is the named stress case: its stage driver becomes a
  plan (aux vectors as `Open`, stages 1–2 as `Batch`, stages 3–4 as `Defer`); if its
  aux-vector decoding cannot go through `setup`, the fallback is a data step `Public` that
  decodes named openings into small integers later steps read — still not a closure. Digests
  unchanged per family PR; `docs/pull-requests.md` gains §5, the verifier checklist (a new op
  is a plan plus relations plus probe-model coverage; no new tolerance; no new lint allow).

### Testing Strategy

Every stage: `cargo nextest run --workspace` (op-surface table, the three sweeps, the zoo, the
digest harness, the relation tests), the `verifier-isolation` job, the clippy job, and the
GPT-2 Akita heavy job once per stage PR. Stage 1 adds nothing but the two lint lines; the
digests prove it moved code and nothing else. Stage 2 adds `proof_types_all_exercised`, the
orphan and IO sweep sites, the fuzz target and its corpus-replay test, and re-pins the digests
once. Stage 3 adds, per family, nothing new: the digests and the sweeps are the gate, and the
compiler holds the privacy and the deleted traits. No stage touches the `atlas-onnx-import`
goldens. Coverage of a new op or proof field stays as `CLAUDE.md` states: a probe model in
`onnx_proof/testing/models.rs` and a `mutate::Site`.

### Performance

Benchmark: `cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2`
(Akita over Fp128, the README table), median of 3 runs, compared with `dev` immediately before
the stage. Gate: total prove time and verify time each within +3% of the baseline median; peak
memory unchanged within noise on the 16 GB reference machine.

- Stage 1 moves code and changes no arithmetic: every stage of the table is unchanged.
- Stage 2 adds O(claims) ledger bookkeeping to verify (sub-millisecond against the 2 s
  verify row) and removes the single-consumer evaluation-reduction entries, so the proof
  shrinks slightly and witness, IOP and opening stages are unchanged; the PR reports the new
  proof size.
- Stage 3 replaces trait dispatch with plan interpretation, one step per relation, negligible
  against the sumcheck rounds; IOP proving and verify are expected flat, and the gate catches
  a prover regression from a lost parallel batch.

## Design

### Architecture

End state:

```text
jolt-atlas-verifier (Trace-free)                    jolt-atlas-core (the prover)
  proof.rs, types.rs, proof_serialization.rs          prover.rs: prove_iop walks each node's Plan
  onnx_proof/                                           Step::Relation/Batch → RelationWitnesses
    verifier.rs   the interpreter: owns proofs,         Step::Tie            → TieClaims
                  transcript, accumulator, queue;       Step::Defer          → batched provers
                  take_proof / take_eval_reduction;    relation.rs: RelationWitnesses, TieClaims
                  runs Steps; drains at the end        witness.rs, preprocessing.rs
    plan.rs       Plan, Step, OpPlan                   ops/*, clamp_split, ...: witness builders
    relation.rs   AtlasRelation, EngineRelation          (from_trace), probe.rs
                  (expressions only), TieRelation
    ops/<op>.rs   relations + fn plan()
    clamp_split.rs, global_clamp.rs, fused_rebase.rs,
    range_checking/, op_lookups/, neural_teleport/:
                  relations + model-only layout
atlas-sumcheck::relation   RelationVerifier, verify_relation{,_batch}   (engine, both sides)
atlas-pcs, atlas-transcript, atlas-algebra, atlas-trace (Model, semantics, IO types)
```

**Stage 1.** The prover-only methods on shared relation types become core extension traits, the
pattern `OpLookupProve`, `RangeCheckProve` and `NodeEvalReductionProve` already use: a core
trait implemented for a verifier-defined type. `EngineRelation` keeps `binding_order`;
`TieRelation` becomes a marker with its claims in `TieClaims`. Witness builders that are free
functions or inherent methods on shared structs (`BucketSplit`, `NodeSplit`, `ClampBucket`)
move to core as free functions or extension traits beside the prover that calls them. The
verifier's `trace_view.rs` has no verifier reader and moves whole.

**Stage 2.** The ledger is a `BTreeSet<ProofId>` and a `BTreeSet<usize>` on `Verifier` plus the
accumulator's existing appended-point set (`mark_appended`); the drain compares them with the
proof's key sets. Fallibility is pushed to the getters: `resolve` returns `Result`, and an
expression is evaluated after every `RelationOpening` it names has been resolved into a map, so
`jolt_claims::Expr::evaluate`'s infallible closures stay as they are. The prover accumulator
implements the same `Result` getters; its callers `expect`, which core allows and the verifier
crate denies.

**Stage 3.** A `Step` is data: a `ProofType` slot and a relation (or a batch of them). The
interpreter maps `Relation` to `take_proof` + `RelationVerifier::new` + `verify_relation`,
`Batch` to the batch driver, `Tie` to `verify_tie`, `Reduce` to the evaluation reduction,
`Open` to the accumulator appends, `Defer` to the queue. The prover's interpreter maps the same
steps to `RelationProver::initialize` + `prove_relation`, the batch provers, `prove_tie`, the
reduction prover, the advice appends, and the deferred provers. A plan is built per node from
model and IO before the transcript runs; challenge-dependent setup stays in each relation's
`setup`, so building a plan reads nothing from the transcript. The clamp exact-output branch
(`exact_output_verifier`) is a plan decision: `Tie` when the public output is unsaturated,
`Defer` otherwise.

### Alternatives Considered

- **Lints and ledger only (stages 1–2 without 3).** Removes the panics and the prover code but
  leaves the hand-written `VerifyOp` impls and three deferral queues as the audit surface, each
  op still touching the transcript directly. Rejected: the question "what does the verifier
  check for op X, in what order" would still be answered by reading each op.
- **A general constraint DSL** describing whole ops, sequencing included, as pure data.
  Rejected as the adoption program did: relations already carry the algebra; the plan is only
  the sequencing, and steps hold existing relation types.
- **A fully precomputed model-level plan** with no per-step accumulator access. Rejected:
  relation setup draws challenges in transcript order, so construction is necessarily
  interleaved with verification; keeping setup inside the relation and the plan as data gets
  the auditability without pretending the dependency away.
- **Feature-gating prover code inside the verifier crate.** Rejected for the reason the
  stage 9b proposal was: a feature flag is not a boundary an auditor can see; code motion is.
- **Splitting `atlas-trace` into model and trace crates** so Trace-freedom is a crate-level
  fact. Viable follow-up; `disallowed-types` is sufficient now and far cheaper.
- **`catch_unwind` at the verify boundary** for panic-freedom. Rejected: hides bugs, does not
  cover rayon workers, and is not a property an auditor can check in the source.
- **Reading the single-consumer `h` polynomial** so the field is live instead of dropping it.
  Rejected: more verifier work to legitimize a dead field; absence is the honest shape.
- **Upstream `jolt-verifier`.** Upstream drives verification as stage lists of
  `ConcreteSumcheck`s, i.e. data-described sequencing over `SymbolicSumcheck`s. The plan is the
  atlas analogue with ops in place of zkVM stages; it was considered and is what stage 3 adopts
  in spirit, not by vendoring (upstream's stages are bound to its zkVM).

## Documentation

- `docs/verifier.md` (new, stage 1; living): the audit reading order, the trust-perimeter
  statement, the stage 1 inventory, then the plan interpreter (stage 3) and the proof-shape
  rules (stage 2) as they land.
- `CLAUDE.md`: the `jolt-atlas-verifier` crate-map paragraph (Trace-free, the interpreter, the
  plan) and a Prover/Verifier Invariants bullet for "the verifier rejects, never panics; the
  proof is consumed exactly" naming the sweeps and lints that hold it (stage 2).
- `docs/pull-requests.md`: §5, the verifier checklist (stage 3).
- `README.md`: only if the stage re-measurement changes the published numbers.

## Execution

Suggested commit sequence.

Stage 1: (1) `RelationWitnesses`/`TieClaims` in core, prover bounds switched, verifier trait
methods deleted; (2) clamp split, bucket and clamp-lookup witnesses moved; (3) range-check,
op-lookup, activation, trig, softmax, sum and einsum helpers moved; (4) `trace_view.rs`,
`probe.rs` and the macros moved, `rayon` dropped; (5) lint lines, `docs/verifier.md`,
`CLAUDE.md`. Digests green at every commit.

Stage 2: (1) `Result` getters and `resolve`, prover call sites `expect`; (2) `take_*` ledger
and drain, new error variants, `Expect` deleted, orphan and IO sites added; (3) dead
`ProofType` tags deleted and `proof_types_all_exercised`; (4) single-consumer reduction entries
dropped — the one re-pin commit; (5) deny lints on, remaining panics converted; (6) fuzz target
and corpus replay.

Stage 3: (1) `plan.rs` and the interpreter beside the old dispatch, the macro-bound relation
ops as the first family on both sides; (2) fused rescale and the lookup ops; (3) reductions,
gather, div and rsqrt; (4) softmax; (5) the deferral queue unified and the old traits, queues
and `ReductionFlow` deleted, `Verifier` fields private, §5 of `docs/pull-requests.md`.

## References

- `specs/adoption-program.md` — stages 1 (verifier crate), 5–6 (relations), 8–10 (verification
  moved, prover items in `prover` modules, legacy drivers retired); its digest harness and
  performance gate are reused here.
- `docs/pull-requests.md` §4 — the relation-migration checklist stage 3 PRs also meet.
- `docs/upstream-inventory.md` — upstream `jolt-verifier`'s `ConcreteSumcheck` stage lists, the
  prior art for data-described sequencing.
- Issues: #218 (softmax aux vectors, non-goal), #201 (committing to inputs, non-goal),
  #230 / #244 (input binding; the transcript binding the interpreter keeps first).
- Branches `claim-lifecycle-hardening` and `testlib-consolidation` (pre-rewrite, reference
  only): the claim-ledger and drain-check design this spec's stage 2 realizes on the current
  tree.
- `atlas-onnx-import/fuzz/README.md` — the fuzzing precedent stage 2's target follows.
