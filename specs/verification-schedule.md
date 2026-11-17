# Spec: One verification schedule per model

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-clamp.md` (no relation is chosen by a proof-carried claim; the committed set does not depend on the public IO), `specs/relations-as-values.md` (a relation is a value a schedule can hold) |
| Supersedes  | `specs/post-loop-batches.md`'s deferral queue (`DeferredBatch`, `DeferredSetup`), as a mechanism; its two-batch ordering is kept |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

What the verifier checks for a model, and in what order, is not written down anywhere. It is
produced while verifying: `verify_iop` walks the nodes in reverse, builds each node's `Plan`
from the model and the public IO, and runs it; `Step::Defer` pushes relations onto a queue,
some already set up (`Deferred::Ready`) and some not (`Deferred::Pending`); after the loop
`deferred_lookups::plan` builds a second plan for the packed buckets, and `run_deferred`
drains the queue batch by batch. The list of committed polynomials is computed by a third
path, `OpLayout::get_committed_polynomials` per op plus the buckets, which must agree with
what the plans open. An auditor who wants the transcript order of GPT-2 has to execute the
verifier in their head.

After `specs/one-clamp.md`, everything in that order is a function of the model: no step
depends on the public IO or on a claim's value. This spec computes it once, as one list of
steps for the whole model, and makes verifying a loop over that list. The schedule can be
printed, diffed and pinned; the deferral queue becomes two explicit steps at the end of the
list; and the committed polynomials are read off the schedule instead of declared beside it.

## Intent

### Goal

Compile a model's entire verification into one `Schedule`, a function of the model alone,
built once when the prover's and the verifier's preprocessing are constructed, that the
verifier interprets with a single loop and the prover walks with the same loop.

Introduced, in `jolt-atlas-verifier/src/onnx_proof/schedule.rs` (replacing `plan.rs`):

- `Schedule<F> { relations: Vec<Rel<F>>, steps: Vec<Step> }`. `Rel` is the relation enum of
  `specs/relations-as-values.md` (`OpRelation`: `Op`, `TableRead`, `OneHot`, the two
  GatherSmall one-hot stages, `SoftmaxRead`, the bucket value and digit relations); a `RelId`
  indexes `relations`.
- `Step`:
  - `Reduce { node }`: the node's evaluation reduction, with its claim count.
  - `Open(OpOpen)`: advice openings at a stated point.
  - `Setup(RelId)`: run a relation's setup now (its draws and appends), verify it later.
  - `Sumcheck { proof: ProofId, members: Vec<RelId> }`: set up each member not yet set up,
    in order, then verify the members as one sumcheck (a batch when there are several).
  - `Tie(RelId)`: a zero-round relation (an `OpRel`), checked by `verify_tie`.
- `ScheduleBuilder`, which the per-op plan functions write to: `reduce`, `open`, `relation`
  (a `Sumcheck` of one), `batch`, `tie`, `lookup` (a `Setup` now, membership in the final
  lookups sumcheck) and `one_hot` (membership in the final one-hot sumcheck). `finish` emits
  the bucket steps and the two final sumchecks and validates the result.
- `Schedule::compile(model) -> Result<Schedule<F>, ProofVerifyError>`,
  `Schedule::committed() -> Vec<CommittedPoly>`, `Schedule::proof_ids()`,
  and `impl Display` (one line per step).
- `AtlasProverPreprocessing<F, PCS>` and `AtlasVerifierPreprocessing<F, PCS>` each hold the
  compiled `Schedule<F>`, built when they are constructed from the shared preprocessing.
  `AtlasSharedPreprocessing` stays `{ model }`: it has no field parameter and is serialized
  (the `prove` example's `--use-cache`), and a schedule holds function pointers
  (`OpRel::custom`) and is never serialized.

Modified:

- `Verifier::run` is one `for step in &schedule.steps`. `Verifier::deferred` is removed.
- `ops::OpPlan::plan(&self, node, model, io) -> Plan` becomes
  `plan(&self, node, model, &mut ScheduleBuilder)`.
- The `Input` op's public-value tie names the input by position and leaves `OpRel::public`
  empty. When the interpreter reaches a `Tie` whose relation is of kind `OpRelKind::Input`, it
  fills `public` from the public IO it holds before calling `verify_tie`; the prover's walk
  fills it from its trace. This is the one step that reads the IO. The input stays public and
  uncommitted (see Non-Goals).
- `jolt-atlas-core`: `prove_plan`, `prove_steps`, `prove_deferred_batches` and
  `deferred_lookups::prove_all` become one walk of the schedule.
- The `prove` example gains `--schedule`, which prints the schedule and exits.

Deleted: `Plan`, `Step::{Relation, Batch, Defer}`, `Deferred`, `DeferredBatch`,
`DeferredSetup`, `Verifier::run_plan`, `run_deferred`, `deferred_lookups.rs` in both crates;
`OpLayout`, `NodeCommittedPolynomials`, `get_models_committed_polynomials`,
`range_bucket::bucket_committed_polys`.

### Invariants

Relied on, not changed:

1. **The `atlas-trace` behavioral contract**, all of it. *Mechanism:* no diff under
   `atlas-trace/`.
2. **Verifier hygiene.** *Mechanism:* the crate-root lints, `clippy.toml`, the
   `verifier-isolation` CI job.
3. **Exact consumption.** The ledgers and the drain check stay as they are.
   *Mechanism:* `ProofLedger`, `Verifier::check_consumed`, the orphan mutation sites.

Preserved, with the mechanism restated:

4. **The protocol is unchanged.** Every proof and the final transcript state are
   byte-identical to the parent commit's. *Mechanism:* `testing/digests.rs` is not edited by
   this PR.
5. **Prover and verifier agree on transcript order.** *Mechanism:* both walk
   `schedule.steps`; there is no other order. The lockstep comparison in the test `verify`.
6. **Two-batch ordering of the lookups.** A table read is set up in its node's turn and
   proven in the lookups sumcheck; a one-hot check is set up when the one-hot sumcheck runs,
   after the lookups sumcheck has produced the read-address claim it reads. *Mechanism:*
   `ScheduleBuilder::finish` emits the lookups `Sumcheck` before the one-hot `Sumcheck`, and
   `lookup` emits a `Setup` while `one_hot` does not; the digests.

New:

- **A. The schedule is a function of the model.** *Mechanism:* the compiler:
  `Schedule::compile` takes the model and nothing else, so no step can depend on the public
  IO, the proof or a claim.
- **B. Every relation is set up once and verified once.** Each `RelId` is the target of at
  most one `Setup`, and a member of exactly one `Sumcheck` or the target of exactly one
  `Tie`; a `Setup` precedes the `Sumcheck` that holds its relation. *Mechanism:*
  `ScheduleBuilder::finish` checks it and returns `Err`; a unit test with a hand-built
  violating schedule for each clause.
- **C. No empty sumcheck.** A `Sumcheck` step has at least one member; a model with no
  lookup has no lookups step and no proof entry for it. *Mechanism:* `finish`; the existing
  drain check rejects a stray entry.
- **D. The committed set is the opened set.** The commitments the verifier expects are
  exactly the committed polynomials the schedule's steps open. *Mechanism:* the compiler
  (`Schedule::committed` is the only source; `OpLayout` is deleted), and the digests, which
  cover the commitment list of four models.
- **E. The schedule is reviewable.** A change to any checked-in model's schedule shows up as
  a changed pin. *Mechanism:* `testing/schedules.rs`, which pins the Blake2b-256 of the
  rendered schedule of `op_gauntlet` and the three tiny models.
- **F. Verification does no planning.** `verify` reads the schedule from the verifier
  preprocessing. *Mechanism:* the compiler (`OpPlan::plan` is called only from
  `Schedule::compile`, and `Schedule::compile` only from the two preprocessing constructors
  and the `prove` example's `--schedule`).

### Non-Goals

- The proof format: `ProofId`, `ProofType`, the ledgers and the claim map stay
  (`specs/stream-proof.md`).
- Any change of transcript order.
- Proving nodes in parallel, or out of schedule order.
- Serializing a schedule or shipping it with a proof. It is recomputed from the model.
- Folding `Open` into relations, or `Reduce` into `Sumcheck`.
- A static check of proof shape against `Schedule::proof_ids` before the transcript runs.
  The drain check already rejects a wrong shape.
- Committing the model's inputs. That would make the `Input` tie independent of the IO, but
  it changes what a proof states (some input behind a commitment, not this public input) and
  adds a commitment to the proof; it needs its own spec.

## Evaluation

### Acceptance Criteria

- [ ] `testing/digests.rs` is not modified, and its four tests pass.
- [ ] `Schedule::compile`'s signature takes the model and no `ModelExecutionIO`, proof or
      accumulator (invariant A).
- [ ] `Plan`, `Deferred`, `DeferredBatch`, `DeferredSetup`, `run_plan`, `run_deferred`,
      `OpLayout`, `NodeCommittedPolynomials` and `get_models_committed_polynomials` do not
      exist; `deferred_lookups.rs` does not exist in either crate.
- [ ] The verifier and the prover each contain one loop over `schedule.steps` and no other
      call that runs a step.
- [ ] Invariant B: the three unit tests (double setup, relation in two sumchecks, setup
      after its sumcheck) return `Err` from `finish`.
- [ ] Invariant C: a model with no lookup compiles to a schedule with no lookups or one-hot
      step, and its proof verifies.
- [ ] Invariant D: on the commit before `OpLayout` is deleted, a test asserts
      `Schedule::committed()` equals `get_models_committed_polynomials` as sets on every
      probe model, `op_gauntlet` and the three tiny models. The test is removed with
      `OpLayout`.
- [ ] Invariant E: `testing/schedules.rs` exists with four pins; `cargo run --release -p
      jolt-atlas-core --example prove -- --model models/op_gauntlet --schedule` prints one
      line per step.
- [ ] The three sweeps pass with no tolerance entry; `proof_types_all_exercised` passes.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets`, `cargo check -p jolt-atlas-verifier` and `cargo nextest run --workspace`
      pass.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing unchanged: the digests; the zoo with its serialization
round trip; the three sweeps; the op-surface table; `fuzz_regressions_replay`;
`relation_kinds_match_their_dispatched_op_kind`.

New tests: the three `finish` rejections of invariant B; the no-lookup model of invariant C;
the four schedule pins of invariant E; the temporary set-equality test of invariant D.

Changed tests: any test that builds a `Plan` by hand or calls `run_plan` builds a schedule
through `ScheduleBuilder`.

CI coverage: the workspace job. Not protocol-level (invariant 4), so the heavy job is not
required. No proof format change, so no new `mutate::Site`.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16`, against this spec's parent commit, on the
benchmark machine.

Expected: prove flat. Verify flat or slightly down, since planning moves from each `verify`
call to the construction of the verifier preprocessing, which goes up by one planning pass. Peak memory flat:
the schedule holds a few expressions per node. Bounds: the gate's defaults (prove and verify
within 3%, peak memory within 5%); proof size equal to the byte.

## Design

### Architecture

```text
before:  for node in reverse: plan(node, model, io) -> run_plan
         deferred_lookups::plan(model, io) -> run_steps
         for batch in [Lookups, OneHot]: drain queue (Ready | Pending) -> verify_batch
         committed: OpLayout per node + buckets

after:   schedule = Schedule::compile(model)          // preprocessing, once
         for step in schedule.steps: run(step)        // verifier and prover
         committed: schedule.committed()
```

The schedule for a model is the concatenation, in reverse topological order, of each node's
steps, followed by the bucket steps, the lookups sumcheck and the one-hot sumcheck. A ReLU
node, for example, contributes `Reduce`, `Setup(read)` and nothing else in its turn; `read`
is a member of the final lookups sumcheck and its one-hot relation a member of the final
one-hot sumcheck.

`Sumcheck` covers the three shapes there are today. `Step::Relation` is a `Sumcheck` of one
member not yet set up. `Step::Batch` is a `Sumcheck` of several. A deferred batch is a
`Sumcheck` some of whose members were set up earlier by `Setup`. The interpreter keeps a
`Vec<Option<verifiers>>` indexed by `RelId` for relations set up and not yet verified; that
vector replaces the queue.

`ProofId`s are assigned by the builder as they are today: `ProofId(node, proof_type)` for a
node's steps, `ProofId(DEFERRED_PROOF_IDX, ..)` for the final ones.

`Schedule::committed` collects the committed polynomial ids of every `Open` step and of every
relation: an `OpRel`'s committed `produced` and `setup_opens` ids, a one-hot relation's
`RaEncoding::committed` chunks, a digit relation's digit columns. A committed polynomial that
no step opens cannot exist.

`Display` renders a step as its kind, its node and its relations' `OpRelKind` or name, for
example `n412 Sumcheck [Product]`. A step's `ProofId` is not rendered: it is where the proof
is stored, not what is checked, and `specs/stream-proof.md` removes it without changing a
pin. The rendering is for people and for the pin; nothing parses it.

### Alternatives Considered

- **Keep per-node plans, add a function that concatenates them for printing.** Rejected: the
  printed order would be a second description of what the interpreter and the queue do, and
  the queue, the `Ready`/`Pending` split and `OpLayout` would all stay.
- **Compile the schedule from the model and the public IO.** Rejected: after
  `specs/one-clamp.md` nothing in the order needs the IO, and a schedule that is a function
  of the model can live in the preprocessing and be pinned.
- **Derive the committed set but keep `OpLayout` as a cross-check.** Rejected: two sources
  that must agree are what this spec removes. The temporary test of invariant D does the
  cross-check once.
- **Pin the rendered schedule's text in the repository** instead of its hash. Rejected: the
  GPT-2-shaped schedules are thousands of lines; `--schedule` prints the text on demand.
- **Make `Setup` implicit in `Sumcheck` always**, by reordering the lookups' setups to the
  end. Rejected: it changes the transcript, and a read's setup draws its challenge in the
  node's turn for a reason (`specs/post-loop-batches.md`).

## Documentation

- `docs/verifier.md`: the reading order (step 2 becomes "the schedule and its interpreter"),
  the "Plans" section rewritten as "The schedule", and "Checked inputs" (the committed list).
- `CLAUDE.md`: the `jolt-atlas-verifier` and `jolt-atlas-core` crate-map entries (`OpLayout`,
  `OpPlan`, `Plan`, `run_plan`/`run_deferred`, `prove_plan`); the essential commands gain the
  `--schedule` line.
- `docs/pull-requests.md` and `docs/op-roadmap.md`: a new op writes one `plan` function; a
  changed schedule pin is expected and reviewed.
- Module docs of `schedule.rs`, `verifier.rs`, `preprocessing.rs`, `ops/mod.rs`.

## Execution

Each commit leaves the digests green.

1. `Schedule`, `Step`, `ScheduleBuilder` and `finish`'s checks, with the per-op `plan`
   functions writing to a builder; `Plan` becomes a per-node view of it for one commit.
2. The bucket steps and the two final sumchecks in `finish`; the interpreter loop; delete
   the queue.
3. The prover's walk.
4. `Schedule::committed`, the set-equality test, then delete `OpLayout` and its callers.
5. The schedule in the prover's and the verifier's preprocessing; the `Input` tie reads the
   IO at run time.
6. `Display`, `--schedule`, the pins, docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/{plan.rs, verifier.rs, deferred_lookups.rs,
  preprocessing.rs, range_bucket.rs, relation.rs, lookup.rs, ops/mod.rs, ops/input.rs}`.
- `jolt-atlas-core/src/onnx_proof/{prover.rs, ops/mod.rs, deferred_lookups.rs}`,
  `testing/digests.rs`, `examples/prove.rs`.
- `specs/verifier-audit-simplification.md` (plans), `specs/post-loop-batches.md` (the two
  batches), `specs/one-clamp.md`, `specs/relations-as-values.md`, `specs/stream-proof.md`.
