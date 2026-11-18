# Spec: The schedule runs in waves

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-18                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/digits-as-trace.md` stage 1 |
| Supersedes  | `specs/digits-as-trace.md` stage 2 (the statement of it; this is its spec) |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The prover walks the model one node at a time. A node's sumcheck cannot start before the
sumchecks of its consumers have ended, because their challenges fix the point of its claim, so
the 756 sumchecks of Qwen (311 of GPT-2) run one after another, each too small to use more than
one core. On the benchmark machine that walk is 3.26 s of Qwen's 11.2 s prove and 1.26 s of
GPT-2's 3.66 s, it runs on about one core of sixteen, and no tuning of the sumchecks themselves
moves it (the Performance section has the measurements).

`specs/digits-as-trace.md` stage 1 removes the reason for the order. A node whose output is
in a range bucket (a digit-backed node) takes the point of its relation fresh from the
transcript, and claims on its output are discharged against the committed digits. Such a node
waits for nobody. What is left of the chain is short: from a digit-backed node through the
nodes that are not digit-backed (table reads, softmax, the structural ties, constants) to the
next digit-backed node or public tensor. A transformer's blocks are the same circuit, so every
block's nodes become ready at the same moments.

This spec uses that. The schedule is compiled in waves: each wave takes every node that is
ready, runs their steps that draw no sumcheck challenge, and proves all their sumchecks as one
batched sumcheck with the nodes as members, in parallel. The number of sumchecks of a model
becomes the number of waves, which is the longest chain between two digit-backed nodes rather
than the number of nodes.

## Intent

### Goal

Compile a model's verification schedule (`Schedule::compile`) as a sequence of waves over the
dependency order of its nodes instead of one node turn after another, with one batched
sumcheck per wave over the relations of the wave's nodes.

Definitions, all functions of the model alone:

- A node is **complete** when every step of its plan has been emitted.
- A node is **ready** when every consumer of its output is complete, or when it is
  digit-backed (stage 1 of `specs/digits-as-trace.md`: its plan starts with `Step::Point` and
  reads no consumer's claim), or when it is a model output.
- A node's plan is a list of steps (`OpPlan::plan` writes it to a `ScheduleBuilder`, as today).
  Its **segments** are the runs of steps between its `Step::Sumcheck`s.
- A **wave** is: for every ready, incomplete node in descending node order, the steps of its
  next segment; then one `Step::Sumcheck { batched: true }` whose members are the next
  sumcheck of each of those nodes that has one, in descending node order. A node with no
  further sumcheck completes in the wave's first half.
- A wave's sumcheck is split into consecutive batches when its members' witness sizes exceed
  the **wave budget**: a constant number of field elements (`WAVE_BUDGET`, in
  `schedule.rs`), counted as the sum over members of `2^rounds · wires`, the member list cut
  greedily in order. A member larger than the budget is a batch of its own.

The three post-loop batches (bucket values, lookups, one-hot) stay where they are, after the
last wave.

Introduced or modified:

- `jolt-atlas-verifier/src/onnx_proof/schedule.rs`: `Schedule::compile` collects each node's
  plan, then emits waves; `ScheduleBuilder` records plans per node; `Schedule::validate` gains
  the readiness rule (a `Reduce { node }` comes after every step that appends a claim on
  `node`); the module header states the wave rule. `Display` renders a wave's sumcheck as
  today's cross-node batch (`model Batch [...]`).
- `jolt-atlas-verifier/src/onnx_proof/verifier.rs`: `Verifier::run` is unchanged in kind (one
  loop over steps); `Step::Sumcheck { batched: true }` already verifies members of different
  nodes.
- `jolt-atlas-core/src/onnx_proof/ops/mod.rs`: `prove_schedule` builds the witnesses of a
  wave's members in parallel. A member's setup (its reads of the accumulator, its draws and its
  appends) runs in member order on the prover's thread; its witness is then built from the
  trace and that setup alone, on the thread pool. `Prepared` gains the variants this needs;
  consecutive `Step::Tie`s compute their claims in parallel and send them in order.
- `atlas-sumcheck`: none expected. `verify_relation_batch` and `prove_relation_batch_parallel`
  already take members of different rounds and degrees, tail-aligned.

### Invariants

Relied on, not changed:

1. **Every relation, every step's meaning, the commitments and the trace semantics.** Only the
   order of steps and the grouping of sumchecks change. *Mechanism:* the importer's golden
   hashes are not re-pinned; `relation_kinds_match_their_dispatched_op_kind`;
   `relation_kinds_all_exercised`.
2. **The schedule is a function of the model.** Readiness, wave membership, member order and
   the budget split read the model's graph and shapes only. *Mechanism:* `Schedule::compile`
   takes the model alone (the type); the mutation sweep's check that two executions of one
   model read the same sequence of kinds.
3. **A claim exists only from the append that binds it, and a batch reads every member's
   input claim before its first round.** *Mechanism:* `verify_relation_batch` (unchanged);
   the accumulator's bind rules.
4. **Verifier hygiene** (rejects, never panics; reads the proof once, in schedule order).
   *Mechanism:* the crate lints, the CI path checks, the mutation sweep with no tolerance
   entry, the cut-and-extend checks.
5. **Stage 1's invariants A to E** of `specs/digits-as-trace.md`.

New:

- **A. Readiness is sound.** No step reads the reduced claim of a node, and no `Reduce` of a
  node runs, before every step that appends a claim on that node. *Mechanism:*
  `Schedule::validate` checks it on every compiled schedule from the steps' own declared
  reads and appends, independently of the wave construction; a compiled schedule that fails
  is a compile `Err`. The accumulator's existing debug assertion (no `NodeOutput` claim after
  the node's reduction) stays as a second line in tests.
- **B. One batching coefficient per member, drawn after every member's input claim is
  absorbed.** Members of one wave do not constrain each other; the batch is sound by the
  usual random-linear-combination argument, with error `members / |F|` added per wave and
  covered by the engine's per-challenge proof of work as for today's batches. *Mechanism:*
  `verify_relation_batch` (unchanged); review of the grinding bound for the largest wave.
- **C. A wave's members are independent of each other's challenges.** A member's setup may
  read only claims appended in earlier waves or in this wave's first half. *Mechanism:*
  follows from A (a node is ready only when its consumers are complete, and a complete node
  has run all its sumchecks); `Schedule::validate`.
- **D. The prover builds a member's witness without the accumulator or the transcript.**
  *Mechanism:* the compiler: the parallel build takes `&Trace` and the member's setup, which
  are `Sync`; the accumulator is not.
- **E. Peak memory is bounded by the wave budget.** At most one batch's witnesses are live.
  *Mechanism:* the gate's peak-memory column on GPT-2, bge-small and Qwen.

### Non-Goals

- Stage 1 of `specs/digits-as-trace.md` (its own PR, landed first).
- Stage 3: giving the remaining virtual outputs a decomposition, deleting `Step::Reduce`.
- Sharing one point among the members of a wave, or stacking same-shape members into one
  relation over a block index. Members keep their own points and their own witnesses.
- Any change to the post-loop batches, the opening reduction or the commitment.
- Detecting a model's blocks. The waves follow from readiness; nothing names a block.

## Evaluation

### Acceptance Criteria

- [ ] `specs/digits-as-trace.md` stage 1 is merged.
- [ ] `Schedule::compile` emits waves as defined above; the schedule of every checked-in
      model passes `Schedule::validate` with invariant A's rule.
- [ ] The number of `Step::Sumcheck`s outside the three post-loop batches is at most a tenth
      of today's for GPT-2 and for Qwen (311 and 756 today), and the PR states the counts.
- [ ] A test builds a schedule by hand that reduces a node before one of its consumers has
      appended its claim, and `validate` rejects it.
- [ ] A test with a budget of one member's size shows the split: every batch has one member,
      and the proof verifies.
- [ ] The three sweeps pass over the probe models with no tolerance entry; `ReadKind`
      coverage is complete.
- [ ] `testing/schedules.rs` and `testing/digests.rs` are re-pinned in a commit declared
      protocol-breaking; the `fuzz_regressions_replay` corpus is regenerated there.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2, bge-small and Qwen prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; every op-surface row; the three
sweeps; the zoo with its serialization round-trip; `relation_kinds_all_exercised`;
`relation_kinds_match_their_dispatched_op_kind`; `schedule_of_a_lookup_free_model_has_no_
lookup_step`.

Changed tests: the schedule and digest pins. Tests in `testing/relations.rs` that assert a
node's sumcheck is unbatched follow the new grouping.

New tests: the hand-built schedule of invariant A; the budget split; a probe model with two
independent digit-backed chains of different lengths, asserting that the shorter chain's nodes
complete in the waves the longer one is still in, and that the proof verifies.

CI coverage: the workspace job; the heavy job, since sumcheck ordering and transcript
absorption change. One manually dispatched exhaustive soundness run before merge.

### Performance

Benchmark: `bench/gate.sh` on `gpt2`, `bge-small-en-v1.5` and `qwen` at 16 tokens, against the
commit stage 1 lands on, on the benchmark machine.

Measured at `5b08e87b3` on the benchmark machine (before stage 1):

| | GPT-2 | Qwen |
|---|---|---|
| prove | 3.66 s | 11.19 s |
| sumcheck stage (`iop`) | 1.66 s | 4.57 s |
| of which the node walk | 1.26 s | 3.26 s |
| sumchecks in the walk | 311 | 756 |
| walk at one thread and at sixteen | 0.35 s and 0.42 s (the sumchecks) | not measured |

Expected: the walk's work is unchanged and spreads over the thread pool. The sumcheck stage
is expected at or under 1.0 s on GPT-2 and 2.5 s on Qwen. Proof size falls: a wave sends one
round message per round for all its members, where each member sent its own. Verify falls by
the round messages no longer hashed; the verifier still evaluates every member's relation.
Commit and openings unchanged.

Bounds: the sumcheck stage at most 70% of the reference on GPT-2 and on Qwen; prove and verify
no worse than the gate's 3% on all three models; peak memory within 5%; proof size no larger.

## Design

### Architecture

```text
today (after stage 1):   n9: Point, Sumcheck   n8: Reduce, Tie   n7: Point, Sumcheck   ...
                         one node turn after another, one sumcheck per node

waves:   wave 1:  n9 Point | n7 Point | n3 Point | ...     then Sumcheck [n9, n7, n3, ...]
         wave 2:  n8 Reduce, Tie | n6 Reduce, Open | ...   then Sumcheck [n6, ...]
         ...
         post-loop: bucket values, lookups, one-hot
```

`Schedule::compile` runs every node's `OpPlan::plan` into its own step list first, with the
relations registered once as today. It then loops: collect the ready, incomplete nodes; emit
each one's next segment; emit the wave's sumcheck, split by the budget; mark nodes with no
step left complete; repeat until every node is complete. Consumers are read from the model's
views. The loop ends because a model's graph is acyclic and a model output is ready at once.

A node whose plan appends a claim on an operand in a late segment holds that operand back
until it completes. That is conservative and simple; the measured chains are short.

`Schedule::validate` derives, for each step, the nodes whose reduced claim it reads and the
nodes it appends a claim on, from the step and its relations (`Reduce` reads all claims on
its node; a relation's `produced` and `setup_opens` name the operands it appends on; an
`OpOpen` and a table read name the reduced output they read). It walks the steps once and
rejects a read that precedes an append on the same node.

On the prover, a wave's sumcheck is `prepare` (in member order: setup, draws, appends) then
`build_all` (parallel: the witness from the trace and the setup) then
`prove_relation_batch_parallel`, the path the post-loop batches already take. The budget keeps
one batch's witnesses live at a time; a batch's witnesses are dropped on the background
thread when its sumcheck ends.

### Alternatives Considered

- **Commit the block boundaries and batch isomorphic blocks.** The first form of this idea.
  Rejected: the boundaries are already committed as digits, stage 1 already cuts the chain
  there and at every other digit-backed node, and readiness gives the same batches without
  matching blocks or assuming a model has them.
- **Keep one sumcheck per node and run independent nodes' sumchecks on separate transcripts.**
  Parallel, but a forked transcript per node is a new object in the trust perimeter, and the
  proof keeps one round message per round per node.
- **Stack same-shape members over a block index into one relation.** Fewer verifier
  evaluations and shared `eq` tables, at the cost of padding 24 blocks to 32 and a relation
  compiler. This is stage 3's direction; the waves are a prerequisite for it, not an
  alternative.
- **Parallelism inside each small sumcheck.** Measured: at these sizes the thread pool makes
  them slower (0.35 s on one thread, 0.42 s on sixteen), and a coarser grain does not help.
- **Upstream `a16z/jolt`.** Its zkVM proves all constraints of a stage as one batched
  sumcheck over committed columns; a wave is that structure for the part of the trace that
  is committed.

## Documentation

- `docs/verifier.md`: the reading order (waves, the readiness rule, what `validate` checks).
- `docs/prover-plan.md`: the route's node-walk item, marked landed with the measured numbers.
- `docs/measurement-plan.md`: the measurements of the Performance section.
- `CLAUDE.md` and `AGENTS.md`: the crate map's description of the schedule and the prover's
  loop.
- Module header of `schedule.rs`.

## Execution

Suggested order: (1) `Schedule::validate`'s readiness rule on today's schedules, digests
unchanged; (2) per-node plans collected, then emitted in today's order, digests unchanged;
(3) the wave loop with a batch per member (waves change the order only), pins moved;
(4) one batch per wave and the budget split, pins moved again, both pin moves in the
protocol-breaking commit; (5) the prover's parallel witness build and lockstep ties;
(6) the tests, the measurements and the docs.

## References

- `specs/digits-as-trace.md` (stage 1, which this depends on; its stage 2 is this spec).
- `specs/post-loop-batches.md`, `specs/verification-schedule.md` (the schedule and the
  batches this extends).
- `jolt-atlas-verifier/src/onnx_proof/schedule.rs`, `verifier.rs`;
  `jolt-atlas-core/src/onnx_proof/ops/mod.rs` (`prove_schedule`, `prepare`, `build_all`);
  `atlas-sumcheck/src/relation/{batch,prover}.rs`.
