# Spec: Lookups in two post-loop batches

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`     |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Lookups are scheduled by family, not by what they need. ReLU, Clamp, the trig downscale and
table, and Gather run their read sumcheck and their one-hot check inside the node's turn, one
small sumcheck each. The clamped activations and softmax's stage 3 are deferred into
per-family batches, each followed by its own one-hot batch, six in all, and the clamp buckets'
slack chunk check is a seventh batch of the same kind. `ProofType` carries a tag per family.
Only one ordering fact is real: a one-hot check reads the read-address claim that the sumcheck
before it produces. This spec schedules by that fact. Every table read goes into one batch,
`Lookups`, and every one-hot check into one batch after it, `OneHot`. The packed buckets' value
and digit batches already follow their own single ordering fact and are left alone.

## Intent

### Goal

Replace the six per-family deferred batches, the in-loop lookup sumchecks and the buckets'
slack chunk batch with two post-loop batches, `Lookups` and `OneHot`, and retire the
per-family `ProofType` tags.

| Batch | Members | Set up | Reads |
|---|---|---|---|
| `Lookups` | every table read whose sumcheck produces only its own read-address claim: ReLU, Clamp, activation clamp, small activation table, trig downscale, trig table, softmax exp-hi, exp-lo and significance clamp | in the node's turn | claims appended in the node loop |
| `OneHot` | the one-hot checks of every `Lookups` member and of Gather; Gather-small's Hamming-weight check; the clamp buckets' slack chunk checks | when the batch runs | the read-address claims `Lookups` and the in-loop Gather read produce; `IND` and `S` from the clamp buckets' split batch |

Unchanged: the four bucket batches (`DeferredClampSplit`, `DeferredClampDigitRange`,
`DeferredRemainderRC`, `DeferredRemainderDigitRange`), which run first, in today's order.
After the node loop the verifier therefore runs at most six batched sumchecks, in a fixed
order: the four bucket batches, `Lookups`, `OneHot`.

Gather's read sumcheck stays in the node loop: it produces a claim on its dictionary operand,
which must exist before that operand's evaluation reduction.

Introduced or modified:

- `jolt-atlas-verifier`: `deferred_lookups.rs` (`DeferredBatch` becomes `Lookups` and
  `OneHot`; the post-loop plan queues the slack chunk checks with `Step::Defer` instead of a
  `Step::Batch`), `types.rs` (two new `ProofType` tags; nine retired), and the plans of
  `relu.rs`, `clamp.rs`, `sin.rs`, `cos.rs`, `gather/small.rs`, `gather/large.rs`,
  `activation_clamped/mod.rs`, `softmax_last_axis/mod.rs`. No relation, and no `OpRelation`
  variant, changes.
- `jolt-atlas-core`: `deferred_lookups.rs`, `ops::prove_deferred_batches`, and a tracing span
  per relation kind inside a batch's witness build.

### Invariants

Relied on, not changed:

1. **Every relation.** No relation's expression, setup, challenges or produced openings change;
   only the batch it is proven in and the transcript position of that batch.
   *Mechanism:* review on the diff, which touches plans and the batch enum, not relations.
2. **Batched sumcheck semantics.** Members are tail-aligned, the batch degree is the maximum
   member degree, and a member cannot read an opening another member of its batch produces.
   *Mechanism:* `atlas_sumcheck::relation::verify_relation_batch`, unchanged.
3. **The bucket batches.** Their members, order and tags. *Mechanism:* the diff does not touch
   their plan steps except the slack chunk check.
4. **Verifier hygiene.** Rejects with `Err`, never panics, consumes the proof exactly, stays
   Trace-free. *Mechanism:* the crate lints, the CI path checks, the mutation sweep with no
   tolerance entry, the drain check.

New:

- **A. Two setup timings, decided by the batch.** A `Lookups` member is set up in its node's
  turn: it draws its challenges and appends its operand claim there. A `OneHot` member is set
  up when `OneHot` runs. `DeferredBatch::setup` returns exactly these two answers.
  *Mechanism:* a claim missing at setup is an `Err` in the verifier and an `expect` in the
  prover, so a wrong assignment fails every test that reaches it.
- **B. Fixed order, dependency-closed.** `run_deferred` runs the bucket plan, then `Lookups`,
  then `OneHot`. Every opening a `OneHot` member reads is produced by `Lookups`, by the
  in-loop Gather read, or by the clamp split batch, all of which precede it.
  *Mechanism:* `DeferredBatch::ALL`; invariant A's failure mode, exercised by the zoo and by
  `proof_types_all_exercised` (both batches non-empty in some checked-in model).
- **C. Operand claims land before their reduction.** A deferred lookup appends its operand
  claim (`NodeOutput(input)`) during its node's turn, as the activation clamp does today, never
  at batch time. *Mechanism:* the accumulator rejects an append to a node after its evaluation
  reduction; an op-surface case per moved family whose operand has a second consumer.
- **D. One-hot checks are independent of each other.** No `OneHot` member reads another's
  output; Gather-small's three checks read only the index operand's point.
  *Mechanism:* review-held; the gather op-surface cases.
- **E. Empty batches carry no proof.** A model with no table lookup, no Gather and no slack
  check has no `Lookups` or `OneHot` entry; a stray entry fails the drain check. Whether a slack
  check exists still depends on the buckets' exact flags, which are absorbed before the batch,
  as today. *Mechanism:* the drain check; a test on a lookup-free model.
- **F. No per-family tag survives.** `ProofType` keeps the node-level tags and the four bucket
  tags and gains `DeferredLookups` and `DeferredOneHot`. *Mechanism:* the compiler;
  `proof_types_all_exercised`; retired tags fail to deserialize.
- **G. Timing stays attributable.** A release run still reports time per lookup family.
  *Mechanism:* a tracing span per relation kind in the prover's witness build for a batch;
  `bench/spans.py` on a GPT-2 run shows a row per kind.

### Non-Goals

- Merging the bucket value or digit batches. At analysis the estimate was that merging the
  clamp and remainder digit range checks keeps about 1.3 GB more alive at once on GPT-2, around
  7% of the 17.6 GB peak, for two fewer proof entries. Not pursued here.
- Any change to a relation, a table, or the one-hot check (`shout::ra_onehot_verifiers`).
- Folding the Hamming-weight check into the read sumcheck, or four-factor virtual selectors.
- Fusing the activation clamp into the table lookup (`docs/prover-plan.md` item 6, B2).
- Deferring node relations other than lookups; the evaluation reduction
  (`specs/digits-as-trace.md`).
- Softmax stages 1 and 2, which stay in-loop batches of the node.
- Any change to trace semantics or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] `DeferredBatch` has exactly two variants, `Lookups` and `OneHot`.
- [ ] No plan of ReLU, Clamp, Sin, Cos, the activations or softmax stage 3 contains an in-loop
      lookup sumcheck or one-hot batch; Gather keeps exactly one in-loop `Step::Relation`.
- [ ] The post-loop plan has no slack chunk `Step::Batch`; the four bucket batches are
      unchanged.
- [ ] `ProofType` has `DeferredLookups` and `DeferredOneHot`; tags 2, 3, 15, 18, 19, 20, 23, 24
      and 25 are in the deleted lists of `types.rs` and `docs/verifier.md`.
- [ ] `proof_types_all_exercised` passes with both batches non-empty across the checked-in
      models.
- [ ] Invariant C: one op-surface case each for ReLU, Clamp, Sin, Cos and an activation where
      the lookup's operand has a second consumer, so its evaluation reduction has two claims.
- [ ] Invariant E: a model with no lookup and no clamped node has no post-loop proof entry.
- [ ] `saturating_probe_saturates` asserts the slack check through the slack chunk openings in
      the proof's claims, not through a proof tag.
- [ ] Invariant G: the PR shows the per-kind rows from `bench/spans.py` on a GPT-2 run.
- [ ] The three sweeps pass over the probe models with no tolerance entry; one `mutate::Site`
      per new proof entry.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit, with the `fuzz_regressions_replay` corpus regenerated there if it no longer
      parses.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the op-surface rows of every lookup
op and of the bucket cases; `exact_output_bucket_commits_nothing`; the three sweeps; the zoo
with its serialization round-trip.

Changed test: `saturating_probe_saturates` names `ProofType::DeferredClampChunkChecks` today;
that tag is retired, so the test checks for the slack chunk openings instead.

New tests: the five two-consumer cases of invariant C; the lookup-free model of invariant E.

CI coverage: the workspace job; the heavy job for GPT-2 and bge-small, since this is a
transcript-order change across every lookup. The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the head
of `specs/one-range-check.md`, on the benchmark machine.

Expected: the IOP stage flat or slightly down (more instances proven in parallel by
`prove_relation_batch_parallel`, fewer small sequential sumchecks); proof size and verify time
slightly down (fewer sumcheck proofs, fewer grinds). On GPT-2 and bge-small every lookup is
already deferred except Gather's one-hot check, so the movement there comes from merging
batches.

Memory: a batch keeps all its members' prover state alive while it is proven, so merging the
activation, softmax and gather families raises the peak of the lookup phase. These families
are a few percent of committed rows; the bucket digit batches, which are not merged, dominate.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better on both
models; peak memory within 5%; proof size no larger.

## Design

### Architecture

```text
node loop (reverse topological):
    lookup op:   Reduce, [Open ...], Defer{Lookups, read}, Defer{OneHot, one-hot checks}
    gather:      Reduce, Relation{Execution, read}, Defer{OneHot, one-hot checks}
    clamped op:  as today (Open(ClampAcc), relation, ...)
post-loop:
    per live clamp bucket: Open(ClampExactFlag), Tie(ClampBucketExact)
    Batch[DeferredClampSplit]             as today
    Batch[DeferredClampDigitRange]        as today
    per live clamp bucket: Defer{OneHot, slack chunk check}
    Batch[DeferredRemainderRC]            as today
    Batch[DeferredRemainderDigitRange]    as today
    Lookups   every deferred read
    OneHot    every one-hot check, the slack chunk checks
```

`DeferredBatch::setup` keeps its two answers: `Lookups` is `InNodeLoop`, `OneHot` is `AtBatch`.
The slack chunk check is queued as `AtBatch` like any one-hot check; it reads `IND` and `S`,
which the split batch has produced by the time `OneHot` runs, and yields no relation when the
bucket's exact flag is set, as today.

Degrees: `Lookups` 2 throughout; `OneHot` up to 9 (the ra-virtualization of a 32-bit address
in eight chunks) with Booleanity at 3 and Gather-small's Hamming-weight check at 1. A node's
one-hot batch already mixes the first two degrees today.

Prover: `prove_deferred_batches` proves each of the two with
`prove_relation_batch_parallel`; `Prover::lookup_addresses` keeps handing read addresses from a
read to its one-hot check, keyed as today. The witness build emits one span per relation kind.

### Alternatives Considered

- **Four batches: also merge the bucket value batches into `Values` and the digit batches
  into `Digits`.** The first draft of this spec. Rejected at analysis: the bucket batches
  already follow their one ordering fact, and merging their digit range checks costs memory on
  the phase that dominates it, for two fewer proof entries.
- **Four batches as a hard requirement, with lazily built digit-range provers.** Rejected:
  it reaches into the sumcheck prover to pay for that same small gain.
- **One batch for everything.** Impossible: a member cannot read an opening produced in its
  own batch.
- **Defer Gather's read.** Rejected: it produces an operand claim.
- **Keep in-loop lookups in the loop.** Rejected: they have no ordering reason to be there,
  and they are the lookups that get no cross-node batching today.
- **Leave the slack chunk check as its own batch.** Rejected: it is a Booleanity check over
  one-hot chunks that reads an earlier batch's output, which is what `OneHot` is.
- **A per-round degree schedule so low-degree members do not pay the batch degree in proof
  size** (a16z/jolt #2336). Deferred: an engine change; revisit if the proof-size bound fails.

## Documentation

- `docs/verifier.md`: the "Plans" section's post-loop paragraph is rewritten around the bucket
  batches followed by `Lookups` and `OneHot`; the deleted-tag list grows.
- `docs/pull-requests.md` §5: a new lookup op defers its read to `Lookups` and its one-hot
  check to `OneHot`.
- `docs/measurement-plan.md`: the span names the stage tables are read from.
- Module docs of `deferred_lookups.rs` in both crates.
- `README.md`: the benchmark table if a row moves outside noise.

## Execution

Suggested order: (1) the two-variant enum with the six existing deferred families mapped onto
`Lookups` and `OneHot`, and the per-kind spans; (2) the slack chunk check into `OneHot`, with
the `saturating_probe_saturates` change; (3) the in-loop families, one op per commit; (4) tag
retirement and the digest re-pin.

## References

- `jolt-atlas-verifier/src/onnx_proof/deferred_lookups.rs`, `verifier.rs` (`run_deferred`),
  `plan.rs`; `atlas-sumcheck/src/relation/batch.rs`; `atlas-sumcheck/src/shout/mod.rs`
  (`ra_onehot_verifiers`).
- `jolt-atlas-core/src/onnx_proof/ops/mod.rs` (`prove_steps`, `prove_deferred_batches`,
  `prepare`, `build_all`).
- `specs/verifier-audit-simplification.md` (the plan interpreter and the single deferral queue).
- a16z/jolt #2336 (per-round degree schedule).
