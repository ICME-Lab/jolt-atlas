# Spec: Four post-loop batches

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

Lookups are scheduled by family, not by what they need. ReLU, Clamp, the trig tables and Gather
run their read sumcheck and their one-hot check inside the node's turn, one small sumcheck
each. The clamped activations, and softmax's stage 3, are deferred into per-family batches,
each followed by its own one-hot batch. The packed buckets add five more batches. After
`specs/one-range-check.md` the post-loop phase is still eleven batched sumchecks in a
hand-ordered list, and `ProofType` has a tag per family. Only two ordering facts are real: a
one-hot check reads the `ra` claim its read sumcheck produces, and a digit range check reads
the value its value sumcheck produces. This spec schedules by those two facts: after the node
loop the verifier runs four batched sumchecks, in a fixed order, whatever the model.

## Intent

### Goal

Replace the per-family deferred batches and the in-loop lookup sumchecks with four post-loop
batches, `Values`, `Digits`, `Lookups` and `OneHot`, and retire the per-family `ProofType`
tags.

| Batch | Members | Reads |
|---|---|---|
| `Values` | each clamp bucket's split or exact-value sumcheck; each remainder bucket's value sumcheck | claims appended in the node loop |
| `Digits` | each bucket's digit range check | the value openings `Values` produces |
| `Lookups` | every table read whose sumcheck produces only its own `ra`: ReLU, Clamp, activation clamp, small activation table, trig downscale, trig table, softmax exp-hi, exp-lo and significance clamp | claims appended in the node loop |
| `OneHot` | the one-hot checks of every `Lookups` member and of Gather; Gather-small's Hamming-weight check; the clamp buckets' slack chunk checks | the `ra` openings `Lookups` and the in-loop Gather read produce; `IND` and `S` from `Values` |

Gather's read sumcheck stays in the node loop: it produces a claim on its dictionary operand,
which must exist before that operand's evaluation reduction.

Introduced or modified:

- `jolt-atlas-verifier`: `deferred_lookups.rs` (`DeferredBatch` becomes the four batches;
  `plan` emits `Defer` steps instead of `Batch` steps), `verifier.rs` (`run_deferred` drains
  four queues), `types.rs` (four new `ProofType` tags; thirteen retired), `plan.rs` (no enum
  change; the lookup arms are unchanged), and the plans of `relu.rs`, `clamp.rs`, `sin.rs`,
  `cos.rs`, `gather/small.rs`, `gather/large.rs`, `activation_clamped/mod.rs`,
  `softmax_last_axis/mod.rs`.
- `jolt-atlas-core`: `deferred_lookups.rs`, `ops::prove_deferred_batches`.

### Invariants

Relied on, not changed:

1. **Every relation.** No relation's expression, setup, challenges or produced openings change;
   only the batch it is proven in and the transcript position of that batch.
   *Mechanism:* review on the diff, which touches plans and the batch enum, not relations.
2. **Setup timing.** A member that reads only node-loop claims is set up in its node's turn
   (`DeferredSetup::InNodeLoop`); a member that reads another batch's output is set up when
   its batch runs (`AtBatch`). *Mechanism:* a missing claim at setup is an `Err` in the
   verifier and an `expect` in the prover, so a wrong assignment fails every test that
   reaches it.
3. **Batched sumcheck semantics.** Members are tail-aligned, the batch degree is the maximum
   member degree, and a member cannot read an opening another member of its batch produces.
   *Mechanism:* `atlas_sumcheck::relation::verify_relation_batch`, unchanged.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. Fixed order.** `run_deferred` runs `Values`, `Digits`, `Lookups`, `OneHot` in that
  order for every model; a batch with no member has no proof entry. *Mechanism:*
  `DeferredBatch::ALL`; the drain check rejects a stray entry.
- **B. Dependency closure.** Every opening a member reads at `AtBatch` setup is produced by an
  earlier batch or by the node loop: `Digits` reads `Values`; `OneHot` reads `Lookups`,
  `Values` and the in-loop Gather read. *Mechanism:* invariant 2's failure mode, exercised by
  `proof_types_all_exercised` (every batch non-empty in some checked-in model) and the zoo.
- **C. Operand claims land before their reduction.** A deferred lookup appends its operand
  claim (`NodeOutput(input)`) during its node's turn, as the activation clamp does today, never
  at batch time. *Mechanism:* the accumulator rejects an append to a node after its evaluation
  reduction; an op-surface case per moved family.
- **D. One-hot checks are independent of each other.** No `OneHot` member reads another's
  output; Gather-small's three checks read only the index operand's point.
  *Mechanism:* review-held; the gather op-surface cases.
- **E. No per-family tag survives.** `ProofType` keeps the node-level tags and gains four.
  *Mechanism:* the compiler; `proof_types_all_exercised`; retired tags fail to deserialize.

### Non-Goals

- Any change to a relation, a table, or the one-hot check (`shout::ra_onehot_verifiers`).
- Folding the Hamming-weight check into the read sumcheck, or four-factor virtual selectors.
- Fusing the activation clamp into the table lookup (`docs/prover-plan.md` item 6, B2).
- Deferring node relations other than lookups; the evaluation reduction
  (`specs/digits-as-trace.md`).
- Softmax stages 1 and 2, which stay in-loop batches of the node.
- Any change to trace semantics or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] `DeferredBatch` has exactly four variants and `run_deferred` has no other batch.
- [ ] `deferred_lookups::plan` emits no `Step::Batch`.
- [ ] No plan of ReLU, Clamp, Sin, Cos, the activations or softmax stage 3 contains an in-loop
      lookup sumcheck or one-hot batch; Gather keeps exactly one in-loop `Step::Relation`.
- [ ] `ProofType` has four new tags; tags 2, 3, 14, 15, 16, 18, 19, 20, 23, 24, 25, 26 and 27
      are in the deleted lists of `types.rs` and `docs/verifier.md`.
- [ ] `proof_types_all_exercised` passes with all four batches non-empty across the checked-in
      models.
- [ ] Invariant C: one op-surface case each for ReLU, Clamp, Sin, Cos and an activation where
      the lookup's operand has a second consumer, so its evaluation reduction has two claims.
- [ ] The three sweeps pass over the probe models with no tolerance entry; one `mutate::Site`
      per new proof entry.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The gate and bounds of the Performance section hold, or the fallback there is taken and
      recorded.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the op-surface rows of every lookup
op and of the bucket cases; `exact_output_bucket_commits_nothing` and
`saturating_probe_saturates` (which name the tags they expect and are updated to the new
ones); the three sweeps; the zoo with its serialization round-trip; `fuzz_regressions_replay`,
its corpus regenerated in the digest commit.

New tests: the five two-consumer cases of invariant C; a test that a model with no lookup and
no clamped node carries no post-loop proof entry.

CI coverage: the workspace job; the heavy job for GPT-2 and bge-small, since this is a
transcript-order change across every lookup. The `zk` feature is not touched.

### Performance

Benchmark: the run table of `docs/measurement-plan.md` on `gpt2 --seq-len 16` and
`bge-small-en-v1.5` against `bench/bin/prove-c65753bfa`.

Expected: the IOP stage flat or slightly down (more instances proven in parallel by
`prove_relation_batch_parallel`, fewer small sequential sumchecks); proof size and verify time
slightly down (fewer sumcheck proofs, fewer grinds). On GPT-2 and bge-small every lookup is
already deferred except Gather, so the movement there comes from merging batches.

**Gate: peak memory.** Merging the clamp and remainder digit range checks into one `Digits`
batch keeps both sets of prover instances alive together, and GPT-2 already peaks at about
11 GB on a 16 GB reference machine. Peak RSS on `gpt2 --seq-len 16` must not exceed the
baseline by more than 5%. If it does, `Digits` is split into two batches by bucket kind (five
batches), the same test is applied to `Values`, and the PR records the measurement. Deferring
ReLU, Clamp and trig lookups also holds their witnesses from the node's turn to the batch; the
same bound applies on a model that has them (`op_gauntlet` is the check, by peak RSS).

Bounds: prove within 3% of baseline or better on both models; verify within 3% or better;
proof size no larger.

## Design

### Architecture

```text
node loop (reverse topological):
    lookup op:   Reduce, [Open ...], Defer{Lookups, read}, Defer{OneHot, one-hot checks}
    gather:      Reduce, Relation{Execution, read}, Defer{OneHot, one-hot checks}
    clamped op:  as today (Open(ClampAcc), relation, ...)
post-loop:
    per live clamp bucket: Open(ClampExactFlag), Tie(ClampBucketExact)
    Values   split / exact-value per clamp bucket, value per remainder bucket
    Digits   digit_range per bucket
    Lookups  every deferred read
    OneHot   every one-hot check, the slack chunk checks
```

`DeferredBatch::setup` keeps its two answers: `Values` and `Lookups` members are
`InNodeLoop` (bucket members are set up at the start of `run_deferred`, before `Values` runs,
which is the same transcript position for all of them); `Digits` and `OneHot` are `AtBatch`.
`deferred_lookups::plan` becomes the list of `Open`, `Tie` and `Defer` steps for the buckets,
run through the same interpreter before the queues drain.

Degrees: `Values` 4 (split) and 2; `Digits` `DIGIT_BASE + 1`; `Lookups` 2 throughout; `OneHot`
up to 9 (the ra-virtualization of a 32-bit address in eight chunks) with Booleanity at 3. The
batch already mixes these two degrees today inside one node's one-hot batch.

Prover: `prove_deferred_batches` proves each of the four with
`prove_relation_batch_parallel`; `Prover::lookup_addresses` keeps handing read addresses from a
read to its one-hot check, keyed as today.

### Alternatives Considered

- **Two batches (`Lookups`, `OneHot`) and leave the bucket batches alone.** Rejected: the
  bucket batches follow the same two ordering facts, and leaving them keeps five tags and a
  second scheduling mechanism (`Step::Batch` in the post-loop plan).
- **One batch for everything.** Impossible: a member cannot read an opening produced in its own
  batch.
- **Defer Gather's read.** Rejected: it produces an operand claim.
- **Keep in-loop lookups in the loop.** Rejected: they have no ordering reason to be there, and
  they are the lookups that get no cross-node batching today.
- **A per-round degree schedule so low-degree members do not pay the batch degree in proof
  size** (a16z/jolt #2336). Deferred: an engine change; revisit if the proof-size bound fails.

## Documentation

- `docs/verifier.md`: the "Plans" section's post-loop paragraph is rewritten around the four
  batches; the deleted-tag list grows.
- `docs/pull-requests.md` §5: a new lookup op defers its read to `Lookups` and its one-hot
  check to `OneHot`.
- Module docs of `deferred_lookups.rs` in both crates.
- `README.md`: the benchmark table if a row moves outside noise.

## Execution

Suggested order: (1) the four-variant enum with the existing deferred families mapped onto
`Lookups` and `OneHot`; (2) the bucket batches onto `Values`, `Digits` and `OneHot`, with the
memory gate measured here; (3) the in-loop families, one op per commit; (4) tag retirement and
the digest re-pin.

## References

- `jolt-atlas-verifier/src/onnx_proof/deferred_lookups.rs`, `verifier.rs` (`run_deferred`),
  `plan.rs`; `atlas-sumcheck/src/relation/batch.rs`; `atlas-sumcheck/src/shout/mod.rs`
  (`ra_onehot_verifiers`).
- `specs/verifier-audit-simplification.md` (the plan interpreter and the single deferral queue).
- a16z/jolt #2336 (per-round degree schedule).
