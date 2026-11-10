# Spec: Dense-digit range check for the clamp output chunks

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-10                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | —                              |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every clamped element (the output of each einsum, Add, Sub, Sum, Mul, Square, Cube and
MeanOfSquares) is range-checked by committing `OUT = out + 2^31` as 8 one-hot chunks (K = 16) in
its clamp bucket. These output chunks are about 71% of the digit rows the prover commits on
GPT-2, and the Akita commit is 59% of GPT-2 proving time (`docs/prover-plan.md` §1). Akita
charges about 144 ns per committed one-hot row against 25–37 ns for the same 4-bit digit held as
two dense base-4 values, and row sparsity cannot help because every row holds a digit. The
one-hot encoding is what Shout needs for a lookup address, but the clamp buckets run no lookup:
a split sumcheck and a Booleanity chunk check. This spec replaces the one-hot output chunks with
dense digit columns, a zero-check that each digit is in range, and the linear recomposition of
`OUT`. It is the first protocol change of the prover plan (item 1) and fixes the encoding that
the remainder buckets, the row check and the digits-as-trace IOP build on.

## Intent

### Goal

Replace the 8 one-hot `OUT` chunks of every live clamp bucket with dense base-`b` digit columns
(`U8Scalars`; 16 columns at `b = 4`), range-checked by a new reusable relation
`atlas_sumcheck::digit_range` that carries `OUT`'s recomposition, leaving the slack chunks, the
virtual indicator `IND` and the split and exact-value sumchecks as they are.

Introduced or modified:

- `atlas-sumcheck`: new module `digit_range` (relation and verifier; prover in its `prover`
  submodule) and a `RelationId::DigitRange`.
- `common`: `CommittedPoly::GlobalClampOutDigit(bucket, j)` added; `GlobalClampOutRaD` removed.
- `jolt-atlas-verifier`: `onnx_proof/clamp_split.rs` (chunk list and check builders),
  `onnx_proof/plan.rs` (`OpRelation` variants), `onnx_proof/deferred_lookups.rs` (plan steps),
  `types.rs` (`ProofType`).
- `jolt-atlas-core`: `onnx_proof/clamp_split.rs` (`BucketSplit::witness_polys`), the deferred
  prover, `examples/prove.rs` (census).

Also in scope, as cleanup of the code this change touches:

- Delete the unreachable `CommittedPoly::ClampRaD`, `RescaleRemainderRaD` and `GlobalClampRaD`
  variants and their arms.
- Correct the comments that still describe a read-raf clamp (`ops/add.rs`, `ops/sub.rs`,
  `ops/sum/mod.rs`, `clamp_lookups/mod.rs`, `types.rs` in the verifier crate).
- Build the bucket witnesses once at commit time and reuse them in the deferred prover (today
  they are built in `global_clamp::bucket_witnesses` and again in `deferred_lookups::prove_all`;
  this covers the remainder buckets' build as well, whose encoding does not change).
- Correct the behavioral-contract comment in `atlas-trace/src/lib.rs` and its copy in
  `CLAUDE.md`: the function is `sat_clamp_i32`, and Add, Sub and Sum are clamped too.

### Invariants

Relied on, not changed:

1. **Clamp semantics.** `output = sat_clamp_i32(quotient)` (the fused-rescale clause of the
   `atlas-trace` contract) and the per-step saturation of Add/Sub/Sum. Model outputs do not move.
   *Mechanism:* the importer's golden hashes (`atlas-onnx-import/tests/golden_zoo.rs`) are not
   re-pinned by this PR; the zoo tests compare against the trace.
2. **Padding.** A padded row of a node (`out = 0`) and a gap row of a bucket both hold
   `OUT = 2^31` (`BucketSplit::assemble` fills gaps with the offset), whose digits
   (`(2, 0, …, 0)` at `b = 4`) satisfy the zero-check with no special case. *Mechanism:* `padded_add_1000` and
   `clamp_bucket_packing_mixed_sizes` in `op_surface.rs`, the zoo tests.
3. **Commit before challenge.** The digit columns are committed before the node points `r_n`
   and the batching challenges are drawn, as the chunks are today; the Schwartz–Zippel step of
   the split argument depends on it. *Mechanism:* review-held; `testing/digests.rs` pins the
   transcript so an unintended reordering is visible.
4. **Split and exact-value relations.** `SplitParams` (`RelationId::ClampSplit`) and
   `ValueParams` (`RelationId::PackedValue`) keep their expressions and their `OUT(r')` opening.
   *Mechanism:* review-held on the diff; the soundness sweeps.
5. **Slack side.** Slack chunks stay one-hot; `IND` stays virtual (the Hamming weight of the
   slack chunks); an all-empty slack column commits nothing. *Mechanism:* review-held; the census
   reports committed rows per family.

New:

- **A. Digit range.** An accepted proof implies every committed `OUT` digit is in `[0, b)` on
  every row of its bucket, and there are exactly `32 / log2(b)` digits, hence `OUT ∈ [0, 2^32)`
  and `out ∈ i32`; the saturation identity `IND·OUT·(OUT − M) = 0` depends on this bound.
  *Mechanism:* a new `witness_tamper_sweep` site that (i) moves a digit out of range while
  keeping the recomposition true (`d_j + b`, next more significant digit `− 1`), and (ii) raises
  the most significant digit by `b`, so the recomposed `OUT ≥ 2^32`, with the rest of the witness
  made consistent with that `OUT`. Both must be rejected with an `Err`. The relation-level test
  in `atlas-sumcheck` covers the same property in isolation.
- **B. Recomposition binding.** The `OUT(r')` opened by the split or exact-value sumcheck equals
  `Σ_j b^(n−1−j)·d_j(r')` over the committed digit columns, and each digit column is opened
  exactly once. *Mechanism:* `opening_forgery_sweep` and `proof_mutation_sweep` over the new
  openings and the digit-range proof, one `mutate::Site` per new proof field; the verifier's
  drain check.
- **C. The saturating path is under the sweeps.** A bucket that saturates carries dense `OUT`
  digits and one-hot slack chunks together, and is checked by both halves. *Mechanism:* a probe
  model in `testing/models.rs` with a saturating internal non-scalar clamped node, and a test
  asserting that it saturates (today no probe is asserted to; among checked-in and real models
  only one GPT-2 Cube saturates).
- **D. No one-hot residue on `OUT`.** Nothing reads or commits the old output chunks.
  *Mechanism:* the compiler; `GlobalClampOutRaD` is deleted.
- **E. Chunk check split.** The Booleanity sumcheck with its linear term splits in two: the
  digit-range check carrying `OUT`'s recomposition (every live bucket), and Booleanity over the
  slack chunks carrying Hamming weight `= IND` and `S`'s recomposition (non-exact buckets only).
  Both run in the deferred plan; "every chunk is opened once" holds per half. *Mechanism:* the
  drain check; `proof_types_all_exercised` (both tags emitted by a checked-in model, the slack
  one by the probe of invariant C).
- **F. One relation, one base constant.** The relation is parameterised by the digit base `b`
  (zero-check `∏_{a<b}(d − a)`, recomposition weight `b^j`), `b` is a single constant, and the
  relation takes its setup (digit opening ids, point, recomposed opening) from the caller and
  knows nothing about buckets. *Mechanism:* review-held; the relation-level test runs both
  `b = 4` and `b = 2`.
- **G. Class layout.** All digit columns of one arity share one dense Akita class, and the
  total number of opened Akita classes does not rise. *Mechanism:* the acceptance measurement on GPT-2
  (class and opening counts from `--stats` against the baseline binary).
- **H. A bucket with nothing to check costs nothing.** A bucket whose nodes are all unsaturated
  model outputs adds no committed cells and no openings. An all-zero dense class is skipped by
  Akita, but an all-zero column inside a nonzero class is not free, so this must hold whatever
  class the bucket's arity falls in. The committed-poly list therefore depends on the public IO:
  `global_clamp::bucket_committed_polys` leaves out every clamp bucket with no live node, with
  liveness derived from the IO by `clamp_split::exact_output_verifier` as
  `deferred_lookups::plan` already does, and the prover commits nothing for such a bucket
  (`BucketSplit::empty_witness_polys` is deleted). *Mechanism:* a named test asserting such a
  bucket is absent from the committed-poly list (see Testing Strategy).
- **I. Verifier hygiene.** The verifier rejects with `Err`, never panics, and consumes the proof
  exactly; `jolt-atlas-verifier` stays Trace-free and prover-free. *Mechanism:* the crate's
  lints and CI path checks, the mutation sweep with no tolerance entry, the drain check.

Bucket relations are model-level: they implement `SymbolicSumcheck`, `Produces` and `Relation`
and carry no `type Spec` (that binding exists only on node-level `AtlasRelation`s), so
`relation_specs_match_their_dispatched_op_kind` does not apply to the new relation.

The field is Fp128; `∏_{a<b}(d − a)` has exactly the roots `0..b` and `Σ_j b^j·d_j < 2^32` never
wraps. Review-held.

### Non-Goals

- The remainder buckets (prover-plan item 2); invariant F is the only concession to them.
- The row check on wide rows (item 3) and declared per-op clamp widths (item 4).
- Any change to the slack chunks, `IND`, or the split and exact-value relations.
- The per-node range checks of Rsqrt, Div, MeanOfSquares, Teleport (item 7), and the activation
  and softmax clamps.
- The Akita class-layout rework (item 8) and any change in `atlas-pcs` beyond what dense
  `U8Scalars` columns already get.
- Any change to trace semantics, `SemVersion`, model outputs or golden hashes.
- BGE and Qwen real-model runs as merge requirements (GPT-2 is the required run).

## Evaluation

### Acceptance Criteria

- [ ] Each live clamp bucket commits dense `U8Scalars` digit columns for `OUT` (16 at `b = 4`)
      and no one-hot output chunks; `CommittedPoly::GlobalClampOutRaD` no longer exists.
- [ ] One relation, `atlas_sumcheck::digit_range`, parameterised by a single base constant,
      proves `∏_{a<b}(d − a) = 0` for every digit column and carries `OUT`'s recomposition; the
      slack half keeps Booleanity, Hamming weight `= IND` and `S`'s recomposition; both run in
      the deferred plan.
- [ ] Base 4 and Booleans are both measured on `gpt2-tiny` and GPT-2 by total proving time; the
      PR states which ships, with the numbers, and `docs/prover-plan.md` decision 3 records it.
- [ ] All digit columns of one arity share one dense Akita class, and the total number of
      opened Akita classes on GPT-2 does not rise against `bench/bin/prove-25c7218f1`.
- [ ] A bucket whose nodes are all unsaturated model outputs adds no committed cells and no
      openings: it is absent from the committed-poly list (the named test of invariant H).
- [ ] Invariant A: the new tamper site, in both forms (in-range recomposition with an
      out-of-range digit; most significant digit raised by `b`), is rejected with an `Err`.
- [ ] Invariant B: one `mutate::Site` per new proof field; `proof_mutation_sweep` and
      `opening_forgery_sweep` reach the digit-range proof and the digit openings with no
      tolerance entry.
- [ ] Invariant C: a probe model has a saturating internal non-scalar clamped node, a test
      asserts that it saturates, and the three sweeps run over it.
- [ ] `proof_types_all_exercised` passes with the digit-range tag and the slack chunk-check tag
      both emitted by checked-in models; any retired tag is in the deleted list in
      `docs/verifier.md`.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit.
- [ ] `CommittedPoly::ClampRaD`, `RescaleRemainderRaD` and `GlobalClampRaD` and their arms are
      deleted.
- [ ] No comment in the verifier's `ops/add.rs`, `ops/sub.rs`, `ops/sum/mod.rs`,
      `clamp_lookups/mod.rs` or `types.rs` describes a read-raf clamp.
- [ ] The bucket witnesses are built once per proof and reused by the deferred prover.
- [ ] The contract comment in `atlas-trace/src/lib.rs` and its copy in `CLAUDE.md` name
      `sat_clamp_i32` and list Add, Sub and Sum among the clamped ops.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 proves and verifies under the `heavy` profile (`zoo::gpt2`), and one manually
      dispatched exhaustive soundness run (`JOLT_SOUNDNESS_EXHAUSTIVE=1`, the
      `workflow_dispatch` job in `rust.yml`) passes on the PR head.
- [ ] The gate and the bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing:

- The workspace run. The op-surface cases that reach the split today:
  `add_saturating_overflow_1k`, `mul_saturates_pos`/`neg`, `sum_saturating_rows`,
  `einsum_mk_kn_mn_saturates_pos`/`neg`, `clamp_bucket_packing_mixed_sizes`, `padded_add_1000`.
- `proof_mutation_sweep`, `witness_tamper_sweep` and `opening_forgery_sweep` over the probe
  models, with no tolerance entry added.
- The checked-in zoo (each case round-trips the proof through serialization) and
  `proof_types_all_exercised`.
- The importer's zoo gates and golden hashes, unchanged.
- `fuzz_regressions_replay`; if the changed proof format no longer parses its corpus, the corpus
  is regenerated in the digest re-pin commit.

New tests:

- Relation-level, in `atlas-sumcheck`: honest digits verify; an out-of-range digit with a true
  recomposition is rejected; a false recomposition is rejected; for `b = 4` and `b = 2`. Ground
  truth is the algebraic property, not a second prover.
- If the prover is a specialized `RelationWitness`, its round messages are compared against the
  engine's `RelationProver` on random inputs.
- The tamper site of invariant A in `witness_tamper_sweep`.
- One `mutate::Site` per new proof field.
- The saturating probe model with its saturation assertion (invariant C).
- The nothing-to-check bucket case (invariant H).

CI coverage: the workspace job covers all of the above; the heavy job (label `ci:heavy`) covers
GPT-2; the exhaustive soundness job is dispatched by hand once before merge. The `zk` feature is
not touched.

### Performance

Benchmark: GPT-2 at 16 tokens with `bench/phase1.sh` (the Phase 1 run table of
`docs/measurement-plan.md`) against the baseline binary `bench/bin/prove-25c7218f1`, on the same
machine in the same session. Baseline figures are in `docs/prover-plan.md` §1.

**Gate (pass/fail):** total time per clamp digit, the Akita commit of the output digits plus the
digit-range and split sumchecks, is below the baseline's (commit of the output chunks plus the
Booleanity and split sumchecks), and the Booleanity sumcheck's time drops to its slack-only
share.

Expected movement (estimates; the run measures them):

| Stage | Baseline | Expected |
|---|---|---|
| Akita commit | 23.6 s | 10–11.5 s |
| Total prove | 40.2 s | 27–29 s |
| Deferred sumchecks | 3.1 s | flat to slightly down |

The deferred-sumcheck row is the uncertain one: Booleanity over 8 one-hot chunks is replaced by
a degree-5 zero-check over 16 dense columns (`∏` is degree 4 in the digit, plus eq), or degree 3
over 32 columns with Booleans. It decides the base.

Bounds on what should not move:

- Opening stage (reduction sumcheck plus class openings) within 5% of baseline.
- Verify time within 5% of baseline.
- Proof size within 2% of baseline.
- Peak memory no higher than baseline (the README reference machine has 16 GB).

`gpt2-tiny` is the completeness and direction check during development; the commit is not
dominant there, so it does not stand in for the gate.

## Design

### Architecture

**`atlas-sumcheck::digit_range`** (new; laid out as `booleanity` and `hamming_booleanity` are).
For digit columns `d_0..d_{n−1}` (most significant first), a point `r'` and an opening `V(r')`
supplied by the caller:

```text
Σ_t eq(r', t) · [ Σ_j γ^j · ∏_{a<b}(d_j(t) − a)  +  β · Σ_j b^(n−1−j) · d_j(t) ]  =  β · V(r')
```

Degree `b + 1`. `γ` and `β` are drawn after `V(r')` is fixed and the digits are committed, so the
identity gives `Σ_t eq(r', t)·∏(d_j − a) = 0` for every `j` and the recomposition at `r'`; `r'`
is itself random after commitment, which gives the zero-check pointwise. Folding the
recomposition into the zero-check (as `LinearTerm` does for Booleanity today) opens each digit
column once. The module header states this argument. The relation is a `SymbolicSumcheck` over
atlas openings with a derived `RelationVerifier`; no hand-written verifier.

**`common`**: `CommittedPoly::GlobalClampOutDigit(bucket, j)`; four variants removed.

**`jolt-atlas-verifier`**:

- `clamp_split.rs`: `chunk_polys` lists the digit columns, then the slack chunks. The
  chunk-check builders split into a digit-range half (for split and exact buckets alike, with
  `V = OUT`) and a slack Booleanity half (non-exact buckets).
- `plan.rs`: `OpRelation::ClampBucketChunks` becomes a digits relation and a slack-chunks
  relation.
- `deferred_lookups::plan`: each is its own `Step::Batch` after the split batch; the slack batch
  is omitted when no live bucket is non-exact.
- `types.rs`: a new `ProofType` tag for the digit-range batch (tags are not reused); the
  existing chunk-check tag keeps the slack half.
- `global_clamp.rs` / `preprocessing.rs`: the committed-poly list becomes IO-aware
  (invariant H): `bucket_committed_polys` takes the public IO and leaves out clamp buckets with
  no live node.

**`jolt-atlas-core`**:

- `clamp_split.rs`: `BucketSplit::witness_polys` emits `U8Scalars` digit columns; the
  digit-range prover instance.
- `prover.rs` / `deferred_lookups.rs`: the witnesses built at commit time are kept on the
  `Prover` and reused.
- `examples/prove.rs`: the census counts the new columns.

**`atlas-pcs`**: no change expected. `U8Scalars` routes to the dense unsigned class, prefix-packed
per arity. To confirm during implementation: buckets below the dense minimum arity (exercised by
the tiny zoo).

**`atlas-trace`, `atlas-onnx-import`**: no behaviour change; the contract comment only.

```text
node:      ... Open(ClampAcc) ... op relation            (unchanged)
deferred:  per live bucket: Open(ClampExactFlag), Tie(ClampBucketExact)
           Batch[DeferredClampSplit]        split / exact-value  →  OUT(r'), S(r'), IND(r')
           Batch[<digit-range tag>]         digit_range(V = OUT)  →  d_j(r'')      (every live bucket)
           Batch[DeferredClampChunkChecks]  Booleanity + HW = IND + S recomposition (non-exact only)
           remainder batches                                       (unchanged)
```

### Alternatives Considered

- **Keep the one-hot chunks.** Rejected: about 144 ns per row against 25–37 ns, and every row
  holds a digit, so sparsity cannot help.
- **Booleans instead of base 4.** Not rejected: 32 columns at degree 3 against 16 at degree 5,
  57–69 ns against 25–37 ns per 4-bit digit to commit. Decided by measured total time; a
  one-constant change by invariant F.
- **One dense base-16 column per chunk.** Cheapest to commit (12 ns), but its zero-check is
  degree 16 in the digit, the cost class that ra-virtualization was removed to avoid.
- **Dense slack with a committed or `OUT`-derived `IND`.** Rejected: empty one-hot slack columns
  are free today and almost every bucket is exact; a dense slack pays cells on every row of a
  saturating bucket and changes the split relation.
- **Recompose `OUT` inside the split sumcheck**, with a separate zero-check. Rejected: every
  digit column is opened twice, adding claims to the opening reduction.
- **A lookup argument for the range** (LogUp, or Shout on a small table). Rejected: the bucket
  checks run no lookup today, and a zero-check on a tiny digit needs no table.
- **Declared widths or the row check first.** Rejected as the first step: both build on this
  encoding, and declared widths are a `SemVersion` decision.
- **A different commitment for the range digits (BitZ).** Decided against on 2026-11-10
  (`docs/prover-plan.md` §5); revisit after this lands and the commit is re-measured.
- **Upstream `a16z/jolt`.** Range checks there go through one-hot Shout lookups; its 8-bit
  one-hot experiment was slower on Akita (#1994). The published dense-digit form is SuperNeo.

## Documentation

- `README.md`: the GPT-2 benchmark table re-measured (stage rows, proof size, peak memory).
- `docs/prover-plan.md`: item 1 marked landed with its measured numbers; decision 3 recorded.
- `docs/measurement-plan.md`: the gate run as a results section, logs under `bench/runs/`.
- `docs/verifier.md`: the clamp sections and the type table updated to the digit-range relation;
  any retired `ProofType` tag added to the deleted list.
- Module docs: the header of `clamp_split.rs` (both crates) rewritten for dense `OUT` digits; a
  header on `atlas_sumcheck::digit_range` with its soundness argument.
- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the contract wording.
- No change to `docs/pull-requests.md` or `CONTRIBUTING.md`: the new-relation checklist applies
  as written.

## Execution

Left to the implementer.

## References

- `docs/prover-plan.md` (§1 measurements, §3 item 1, §5 decision 3) and
  `docs/measurement-plan.md` (Phase 1 run table, Phase 3 commit benchmark).
- `atlas-pcs/examples/commit_bench.rs`; `bench/runs/commit-bench-2^20x64.txt`,
  `bench/runs/census-gpt2-*.txt`.
- `jolt-atlas-verifier/src/onnx_proof/clamp_split.rs` (the current split protocol and its
  soundness argument); `atlas-sumcheck/src/booleanity` (`LinearTerm`) and
  `atlas-sumcheck/src/hamming_booleanity` (the nearest zero-check).
- Neo / SuperNeo, ePrint 2026/242 (dense digit decomposition with a zero-check).
- Akita, ePrint 2026/1983; a16z/jolt #1994 (8-bit one-hot digits on Akita).
- BitZ, ePrint 2026/2141 (the alternative kept in view).
