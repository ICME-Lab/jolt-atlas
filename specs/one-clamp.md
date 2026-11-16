# Spec: One clamp (exact arithmetic, checked range)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`, `specs/post-loop-batches.md`, `specs/log-partition-softmax.md` (`SemVersion::V2`), `specs/fused-gelu.md` (no saturating node left) |
| Supersedes  | `docs/prover-plan.md` §5 decision 2 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

**Decision (2026-11-16).** Saturating arithmetic is replaced by partial-function semantics: a
value that leaves `i32` is an execution error, not a clamp. This holds for the eight
saturating ops and for `Neg`.

## Summary

Add, Sub, Sum, Einsum, Mul, Square, Cube and MeanOfSquares compute in `i64` and emit
`sat_clamp_i32` of the result. Proving that sentence is the largest single mechanism in the
verifier. Each node opens a second polynomial, its pre-clamp accumulation; each bucket commits
one-hot slack chunks beside its output digits; a degree-4 split sumcheck separates interior
from saturated elements through a virtual indicator; a Booleanity check ties the indicator and
the slack to the chunks; the prover may declare a bucket exact, which switches it to a
different sumcheck and adds a tie per member; and a model output that happens not to saturate
is tied directly, which makes the list of committed polynomials depend on the public IO.
Buckets are grouped by a static width of 40, 48, 56 or 64 bits that exists only to size the
slack. Since the fused activations removed GPT-2's Cube, no model we run saturates anywhere:
all of it serves an event that does not occur. With exact arithmetic the statement becomes
"the result is the output, and the output is an `i32`", which the dense digit range check on
the output already proves.

`Neg` has the same defect in a smaller form: its kernel saturates at `i32::MIN` while its
relation is an exact tie, a completeness gap pinned today as
`known_gap_neg_i32_min_saturates`. The same decision closes it.

## Intent

### Goal

Change the semantics of the eight clamped ops and of `Neg`, under `SemVersion::V3`, so that a
result outside `i32` is a `KernelError`, and reduce the clamp protocol to the output's digit
range check: no pre-clamp accumulation, no slack, no indicator, no split sumcheck, no exact
flag, no IO-dependent commitment, no width ladder.

Introduced or modified:

- `atlas-trace`: `SemVersion::V3`, the default of `CompileOptions`; `sat_clamp_i32` replaced
  by a checked narrowing that returns `KernelError::WitnessOverflow` (the executor attaches
  the node, as for every kernel error); `linear.rs` (Add, Sub, Sum, Neg), `fused.rs`
  (`rebase_eval`), `einsum.rs`; `FusedIntermediates::quotient` removed (it equals the output);
  `ClampWidth`, `NodeMeta::sat_clamp_bits` and `annotate_clamp_widths` removed; the behavioral
  contract in `lib.rs`. The softmax kernel and the table generators accept `V3` as they
  accept `V2`.
- `jolt-atlas-verifier`: `check_model_shape` requires `V3`; `clamp_split.rs` reduced to the
  value sumcheck and the digit range builders; `clamp_lookups/` deleted; `global_clamp.rs`
  (one bucket kind; a new range-claim kind for `out + 2^31` at width 32, read from the node's
  reduced output opening, which today's claims cannot express since they bound advice openings
  only; model outputs are never members); `plan.rs` (`OpOpen::{ClampAcc, ClampExactFlag}`,
  `OpTie::{ExactClampOutput, ClampBucketExact}`, `OpRelation::{ClampBucketValue,
  ClampBucketSlack}` and `clamp_steps` deleted; `OpPlan::plan` and `deferred_lookups::plan`
  no longer read the IO for the clamp); `fused_rebase::input_expression`, which takes the
  reduced output where it takes `ClampAcc` (this one function covers Einsum, Mul, Square and
  Cube); `AccumulationRelation`, `SumAxisRelation` and `MeanOfSquaresRelation` likewise;
  `preprocessing.rs` (`get_models_committed_polynomials` loses its `io` argument);
  `types.rs` (`DeferredClampSplit` retired).
- `common`: `CommittedPoly::GlobalClampSlackRaD`, `VirtualPoly::{ClampAcc, GlobalClampSlack,
  GlobalClampInd, GlobalClampExact}` deleted.
- `jolt-atlas-core`: the mirrors; `clamp_intermediate` deleted; the `prove` example's census
  reports the output width needed, since nothing saturates.
- `atlas-onnx-import`: models compile at `V3`.

### Invariants

Changed:

1. **The clamp clauses of the `atlas-trace` contract.** "emit `sat_clamp_i32(quotient)`" and
   "Add and Sub saturate per step, Sum clamps once" become "the quotient, sum, difference or
   negation must lie in `i32`; otherwise execution fails". On every input where `V2` does not
   saturate, `V3` produces the same output. *Mechanism:* the importer's golden hashes are not
   re-pinned (no zoo model saturates); tracer tests that each former saturation input is now an
   error and its neighbour inside the range is unchanged.

Relied on, not changed:

2. **The Euclidean split** `acc = quotient·D + remainder`, `remainder ∈ [0, D)`, and the
   remainder claims. *Mechanism:* unchanged tests.
3. **The digit range check and the packed value sumcheck.** *Mechanism:* unchanged relations.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. Soundness of the reduced statement.** An accepted proof implies, for every clamped node
  that is not a model output, that `out + 2^31 ∈ [0, 2^32)` on every element and that the op's
  identity holds over the integers with `out` in place of the pre-clamp value
  (`Σ L·R = out·2^S + rem`, `out = left ± right`, the axis sum). Each relation's doc states its
  no-wrap bound: the magnitudes of both sides from the `i32` bounds on operands, output and
  remainder, against the field modulus. No relation states one today; the only bound in the
  code is the split protocol's, which rests on the slack width. *Mechanism:*
  `witness_tamper_sweep`, and a forged test per op that sets an output to `i32::MAX + 1` and to
  `i32::MIN − 1` with the op identity kept true.
- **B. An overflowing execution has no proof.** The tracer returns an error, so there is no
  witness; a prover that builds one anyway is rejected by A. *Mechanism:* op-surface rows for
  each former saturation case, declared as expected rejections at the trace.
- **C. Model outputs need no range check.** A model output is a public `i32` tensor, so it is
  never a bucket member, whatever its values. *Mechanism:* a test that the committed
  polynomial list is equal for two public IOs of the same model, one of which holds
  `i32::MAX`.
- **D. The committed set is a function of the model.** *Mechanism:* the compiler
  (`get_models_committed_polynomials` takes no IO) and the test of C.
- **E. One opening per clamped node.** No node has an advice opening for its pre-clamp value.
  *Mechanism:* the compiler (`VirtualPoly::ClampAcc` is deleted); the drain check.
- **F. Version binding.** The verifier requires `SemVersion::V3` and rejects `V1` and `V2`.
  *Mechanism:* the existing version test, extended.
- **G. The zoo still proves.** No checked-in model and no real model overflows on its
  benchmark inputs. *Mechanism:* the census line `saturated elements: 0` on the baseline
  commit (GPT-2 at 16 tokens reads 0 of 7,325,248 since the fused activations; bge-small and
  qwen2-tiny read 0 before them); the heavy job.
- **H. `Neg` is exact or absent.** `Neg` of `i32::MIN` is an execution error, and its tie
  holds on every execution that exists. *Mechanism:* `known_gap_neg_i32_min_saturates` becomes
  an expected-rejection row; the `Neg` relation's doc loses its completeness caveat.

### Non-Goals

- The lookup-table clamps: the `Clamp` op, the activation input clamps and the softmax address
  clamp. They are table semantics and stay.
- The wrapping forms: raw `Mul`, `Square` and `Cube` (`rebase: false`), and `Div` and
  `ScalarConstDiv` at `i32::MIN / −1`. They wrap today and are not range-checked; a separate
  change.
- Declared per-op widths narrower than 32 bits (`docs/prover-plan.md` item 4) and the row
  check (item 3). Both become simpler after this spec and are specified separately.
- Supporting `V2` beside `V3` in the prover or the verifier.
- Add and Sub with more than two operands, which the proof does not support today.

## Evaluation

### Acceptance Criteria

- [ ] The decision is recorded in `docs/prover-plan.md` §5 decision 2.
- [ ] The census reports zero saturating elements on GPT-2 (16 tokens and three texts),
      bge-small and qwen2-tiny at the baseline commit of this PR (invariant G).
- [ ] Invariant 1: the importer's golden hashes are untouched; the tracer's saturation-edge
      tests pass.
- [ ] `clamp_lookups/` does not exist in either crate; `clamp_split.rs` contains no split,
      slack, indicator or exact-flag item.
- [ ] The `plan.rs` variants and the `common` variants listed in the Goal do not exist;
      `ProofType::DeferredClampSplit` is retired and the one-hot batch has no slack member.
- [ ] `ClampWidth`, `sat_clamp_bits`, `sat_clamp_i32` and `FusedIntermediates::quotient` do
      not exist.
- [ ] `get_models_committed_polynomials` takes no `ModelExecutionIO`; the test of invariant C
      passes.
- [ ] Invariant A: both forged outputs are rejected for a node of each of the eight ops; each
      of the eight relations' docs states a no-wrap bound.
- [ ] Invariant B: every op-surface row that saturated is an expected trace-level rejection:
      `add_scalar_sat_pos`/`neg`, `add_saturating_overflow_1k`, `sub_scalar_sat_pos`/`neg`,
      `sub_saturating_overflow_1k`, `mul_saturates_pos`/`neg`, `mul_scalar_rescale_sat`,
      `square_saturates`, `cube_saturates_pos`/`neg`, `mean_of_squares_saturates`,
      `sum_saturating_rows`, `sum_scalar_saturating`,
      `einsum_mk_kn_mn_saturates_pos`/`neg`.
- [ ] The `saturating` probe model is removed and the `scalars` probe rebuilt without its
      saturating Add; `saturating_probe_saturates` and `scalars_probe_layout` follow.
- [ ] Invariant F: `V1` and `V2` models are rejected by the verifier with `Err`.
- [ ] Invariant H: the `Neg` row passes as an expected rejection and is no longer `#[ignore]`.
- [ ] `proof_types_all_exercised` and the three sweeps pass with no tolerance entry.
- [ ] `testing/digests.rs` is re-pinned in its own commit; the `fuzz_regressions_replay`
      corpus is regenerated there if it no longer parses.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile; Qwen in the CI heavy job.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the non-saturating op-surface rows
of the nine ops; `clamp_bucket_packing_mixed_sizes`, `padded_add_1000`; the three sweeps; the
zoo with its serialization round-trip; the importer's zoo gates and golden hashes.

Changed tests: the rows of invariant B; `exact_output_bucket_commits_nothing` becomes the test
of invariant C; the tests that read the fused quotient (`FusedQuotient` tamper sites,
`mean_of_squares_remainder_is_bounded`, `clamp_width_soundness`, the kernel goldens' trace
hash) are rewritten against the output; the clamp buckets' digit-tamper sites become range
bucket sites.

New tests: the forged outputs of invariant A; the tracer's edge tests of invariant 1; the
version test of invariant F.

CI coverage: the workspace job; the heavy job on all three real models, since this is a
semantics change. The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against this
spec's parent commit, on the benchmark machine.

Expected: commit flat (the slack columns are empty and free today; the output digits are
unchanged, 16 per element); the post-loop sumchecks down (the degree-4 split becomes the
degree-2 value sumcheck for every bucket, and the slack Booleanity disappears); proof size down
by one opening claim per clamped node and one flag per bucket; verify time down. Fewer
buckets, since all output claims share one width with the width-32 remainder claims.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better on both
models; peak memory within 5%; proof size no larger.

## Design

### Architecture

```text
before:  Reduce, Open(ClampAcc), [Tie(ExactClampOutput)], op relation on acc,
         post-loop: Open(flag), Tie(exact), split | exact-value, digit range, slack chunks
after:   Reduce, op relation on out,
         post-loop: value sumcheck (range claim out + 2^31, width 32), digit range
```

Every op relation that today has `acc` as its input claim takes the reduced output opening
instead; nothing else in the relation changes. `AccumulationRelation` for Add and Sub ties
`out(r) = left(r) ± right(r)` directly.

A clamped node that is not a model output contributes one range claim, the affine expression
`out(r_n) + 2^31` at width 32, to the range buckets. The claim is read from the node's reduced
output opening at the node's reduced output point; `clamp_split::exact_value_params` computes
this same combination today for buckets declared exact, so the new claim kind reuses that
path. The clamp bucket as a separate kind, with its own value relation and columns, is gone:
`GlobalClampOutDigit` and `GlobalRemainderDigit` become one digit family.

In `atlas-trace`, the kernels call a checked narrowing where they call `sat_clamp_i32`. Add
and Sub check each pairwise step, where saturation happened. `Neg` uses a checked negation.

### Alternatives Considered

- **Keep saturation, remove only the special cases** (the exact flag and the
  unsaturated-output tie), so every bucket runs the split, digit range and slack check and the
  committed set is model-only. Rejected with the decision: it keeps the pre-clamp opening, the
  slack chunks, the indicator, the split sumcheck and the width ladder for an event no model
  produces, and it commits output digits for large public outputs.
- **Partial-function for the eight ops only, leaving `Neg`.** Rejected at analysis: the same
  decision applied to `Neg` closes a pinned gap.
- **Saturate at a declared width** (`docs/prover-plan.md` item 4, first option). Rejected as a
  simplification: it keeps every mechanism and adds a per-op parameter.
- **Decompose the pre-clamp value itself** (64 bits of digits) and derive the output from the
  digits. Rejected: twice the committed digits, and a non-linear relation between digits and
  output.
- **Keep `V2` provable beside `V3`.** Rejected: both protocols would stay in the verifier.
- **Published systems.** FlashProve and Sable use partial-function semantics; total exact
  semantics is a property the plan lists as unclaimed, and this spec gives it up for the
  clamp.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the behavioral contract's two clamp clauses.
- `README.md`: the semantics statement (overflow is an error), and the benchmark table.
- `docs/verifier.md`: the "Plans" post-loop paragraph, the version check and the deleted-tag
  list.
- `docs/prover-plan.md`: decision 2 recorded; "total exact semantics" removed from the
  unclaimed positions; items 3 and 4 restated against the new base.
- `docs/op-roadmap.md`: "what adding an op requires" names the overflow error.
- Module docs of `clamp_split.rs` and `global_clamp.rs` in both crates, and the eight
  relations' S ⊆ R notes.

## Execution

Suggested order: (1) `SemVersion::V3` in `atlas-trace` with the checked kernels and `Neg`, the
saturating tests turned into rejections, the proof still on the split path with every bucket
exact; (2) op relations on `out`, with their no-wrap notes; (3) the new range-claim kind, the
single digit family and model outputs out of the buckets; (4) deletions in the verifier, core
and `common`; (5) the width ladder and `quotient` out of `atlas-trace`; (6) the digest re-pin
and docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/clamp_split.rs` (module header: the split protocol;
  `exact_value_params`), `clamp_lookups/mod.rs`, `global_clamp.rs`, `fused_rebase.rs`,
  `plan.rs` (`clamp_steps`, `fused_rescale_plan`), `ops/{add,neg,mean_of_squares}.rs`,
  `ops/sum/axis.rs`.
- `atlas-trace/src/semantics/fixed.rs`, `op/linear.rs`, `op/fused.rs`, `model.rs`
  (`ClampWidth`), `error.rs` (`KernelError`).
- `docs/prover-plan.md` §3 items 3 and 4, §5 decision 2.
- `specs/clamp-dense-digit-range-check.md` (invariant H, the IO-dependent commitment this
  spec removes).
