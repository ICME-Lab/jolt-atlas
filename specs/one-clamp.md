# Spec: One clamp (exact arithmetic, checked range)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`; a GELU that does not overflow (see Invariants, G) |
| Supersedes  | `docs/prover-plan.md` §5 decision 2 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

**Decision this spec assumes, not yet made by the maintainer.** It replaces saturating
arithmetic by partial-function semantics: a value that leaves `i32` is an execution error, not
a clamp. If the decision is to keep saturation, this spec is withdrawn; the fallback that keeps
saturation and removes only the special cases is recorded under Alternatives Considered.

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
slack. All of it serves an event that, across the checked-in zoo and the real models, occurs
on two elements of one GPT-2 Cube. With exact arithmetic the statement becomes "the result
is the output, and the output is an `i32`", which the dense digit range check on the output
already proves.

## Intent

### Goal

Change the semantics of the eight clamped ops, under a new `SemVersion`, so that a result
outside `i32` is a `KernelError`, and reduce the clamp protocol to the output's digit range
check: no pre-clamp accumulation, no slack, no indicator, no split sumcheck, no exact flag, no
IO-dependent commitment, no width ladder.

Introduced or modified:

- `atlas-trace`: `SemVersion::V2`; `semantics::fixed::sat_clamp_i32` replaced on the V2 path by
  a checked narrowing that returns `KernelError::Overflow { node }`; `linear.rs` (Add, Sub,
  Sum), `fused.rs` (`rebase_eval`), `einsum.rs`; `FusedIntermediates::quotient` removed (it
  equals the output); `ClampWidth`, `NodeMeta::sat_clamp_bits` and `annotate_clamp_widths`
  removed; the behavioral contract in `lib.rs`.
- `jolt-atlas-verifier`: `clamp_split.rs` reduced to the exact-value sumcheck and the digit
  range builders; `clamp_lookups/` deleted; `global_clamp.rs` (one bucket kind: a range claim
  on `out + 2^31` at width 32 joins the remainder claims; model outputs are never members);
  `plan.rs` (`OpOpen::{ClampAcc, ClampExactFlag}`, `OpTie::{ExactClampOutput,
  ClampBucketExact}`, `OpRelation::{ClampBucketValue, ClampBucketSlack}` deleted;
  `clamp_steps` deleted); every clamped op's relation takes the reduced output as its input
  claim; `preprocessing.rs` (`get_models_committed_polynomials` loses its `io` argument).
- `common`: `CommittedPoly::GlobalClampSlackRaD`, `VirtualPoly::{ClampAcc, GlobalClampSlack,
  GlobalClampInd, GlobalClampExact}` deleted.
- `jolt-atlas-core`: the mirrors; `clamp_intermediate` deleted.
- `atlas-onnx-import`: models compile at V2; golden hashes as stated below.

### Invariants

Changed:

1. **The clamp clauses of the `atlas-trace` contract.** "emit `sat_clamp_i32(quotient)`" and
   "Add and Sub saturate per step, Sum clamps once" become "the quotient, sum or difference
   must lie in `i32`; otherwise execution fails with `KernelError::Overflow`". On every input
   where V1 does not saturate, V2 produces the same output. *Mechanism:* a property test in
   `atlas-trace` running both versions on random inputs and asserting equal outputs or a V2
   error exactly where V1 saturates; the importer's golden hashes, which are not re-pinned for
   any fixture that does not saturate under V1.

Relied on, not changed:

2. **The Euclidean split** `acc = quotient·D + remainder`, `remainder ∈ [0, D)`, and the
   remainder buckets. *Mechanism:* unchanged tests.
3. **The digit range check and the packed value sumcheck.** *Mechanism:* unchanged relations.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. Soundness of the reduced statement.** An accepted proof implies, for every clamped node
  that is not a model output, that `out + 2^31 ∈ [0, 2^32)` on every element and that the op's
  identity holds over the integers with `out` in place of the pre-clamp value
  (`Σ L·R = out·2^S + rem`, `out = left ± right`, the axis sum). The bound on `out` and on the
  remainder is what keeps each identity from wrapping in Fp128; each relation's doc restates
  its no-wrap bound without reference to a slack. *Mechanism:* `witness_tamper_sweep` sites
  that set an output to `i32::MAX + 1` and to `i32::MIN − 1` with the op identity kept true;
  review of the eight S ⊆ R notes.
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
- **F. Version binding.** The verifier requires `model.version() == SemVersion::V2` and
  rejects any other with `Err`; today it reads no version. *Mechanism:* a verifier test.
- **G. The zoo still proves.** No checked-in model and no real model overflows on its
  benchmark inputs. GPT-2 does today: the Cube inside its GELU saturates on two elements. That
  Cube must be gone, or its input bounded, before this spec merges, by the fused GELU of
  `docs/op-roadmap.md` tier 1 or by an importer rewrite, in a separate change. *Mechanism:*
  the heavy job; the census's saturation count per model, which must be zero.

### Non-Goals

- The lookup-table clamps: the `Clamp` op, the activation input clamp and the softmax
  significance clamp. They are table semantics and stay.
- `Neg`, `Div`, `Rsqrt` and the other unclamped ops.
- Declared per-op widths narrower than 32 bits (`docs/prover-plan.md` item 4) and the row
  check (item 3). Both become simpler after this spec and are specified separately.
- The GELU change of invariant G.
- Supporting V1 and V2 side by side in the prover. V1 is removed from the proving path; its
  tracer code is deleted once the property test of invariant 1 has served.

## Evaluation

### Acceptance Criteria

- [ ] The maintainer's decision for partial-function semantics is recorded in
      `docs/prover-plan.md` §5 decision 2.
- [ ] The census reports zero saturating elements on GPT-2 (three texts), bge-small and
      qwen2-tiny at the baseline commit of this PR (invariant G).
- [ ] Invariant 1: the V1/V2 property test passes; the golden hashes of non-saturating
      fixtures are untouched, and any re-pinned hash is listed in the PR with the reason.
- [ ] `clamp_lookups/` does not exist in either crate; `clamp_split.rs` contains no split,
      slack, indicator or exact-flag item.
- [ ] The `plan.rs` variants and the `common` variants listed in the Goal do not exist.
- [ ] `ClampWidth`, `sat_clamp_bits` and `FusedIntermediates::quotient` do not exist.
- [ ] `get_models_committed_polynomials` takes no `ModelExecutionIO`; the test of invariant C
      passes.
- [ ] Invariant A: both tamper sites are rejected with `Err` for a node of each of the eight
      ops.
- [ ] Invariant B: every op-surface row that saturated under V1 is an expected trace-level
      rejection with its reason; the `saturating` probe model and
      `saturating_probe_saturates` are replaced by an overflow-rejection test.
- [ ] Invariant F: a V1 model is rejected by the verifier with `Err`.
- [ ] `ProofType::DeferredClampChunkChecks` (15) is retired, or its successor from
      `specs/post-loop-batches.md` no longer has a slack member; `proof_types_all_exercised`
      passes.
- [ ] The three sweeps pass with no tolerance entry.
- [ ] `testing/digests.rs` is re-pinned in its own commit; the `fuzz_regressions_replay`
      corpus is regenerated there.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2, bge-small and Qwen prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the non-saturating op-surface rows
of the eight ops; `clamp_bucket_packing_mixed_sizes`, `padded_add_1000`; the three sweeps; the
zoo with its serialization round-trip; the importer's zoo gates.

Changed tests: `add_saturating_overflow_1k`, `mul_saturates_pos`/`neg`, `sum_saturating_rows`,
`einsum_mk_kn_mn_saturates_pos`/`neg`, the scalar `*_sat_*` rows and `sum_scalar_saturating`
become expected rejections; `exact_output_bucket_commits_nothing` becomes the test of
invariant C.

New tests: the V1/V2 property test; the tamper sites of invariant A; the version test of
invariant F.

CI coverage: the workspace job; the heavy job on all three real models, since this is a
semantics change. The `zk` feature is not touched.

### Performance

Benchmark: the run table of `docs/measurement-plan.md` on `gpt2 --seq-len 16` and
`bge-small-en-v1.5` against `bench/bin/prove-c65753bfa`.

Expected: commit flat (the slack columns are empty and free today; the output digits are
unchanged); the post-loop sumchecks down (the degree-4 split becomes the degree-2 value
sumcheck for every bucket, and the slack Booleanity disappears); proof size down by one
opening claim per clamped node and one flag per bucket; verify time down. Fewer buckets, since
all output claims share one width.

Bounds: prove and verify within 3% of baseline or better on both models; proof size no larger;
peak memory no higher.

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
`out(r_n) + 2^31` at width 32, to the range buckets of `specs/one-range-check.md`. The clamp
bucket as a separate kind, with its own value relation and columns, is gone:
`GlobalClampOutDigit` and `GlobalRemainderDigit` become one digit family.

In `atlas-trace`, the V2 kernels call a checked narrowing where V1 calls `sat_clamp_i32`.
Add and Sub with more than two operands check each pairwise step, matching where V1 would
saturate. The error names the node, so the `prove` example can report which op overflowed.

### Alternatives Considered

- **Keep saturation, remove only the special cases** (the exact flag and the
  unsaturated-output tie), so every bucket runs the split, digit range and slack check and the
  committed set is model-only. This is the fallback if the semantics decision goes the other
  way. It keeps total semantics and deletes the flag's `Open`, `Tie` and branch, but keeps the
  pre-clamp opening, the slack chunks, the indicator, the split sumcheck and the width ladder,
  and it commits output digits for large public outputs (GPT-2's logits are 2^20 padded
  elements). It needs a measurement before it can be recommended.
- **Saturate at a declared width** (`docs/prover-plan.md` item 4, first option). Rejected as a
  simplification: it keeps every mechanism and adds a per-op parameter.
- **Decompose the pre-clamp value itself** (64 bits of digits) and derive the output from the
  digits. Rejected: twice the committed digits on 71% of the rows, and a non-linear relation
  between digits and output.
- **Keep V1 provable beside V2.** Rejected: both protocols would stay in the verifier.
- **Published systems.** FlashProve and Sable use partial-function semantics; total exact
  semantics is a property the plan lists as unclaimed, and this spec gives it up for the
  clamp. The trade is the decision recorded at the top.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the behavioral contract's two clamp clauses.
- `README.md`: the semantics statement (overflow is an error), and the benchmark table.
- `docs/verifier.md`: the "Plans" post-loop paragraph and the deleted-tag list.
- `docs/prover-plan.md`: decision 2 recorded; items 3 and 4 restated against the new base.
- `docs/op-roadmap.md`: "what adding an op requires" names the overflow error.
- Module docs of `clamp_split.rs` and `global_clamp.rs` in both crates.

## Execution

Suggested order: (1) `SemVersion::V2` in `atlas-trace` with the property test, nothing else
changed; (2) op relations on `out`, with the buckets always taking the exact-value path;
(3) deletions in the verifier, core and `common`; (4) output claims into the range buckets and
the single digit family; (5) the width ladder and `quotient` out of `atlas-trace`;
(6) the digest re-pin and docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/clamp_split.rs` (module header: the split protocol),
  `clamp_lookups/mod.rs`, `global_clamp.rs`, `plan.rs` (`clamp_steps`, `fused_rescale_plan`).
- `atlas-trace/src/semantics/fixed.rs`, `op/linear.rs`, `op/fused.rs`, `model.rs`
  (`ClampWidth`).
- `docs/prover-plan.md` §3 item 4 and §5 decision 2; `docs/measurement-plan.md` Phase 2 (the
  clamp-width census: one saturating node).
- `specs/clamp-dense-digit-range-check.md` (invariant H, the IO-dependent commitment this
  spec removes).
