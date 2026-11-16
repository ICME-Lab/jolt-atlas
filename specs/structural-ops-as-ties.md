# Spec: Reshape, Slice and Concat as zero-round ties

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | —                              |
| Supersedes  | `docs/prover-plan.md` §3 item 5, "Structural ops as ties (B4)", first half |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Reshape, Slice and Concat move no information: each output element is an input element.
MoveAxis, Broadcast and Identity are already verified that way, as zero-round ties that
re-point a claim. Reshape, Slice and Concat instead run a degree-2 sumcheck against a selector
table the verifier builds and evaluates itself, at a cost linear in the tensor, because on
arbitrary dimensions per-dimension padding makes the index map non-affine. But every model the
prover accepts is compiled with `pad_pow2`, and `pad_graph` already rejects a Reshape whose
layout changes under padding, a Slice whose extent is not a power of two, and a Concat whose
leading operands are not powers of two on the axis or whose padded operand extents do not sum
to the padded output. On those models a Reshape is the identity on padded indices, and an
aligned Slice or Concat selects a subcube. The sumchecks and their
selector tables serve only unpadded test models. GPT-2 has 176 Reshapes and 12 Slices,
bge-small 194 Reshapes, qwen2-tiny 52 Reshapes, 10 Slices and 5 Concats. This spec verifies
the three ops as ties and deletes the selector relations.

## Intent

### Goal

Verify Reshape, Slice and Concat by `TieRelation`s whose opening points are functions of the
node's reduced output point, require the alignment that makes them ties at compile time, and
delete `ReshapeRelation`, `SliceRelation`, `ConcatRelation` and their selector builders.

| Op | Tie identity at the reduced output point `r` | Requires |
|---|---|---|
| Reshape | `out(r) = in(r)` | the padded layouts agree (`check_pad_coherent_reshape`, enforced today) |
| Slice | `out(r) = in(r with the axis's high bits fixed to start / extent)` | extent a power of two (enforced today) and `start` a multiple of it (new) |
| Concat | `out(r) = Σ_i eq(prefix_i, r_hi,i) · in_i(r_lo,i)` | the padded operand extents sum to the padded output (enforced today) and each operand's offset on the axis is a multiple of its padded extent (new; rejects `4 + 8 + 4`) |

Introduced or modified:

- `atlas-trace`: `pad_graph`'s Slice and Concat checks (`op/structural.rs`) gain the alignment
  conditions; a violation is a compile error like the existing ones.
- `jolt-atlas-verifier`: `ops/reshape.rs`, `ops/slice.rs`, `ops/concat.rs` (tie relations
  replace the engine relations), `plan.rs` (`OpRelation::{Reshape, Slice, Concat}` become
  `OpTie` variants), `node_view::check_model_shape` (today it checks only topological order;
  it gains, for these three ops, that every dim of the node and of its operands is a power of
  two, and the Slice and Concat alignment rules).
- `atlas-sumcheck`: one `RelationDerived` variant for Concat's per-operand coefficient. No
  existing tie combines several operands at different points with derived coefficients
  (Broadcast has the coefficient, `AccumulationRelation` the two operands); `verify_tie`
  already supports it.
- `jolt-atlas-core`: `TieClaims` impls replace the `RelationWitnesses` impls for the three ops.
- `atlas-onnx-import`: the `dump_lowered_ops` example prints each Slice's axis, start, end and
  input extent and each Concat's axis and operand extents (it prints counts only today).
- Tests: the op-surface rows and the `tables_and_layout` probe model, as the Testing Strategy
  lists.

### Invariants

Relied on, not changed:

1. **Padding.** Dimensions are padded per dimension to powers of two at compile time, and every
   op declares how it behaves under padding (the `atlas-trace` contract). This spec tightens two
   ops' declarations and changes no padded value. *Mechanism:* the importer's golden hashes are
   not re-pinned.
2. **Tie mechanics.** `verify_tie` appends each produced opening at its point and checks the
   input claim against the output expression; a tie's points and derived values use no
   challenge. *Mechanism:* `onnx_proof/relation.rs`, unchanged.
3. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. Completeness on every accepted model.** For every model `Model::compile` accepts with
  `pad_pow2`, each tie identity holds at every `r`: the output's multilinear extension equals
  the stated combination of operand extensions. *Mechanism:* a property test in `atlas-trace`
  that, for random shapes passing `pad_graph`, evaluates both sides at random points; the
  op-surface rows; the zoo.
- **B. S ⊆ R per op.** Each tie's doc states why the identity at a random `r` implies the
  elementwise statement of the op's semantics; an accepted Concat's operands tile the padded
  axis exactly, so there is no uncovered region. On power-of-two dims every Reshape is the flat
  identity, so the verifier needs no coherence check of its own. *Mechanism:* review-held (`docs/pull-requests.md` §4.2);
  `witness_tamper_sweep` over the probe models.
- **C. Alignment is decided at compile time.** A model with an unaligned Slice or Concat does
  not compile; the verifier independently rejects such a node with `Err`, since it does not
  trust that the model was compiled by us. *Mechanism:* `pad_graph` tests that assert the
  rejection reason; a verifier test on a hand-built model. (`import_full_zoo` asserts only that
  `reject_slice` and `reject_concat` fail to import, so it does not pin the new rules.)
- **D. Nothing in the zoo is lost.** Every Slice and Concat in the checked-in zoo and in GPT-2,
  bge-small and Qwen is aligned. *Mechanism:* the census of the first acceptance criterion; the
  importer's zoo gates.
- **E. No selector code survives.** `build_reshape_selectors`, `build_slice_selector` and the
  Concat selector builder are deleted, with no test-only copy. *Mechanism:* the compiler.

### Non-Goals

- Proving unpadded models. After this spec the three ops' verifiers require padded dimensions;
  the other ops' handling of non-power-of-two dimensions is not touched.
- Rewriting IsNan to a constant and folding the dead Iff in the importer, and merging Reshape
  and MoveAxis chains (the second half of B4): importer passes, separate.
- MoveAxis, Broadcast, Identity, IsNan, Iff.
- Supporting unaligned Slice or Concat by any other protocol.
- Any change to padded outputs, `SemVersion` or golden hashes.

## Evaluation

### Acceptance Criteria

- [ ] A census of Slice starts and Concat offsets over the checked-in zoo, GPT-2, bge-small and
      Qwen, from the extended `dump_lowered_ops`, is in the PR, and every one is aligned. The
      census had not been run when this spec was analyzed. If one is not, implementation stops and this spec is amended.
- [ ] `ReshapeRelation`, `SliceRelation`, `ConcatRelation` and the three selector builders do
      not exist; `OpRelation` has no `Reshape`, `Slice` or `Concat` variant.
- [ ] The plans of the three ops are `[Reduce, Tie]`.
- [ ] Invariant A: the property test passes for the three ops.
- [ ] Invariant C: `pad_graph` rejects an unaligned Slice (`8..40` of 64) and an unaligned
      Concat (`4 + 8 + 4`), each with its stated reason; the verifier returns `Err` on a
      hand-built model with each, and on one with a non-power-of-two dim.
- [ ] The op-surface table has exactly one expected-rejection row per rule (non-power-of-two
      dims, unaligned Slice start, unaligned Concat offset) and the proving rows the Testing
      Strategy lists.
- [ ] The probe model `tables_and_layout` is rebuilt with power-of-two dims and aligned Slice
      and Concat nodes (today its Concat output is `[48]` and its Slice `8..40`), keeps a
      Reshape, a Slice at a nonzero start and a three-operand Concat, and stays under the three
      sweeps, which pass with no tolerance entry.
- [ ] `relation_specs_match_their_dispatched_op_kind` passes with the ties bound to their ops.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 proves and verifies under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the MoveAxis, Broadcast and Identity
rows (untouched ties); the three sweeps; the zoo with its serialization round-trip; the
importer's zoo gates including `reject_slice` and `reject_concat`; `fuzz_regressions_replay`,
its corpus regenerated in the digest commit.

Changed tests (op-surface). Rows with power-of-two shapes and aligned layouts keep proving on
the unpadded builder and do not move: `reshape_16_to_4x4`, `reshape_4x8_to_8x4`,
`reshape_2x4x8_to_64`, `reshape_2x4x8_to_8x8`, the five `concat_pow2_*` rows, and
`clamp_bucket_packing_mixed_sizes` (a Slice `0..256` of 1024). The rows the new rules reject
are replaced, not kept as rejections: all six `slice_*` rows, the five `concat_arbitrary_*`
rows, `concat_three_inputs_*`, `concat_four_inputs_axneg1`, `concat_single_input` and
`reshape_non_pow2_10x10_to_20x5`. In their place:

- one expected-rejection row per rule: a non-power-of-two dim on an unpadded model, an
  unaligned Slice start, an unaligned Concat offset;
- proving rows built with `build_padded` for what padding makes legal: Reshape
  `[1,7,8] → [7,8]`; Concat `4 + 3`; Slices at start 0 and at a nonzero aligned start, on a
  first, middle and last axis; aligned three- and four-operand Concats (`8 + 4 + 4`,
  `4 + 4 + 4 + 4`); a single-operand Concat.

New tests: the property test of invariant A; the rejection tests of invariant C.

CI coverage: the workspace job; the heavy job for GPT-2. The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the
commit this lands on, on the benchmark machine.

Expected: the IOP stage and verify time slightly down (188 sumchecks on GPT-2 and 194 on
bge-small become ties, and the verifier stops building selector tables linear in each tensor);
proof size down by those sumcheck proofs; commit and openings unchanged. The magnitude is
taken from the baseline's per-op spans, not estimated here.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better; peak
memory within 5%; proof size no larger.

## Design

### Architecture

Each tie follows `AxisPermutationRelation` (`ops/moveaxis.rs`): `rounds() = 0`, the incoming
point is the reduced output point, and `opening_points` maps it to the operand points.

- **Reshape.** One operand opening at `r`. Output expression: the operand opening.
- **Slice.** `r` splits into per-axis coordinate groups. On the sliced axis the operand has
  `log₂(input extent)` variables and the output `log₂(extent)`; the operand point is the
  output's group prefixed with the bits of `start / extent`. Output expression: the operand
  opening.
- **Concat.** For operand `i` with offset `o_i` and padded extent `e_i` on the axis, the
  output's axis group splits into `log₂(output extent / e_i)` high coordinates and
  `log₂ e_i` low ones. The operand is opened at `r` with the axis group replaced by the low
  coordinates; its coefficient is `eq(bits(o_i / e_i), high coordinates)`, a new
  `RelationDerived` variant the relation's `derive` computes from `r` in `O(log)` time (the
  existing `Table` variant evaluates at the sumcheck point, which a tie does not have). Output expression:
  `Σ_i coefficient_i · opening_i`.

Each produced opening is a `NodeOutput` claim on an operand, appended in the node's turn, so
the operand's evaluation reduction sees it as it sees the sumcheck's claim today.

`check_model_shape` gains the three ops' shape conditions so the verifier rejects, rather than
mis-verifies, a model that was not compiled with padding. A point is split into per-axis
groups with one shared helper; `permute_challenge_groups` (MoveAxis) and
`split_broadcast_vars` (Broadcast) each write that split inline today and are not changed.

### Alternatives Considered

- **Keep the sumchecks for unaligned Slice and Concat, ties for aligned.** Rejected: two
  protocols per op and no code deleted, for a case no model in the zoo has. If the census
  finds one, this is the amendment to consider first.
- **Keep the sumchecks behind the unpadded test builder.** Rejected: test-only verifier code
  inside the trust perimeter.
- **Rewrite an unaligned Slice in the importer** as an aligned Slice of a shifted tensor.
  Rejected: the shift is itself an unaligned selection.
- **Fold Reshape away in the importer.** Rejected for now: a Reshape changes the dims its
  consumers see, so removing the node means rewriting consumers; the tie costs nothing.
- **Upstream `a16z/jolt`.** No analogue; a zkVM has no tensor layout ops.

## Documentation

- `docs/verifier.md`: the "Plans" section lists the three ops with the other ties.
- `docs/op-roadmap.md` and the importer docs: the Slice and Concat alignment rules beside the
  existing padding rejections.
- Module docs of the three op files in both crates, each with its S ⊆ R note.
- `docs/prover-plan.md`: item 5's B4 line marked landed for the tie half.
- `README.md`: the supported-op notes if they describe Slice or Concat constraints.

## Execution

Suggested order: (1) `dump_lowered_ops` prints Slice and Concat layouts, and the census; (2) Reshape, which needs no new rule; (3) the alignment rules
in `pad_graph` with their tests; (4) Slice; (5) Concat; (6) the op-surface migration to padded
builds and the probe model; (7) the digest re-pin.

## References

- `jolt-atlas-verifier/src/onnx_proof/ops/{reshape,slice,concat,moveaxis,broadcast}.rs`;
  `onnx_proof/relation.rs` (`TieRelation`, `verify_tie`).
- `atlas-trace/src/model.rs` (`pad_graph`), `atlas-trace/src/op/structural.rs`
  (`check_pad_coherent_reshape`, the Slice and Concat padding checks).
- `bench/runs/lowered-*.txt` (op counts); `docs/prover-plan.md` §3 item 5.
- `models/reject_slice`, `models/reject_concat`; `atlas-onnx-import/tests/decode_zoo.rs`.
