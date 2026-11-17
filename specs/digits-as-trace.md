# Spec: The digits are the trace

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`, `specs/one-clamp.md` |
| Supersedes  | `docs/prover-plan.md` §3 item 9 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

This spec describes a program of three stages and is implementable for stage 1 only. Stages 2
and 3 are stated so that stage 1 is built toward them; each gets its own spec before it starts.

## Summary

The proof is a chain. No node output is committed; a claim on one node's output at a random
point is reduced by that node's relation to claims on its operands, node by node, in reverse
topological order. Everything about the verifier's control flow follows from that: the node
loop's order, the rule that an operand claim must be appended before the operand's turn, the
per-node evaluation reduction that merges the claims of several consumers into one (a
line-restriction proof per multi-consumer node), the distinction between relations that may be
deferred and relations that may not, and the fact that the node relations of a model cannot be
batched or proven in parallel. Yet since `specs/one-clamp.md`, most of the trace is committed:
every node whose op is exact arithmetic narrowed to `i32`, and that is not a model output, has
its output in a range bucket as the 16 base-4 digit columns of `out + 2^31`, fixed before any
challenge. A claim on such an output can be checked against the commitment directly, at any
point, as one more term of the bucket's value sumcheck. Then the node's relation no longer
depends on its consumers, and the chain falls apart into independent statements about committed
polynomials. This spec takes that step for the nodes that have digits.

## Intent

### Goal

Stage 1 (this spec's PR): for every node whose output is a range-bucket member (a
**digit-backed** node: `global_clamp::node_range_claims` gives it a `Remainder::Output` claim,
which is an `Add`, `Sub`, `Sum`, rescaling `Mul`, `Square` or `Cube`, `MeanOfSquares` or
`Einsum` that is not a model output), discharge every claim on its output at the bucket's value
sumcheck, draw the point of the node's own relation fresh from the transcript, and remove its
evaluation reduction.

Today such a node has one bucket member, at its reduced output point: `Step::Reduce` adopts the
single claim of a one-consumer node as is (no proof), and runs the line-restriction protocol
(`EvalReductionProof`) for a node with several.

Concretely, for a digit-backed node `n`:

- each claim `out_n(r) = c` appended by a consumer becomes a member of the bucket's value
  sumcheck (one more `PackedSegment` with its own `γ`), instead of an input to `Step::Reduce`;
- `n`'s plan starts with `Step::Point`, which draws `τ_n`, takes the prover's claim
  `out_n(τ_n)` and adds it as one more bucket member; the node's relation runs on that claim;
- `n`'s other range claims (a fused node's `RescaleRemainder`), which are opened at the point
  of its relation, are opened at `τ_n`;
- `n` has no `EvalReductionProof`.

Nodes that are not digit-backed (lookup outputs including the smooth ReLUs, softmax, Gather,
Div, Rsqrt, `Neg`, the raw products, the structural ties, constants, inputs and model outputs)
keep `Step::Reduce` and the chain, and the node loop keeps its order.

Introduced or modified:

- `jolt-atlas-verifier`: `plan.rs` (`Step::Point`; `fused_rescale_plan`, `accumulation_plan`
  and the `Sum`, `MeanOfSquares` and `Einsum` plans start with it when the node is
  digit-backed), `verifier.rs` (the interpreter), `global_clamp.rs` (`ClampBucket::
  remainder_inputs` builds one `PackedSegment` per `(node, point)` output member, where it
  builds one per node from `Remainder::opening`), `clamp_split.rs` (`ValueParams` takes the
  member count from the accumulator's claims), `ops/eval_reduction.rs` (not reached for
  digit-backed nodes).
- `atlas-sumcheck`: `opening_proof` (the accumulator returns every `NodeOutput(n)` claim of a
  digit-backed node, and `get_node_output_opening(n)` returns the `Step::Point` claim);
  `relation::RelationOpening::ReducedNodeOutput` therefore resolves to that claim and no op
  relation changes.
- `jolt-atlas-core`: the mirrors; the prover evaluates `out_n(τ_n)` and each consumer claim
  from the trace as it does today.

Later stages, each under its own spec:

- **Stage 2, independence used.** The relations of digit-backed nodes leave the node loop and
  are proven as batched sumchecks over groups of nodes, in parallel. Requires a memory plan:
  all witnesses of a batch are live at once.
- **Stage 3, everything digit-backed or public.** The remaining virtual outputs either gain a
  decomposition (table outputs, softmax) or are resolved through zero-round ties to a
  digit-backed or public node. The node loop, `Step::Reduce`, the evaluation-reduction protocol
  and the reverse-topological order are deleted; same-shape relations stack across layers.

### Invariants

Relied on, not changed:

1. **Trace semantics and every op relation's expression.** Only the origin of a relation's
   input claim and the destination of its operand claims change. Outputs do not move.
   *Mechanism:* the importer's golden hashes are not re-pinned;
   `relation_specs_match_their_dispatched_op_kind`.
2. **Commit before challenge.** All digit columns are committed before the first challenge.
   *Mechanism:* `testing/digests.rs`; review.
3. **The packed value sumcheck and the digit range check.** *Mechanism:* unchanged relations.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. A digit-backed output is the committed polynomial.** For a digit-backed node, every
  claim `out_n(r) = c` the verifier relies on, whether appended by a consumer or taken at
  `τ_n`, is a member of its bucket's value sumcheck, and therefore holds for the polynomial
  defined by the bucket's digit columns on the node's segment. *Mechanism:* the drain check is
  extended: a `NodeOutput` claim on a digit-backed node that no bucket member covers is an
  `Err`. `opening_forgery_sweep` forges each such claim and must see a rejection.
- **B. The node relation is checked at a point the prover could not anticipate.** `τ_n` is
  drawn after the commitments and after the claim it will be compared against cannot be
  chosen; with A, the relation at `τ_n` implies the op's identity on every element of the
  segment by Schwartz–Zippel, without reference to any consumer. The module header of
  `plan.rs` states this argument once, for all ops. *Mechanism:* review-held;
  `witness_tamper_sweep` over the probe models.
- **C. No evaluation-reduction proof for a digit-backed node.** *Mechanism:* the proof carries
  no entry; a stray entry fails the drain check; a `mutate::Site` inserts one.
- **D. Mixed regime is ordered.** A digit-backed node's relation still appends its operand
  claims during its turn in the node loop, so a non-digit-backed operand's `Reduce` sees them,
  and every claim on a digit-backed node exists before the post-loop value sumcheck draws its
  batching coefficients. *Mechanism:* the accumulator's existing check that no claim is
  appended to a reduced node ("cannot append opening for node … after evaluation reduction"),
  which stays for non-digit-backed nodes; the zoo.
- **E. Members are a function of the model.** The output members of a bucket are, for each of
  its digit-backed nodes, the claims its consumers' plans append plus the one at `τ_n`, in
  one fixed order (node, then opening id). Prover and verifier enumerate them from their
  accumulators, and the count never depends on the proof. *Mechanism:* a test on the
  `relation_flows` probe's shared `Add` (three consumers): four members.

### Non-Goals

- Stages 2 and 3.
- Any change to non-digit-backed nodes, to the lookups, or to the post-loop batch structure.
- Giving table outputs or softmax a digit decomposition.
- A per-round degree schedule in the sumcheck engine.
- Performance beyond the bounds below; the gain of this stage is the removed evaluation
  reductions, and the larger gains are stage 2's.

## Evaluation

### Acceptance Criteria

Stage 1, this PR:

- [ ] The measurement of the Performance section is reproduced in the PR at the commit this
      lands on, if that is not `c90a4652e`.
- [ ] No digit-backed node's plan contains `Step::Reduce`; each begins with `Step::Point`.
- [ ] No proof of a checked-in model carries an `EvalReductionProof` for a digit-backed node.
- [ ] Invariant A: the extended drain check exists; the forgery sweep reaches every
      digit-backed output claim and each forgery is rejected with `Err`.
- [ ] Invariant C: the `mutate::Site` that inserts an evaluation-reduction entry for a
      digit-backed node is rejected.
- [ ] Invariant E: the three-consumer test passes.
- [ ] The three sweeps pass over the probe models with no tolerance entry.
- [ ] `produced_opening_joins_a_shared_operand_reduction` in `testing/relations.rs` is
      rewritten to the new flow; `raw_product_reads_a_multi_consumer_reduction` (a raw product
      is not digit-backed) passes unchanged.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit; the `fuzz_regressions_replay` corpus is regenerated there.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

Later stages (not ticked by this PR):

- [ ] Stage 2 PR: digit-backed node relations are proven outside the node loop in batches.
- [ ] Stage 3 PR: `Step::Reduce`, `atlas_sumcheck::evaluation_reduction` and the node loop's
      ordering constraint do not exist.

### Testing Strategy

Existing tests that must keep passing: the workspace run; every op-surface row; the three
sweeps; the zoo with its serialization round-trip; `relation_prover_equiv_*`.

Changed tests: `produced_opening_joins_a_shared_operand_reduction` in `testing/relations.rs`
pins the flow this spec replaces (its shared operand is an `Add`); it is rewritten to assert
that the claims become bucket members. `raw_product_reads_a_multi_consumer_reduction` keeps
pinning the reduction for a node that is not digit-backed. The witness sweep's `BucketDigit`
sites and `scalars_probe_layout` follow the member layout.

New tests: the forgery coverage of invariant A; the `mutate::Site` of invariant C; the
three-consumer case; a digit-backed node with no consumer other than the model output check.

CI coverage: the workspace job; the heavy job, since the claim lifecycle changes. One manually
dispatched exhaustive soundness run before merge. The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against this
spec's parent commit, on the benchmark machine.

Measured at `c90a4652e` on the benchmark machine (16 tokens; node counts from the lowered
graph, times from the `NodeEvalReduction::prove` spans of the gate's trace):

| | GPT-2 | bge-small |
|---|---|---|
| digit-backed nodes with one consumer | 380 (6.51 M cells) | 355 (2.58 M cells) |
| digit-backed nodes with several consumers | 50 (0.82 M cells) | 75 (0.61 M cells) |
| output member cells today | 7.33 M | 3.19 M |
| output member cells after | 15.86 M | 7.18 M |
| evaluation reduction of digit-backed nodes | 0.19 s of 25.0 s prove | 0.14 s of 16.1 s prove |

Expected: prove and verify flat. The removed line-restriction proofs are under 1% of prove
time. Against them, every digit-backed node gains one member at `τ_n` and a node with `k`
consumers goes from one member to `k + 1`: one more evaluation of the output by the prover
and one more `eq` table of the node's size in the bucket's cycle weight per added member,
about 8.5 M cells on GPT-2, which is linear work of the size of one pass over the outputs.
Proof size: down by 50 (GPT-2) and 75 (bge-small) line-restriction proofs, up by one claim
per digit-backed node. Commit and openings unchanged.

Stage 1 is not a performance change. It is justified by what stage 2 builds on it and by the
verifier's simpler claim lifecycle for these nodes.

Bounds (the gate's defaults): prove and verify within 3% of the reference on both models; peak
memory within 5%; proof size within 2%.

## Design

### Architecture

```text
today:     consumer relation → claim on out_n ─┐
           consumer relation → claim on out_n ─┼→ Reduce (line restriction) → n's relation → operand claims
           ...                                 ─┘

stage 1:   consumer relation → claim on out_n ──→ bucket member
           consumer relation → claim on out_n ──→ bucket member
           Point: τ_n, claim out_n(τ_n)       ──→ bucket member
                                                └→ n's relation → operand claims
           post-loop: value sumcheck over all members, digit range
```

A bucket's cycle weight today is `W(t) = Σ_m γ_m·eq(prefix_m, hi)·eq(r_m, lo)`
(`CycleWeight::Packed`, one `PackedSegment` per range claim, built by
`ClampBucket::remainder_inputs`). An output claim contributes one term per `(node, point)`
member where it contributes one per node; the other range claims (remainders, slacks, softmax
gaps) keep one term each. The relation (`ValueParams`), its degree and its output opening are
unchanged. The input claim is the `γ`-combination of the members' claims, each
`out(r) + 2^31` (`Remainder::opening`, `OUTPUT_OFFSET`).

`Step::Point` is the only new step: draw `τ_n` of the node's output arity, append the
`NodeOutput(n)` claim at `τ_n`, register the member. `RelationOpening::ReducedNodeOutput(n)`
resolves to it, so no op relation changes.

The interpreter knows which nodes are digit-backed from the bucket layout, which is a function
of the model. A consumer's claim on a digit-backed node is registered as a member when it is
appended; on any other node it waits for `Reduce` as today.

What `specs/one-clamp.md` made possible: the value relation of every bucket is `Σ W·value`
over one digit family, so an output member is a term of `W` and nothing else. (The split
relation it removed weighted the output, the indicator and the slack by one cycle weight built
from one point per node.)

### Alternatives Considered

- **Keep the chain; speed up the evaluation reduction** (`docs/prover-plan.md` item 10, A3).
  A valid optimisation, but it keeps the ordering and the non-batchable node loop.
- **Reuse one consumer's point for the node's own relation** instead of a fresh `τ_n`, saving
  one member per node; for a one-consumer node (380 of 430 on GPT-2) this is exactly today's
  flow. Rejected: it ties the node to a consumer again, which is what stage 2 needs gone, it
  leaves two regimes among digit-backed nodes, and the measurement above puts the cost of the
  uniform rule well inside the bounds.
- **Do stages 1–3 in one change.** Rejected: stage 3 needs a relation compiler and a decision
  on table outputs; stage 2 needs a memory plan. Stage 1 changes the claim lifecycle alone and
  is testable by the existing sweeps.
- **Commit every node output as a dense polynomial.** Rejected: the digits are already
  committed and already range-checked; a second commitment of the same values costs commit
  time, which is the prover's largest stage.
- **Upstream `a16z/jolt`.** Its zkVM commits its trace columns and proves all constraints in
  staged batches; this is the same structure, reached from a GKR-style chain. Its per-round
  degree schedule (#2336) is relevant to stage 2.

## Documentation

- `docs/verifier.md`: the reading order and "Plans" sections (a second kind of node turn;
  the extended drain check).
- `docs/prover-plan.md`: item 9 restated as the three stages, stage 1 marked landed.
- `docs/pull-requests.md` §5: a new op states whether its output is digit-backed.
- `CLAUDE.md`: the invariants paragraph on claim lifecycle, if it names the evaluation
  reduction.
- Module header of `plan.rs` (invariant B's argument).

## Execution

Suggested order: (1) the measurement of the first criterion; (2) `(node, point)` output
members in `remainder_inputs` and the prover's `BucketWitnesses`, with one member per node,
digests unchanged; (3) `Step::Point` and consumer routing for `Add` and `Sub`
(`accumulation_plan`) behind the probe models; (4) the fused products, `Sum`, `MeanOfSquares`
and `Einsum`; (5) the drain-check extension and the sweeps' new sites; (6) the digest re-pin
and docs.

## References

- `docs/prover-plan.md` §3 item 9 (the idea) and item 10 A3 (the alternative).
- `atlas-sumcheck/src/evaluation_reduction/`, `atlas-sumcheck/src/opening_proof/`;
  `jolt-atlas-verifier/src/onnx_proof/ops/eval_reduction.rs`, `verifier.rs`.
- `jolt-atlas-verifier/src/onnx_proof/global_clamp.rs` (`Remainder::Output`,
  `node_range_claims`, `ClampBucket::remainder_inputs`), `clamp_split.rs` (`ValueParams`);
  `atlas-sumcheck/src/ps_shout/mod.rs` (`PackedSegment`, `CycleWeight`).
- `specs/verifier-audit-simplification.md` (the plan interpreter this extends).
- a16z/jolt #2336 (per-round degree schedule).
