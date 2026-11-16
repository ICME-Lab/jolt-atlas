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
batched or proven in parallel. Yet since the dense-digit change, most of the trace is
committed: every clamped node's output sits in a bucket as digit columns, fixed before any
challenge. A claim on such an output can be checked against the commitment directly, at any
point, as one more term of the bucket's value sumcheck. Then the node's relation no longer
depends on its consumers, and the chain falls apart into independent statements about committed
polynomials. This spec takes that step for the nodes that have digits.

## Intent

### Goal

Stage 1 (this spec's PR): for every node whose output is a range-bucket member (a
**digit-backed** node), discharge every claim on its output at the bucket's value sumcheck,
draw the point of the node's own relation fresh from the transcript, and remove its evaluation
reduction.

Concretely, for a digit-backed node `n`:

- each claim `out_n(r) = c` appended by a consumer becomes a member of the bucket's value
  sumcheck (one more `PackedSegment` with its own `γ`), instead of an input to `Step::Reduce`;
- `n`'s plan starts with `Step::Point`, which draws `τ_n`, takes the prover's claim
  `out_n(τ_n)` and adds it as one more bucket member; the node's relation runs on that claim;
- `n` has no `EvalReductionProof`.

Nodes that are not digit-backed (lookup outputs, softmax, Gather, Div, Rsqrt, the structural
ties, constants, inputs and model outputs) keep `Step::Reduce` and the chain, and the node loop
keeps its order.

Introduced or modified:

- `jolt-atlas-verifier`: `plan.rs` (`Step::Point`), `verifier.rs` (the interpreter; claims on
  digit-backed nodes route to a per-bucket member list), `global_clamp.rs` and `clamp_split.rs`
  (a member is a `(node, point)` pair, not a node), `ops/eval_reduction.rs` (not reached for
  digit-backed nodes), the plans of the eight clamped ops.
- `atlas-sumcheck`: `opening_proof` (an append of a `NodeOutput` claim after the node's turn is
  legal for a digit-backed node); `relation::RelationOpening::ReducedNodeOutput` resolves to
  the `Step::Point` claim for such a node.
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
  claims during its turn in the node loop, so a non-digit-backed operand's `Reduce` sees them.
  *Mechanism:* the accumulator's existing check that no claim is appended to a reduced node,
  which stays for non-digit-backed nodes; the zoo.
- **E. Member count is a function of the model.** The number of members of each bucket's value
  sumcheck is the node's consumer-claim count plus one, computed from the model before the
  transcript runs. *Mechanism:* the plan builder; a test on a node with three consumers.

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

- [ ] Before implementation, the baseline's evaluation-reduction time and the count of
      multi-consumer digit-backed nodes on GPT-2 and bge-small are recorded in the PR (from the
      spans of `bench/runs/c65753bfa/`), so the expected movement is a number.
- [ ] No digit-backed node's plan contains `Step::Reduce`; each begins with `Step::Point`.
- [ ] No proof of a checked-in model carries an `EvalReductionProof` for a digit-backed node.
- [ ] Invariant A: the extended drain check exists; the forgery sweep reaches every
      digit-backed output claim and each forgery is rejected with `Err`.
- [ ] Invariant C: the `mutate::Site` that inserts an evaluation-reduction entry for a
      digit-backed node is rejected.
- [ ] Invariant E: the three-consumer test passes.
- [ ] The three sweeps pass over the probe models with no tolerance entry.
- [ ] `relation_flows` and the tests of `testing/relations.rs` that pin multi-consumer
      reductions are updated to the new flow and pass.
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

Changed tests: `raw_product_reads_a_multi_consumer_reduction` and
`produced_opening_joins_a_shared_operand_reduction` in `testing/relations.rs` pin the flow
this spec replaces for digit-backed nodes; each is rewritten to assert the new flow (the
claims become bucket members) and kept for a non-digit-backed operand.

New tests: the forgery coverage of invariant A; the `mutate::Site` of invariant C; the
three-consumer case; a digit-backed node with no consumer other than the model output check.

CI coverage: the workspace job; the heavy job, since the claim lifecycle changes. One manually
dispatched exhaustive soundness run before merge. The `zk` feature is not touched.

### Performance

Benchmark: the run table of `docs/measurement-plan.md` on `gpt2 --seq-len 16` and
`bge-small-en-v1.5` against `bench/bin/prove-c65753bfa`.

Expected: the IOP stage down by the evaluation reductions of digit-backed nodes (the prover
evaluates a node's polynomial along a curve for each; the amount is the first acceptance
criterion's measurement, not estimated here); the value sumchecks slightly up (more members
per bucket, each one more `eq` table in the cycle weight); proof size down by the reduction
proofs, up by one claim per digit-backed node. Commit and openings unchanged.

Bounds: prove and verify within 3% of baseline or better on both models; proof size within 2%;
peak memory no higher.

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

A bucket's cycle weight today is `W(t) = Σ_n γ_n·eq(prefix_n, hi)·eq(r_n, lo)` with one term
per node. It becomes one term per `(node, point)` member; the relation, its degree and its
output opening are unchanged. The input claim is the `γ`-combination of the members' claims
(each `out(r) + 2^31` after `specs/one-clamp.md`).

`Step::Point` is the only new step: draw `τ_n` of the node's output arity, append the
`NodeOutput(n)` claim at `τ_n`, register the member. `RelationOpening::ReducedNodeOutput(n)`
resolves to it, so no op relation changes.

The interpreter knows which nodes are digit-backed from the bucket layout, which is a function
of the model. A consumer's claim on a digit-backed node is registered as a member when it is
appended; on any other node it waits for `Reduce` as today.

Why stage 1 depends on `specs/one-clamp.md`: with saturation, the split relation weights
`OUT`, the indicator and the slack by one cycle weight built from one point per node. Extra
output-only members would need a second weight on `OUT` alone. With exact arithmetic the
value relation is `Σ W·OUT`, and a member is just a term of `W`.

### Alternatives Considered

- **Keep the chain; speed up the evaluation reduction** (`docs/prover-plan.md` item 10, A3).
  A valid optimisation, but it keeps the ordering and the non-batchable node loop.
- **Reuse one consumer's point for the node's own relation** instead of a fresh `τ_n`, saving
  one member per node. Rejected: it ties the node to a consumer again, which is what stage 2
  needs gone, and model outputs and unconsumed nodes would need a special case.
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

Suggested order: (1) the measurement of the first criterion; (2) `(node, point)` members in
the bucket code with one member per node, digests unchanged; (3) `Step::Point` and consumer
routing for one op (Add) behind the probe models; (4) the other seven ops; (5) the drain-check
extension and the sweeps' new sites; (6) the digest re-pin and docs.

## References

- `docs/prover-plan.md` §3 item 9 (the idea) and item 10 A3 (the alternative).
- `atlas-sumcheck/src/evaluation_reduction/`, `atlas-sumcheck/src/opening_proof/`;
  `jolt-atlas-verifier/src/onnx_proof/ops/eval_reduction.rs`, `verifier.rs`.
- `jolt-atlas-verifier/src/onnx_proof/global_clamp.rs` (`PackedSegment`), `clamp_split.rs`
  (`ValueParams`).
- `specs/verifier-audit-simplification.md` (the plan interpreter this extends).
- a16z/jolt #2336 (per-round degree schedule).
