# Spec: [Feature Name]

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   |                                |
| Created     | YYYY-MM-DD                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | (other specs/PRs this one builds on, or —) |
| Supersedes  | (specs or sections this one replaces, or —) |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

One paragraph: what is this feature and why does it matter? State the problem being solved, not
just the solution.

## Intent

### Goal

What are we building? State the primary objective in one sentence without qualifiers. Name the
key types, modules, and boundaries this feature introduces or modifies.

### Invariants

What properties must hold? List the correctness and soundness invariants the implementation must
preserve. For anything the prover touches, state the prover/verifier consistency requirement and
what a malicious prover must not be able to do. Name the parts of the `atlas-trace` behavioral
contract (`atlas-trace/src/lib.rs`) that the feature relies on or changes: the global fixed-point
scale, the fused-rescale quotient/remainder split, lookup-table semantics, padding behavior.

**For each invariant, name the mechanism that holds it**: the compiler, a CI job, a named test
or sweep, or review. Say "review-held" out loud when it is review — an invariant whose only
mechanism is vigilance is a different promise than one the build enforces, and reviewers will
find the configuration where an unnamed mechanism doesn't exist. "Unchanged" and "enforced"
without a mechanism are the two words spec reviews reject most.

Facts that can go stale (upstream revisions, line counts, commit lists) belong in a referenced
living document (e.g. under `docs/`), not in the spec body: the spec freezes at merge, the facts
don't.

### Non-Goals

What is explicitly out of scope? Listing non-goals prevents scope creep and clarifies the
feature's boundaries.

## Evaluation

### Acceptance Criteria

Concrete, testable criteria. Each should be verifiable by a test, benchmark, or assertion.

**Scope every criterion to the PR that ticks it.** A criterion in the main list must be
verifiable on *this spec's PR*; if the spec describes a multi-PR program, put later outcomes in
a separate "Later phases/stages" list that names which PR ticks each box. A spec whose
acceptance list mixes this-PR and end-state criteria cannot be implemented or reviewed against:
`/implement-spec` either over-builds or cannot finish.

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Testing Strategy

Which existing tests must continue passing? What new tests are needed? Say which CI jobs cover
the change (`cargo nextest run --workspace`: unit tests, the op-surface table, the soundness
sweeps, the checked-in zoo; the `atlas-onnx-import` zoo gates and golden hashes; the real-model
GPT-2 / BGE / Qwen prove & verify job). For a new op, name its rows in
`jolt-atlas-core/src/onnx_proof/testing/op_surface.rs` (including saturation and non-power-of-two
stances) and which probe model in `onnx_proof/testing/models.rs` puts it under the soundness
sweeps. For a changed proof format, name the new `mutate::Site`. For changes behind the `zk`
feature, which CI does not build, say how they are verified.

### Performance

What are the performance expectations? Name the benchmark that verifies them: the `prove`
example (`cargo run --release -p jolt-atlas-core --example prove -- --model models/<dir>
--trace-terminal`, with `--pcs dory` for the production scheme) on a named model, and the stage
from the README benchmark table (witness + commitment, IOP proving, opening proof, verify) that
is expected to move, with direction and rough magnitude. "No regression" is acceptable if
you name the benchmark that verifies it. Memory budgets matter: the README's reference machine
has 16 GB.

## Design

### Architecture

How does this feature fit into the existing system? Describe which crates, modules, and types are
affected: the import pipeline (`atlas-onnx-import`), the IR and trace engine (`atlas-trace`), the
proof system (`jolt-atlas-core/src/onnx_proof/`), or the shared toolkit (`joltworks`). Include a
diagram if the interaction is non-trivial.

### Alternatives Considered

What other approaches were evaluated? Why was this design chosen over them? This section prevents
re-litigating decisions during implementation review.

## Documentation

What changes to `README.md`, `docs/`, or crate-level docs are required? If none, state why
(e.g., internal refactor with no user-facing impact).

## Execution

Optional implementation direction: algorithmic approach, optimizations to consider, modules to
touch, suggested commit sequence. The implementer should be able to derive most of this from
Intent and Evaluation.

## References

Links to papers, related specs, relevant issues/PRs, prior art (including the upstream `a16z/jolt`
spec or code this builds on, if any).
