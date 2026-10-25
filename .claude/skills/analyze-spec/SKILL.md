---
name: analyze-spec
description: Interactive Socratic analysis of a spec in specs/ with ambiguity scoring. Use when asked to analyze, review, score, or tighten a spec before implementation.
argument-hint: "[spec file path]"
---

<Purpose>
Analyze a spec file using weighted ambiguity scoring. The goal: ensure the spec is clear enough
for a one-shot implementation with zero clarifying questions.

This skill runs interactively in Claude Code: one question at a time, iterative refinement with
the spec author. There is no CI mode; maintainers review specs by hand.
</Purpose>

<Execution_Policy>
- Gather codebase facts via an `Explore` agent BEFORE asking about them.
- Cite repo evidence (file path, symbol, or pattern) instead of asking the spec author to
  rediscover it.
- Score ambiguity transparently after every round.
- Do not approve until ambiguity ≤ 20%.
- Allow early approval with a clear warning if ambiguity is still high.
</Execution_Policy>

<Steps>

## Phase 1: Initialize

1. **Locate the spec**:
   - If a path is provided in `{{ARGUMENTS}}`, use that directly.
   - Otherwise, find `specs/*.md` files changed on the current branch versus `main`
     (`git diff --name-only main...HEAD -- specs/`), excluding `TEMPLATE.md`.
   - If none or several match, list `specs/` and ask the user which one.
2. **Read the spec** thoroughly: Summary, Intent, Evaluation, Design, Documentation, Execution.
3. **Read `CLAUDE.md`** for the crate map, the `atlas-trace` behavioral contract, and the
   prover/verifier invariants, so you can judge whether the spec's Invariants section is complete.
4. **Explore the codebase**: run an `Explore` agent to map the crates, modules, tests, and
   examples relevant to the spec's intent.
5. **Read prior context**: if the branch has a PR (`gh pr view --json comments`), read its
   comments so you don't re-ask questions already answered.

## Phase 2: Analyze

Score clarity across four dimensions (0.0 to 1.0 each):

| Dimension | Weight | What to assess |
|-----------|--------|---------------|
| Goal Clarity | 0.35 | Is the primary objective unambiguous? Can you state it in one sentence? Are the key types and boundaries clear? |
| Constraint Clarity | 0.20 | Are boundaries, limitations, and non-goals clear? Does the Invariants section cover prover/verifier consistency and what a malicious prover must not do? |
| Success Criteria | 0.30 | Could you write a test for each acceptance criterion? Is the benchmark named? Are the CI jobs that cover the change named? |
| Context Clarity | 0.15 | Do we understand the existing system well enough to modify it safely? Does the Architecture section name the right crates and modules? |

**Calculate ambiguity:**
`ambiguity = 1 - (goal × 0.35 + constraints × 0.20 + criteria × 0.30 + context × 0.15)`

For each dimension below 0.9, generate a targeted question that would improve it:
- Questions should expose ASSUMPTIONS, not gather feature lists.
- If the scope is conceptually fuzzy, ask an ontology-style question about what the thing
  fundamentally IS.
- Cite specific codebase context (files, types, patterns) when relevant.

## Phase 3: Interview

Full iterative loop, one question at a time.

**Each round:**
1. Identify the dimension with the LOWEST clarity score.
2. State why this dimension is the bottleneck.
3. Ask ONE targeted question.
4. Wait for the author's response.
5. Re-score all dimensions.
6. Report progress:

```
Round {n} complete.

| Dimension | Score | Weight | Weighted | Gap |
|-----------|-------|--------|----------|-----|
| Goal | {s} | 0.35 | {s*w} | {gap or "Clear"} |
| Constraints | {s} | 0.20 | {s*w} | {gap or "Clear"} |
| Success Criteria | {s} | 0.30 | {s*w} | {gap or "Clear"} |
| Context | {s} | 0.15 | {s*w} | {gap or "Clear"} |
| **Ambiguity** | | | **{score}%** | |

Next target: {weakest_dimension}: {rationale}
```

**Challenge modes** (each used once):

- **Round 4+, Contrarian:** challenge the spec's core assumption. "What if this constraint
  doesn't exist?" Test whether the framing is correct or habitual.
- **Round 6+, Simplifier:** probe whether complexity can be removed. "What's the simplest version
  that satisfies the invariants?"
- **Round 8+, Ontologist (if ambiguity > 30%):** "What IS this, really?" Find the essence.

**Stop conditions:**

- All dimensions at 0.9+: approve immediately.
- Round 3+: allow early approval if the author says "enough" or "looks good", with a warning
  stating the remaining ambiguity.
- Ambiguity stalls (same score ±0.05 for 3 rounds): activate Ontologist mode.
- Round 10: soft warning about round count.
- Round 15: hard cap.

## Phase 4: Approve

When approved, print the summary (what will be built, key invariants, critical evaluation
criteria) and offer to update the spec file with the refinements discovered during the interview.
Apply the edits the author accepts. Do not change `Status`; that happens at implementation.

</Steps>

<Examples>
<Good>
Probing an invariant the spec left implicit:
```
The Intent → Invariants section says the new softmax path "produces the same output as the
trace." That is the honest-prover direction only. What must a malicious prover be unable to do:
can it pick a different max, or a quotient/remainder split that is not the Euclidean one? The
`witness_tamper_sweep` in `onnx_proof/testing/soundness.rs` pins this once the op is in a probe
model; say so, or name the dedicated `known_gap_*` test if it is deferred. If
soundness is out of scope for this PR, the Invariants section should say so explicitly.
```
Why good: names the gap precisely, points at a concrete existing test as the model, leaves the
door open to an explicit non-goal.
</Good>

<Good>
Citing codebase context:
```
I found that fused-rescale ops share one module, `onnx_proof/fused_rebase.rs`, used by einsum,
Mul, Square, and Cube. The spec says "add rescale to mean-of-squares" but doesn't say whether
mean-of-squares joins that shared module or gets its own protocol. Which boundary should own it?
```
Why good: explored first, cited the specific module and its callers.
</Good>
</Examples>

Task: Analyze the spec. {{ARGUMENTS}}
