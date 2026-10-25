---
name: new-spec
description: Create a new spec in specs/ through a Socratic interview, filling each template section to zero ambiguity. Use when asked to write, start, or draft a spec for a feature or large change.
arguments: "<feature-name>"
---

Create a new spec file in `specs/` by interviewing the user to fill each section of the template.
The goal: produce a spec that would pass `/analyze-spec` with zero ambiguity on the first try.

## Steps

### 1. Initialize

1. Validate the argument: must be lowercase alphanumeric with dashes (e.g. `softmax-division-sumcheck`).
   Reject otherwise.
2. Get the GitHub username: run `gh api user --jq .login`.
3. Get today's date in `YYYY-MM-DD` format.
4. Read `specs/TEMPLATE.md` to understand the required sections.
5. Read `CLAUDE.md` for the crate map and the invariants the project already states.
6. Explore the codebase to understand what areas the feature name suggests. Run an `Explore`
   agent to gather context: which crates and modules are involved, which tests cover them today,
   which examples exercise them. This informs your questions.

### 2. Interview, section by section

Walk through each template section with the user. For each section, ask targeted questions until
you have enough to write it with zero ambiguity. Do NOT move to the next section until the current
one is clear.

**One question at a time. Never batch.**

#### Summary
Ask: "In one paragraph, what problem does this solve and why does it matter?"

#### Intent: Goal
Ask: "What are we building? Can you state the primary objective in one sentence?" Follow up on
types, modules, and boundaries if unclear.

#### Intent: Invariants

Before asking the user anything, form a hypothesis from the Summary and Goal:
1. Read the behavioral contract at the top of `atlas-trace/src/lib.rs`. For each clause, judge
   whether this feature relies on it, preserves it, or changes it.
2. If the feature touches the prover, list the op protocols under
   `jolt-atlas-core/src/onnx_proof/ops/` it affects and the soundness tests
   (`onnx_proof/soundness_tests.rs`, `onnx_proof/malicious_prover.rs`) that pin them today.
3. Independently, imagine the binary properties that would have to hold for the Goal to be
   correct. Which are not pinned by an existing test?

Present the hypothesis to the user in one pass:
```
Based on the Summary and Goal, I think this feature likely:
- Relies on: {contract clause / existing invariant}
- Changes: {contract clause or test X, because ...}
- Needs new invariants: {A: "...", B: "..."}

Does this match your intent? What would you add, remove, or change?
```

Then follow up with: "What other properties must always hold? What must a malicious prover be
unable to do? What would break if the implementation is wrong?"

#### Intent: Non-Goals
Ask: "What is explicitly out of scope?" Push back if non-goals are vague. "You said 'not
performance-critical', but this sits in the sumcheck inner loop that runs every round. Is there a
concrete budget?"

#### Evaluation: Acceptance Criteria
Ask: "What check would prove this works? Give me concrete, checkable criteria." Each criterion
must be a checkbox item. Name the behavior and evidence required; reuse existing coverage when
sufficient and add tests only for distinct gaps.

#### Evaluation: Testing Strategy
Ask: "Which existing tests must keep passing? What new tests are needed? Which CI jobs cover this:
the workspace tests, the zoo gates and golden hashes, the GPT-2 or BGE prove & verify? If it
touches the `zk` feature, how is that verified, since CI does not build it?"

#### Evaluation: Performance

Before asking the user anything, form a hypothesis:
1. Read the benchmark tables in `README.md`. Judge which stages (witness + commitment, IOP
   proving, opening proof, verify) this feature is likely to move, and in which direction.
2. Decide which example under `jolt-atlas-core/examples/` is the right benchmark.

Present the hypothesis in one pass:
```
Based on the Summary, Goal, and Invariants, I think this feature likely:
- Moves: {stage, direction, rough magnitude} on {example}
- Leaves unchanged: {stages}

Does this match your expectations? What would you add, remove, or change?
```

Then follow up with: "Beyond these, are there other expectations? Memory budget, proof size,
verifier time? Or is 'no regression' sufficient, and if so, which benchmark verifies that?"

#### Design: Architecture
Ask: "How does this fit into the existing system?" Use codebase exploration to suggest which
crates and modules are affected. Ask about type parameters, trait implementations, module
boundaries, and whether `joltworks` (shared toolkit) or `onnx_proof` (atlas-specific) should own
the new code.

#### Design: Alternatives Considered
Ask: "What other approaches did you consider? Why this one?" If the user says "none", probe:
"What's the simplest approach that could work? Why isn't that sufficient?" If upstream
`a16z/jolt` solves a similar problem, name how and ask whether that was considered.

#### Documentation
Ask: "Does this need changes to `README.md`, `docs/`, or crate-level docs?"

#### Execution (optional)
Ask: "Any implementation direction you want to capture? Algorithmic approach, optimizations,
modules to touch, a suggested commit sequence? Or should the implementer derive this from intent
and evaluation?"

#### References
Ask: "Any papers, issues, PRs, upstream Jolt code, or prior art to link?"

### 3. Write the spec

Once all sections are filled, create `specs/<feature-name>.md` with the template fully populated:
- `[Feature Name]` → feature name from the interview
- `Author(s)` → `@<github-username>`
- `Created` → today's date
- `Status` → `proposed`
- `PR` → empty until the PR exists; fill it in when known
- All sections filled from interview answers. Drop the template's guidance paragraphs; keep the
  Status rule paragraph under the header table.

### 4. Review with the user

Show the complete spec to the user. Ask: "Does this capture everything? Anything to add or
change?" Make edits until the user is satisfied.

### 5. Next steps

Print:
```
Spec created: specs/<feature-name>.md

Next steps:
1. Run /analyze-spec to score and tighten it
2. Open a PR containing only the spec; put the PR number in the header table
3. After maintainers approve the spec, run /implement-spec on the same branch
```

Do NOT modify `TEMPLATE.md` itself.
