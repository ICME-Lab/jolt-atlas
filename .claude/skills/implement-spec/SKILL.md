---
name: implement-spec
description: Autonomous implementation of an approved spec from specs/, with a QA loop mirroring CI and a final clean-branch audit. Use when asked to implement, build, or execute a spec.
argument-hint: "[spec file path]"
---

<Purpose>
Take an approved spec and implement it: plan the work, execute, run the CI checks until they
pass, validate against the spec's acceptance criteria, and leave a branch that passes
`/clean-branch`. Produces working, verified code from the spec in a single pass.
</Purpose>

<Execution_Policy>
- The spec is the source of truth. Implement what it says, not more.
- Read `CLAUDE.md` for commands, crate map, invariants, and conventions.
- Each phase must complete before the next begins.
- Resolve routine implementation choices from the spec and repository conventions. For ambiguity
  affecting correctness, scope, or acceptance criteria, ask the user (or post a focused PR
  question if running unattended on a PR branch) and continue unaffected work.
- Do not add features, refactor code, or make improvements beyond the spec.
- If the spec has not been through `/analyze-spec` or maintainer review, say so. An explicit user
  instruction to implement it is sufficient authorization; otherwise ask.
- Commit as logical units that follow `docs/pull-requests.md`. Never push unless the user asked
  for it.
</Execution_Policy>

<Steps>

## Phase 1: Plan

1. **Read the spec**: use the path in `{{ARGUMENTS}}`, else the `specs/*.md` file changed on
   this branch versus `main` (excluding `TEMPLATE.md`). Ask if ambiguous.
2. **Read `CLAUDE.md`**: commands, crate map, the `atlas-trace` behavioral contract, the
   prover/verifier invariants.
3. **Explore relevant code**: use `Explore` agents to understand the modules, types, tests, and
   patterns the implementation will touch.
4. **Create the implementation plan** from the spec's Intent, Evaluation, and Execution sections:
   - Files to create, modify, or remove
   - Order of changes (dependencies first), and the commit sequence. Keep mechanical changes
     (renames, moves, plumbing) in their own commits, ahead of behavior changes.
   - How existing patterns and abstractions should be extended
   - Which acceptance criterion each change satisfies
   - Which CI jobs and which benchmark apply (from the spec's Testing Strategy and Performance)
5. **Show the plan** to the user, or post it as a PR comment if running unattended on a PR branch:

```
**Implementation plan for: {spec title}**

**Changes:**
1. {file/module}: {what changes and why}
2. ...

**Commit sequence:** {ordered list}
**Checks:** {CI jobs and benchmark that apply}
**Estimated scope:** {number of files, rough line count}
```

## Phase 2: Execute

1. **Implement** all changes from the plan, in the planned commit order.
   - Follow the code style and invariants in `CLAUDE.md`.
   - Run independent tasks in parallel using agents where beneficial, but run Cargo commands
     sequentially.
   - New or changed op protocols get a soundness test showing a malicious prover is rejected,
     per the spec's Testing Strategy.
2. **Commit** each logical unit as it completes, with a message that describes the change.
   A fix to an earlier local commit is squashed into it, not stacked on top.

## Phase 3: QA

Run the checks until all pass (up to 5 cycles). Repeat a check only when a failure or a
subsequent change requires it; complete the full applicable set before handoff. Tests run in the
dev profile, never `--release`.

1. **Format**: `cargo fmt --all`
2. **Lint**: `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets`
3. **Test**: `cargo nextest run --workspace`
4. **Zoo gates**, if `atlas-onnx-import` or `atlas-trace` changed:
   `cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'not test(golden_zoo_big)'`
5. **Real-model prove & verify**, if the proof system or trace semantics changed (needs the
   downloaded models; see `CLAUDE.md`):
   `cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::gpt2$/)'`
   and the BGE / Qwen equivalents if the spec names them. A new op also needs its
   `op_surface.rs` rows and a place in a soundness probe model.
6. **`zk` feature**, if the spec touches it: `cargo clippy -p jolt-atlas-core --features zk`
   plus whatever verification the spec's Testing Strategy names.
7. **Fix** any failures and repeat.

If the same error persists 3 times, stop and report the fundamental issue to the user (or as a
PR comment when unattended).

## Phase 4: Validate

1. **Correctness**: every acceptance criterion in the spec passes, with the evidence named.
2. **Performance**: run the benchmark the spec's Performance section names (the `prove`
   example on the named model with `--trace-terminal`, in `--release`) and confirm the named
   stage moved in the declared direction, or stayed within the declared tolerance. Record the
   numbers.
3. **Self-review**: consistency with existing patterns, missing edge cases, anything beyond the
   spec that crept in.
4. **Branch audit**: run `/clean-branch` and fix every finding on local commits (reshape history
   as it recommends). Findings on pushed commits become follow-up commits.

Fix any issues found and re-validate.

## Phase 5: Finalize

1. **Update spec status**: change `Status` from `proposed` to `implemented` in the spec file and
   fill in `PR` if known. Commit this with the last change, not as a fixup.
2. **Push** only if the user asked for it.
3. **Report**:

```
**Implementation complete for: {spec title}**

**Changes made:**
- {file}: {summary}
- ...

**Acceptance criteria:**
- {criterion 1}: PASS ({evidence})
- ...

**Checks run:** {list, with results}
**Benchmark:** {example, stage, before → after}
**clean-branch:** {no findings | findings fixed}
```

</Steps>

Task: Implement the spec on this branch. {{ARGUMENTS}}
