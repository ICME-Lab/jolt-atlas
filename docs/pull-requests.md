# Branch standard for pull requests

PRs are squash-merged into `main`. The **net diff** (`git diff main...HEAD`) is what lands on
`main`, so it must be clean; the **commit history** only exists for review, so it must be easy to
review. CI already checks formatting, lints, tests and builds; this standard covers what it can't.

## 1. Scope

- **1.1** The branch is based on `main`, or on a parent PR named in the description.
- **1.2** It contains only commits serving the PR's purpose. Unrelated work, even useful, goes in
  its own PR.

## 2. History

A **local** commit hasn't been pushed yet; a **pushed** commit has been seen by others.

**Local commits** can and should be reshaped before pushing:

- **2.1 No fixup commits.** A commit whose change only makes sense as a correction to an earlier
  local commit (formatting, lint, a typo, a forgotten file, a bug the earlier commit introduced) is
  squashed into it.
  - *One signal among others:* a commit that fails `cargo fmt --check` while a later commit makes
    it pass.
- **2.2 Only deliberate scaffolding is temporary.** Code added and later removed or replaced within
  the PR is allowed only as scaffolding: announced as temporary when introduced (in a comment or the
  commit message), and making a step easier to review. Code from an approach that was later
  abandoned does not appear at all.
- **2.3 One concern per commit.** Mechanical changes (renames, moves, plumbing) are kept separate
  from behavior changes.
- **2.4 No forward dependencies.** Each commit stands on what precedes it: it never relies on, nor
  describes as existing, something introduced later, in its message or its code. A test lands with
  or after the change it tests. Announcing a planned later change is fine.

**Pushed commits** are frozen:

- **2.5** Changes after a push, including answers to review, go in new commits on top. Rewriting
  pushed history breaks reviewers' "changes since last review" view and their comment threads.
- **2.6** Rebase onto `main` only to resolve a conflict or pick up a needed change, not routinely.

## 3. Net diff

The final diff contains only what the merged code needs:

- **3.1 No leftovers.** Nothing remains that only an intermediate state of the branch needed:
  helpers, parameters, widened visibility, bindings. For each item the PR adds or widens, something
  in the final code must need it.
- **3.2 Comments describe the code, not the change.** No "now…", "previously…", "replaces X" —
  that belongs in commit messages. No comment names an identifier that was renamed or removed.
- **3.3 No temporary material.** No debug output, ad-hoc instrumentation, one-off measurement
  tests or commented-out code. A new `#[ignore]` or `TODO` carries its reason.

Scaffolding allowed by 2.2 must therefore be gone by the tip, unless it is a `TODO` whose reason
names its follow-up.

## 4. Relations

A PR that adds an op relation (`onnx_proof/relation.rs`) also meets:

- **4.1 A constructor.** The relation is an `OpRel` value built by a constructor function in
  the op's module of `jolt-atlas-verifier` (`ops::mul::product`, `ops::add::accumulation`),
  starting from `OpRel::base` and called from the op's plan. A tie is a constructor with zero
  rounds. What is not the default goes in `Custom`, as functions, never closures.
- **4.2 A kind.** The relation has an `OpRelKind` variant, with its arm in
  `OpRelKind::relation_id` (the name it is registered under) and in `OpRelKind::serves` (the
  ops it may be built for). A node of the op appears in a probe model or the op gauntlet, so
  `relation_kinds_match_their_dispatched_op_kind` sees it.
- **4.3 A prover arm.** The kind has its arm in each of `jolt-atlas-core`'s matches on
  `OpRelKind` (`ops::{op_rel_witnesses, op_rel_tie_claims, op_rel_prover,
  op_rel_setup_claims}`), which have no wildcard: the engine's witnesses, a tie's claims, or a
  specialized prover. A specialized prover is completeness-only and has a
  `relation_prover_equiv_*` test comparing it with the relation's reference messages.
- **4.4 S ⊆ R note.** The constructor's doc argues that every execution of the op's semantics
  satisfies the relation, naming the `atlas_trace::semantics` helper it relies on where one
  exists, and names the protocols outside the relation that its argument needs (range checks,
  lookups).
- **4.5 Digests.** A PR that restates an existing relation keeps the input claim, opening
  points and opening append order, so `testing/digests.rs` passes unchanged. Changing an op's
  identity is a separate PR declared protocol-breaking.


## 5. Verifier changes

A PR that adds an op, a proof field or a check to `jolt-atlas-verifier` also meets
(`docs/verifier.md` is the reading order and the trust-perimeter statement):

- **5.1 A plan.** The op's verification is its `OpPlan::plan` (`onnx_proof/plan.rs`): ordered
  `Step`s over values (`OpRelation` for a sumcheck, `OpRel` for a tie, `OpOpen` for advice
  openings), built from the model and the public IO alone. No step holds a closure; whatever
  depends on earlier claims or on challenges is read and drawn in the relation's `setup` when
  the interpreter reaches the step. The prover walks the same plan (`jolt-atlas-core`'s
  `ops::prove_plan`), attaching a specialized witness at its `OpRelation` variant, or at its
  `OpRelKind` for an op relation. A table read whose sumcheck produces only its own
  read-address claim is a `Defer` to `DeferredBatch::Lookups`, and its one-hot check a `Defer`
  to `DeferredBatch::OneHot`; neither runs in the node's turn.
- **5.2 Relations.** Every sumcheck of the plan is a relation meeting §4; an in-the-clear check
  is a `Step::Tie` over a zero-round `OpRel`, which `verify_tie` checks against claims the
  accumulator holds, returning `Err` when the identity fails.
- **5.3 Coverage.** A node of the op is in a probe model (`onnx_proof/testing/models.rs`) or the
  op gauntlet, so the soundness sweeps, `op_surface.rs` and `proof_types_all_exercised` see it; a
  new proof field has a `mutate::Site`, and every site is rejected (no tolerance list).
- **5.4 No new allow.** The crate-root `deny` lints stay on with no `#[allow]` outside
  `#[cfg(test)]`; a missing claim, proof entry or malformed input is an `Err`. The crate stays
  Trace-free: witness code goes to `jolt-atlas-core`.
- **5.5 Digests.** A behavior-preserving change keeps `testing/digests.rs` unchanged; a
  protocol-breaking change re-pins them in its own commit and says so.
