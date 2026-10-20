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
