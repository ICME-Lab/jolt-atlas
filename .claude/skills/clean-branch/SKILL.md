---
name: clean-branch
description: Audit the current branch against the repo's PR branch standard (docs/pull-requests.md) before opening or updating a pull request, and recommend how to fix each finding. Audit-only; never rewrites history. Use when asked to clean, audit, review or prepare a branch for a PR.
---

# clean-branch

Audit a branch against `docs/pull-requests.md` and report findings with recommended fixes.
Read that file first, every time: it is the authoritative rule list, and findings cite its rule
numbers. This skill only adds the procedure.

**Audit-only.** Never commit, amend, rebase, reset, push or edit files. Do not run what CI already
runs (lints, tests, builds); the one exception is per-commit `cargo fmt --check`, which CI only
runs on the tip.

## 1. Establish the range

- Branch: `git branch --show-current`, or the one the user names.
- Base: `main`, unless the branch sits on a parent PR branch. Find the nearest branch containing
  the merge base, and confirm with the user if ambiguous. Range = `<base>..HEAD`.
- Local vs pushed: if `origin/<branch>` exists, local commits are `git log origin/<branch>..HEAD`
  and the rest are pushed. Otherwise every commit is local.

## 2. Scope (rules 1.x)

List the range with `git log --oneline --stat`. Flag commits whose content doesn't serve the PR's
purpose (1.2), and a base that is neither `main` nor a declared parent (1.1).

## 3. History (rules 2.x)

Check **local** commits for 2.1–2.4. For pushed commits, still report what you find, but the
recommendation is a follow-up commit (2.5), never a rewrite.

- **2.1 fixups:** for each later commit, ask whether its change only corrects an earlier local
  commit (same lines, typo, forgotten file, bug that commit introduced); compare `git show` of
  both. Also run `cargo fmt --all --check` at each commit in a temporary detached worktree
  (`git worktree add --detach <dir> <commit>`, removed afterwards). A failure fixed by a later
  commit is a fixup signal.
- **2.2 introduce-then-remove:** find code added by one commit and removed or replaced by a later
  one (`git log -p` on the touched files). Allowed only if announced as temporary when introduced;
  otherwise it's an abandoned approach.
- **2.3 mixed concerns:** commits that mix a mechanical change (rename, move, plumbing) with a
  behavior change.
- **2.4 forward dependencies:** identifiers, files or fixtures a commit's message or code relies on
  that don't exist yet at that commit (`git grep <name> <commit>`); tests added before the change
  they test. Announcements of later changes are fine.

## 4. Net diff (rules 3.x)

On `git diff <base>...HEAD`:

- **3.1 leftovers:** for each item the diff adds or widens (functions, fields, parameters,
  `pub`/`pub(crate)` visibility, imports), find what in the final code needs it. The compiler
  misses `pub` items and items only used by other leftovers. Comparing against code the branch
  replaces (an earlier version of the branch, or a superseded PR) is the most reliable way to
  find them.
- **3.2 comments:** added comments narrating the change ("now", "previously", "replaces", "no
  longer", "new"/"old" approach), and comments naming identifiers that no longer exist.
- **3.3 temporary material:** `dbg!`/`println!`/`eprintln!` outside tests and examples, ad-hoc
  timing, one-off measurement tests, commented-out code, new `#[ignore]` or `TODO` without a
  reason. Scaffolding allowed by 2.2 that's still present at the tip is a finding here.

## 5. Report

Group findings by rule number. For each one, give:
- the rule (e.g. `2.2`);
- the location: commit hash for history findings, `file:line` for net-diff findings;
- what's wrong, in one or two sentences;
- the recommended fix:
  - **local commit:** the concrete reshaping, e.g. `git commit --fixup=<target>` then
    `GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base>`, drop, reorder, or split;
  - **pushed commit:** a follow-up commit and what it changes.

Only report findings you verified. Mark judgment calls (whether a commit serves the PR's purpose,
or mixes concerns) as such. End with a one-line count per section, and say plainly when a section
has no findings.
