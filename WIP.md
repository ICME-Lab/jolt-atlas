# WIP: single Gather

Spec for merging `GatherSmall` and `GatherLarge` into one `Op::Gather`:
[`specs/single-gather.md`](specs/single-gather.md). Remove this file before opening the spec PR.

The branch is based on `main` at `329b21ef` and holds only the spec. It is not pushed.

## Status

`/new-spec` is done: every section was settled with the author, and the spec is written. Next is
`/analyze-spec`.

The spec depends on #406 (address selector), which is still open. The spec PR waits for #406 to
merge, so that the Summary's claims describe `main` as it then stands.

## Next steps

1. `/analyze-spec` on `specs/single-gather.md`.
2. After #406 merges: rebase on `main`, delete this file, open a PR containing only the spec, and put
   its number in the spec's header table. Apply the `ci:heavy` label to the implementation PR.
3. After maintainers approve the spec: `/implement-spec` on this branch, following the spec's
   Execution section (baseline first, then six commits).

## Claims to check before or during `/analyze-spec`

- `zk.rs` not compiling on `main` was last checked on 2026-10-27. The spec states it in three places.
- The commit 1 hook (a test-only flag and one adjusted `index_onehot` entry in
  `GatherProver::initialize`) has not been run. Commit 1 tests it: the forgery has to be accepted.
- The zero-round execution sumcheck for a one-row dictionary was traced through the code, not run.
- The Performance table is an estimate from round counts and table sizes.

## Evidence behind the Summary

On `main` at `329b21ef`, a throwaway test (not committed) ran a full `proof.verify` with HyperKZG on a
model with one Gather, indices `[0, 1]`, a word size of 2, and the trace output for index 0 tampered
from `[1, 2]` to `[7777, 7777]`:

| Case | Result |
|---|---|
| `GatherSmall`, dict 16, tamper only | rejected (`SumcheckVerificationError`) |
| `GatherSmall`, dict 16, tamper and solved `ra` claim | accepted |
| `GatherLarge`, dict `2^16 + 1`, tamper and solved `ra` claim | rejected (`SumcheckVerificationError`) |

The hook used then stored each round message of `GatherProver` and evaluated it in
`ingest_challenge`, then sent `running_claim / (dict_final + γ·identity_final)` as the `ra` claim in
`cache_openings`. The test used `prove_tampered` and `run_verify` from `soundness.rs`. The patch is
lost; rebuild it from this description if the smaller hook of commit 1 does not reproduce the
forgery.
