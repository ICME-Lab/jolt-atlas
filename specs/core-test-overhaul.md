# Spec: Core Test Overhaul

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | #392                           |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The `jolt-atlas-core` test suite predated the atlas-trace rewrite: one `mod tests` per op, ten
single-op tests at `T = 2^16` costing 50–100 s each (975 of the workspace's 1,022 CPU-seconds),
36 zk tests that never compiled in CI, soundness coverage hand-written for `Sub` only, and no
Qwen prove test at all. PR #382 shipped a prove-time panic on Qwen's `amk,ank->amn` einsum past
340 green tests. The GPT-2 and BGE CI jobs each recompiled the workspace for a few minutes of
proving. This spec replaces the suite with fewer tests that cover more: an exhaustive op table,
generic soundness sweeps that can find verifier gaps, tiny stand-in models with the production
architectures, and a CI layout where PRs run in minutes and real models run off the PR path.

## Intent

### Goal

Replace every test in `jolt-atlas-core` with one directory, `onnx_proof/testing/`, whose
coverage is enforced structurally (a new `atlas_trace::Op` variant or proof field cannot go
untested) and whose PR run completes in about a minute, while real GPT-2 / BGE / Qwen proofs move
to a heavy CI job.

### Invariants

- Every `atlas_trace::Op` variant has a declared stance in `op_surface.rs`: a proving case, an
  `#[ignore]` with its reason, or an expected rejection. `OpKind::of` is a wildcard-free match
  and the `dispatch_operator!` macro loses its wildcard arm, so both fail to compile otherwise.
- Every end-to-end proof goes through one harness: reference execution, prove, verify with
  transcript lockstep, serialize, deserialize, verify again, re-serialize byte-identically.
- Soundness is tested generically, not per op: every field of a finished proof is either absorbed
  by the transcript or checked (`proof_mutation_sweep`); every effective witness tamper is
  rejected (`witness_tamper_sweep`); every forged opening claim is rejected
  (`opening_forgery_sweep`). A verifier panic on a value mutation is a failure. Prover
  self-checks must not hide verifier behavior (probe mode).
- Tiny stand-ins must be structurally the real models: same architecture family, same ONNX
  exporter and flags, and each e2e asserts the importer passes the real model relies on fired.
- Big-model golden hashes in `atlas-onnx-import/tests/golden_zoo.rs` do not change (the shared
  input builder keeps its RNG draw order).
- Open gaps found by the new tests are pinned as `#[ignore]`d `known_gap_*` tests that fail today;
  nothing is allowlisted silently.
- Trace semantics, the fused-rescale split, lookup tables and padding behavior are unchanged.
  The only production-code changes are the `dispatch_operator!` wildcard removal, the
  `prove`/`prove_from_trace` split, a test-feature forgery hook in `joltworks`, and the sumcheck
  round-count assertion becoming an `Err`.

### Non-Goals

- Fixing the gaps the sweeps find (Neg of `i32::MIN`, one-row `GatherSmall` dictionaries,
  non-power-of-two sequence lengths for Qwen2, the Broadcast padded-lane gap, dead
  `BlindFoldBatchOpening` claims, verifier panics on missing opening claims). Each is pinned or
  documented; fixes are their own PRs.
- The `zk` feature. Its tests never compiled in CI and are deleted; `zk.rs` is untouched.
- In-model `GatherLarge` coverage on every PR: a tiny Qwen2 with a vocab above 65,536 would put
  about 16 MB in git. `GatherLarge` is covered by a single-op case and by the real-Qwen heavy job.
- Keeping the legacy microgpt / nanoGPT / transformer / multihead_attention models and their
  examples.

## Evaluation

### Acceptance Criteria

- [x] `cargo nextest run --workspace` passes in about a minute of wall time on a laptop (measured
  49 s; the previous suite's test phase took 274 s).
- [x] Removing an arm from `OpKind::of` or a row from the op table fails to compile or fails
  `every_op_kind_has_a_declared_case`; `every_case_contains_its_declared_op` passes.
- [x] Saturation cases assert that saturation happened (the previous magnitudes were sized for
  scale 8 and no longer saturated at 14).
- [x] The three sweeps pass on the probe models; the forgery sweep rejects every targeted forgery.
- [x] `gpt2-tiny`, `qwen2-tiny` and `bge-tiny` import, fire the asserted passes
  (`repeat_to_gather` and the `amk,ank->amn` einsum for Qwen2), and prove and verify.
- [x] The real GPT-2 (Dory), BGE and Qwen (Dory, seq 8) e2es pass locally (55 s, 15.6 s, 95.6 s).
- [x] Every `known_gap_*` test fails for the reason in its `#[ignore]` message.
- [x] `golden_zoo_big` hashes are unchanged.

### Testing Strategy

The PR is the test suite. CI: `Test` (workspace run plus full-zoo import gates) on every PR;
`Real model (gpt2 / bge / qwen)` on main, nightly, `workflow_dispatch` and PRs labelled `ci:heavy`;
`Exhaustive soundness sweeps` (`JOLT_SOUNDNESS_EXHAUSTIVE=1`) nightly. Dory tests share one
nextest group because dory-pcs races on its on-disk setup cache.

### Performance

Test-suite time only; prover performance is unchanged. The `prove` example replaces the
per-model examples as the benchmark driver (`--model models/gpt2 --pcs dory --trace-terminal`).

## Design

### Architecture

`jolt-atlas-core/src/onnx_proof/testing/` (a descendant of `onnx_proof`, so private proof fields
are reachable without widening visibility):

- `harness.rs`: `prove_verify_roundtrip<PCS>`.
- `op_kind.rs` + `op_surface.rs` (+ `op_surface_dory.rs`): the table and its meta-tests.
- `probe.rs`, `models.rs`, `mutate.rs`, `soundness.rs`: probe mode, probe models, mutation sites,
  the sweeps, binding tests and `known_gap_*` pins.
- `zoo.rs`: tiny stand-ins on every run, real models behind `#[ignore]`.
- `model_builder.rs`: the former `test_builder.rs`, no longer public API.

`atlas_onnx_import::fixtures` (feature `test-util`) owns the descriptor-driven input builder,
shared by the golden tests, the core e2es and the `prove` example. `scripts/gen_tiny_models.py`
builds the stand-ins from random-init Hugging Face configs with the pinned toolchain in
`scripts/requirements-models.txt`; the download scripts share `scripts/onnx_export_utils.py`.

### Alternatives Considered

- One giant test per sweep or per op table: rejected; one `#[test]` per case keeps nextest's
  process isolation and named failures.
- Keeping `T = 2^16` for a few ops: rejected after auditing the prover for size-gated branches
  (none above `T = 8` besides bucket packing, which has its own case); one `2^15` smoke case stays.
- A nextest archive built once and shared with the heavy jobs: rejected for the PR path, since it
  adds an upload to the critical path for a benefit only the off-PR jobs see; rust-cache with a
  shared key is enough.
- Generating the tiny models in CI instead of checking them in: rejected; exporter drift would
  make the golden hashes flaky, and the files are under 1 MB each.

## Documentation

`README.md` (examples, zoo, testing), `CLAUDE.md` / `AGENTS.md` (commands, invariants),
`CONTRIBUTING.md` (CI table), `specs/TEMPLATE.md` and the spec skills (test and benchmark
references) are updated in the same PR.

## References

- PR #392 (this change); PR #382 (the Qwen einsum panic that motivated a Qwen prove test).
- `atlas-trace/src/model.rs::broadcast_fill_inflates_padded_reduction` (the padded-lane gap).
