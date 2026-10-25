# Spec: Verifier Boundary — the Proof Type Gets Its Own Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | —                              |
| Supersedes  | PR #396 (closed; this is its phase 1, freed of the program it was entangled with) |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Soundness depends only on what the verifier checks, but verification is interleaved with the
prover across `jolt-atlas-core`, so an auditor asking "what exactly does the verifier accept?"
has no boundary to stand on. The full separation is a program (see
`specs/adoption-program.md`); this spec is its first, independent step: a `jolt-atlas-verifier`
crate that owns the **proof artifact** — `ONNXProof` and the types its fields reference — with
a CI build that compiles the crate in isolation, so the boundary exists, is enforced from day
one, and tightens monotonically as the program moves verification logic across it. No
verification logic moves in this PR; every candidate for that was traced to a cargo dependency
cycle (Design).

## Intent

### Goal

Create `jolt-atlas-verifier` owning the proof artifact closed under its field types —
`ONNXProof`, `Claims`, `ProofId`/`ProofType`, `ReducedOpeningProof`, their serialization,
production `from_parts` constructors and public read accessors — with a CI isolation build, and
convert every inherent `ONNXProof` method in `jolt-atlas-core` (proving *and* verification)
into core-owned free functions over the moved type.

### Invariants

Each invariant names the mechanism that holds it.

- **Behavior-preserving, checked across the move, not within one build**: the in-build zoo
  harness proves and verifies with the same code and cannot see a coordinated change, so this
  PR pins proof-bytes and transcript digests from deterministic zoo inputs at the pre-move
  commit and asserts the post-move build reproduces them byte for byte. Mechanism: the pinned
  digests, plus the zoo's transcript-lockstep and byte-identical re-serialization checks.
- **The verifier crate does not depend on `jolt-atlas-core`.** Mechanism: cargo, plus a CI job
  building `cargo check -p jolt-atlas-verifier` in isolation. Two reachable-but-unused
  exceptions are review-held, not compiler-held, until `specs/adoption-program.md` closes them:
  `joltworks` prover APIs (the crate is depended on whole) and `atlas-trace` execution
  (`Model::trace` shares a crate with the graph and `semantics` the verifier legitimately
  needs).
- **The proof serialization format is unchanged.** Mechanism: the zoo round-trips every proof
  through serialize/deserialize/re-serialize byte-identically.
- **Every soundness-sweep assertion and mutation/forgery site is preserved**; only harness
  wiring changes (`testing/mutate.rs` loses module-privacy access to proof fields and goes
  through `test-util` accessors). Mechanism: the sweeps themselves, run unmodified in count and
  expectation.
- **No prover type appears in the verifier crate's API.** `ProverDebugInfo` stays core-owned;
  the `#[cfg(test)]` lockstep comparison stays in core's free `verify`, untouched. Mechanism:
  the isolation build (core types cannot be named) and review.

### Non-Goals

- Moving verification logic. `verify`, the `Verifier` state, `verify_iop`, deferred
  verification, verifier preprocessing and all per-op verifiers stay in core; each would create
  a dependency cycle today (Design), and they cross later under `specs/adoption-program.md`.
- The `zk` feature entirely. It does not compile today (57 pre-existing errors under
  `cargo check -p jolt-atlas-core --features zk`; the tracer rewrite left it behind and the
  test overhaul deleted its never-compiling tests), so this PR does not update `zk.rs`'s
  inherent-method call sites (`ONNXProof::polynomial_map`/`commit_to_polynomials`/
  `output_claim` at `zk.rs:1601–1620`) and cannot regress it. zk is owned by a dedicated
  revival spec scheduled **after** the adoption program (`specs/adoption-program.md` records
  it), which will restore the build on the by-then-current architecture — most likely by
  adopting upstream `jolt-blindfold` rather than repairing the fork.
- Minimizing verifier dependencies (no-std, an `atlas-trace` feature split): follow-ups.
- Changing any verification logic or identity.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-atlas-verifier` exists and owns `ONNXProof`, `Claims`, `ProofId`/`ProofType` and
  `ReducedOpeningProof` with their serialization; each struct with private fields has a
  production `from_parts` taking **every** field (core's prover constructs
  `ReducedOpeningProof` through its private `joint_opening_proof`) and public read accessors
  for what core's free functions consume. Nothing about construction or reads is test-gated.
- [ ] `cargo check -p jolt-atlas-verifier` builds in isolation with no `jolt-atlas-core`
  dependency; CI gains that build as a job.
- [ ] Core's inherent `ONNXProof` methods are free functions (`jolt_atlas_core::prove`,
  `prove_from_trace`, `verify` and their helpers) — both sides at once, because a type defined
  in another crate admits no inherent impls in core.
- [ ] A `test-util` feature gates `&mut` proof-field accessors; `testing/mutate.rs` uses only
  them. Hooks are inert unless invoked (examples build with dev-dependencies, so the benchmark
  binary carries the feature; it must measure the verifier downstream users build).
- [ ] Pinned pre-move proof and transcript digests reproduce byte for byte post-move.
- [ ] `cargo nextest run --workspace` green with sweep sites intact; the real-model heavy job
  (GPT-2 Dory minimum) passes once on the branch.

### Testing Strategy

The existing suites prove the reorganization changed nothing: op-surface table, the three
soundness sweeps, the zoo, importer goldens — all unmodified except `testing/mutate.rs`'s
accessor rewiring. New: the pinned-digest cross-move check. No new tolerance anywhere.

### Performance

Code motion only. Verified by the `prove` example on `models/gpt2 --pcs dory` (README benchmark
table, no regression); the `test-util` inertness requirement above keeps the measurement
honest.

## Design

### Architecture

```
common ─▶ joltworks ─▶ jolt-atlas-verifier ─▶ jolt-atlas-core (prover, witness, testing)
             ▲                 ▲
             └── atlas-trace ──┘        (atlas-onnx-import unchanged)
```

What moves is exactly the no-core-dependency set, closed under field types. Everything else
fails a dependency check today:

- *Verification logic* calls core's per-op `OperatorVerifier` — moving it is a cargo cycle. It
  stays in core as free functions over the moved type (forced by coherence anyway, and done for
  prove and verify symmetrically in one commit).
- *Verifier preprocessing* (`AtlasVerifierPreprocessing`) embeds `AtlasSharedPreprocessing`,
  whose `get_models_committed_polynomials` calls the per-op registry
  (`NodeCommittedPolynomials` in `ops/`) — the same cycle.

Those cycles dissolve when `specs/adoption-program.md`'s relation-migration stages empty core
of per-op verifiers; the loop, state and preprocessing cross then, and `verify` re-attaches to
`ONNXProof`. This spec deliberately contains no schedule for that: the program spec owns the
order of operations end to end.

**Field-access design.** `from_parts` takes every field (prover construction is production
code); reads used by core's verification free functions are public read-only accessors
(production code); `test-util` gates only mutation, keeping the production proof API immutable
after construction.

### Alternatives Considered

- **A `verifier-only` feature flag instead of a crate**: rejected — features are additive and
  give no auditable dependency edge.
- **Moving `verify` now behind a dependency-inversion trait** (core registers per-op verifiers
  at runtime): rejected — a dynamic seam inside the audit surface that the end state doesn't
  want, bought only to claim an entry point early.
- **Bundling this with the adoption program in one spec**: rejected — tried (closed #396/#397);
  the boundary is independent of and smaller than everything else, and entangling it coupled
  its approval to months of unresolved questions.

## Documentation

`README.md` (crate map, proof-type ownership), `CLAUDE.md`/`AGENTS.md` crate map,
`CONTRIBUTING.md` CI table (the isolation job).

## Execution

Commits: (1) convert core's inherent `ONNXProof` methods to free functions in place
(mechanical, reviewable alone); (2) create the crate, move the types + serialization +
`from_parts` + read accessors, update call sites; (3) `test-util` mutation accessors and
`testing/mutate.rs` rewiring; (4) pinned digests; (5) CI isolation build.

## References

- `specs/adoption-program.md`: the program this is step 1 of; owns the order of operations and
  the definition of done for the full prover/verifier split.
- `specs/core-test-overhaul.md`: the sweeps and zoo gating this reorganization.
- Closed PR #396: design history, including the review rounds that produced the coherence,
  field-access and cycle analyses above.
