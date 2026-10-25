# Spec: Standalone Verifier Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | #396                           |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Soundness depends only on what the verifier checks, but today the verifier is interleaved with
the prover across `jolt-atlas-core`: every op module holds a prover/verifier pair, witness
generation sits next to verification, and nothing stops verifier code from reaching prover
internals. An auditor asking "what exactly does the verifier accept?" has no boundary to stand
on. This spec creates a `jolt-atlas-verifier` crate — as upstream a16z/jolt separates its
verifier — so the trust perimeter becomes a cargo dependency edge. **This PR ships phase 1,
the proof-type boundary**: the crate exists, owns the proof artifact, and carries the CI
isolation build; verification logic crosses in later phases as the per-op code is freed from
its prover halves (via the expression-engine migration) and the toolkit underneath is replaced
by upstream adoption (`specs/upstream-adoption.md`).

## Intent

### Goal

Phase 1 (this PR): create `jolt-atlas-verifier` owning the proof artifact — `ONNXProof`, its
serialization, construction (`from_parts`) and read accessors — with a CI build that compiles
the crate in isolation, and convert every inherent `ONNXProof` method in `jolt-atlas-core`
(proving *and* verification) into core-owned free functions over the moved type.

End state (later phases, closed jointly with `specs/upstream-adoption.md`): the crate owns
verifier preprocessing, the `Verifier` state and verification loop, per-op verification and
deferred-lookup verification, with an inherent `verify` back on the type, and a dependency cone
containing no prover code.

### Invariants

- Behavior-preserving in every phase: a proof that verifies before a move verifies after it;
  transcript byte streams are unchanged (zoo e2es with transcript lockstep and serialization
  round-trips).
- The verifier crate must not depend on `jolt-atlas-core`. This is the boundary `cargo` and the
  CI isolation build actually enforce. Two reachable-but-unused exceptions are held by review,
  not by the compiler, until closed: (a) `joltworks` prover APIs (e.g.
  `CommitmentScheme::commit`) — closed compile-time by upstream adoption; (b) `atlas-trace`
  *execution* (`Model::trace` lives in the same crate as the graph and `semantics` the verifier
  legitimately needs) — closed by a future `atlas-trace` feature split; until then, "the
  verifier never executes the trace" is a review-held line, not a compiler-held one.
- The proof serialization format is unchanged, or changes land in their own PR before this one.
- Every existing soundness-sweep assertion and mutation/forgery site is preserved; only harness
  wiring may change (see Testing Strategy).
- No prover type appears in the verifier crate's API, in any phase. `ProverDebugInfo` stays
  core-owned; lockstep comparison happens on core's side (see Testing Strategy).

### Non-Goals

- Moving verification logic in this PR. The `Verifier` state, `verify` loop, per-op verifiers,
  deferred-lookup verification and verifier preprocessing stay in core until later phases (see
  Design for why each would otherwise create a dependency cycle).
- Splitting `joltworks` at all: superseded by the upstream-adoption program
  (`specs/upstream-adoption.md`), whose final stage closes this spec's dependency cone.
- The `zk` feature. BlindFold verification stays in `jolt-atlas-core` behind `zk`.
- Minimizing verifier dependencies (no-std, the `atlas-trace` feature split): follow-ups.
- Changing any verification logic. Pure reorganization; protocol changes land separately.

## Evaluation

### Acceptance Criteria

Phase 1 — this PR:

- [ ] `jolt-atlas-verifier` exists, owns `ONNXProof` + serialization + `ONNXProof::from_parts`
  (taking **every** field: construction is production prover code, so nothing about it is
  test-gated) + public read accessors for the fields core's verification free functions
  consume.
- [ ] `cargo check -p jolt-atlas-verifier` builds in isolation with no dependency on
  `jolt-atlas-core`; CI gains that build as a job, so a future core-dependency leak fails CI.
- [ ] Core's inherent `ONNXProof` methods are free functions (`jolt_atlas_core::prove`,
  `prove_from_trace`, `verify`, and the verification helpers they call), over the moved type —
  both sides at once, because a type defined in another crate admits no inherent impls in core
  (the same coherence rule either way).
- [ ] A `test-util` feature on the verifier crate gates `&mut` accessors for proof fields, and
  `testing/mutate.rs` reaches proof internals only through them (it loses module-privacy access
  when the type moves). Reads need no gating (public accessors); construction needs none
  (`from_parts`).
- [ ] `cargo nextest run --workspace` passes with every sweep assertion and mutation/forgery
  site intact; the zoo's transcript-lockstep and byte-identical re-serialization checks pass.
- [ ] The real-model heavy job (GPT-2 Dory minimum) passes once on the branch.

Later phases (recorded here for the program's shape; ticked by their own PRs, not this one):

- [ ] Verifier preprocessing moves once the per-op committed-polynomial registry
  (`NodeCommittedPolynomials`) moves with the per-op migration.
- [ ] The `Verifier` state, verification loop and deferred verification move when no per-op
  verifier remains in core; `verify` re-attaches to `ONNXProof` as an inherent method and
  core's free `verify` becomes a re-export shim, then disappears.
- [ ] With `verify` in the crate: lockstep becomes a `test-util` **state recorder** on the
  verifier's transcript/accumulator — the verifier never sees `ProverDebugInfo`; core's test
  harness installs the recorder and compares the recording against prover state on its side. A
  canary test forces a transcript divergence and asserts the recorder-based lockstep catches
  it, so silently compiled-out hooks fail CI.
- [ ] The isolation build compiles `default-features` (no `test-util`) in every phase.

### Testing Strategy

The suites prove the reorganization changed nothing: op-surface table, the three soundness
sweeps, the zoo, importer goldens. Harness wiring changes are confined to `testing/mutate.rs`
(test-util mutation accessors replace direct field access) in phase 1, and to the lockstep
recorder when `verify` moves. In phase 1 the lockstep path is untouched: core's free `verify`
keeps the existing `#[cfg(test)]` comparison block; the recorder and canary land only in the
phase that moves `verify`, because until then there is nothing in the verifier crate to gate —
a phase-1 canary would exercise core's `cfg(test)` path and tick a box without protecting
anything.

### Performance

Code motion only; verified by the `prove` example on `models/gpt2 --pcs dory` (README benchmark
table, verify stage, no regression). Note: examples build with dev-dependencies, so a core
dev-dep enabling `test-util` puts the hook-bearing configuration in the benchmark binary. The
spec therefore requires `test-util` hooks (mutation accessors now, the state recorder later) to
be **inert unless explicitly invoked** — no recording, no allocation, nothing beyond an
installed-recorder check — and the benchmark runs with no recorder installed, so the measured
verifier is the one downstream users build.

## Design

### Architecture

Target dependency graph (end state):

```
common ─▶ joltworks* ─▶ jolt-atlas-verifier ─▶ jolt-atlas-core (prover, witness, testing)
             ▲                 ▲
             └── atlas-trace ──┘        (atlas-onnx-import unchanged)
```

\* `joltworks` is being replaced per subsystem by vendored upstream crates
(`specs/upstream-adoption.md`); the edge and the boundary drawn here are unchanged by that
substitution.

**Phase 1 — the proof-type boundary.** What moves is exactly the no-core-dependency set: the
`ONNXProof` struct, its serialization, `from_parts`, read accessors, and `test-util` mutation
accessors. Everything else once considered for the skeleton fails a dependency check:

- *Verification logic* (`verify`, `Verifier`, `verify_iop`, deferred verification) calls core's
  per-op `OperatorVerifier` — moving it is a cargo cycle. It stays in core as free functions
  over the moved type. The free-function conversion is forced by coherence anyway: once
  `ONNXProof` is defined in another crate, core can keep inherent impls for neither `prove` nor
  `verify`, so both sides convert symmetrically in the same commit.
- *Verifier preprocessing* (`AtlasVerifierPreprocessing`) embeds `AtlasSharedPreprocessing`,
  whose `get_models_committed_polynomials` calls the per-op committed-polynomial registry
  (`NodeCommittedPolynomials` in `ops/`) — the same cycle. It stays in core until the per-op
  migration moves the registry.

**Field-access design.** `from_parts` takes every proof field: the prover constructs complete
proofs in production code, so a partial constructor plus test-gated setters cannot serve it.
Reads used by core's verification free functions (the reduced opening proof, the joint opening
proof, claims, commitments) are public read-only accessors — production code, not test code.
The `test-util` feature gates only *mutation* (the sweep's tampering), keeping the production
proof API immutable after construction.

**Per-op verifier logic arrives via the expression-engine migration** (the
verifier-simplification work): once op identities are data consumed by a generic interpreter,
per-op verifier code largely disappears, the interpreter and op specs land in the verifier
crate, and the verification loop, `Verifier` state and preprocessing move last, when nothing
they dispatch to remains in core.

### End state

The end state is complete decomposition: no crate in the verifier's dependency cone contains
prover code. With `jolt-atlas-verifier` depending on `joltworks` whole, cargo cannot stop
verifier code from calling `joltworks`' prover halves — the boundary has a hole one layer
down. The hole is closed by the upstream-adoption program (`specs/upstream-adoption.md`),
which replaces `joltworks` subsystem-by-subsystem with vendored upstream verifier-side crates;
that program's final stage is exactly this spec's end state, proven by the same CI isolation
build. The two specs divide cleanly: this one draws the box around our ONNX verifier (which
upstream, whose `jolt-verifier` verifies RISC-V execution, can never supply); the adoption
program swaps the machinery underneath it.

### Alternatives Considered

- **A feature flag (`verifier-only`) instead of a crate**: rejected — features are additive and
  easy to mis-combine; they don't give an auditable dependency edge, and nothing stops a later
  change from leaking prover code into the "verifier-only" cfg.
- **Extracting a verifier-primitives crate out of `joltworks`** (or splitting it behind a
  `prover` feature): superseded — upstream ships the decomposed toolkit; the subprotocol-level
  split arrives by adoption (`specs/upstream-adoption.md`) instead of by carving up our fork.
- **Splitting every op file by hand now**: rejected as double work; the expression-engine
  migration separates per-op prover/verifier code as a by-product.
- **Moving `verify` in phase 1 behind a dependency-inversion trait** (core registers per-op
  verifiers into the crate at runtime): rejected — it buys the entry point early at the cost of
  a dynamic dispatch seam inside the audit surface that the end state doesn't want, and the
  cycle analysis above shows the honest phase-1 boundary is the proof type.

## Documentation

`README.md` (crate map, proof-type ownership), `CLAUDE.md`/`AGENTS.md` crate map, and
`CONTRIBUTING.md` CI table gain the new crate and the isolation build job.

## Execution

Suggested commit sequence for the phase-1 PR: (1) convert core's inherent `ONNXProof` methods
to free functions in place (mechanical, reviewable alone); (2) create the crate and move the
type, serialization, `from_parts` and read accessors, updating call sites; (3) `test-util`
mutation accessors and the `testing/mutate.rs` rewiring; (4) the CI isolation build
(`default-features`). Later phases ride their own PRs per the sequencing above.

## References

- Upstream precedent: a16z/jolt's separation of verification from the prover crate.
- `specs/upstream-adoption.md` (#397): retires `joltworks` via vendored upstream crates and
  closes this spec's dependency cone; `docs/upstream-inventory.md` is its factual basis.
- `specs/core-test-overhaul.md` (the sweeps that gate this reorganization).
- Verifier-simplification arc: claim-lifecycle hardening PRs (#394, #395), expression-engine
  design (spec to follow).
