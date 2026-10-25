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
on. This spec extracts verification into a `jolt-atlas-verifier` crate — as upstream a16z/jolt
separates its verifier — so the trust perimeter becomes a cargo dependency edge: the audit
surface is one crate, the compiler enforces that verification cannot depend on prover-only
code, and lightweight verification targets (wasm, embedded, constrained environments) build
without the prover.

## Intent

### Goal

Create a `jolt-atlas-verifier` crate containing everything required to deserialize and verify an
`ONNXProof` against a model and its public IO — verifier preprocessing, the verifier opening
accumulator, sumcheck verification, PCS verification, transcripts, and the per-op verification
logic — and nothing a verifier does not need: no trace execution, no witness generation, no
commitment or proving code.

### Invariants

- Behavior-preserving: a proof that verifies before the split verifies after it, and vice versa;
  transcript byte streams are unchanged (the zoo e2es verify with transcript lockstep and
  serialization round-trips).
- The verifier crate must not depend on `jolt-atlas-core` or `atlas-trace` execution: no witness
  generation, no proving, no per-op prover code, no trace execution. This is the boundary `cargo`
  and the CI isolation build actually enforce in phase 1. Named exception: the crate depends on
  `joltworks` whole, whose prover APIs (e.g. `CommitmentScheme::commit`) remain *reachable* though
  unused; review holds that line until the upstream-adoption program
  (`specs/upstream-adoption.md`) closes it compile-time by replacing `joltworks` with vendored
  upstream verifier-side crates.
- The verifier still derives everything it trusts from the model and the proof alone: it may
  depend on `atlas-trace` for graph structure and `semantics` (lookup-table generation is part of
  the statement), never on trace *execution*.
- The proof serialization format is unchanged, or changes land in their own PR before this one.
- Soundness sweeps, the op-surface table, and the zoo run against the extracted verifier — the
  sweeps' subject is precisely the code that moves.

### Non-Goals

- Splitting `joltworks` at all. The end state is complete decomposition — the verifier's whole
  dependency cone prover-free, as upstream a16z/jolt separates it — reached by the
  upstream-adoption program (`specs/upstream-adoption.md`), not by splitting our fork. This spec
  lands only the product boundary and its CI check.
- The `zk` feature. BlindFold verification stays in `jolt-atlas-core` behind `zk` until the
  feature has tests.
- Minimizing verifier dependencies (e.g. a no-std story). The crate boundary comes first;
  dependency dieting is follow-up.
- Changing any verification logic. Pure reorganization; protocol changes land separately.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-atlas-verifier` builds in isolation (`cargo check -p jolt-atlas-verifier`) with no
  dependency on `jolt-atlas-core` (witness generation, proving, per-op prover code) or on trace
  execution. Prover APIs inside `joltworks` remain reachable in phase 1; the criterion is the
  crate edge, not API reachability (see End state).
- [ ] `ONNXProof::verify` call sites migrate to the new crate's entry point; `jolt-atlas-core`
  re-exports it during the transition so downstream code moves at its own pace.
- [ ] `cargo nextest run --workspace` passes; the zoo e2es' transcript-lockstep and
  byte-identical re-serialization checks pass against the extracted verifier. The lockstep
  comparison hooks are gated by a `test-util` cargo feature on the verifier crate (not
  `#[cfg(test)]`, which is off when the crate compiles as a dependency of core's tests), enabled
  through core's dev-dependency. A canary test forces a transcript divergence and asserts the
  lockstep check fires, so silently compiled-out hooks fail CI.
- [ ] The real-model heavy job (GPT-2 Dory, BGE, Qwen) passes once on the extraction branch.
- [ ] CI gains a job (or build step) that builds the verifier crate alone, so a future
  prover-dependency leak fails CI rather than an audit.

### Testing Strategy

The suite's value here is proving the reorganization changed nothing: every existing assertion
and mutation/forgery site is preserved. Harness *wiring* may change — the mutation sweep reaches
proof internals (`reduced_opening_proof`, `joint_opening_proof`) through module privacy today,
which a crate boundary removes — so the verifier crate exposes `test-util`-gated mutation
accessors for the proof fields the sweep targets, and `testing/mutate.rs` goes through them. The
sweeps, tamper harness and op-surface table stay in `jolt-atlas-core`'s `onnx_proof/testing/`
(they need the prover to build proofs to attack) and drive the extracted verifier through the
new crate. One new test: the lockstep canary named in the acceptance criteria.

### Performance

None expected: code motion, no algorithmic change. Verified by the `prove` example on
`models/gpt2 --pcs dory` (verify stage of the README benchmark table, no regression).

## Design

### Architecture

Target dependency graph:

```
common ─▶ joltworks* ─▶ jolt-atlas-verifier ─▶ jolt-atlas-core (prover, witness, testing)
             ▲                 ▲
             └── atlas-trace ──┘        (atlas-onnx-import unchanged)
```

\* `joltworks` is being replaced per subsystem by vendored upstream crates
(`specs/upstream-adoption.md`); the edge and the boundary drawn here are unchanged by that
substitution.

At the end state, `jolt-atlas-verifier` owns: verifier preprocessing
(`AtlasVerifierPreprocessing`), proof types and deserialization, the `Verifier` state and
verification loop, per-op verifier logic, the deferred-lookup verification, and
range-check/clamp verification parameters. `jolt-atlas-core` keeps the prover, witness
generation, trace integration, the `prove` example, and all of `testing/` (which needs both
sides).

**Proof-type ownership.** `ONNXProof` and its inherent `verify` move with the verifier crate.
Core cannot then keep inherent `prove` / `prove_from_trace` impls on the type (re-exports don't
bypass coherence), so proving becomes core-owned free functions (`jolt_atlas_core::prove`,
`prove_from_trace`) returning an `ONNXProof` built through a public `ONNXProof::from_parts`
constructor; call sites migrate mechanically. The `test-util` mutation accessors above cover the
fields `from_parts` doesn't expose.

**Buildable intermediate boundary.** Per-op prover and verifier code share files and private
helpers, and the verification loop (`verify_iop`) dispatches into per-op verifiers: moving the
loop while per-op verifiers remain in core would create a `cargo` dependency cycle (verifier
crate → core's `OperatorVerifier`, core → verifier crate's proof types). So the loop does not
move in phase 1:

1. **Skeleton first**: the crate starts with what has no core dependencies — proof types,
   deserialization, `from_parts`, verifier preprocessing — plus the CI isolation build. Core
   re-exports the moved types; the top-level `verify`, the `Verifier` state and the
   deferred-lookup verification stay in core for now.
2. **Per-op verifier logic arrives via the expression-engine migration** (see the
   verifier-simplification work): once op identities are data consumed by a generic interpreter,
   per-op verifier code largely disappears, the interpreter and op specs land in the verifier
   crate, and the verification loop and `Verifier` state move last, when nothing they dispatch to
   remains in core — far fewer files, split once instead of twice.

The enforcement value (the crate edge and its CI check) lands with step 1; the boundary tightens
monotonically until `verify` itself crosses in step 2's final move.

### End state

The end state is complete decomposition: no crate in the verifier's dependency cone contains
prover code. With `jolt-atlas-verifier` depending on `joltworks` whole, cargo cannot stop
verifier code from calling `joltworks`' prover halves, and light targets link prover machinery
they never run — the boundary has a hole one layer down.

This spec lands the product boundary and its CI isolation build: the layer where the audit risk
concentrates (per-op verification, claim plumbing). The hole below it is closed by the
upstream-adoption program (`specs/upstream-adoption.md`), which replaces `joltworks`
subsystem-by-subsystem with vendored upstream verifier-side crates (`jolt-sumcheck`,
`jolt-poly`, `jolt-openings`/`jolt-claims`, the PCS crates); that program's final stage is
exactly this spec's end state — `jolt-atlas-verifier`'s dependency cone prover-free, proven by
the isolation build. The two specs divide cleanly: this one draws the box around our ONNX
verifier (which upstream, whose `jolt-verifier` verifies RISC-V execution, can never supply);
the adoption program swaps the machinery underneath it.

### Alternatives Considered

- **A feature flag (`verifier-only`) instead of a crate**: rejected — features are additive and
  easy to mis-combine; they don't give an auditable dependency edge, and nothing stops a later
  change from leaking prover code into the "verifier-only" cfg.
- **Extracting a verifier-primitives crate out of `joltworks`** (or splitting it behind a
  `prover` feature): superseded — upstream a16z/jolt now ships the decomposed toolkit, so the
  subprotocol-level split arrives by adoption (`specs/upstream-adoption.md`) instead of by
  carving up our fork. The product-level boundary (what does *jolt-atlas* verification need?)
  remains this spec's job and is the one audits and light targets ask for first.
- **Splitting every op file by hand now**: rejected as double work; see sequencing above.

## Documentation

`README.md` (crate map, verification usage), `CLAUDE.md`/`AGENTS.md` crate map, and
`CONTRIBUTING.md` CI table gain the new crate and the isolation build job.

## Execution

Suggested commit sequence for the skeleton PR: (1) create the crate with proof types,
deserialization and `ONNXProof::from_parts` moved, core's proving as free functions, re-exports
in place; (2) move verifier preprocessing; (3) `test-util` feature with the lockstep hooks and
mutation accessors, plus the canary test; (4) CI isolation build. The `Verifier` state,
verification loop and per-op verifier moves ride later PRs per the sequencing above.

## References

- Upstream precedent: a16z/jolt's separation of verification from the prover crate.
- `specs/upstream-adoption.md` (#397): retires `joltworks` via vendored upstream crates and
  closes this spec's dependency cone; `docs/upstream-inventory.md` is its factual basis.
- `specs/core-test-overhaul.md` (the sweeps that gate this reorganization).
- Verifier-simplification arc: claim-lifecycle hardening PRs, expression-engine design (spec to
  follow).
