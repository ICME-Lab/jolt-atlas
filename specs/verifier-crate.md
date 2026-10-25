# Spec: Standalone Verifier Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          |                                |

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
- The verifier crate must not depend on prover-only code. Enforced by `cargo`, and by CI building
  `jolt-atlas-verifier` in isolation.
- The verifier still derives everything it trusts from the model and the proof alone: it may
  depend on `atlas-trace` for graph structure and `semantics` (lookup-table generation is part of
  the statement), never on trace *execution*.
- The proof serialization format is unchanged, or changes land in their own PR before this one.
- Soundness sweeps, the op-surface table, and the zoo run against the extracted verifier — the
  sweeps' subject is precisely the code that moves.

### Non-Goals

- Splitting `joltworks` in the same PR. `joltworks` mixes prover and verifier halves of each
  subprotocol; the extraction initially depends on `joltworks` whole, with its prover/verifier
  split tracked as follow-up work (feature-gating or a `joltworks-verifier` sibling).
- The `zk` feature. BlindFold verification stays in `jolt-atlas-core` behind `zk` until the
  feature has tests.
- Minimizing verifier dependencies (e.g. a no-std story). The crate boundary comes first;
  dependency dieting is follow-up.
- Changing any verification logic. Pure reorganization; protocol changes land separately.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-atlas-verifier` builds in isolation (`cargo check -p jolt-atlas-verifier`) with no
  path to prover-only modules (`witness`, `prover`, commitment code, trace execution).
- [ ] `ONNXProof::verify` call sites migrate to the new crate's entry point; `jolt-atlas-core`
  re-exports it during the transition so downstream code moves at its own pace.
- [ ] `cargo nextest run --workspace` passes unchanged; the zoo e2es' transcript-lockstep and
  byte-identical re-serialization checks pass against the extracted verifier.
- [ ] The real-model heavy job (GPT-2 Dory, BGE, Qwen) passes once on the extraction branch.
- [ ] CI gains a job (or build step) that builds the verifier crate alone, so a future
  prover-dependency leak fails CI rather than an audit.

### Testing Strategy

No new tests: the suite's value here is proving the reorganization changed nothing. The soundness
sweeps (`proof_mutation_sweep`, `witness_tamper_sweep`, `opening_forgery_sweep`), the op-surface
table, and the zoo all exercise the verifier through its public entry point and must pass
unmodified. The sweeps and tamper harness stay in `jolt-atlas-core`'s `onnx_proof/testing/`
(they need the prover to build proofs to attack) and drive the extracted verifier through the
new crate.

### Performance

None expected: code motion, no algorithmic change. Verified by the `prove` example on
`models/gpt2 --pcs dory` (verify stage of the README benchmark table, no regression).

## Design

### Architecture

Target dependency graph:

```
common ─▶ joltworks ─▶ jolt-atlas-verifier ─▶ jolt-atlas-core (prover, witness, testing)
             ▲                 ▲
             └── atlas-trace ──┘        (atlas-onnx-import unchanged)
```

`jolt-atlas-verifier` owns: verifier preprocessing (`AtlasVerifierPreprocessing`), proof types
and deserialization, the `Verifier` state and verification loop, per-op verifier logic, the
deferred-lookup verification, and range-check/clamp verification parameters. `jolt-atlas-core`
keeps the prover, witness generation, trace integration, the `prove` example, and all of
`testing/` (which needs both sides).

The main obstacle is that per-op prover and verifier code share files and private helpers.
Two viable sequencings:

1. **Skeleton first**: create the crate with the top-level verify entry point, proof types, and
   whatever moves cleanly (verifier.rs, preprocessing, deferred verification); per-op verifier
   structs move file-by-file as they are touched. The dependency-direction CI check lands
   immediately; the boundary tightens over time.
2. **After the expression-engine migration** (see the verifier-simplification work): once op
   identities are data consumed by a generic interpreter, per-op verifier code largely
   disappears, and the extraction moves the interpreter plus op specs — far fewer files, split
   once instead of twice.

Recommendation: do (1)'s skeleton and CI check early for the enforcement value, and let the
bulk of per-op code arrive via (2), avoiding splitting 30+ op files by hand only to delete the
split halves during the engine migration.

### Alternatives Considered

- **A feature flag (`verifier-only`) instead of a crate**: rejected — features are additive and
  easy to mis-combine; they don't give an auditable dependency edge, and nothing stops a later
  change from leaking prover code into the "verifier-only" cfg.
- **Extracting out of `joltworks` instead (verifier primitives crate)**: deferred — the
  subprotocol prover/verifier split inside `joltworks` is real but much larger; the product-level
  boundary (what does *jolt-atlas* verification need?) is the one audits and light targets ask
  for first.
- **Splitting every op file by hand now**: rejected as double work; see sequencing above.

## Documentation

`README.md` (crate map, verification usage), `CLAUDE.md`/`AGENTS.md` crate map, and
`CONTRIBUTING.md` CI table gain the new crate and the isolation build job.

## Execution

Suggested commit sequence for the skeleton PR: (1) create the crate with proof types +
deserialization moved; (2) move `verifier.rs`, verifier preprocessing, and deferred verification,
re-exporting from `jolt-atlas-core`; (3) CI isolation build. Per-op moves ride later PRs.

## References

- Upstream precedent: a16z/jolt's separation of verification from the prover crate.
- `specs/core-test-overhaul.md` (the sweeps that gate this reorganization).
- Verifier-simplification arc: claim-lifecycle hardening PRs, expression-engine design (spec to
  follow).
