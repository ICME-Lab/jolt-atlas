# Spec: own-sumcheck-engine — jolt-sumcheck's engine becomes atlas-sumcheck::engine

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-pcs-adapter.md` (phase 2; removes the `committed` feature's only would-be consumer) |
| Supersedes  | `third_party/PINS.md` section `jolt-sumcheck`; `specs/adoption-program.md` stage 4b's "run on upstream's engine through a facade" |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every atlas sumcheck is proven by `jolt-sumcheck`'s `prove_batch` and verified by its
`verify_compressed`, reached through `atlas_sumcheck::upstream_sumcheck`, a 232-line facade
that presents our transcript as upstream's and grinds before each challenge. The engine is
4,900 lines, a third of which (`committed.rs`, `r1cs.rs`, the committed recorder) is behind
features we declare empty, and `tests.rs` alone is 1,554 lines we cannot run with the vendored
manifest. The facade exists only because the engine is not ours. This phase moves the engine
in, as `atlas-sumcheck::engine`, with our transcript as its native transcript, and deletes
the facade and the dead feature-gated code.

## Intent

### Goal

Copy `jolt-sumcheck`'s `prover`, `verifier`, `batch`, `claim`, `domain`, `error`, `proof`,
`round_proof` and `recorder` modules into `atlas-sumcheck/src/engine/`, make them take
`atlas_transcript`'s prover and verifier transcripts directly (grinding inside the engine's
challenge draw), delete `upstream_sumcheck.rs` and `third_party/jolt-sumcheck`, with every
round message, challenge and proof byte unchanged.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job. The engine's round
  order, the compressed round-proof encoding and the challenge draw (grind, then sample) are
  what the transcript sees; any drift is a digest change.
- **Prover/verifier split holds at item level.** The engine's prover items (`prove_batch`,
  `ProveRounds`, `RoundScheduler`, `MemberRound`, `MemberFinish`, `ClearSumcheckRecorder`,
  `RecordedSumcheck`, `ProvedBatch`) live in `engine::prover`; the verifier side
  (`verify_compressed`, `SumcheckProof`, `RoundProof`, `SumcheckError`, `SumcheckDomain`,
  `BatchPrelude`) in `engine` proper. Mechanism: the CI isolation job's `::prover` path grep
  on `jolt-atlas-verifier`; `clippy.toml`'s `jolt_sumcheck::*` rows are rewritten to the
  `atlas_sumcheck::engine::prover::*` paths in the same PR (review-held that the list is
  complete; the path grep is the backstop).
- **The relation core's contract is unchanged.** `RelationWitness` still extends the engine's
  `ProveRounds`; `RelationVerifier` still drives `verify_compressed`; `prove_relation{,_batch,
  _batch_parallel}` and `verify_relation{,_batch}` keep their signatures. Mechanism: the
  compiler (every op relation and subprotocol compiles unchanged) and
  `relation_specs_match_their_dispatched_op_kind`.
- **The proof-of-work grind happens exactly where it does today**: before each challenge the
  verifier draws, with `grind_bits_for_loss` from the transcript crate. Mechanism: digests
  (a moved or skipped grind changes every challenge).
- **Upstream's engine tests run.** `tests.rs` and `round_scheduler_tests.rs` (1,756 lines)
  are ported to our transcript and field and run under `cargo nextest run -p atlas-sumcheck`.
  Mechanism: Test job.

### Non-Goals

- Changing the batching strategy, round scheduling, the `GruenSplitEqPolynomial` fold path or
  the compressed round proof. This is a move, not a redesign.
- Owning `jolt-poly`'s `UnivariatePoly`/`CompressedPoly`, which the engine's proof type uses:
  the engine keeps importing `jolt_poly` until phase 5.
- The `committed`/`r1cs` features and their code: not copied (atlas commits through the PCS,
  not through the sumcheck recorder; R1CS is the zkVM's).

## Evaluation

### Acceptance Criteria

- [ ] `atlas-sumcheck/src/engine/` holds the copied modules; `atlas-sumcheck/src/upstream_sumcheck.rs`
      is deleted; `third_party/jolt-sumcheck` is deleted; no source names `jolt_sumcheck`.
- [ ] `jolt-atlas-verifier/clippy.toml` has no `jolt_sumcheck` rows and lists the engine's
      prover items under their new paths; the isolation job passes.
- [ ] `cargo nextest run -p atlas-sumcheck` passes, including the ported engine tests.
- [ ] Digest pins unchanged; CI green; heavy job green on `main`.
- [ ] `docs/verifier.md` reading order names `atlas_sumcheck::engine` in place of
      `jolt_sumcheck::verify_compressed`; `PINS.md`, inventory, crate maps updated.

### Testing Strategy

Workspace Test job (digests, zoo, soundness sweeps unchanged), isolation job, heavy job; the
ported engine unit tests. No proof-format change, so no `mutate::Site`; no op change.

### Performance

The engine's inner loop is the prover's hot path. Gate: the `prove` example on `models/gpt2`
(real download) with `--trace-terminal`, IOP proving stage within 3% of the `dev` baseline
taken on the same machine the same day; `models/gpt2-tiny` for iteration. Expected: no
change (same code). Removing the facade's per-challenge indirection may show a sub-1%
improvement; not a target.

## Design

### Architecture

`atlas-sumcheck/src/engine/{mod,prover,verifier,batch,claim,domain,error,proof,round_proof,
recorder}.rs`. The engine's transcript bound changes from upstream's `Transcript` facade to
`atlas_transcript::ProvingTranscript`/`VerifyingTranscript`; the grind-before-challenge step
moves from `upstream_sumcheck::GrindingTranscript` into the engine's challenge draw. The
relation core (`relation/`) and every subprotocol keep their current imports with the path
renamed. `jolt-atlas-verifier` sees `verify_compressed` through `atlas_sumcheck::engine`.

### Alternatives Considered

- **Keep the facade and only move the crate.** Rejected: the facade is the symptom (two
  transcript traits meeting in one adapter) and goes with ownership.
- **Rewrite the engine around our relation core.** Rejected for this phase: the digest gate
  wants a move. A redesign, if wanted, is a later spec with the engine already ours.

## Documentation

`PINS.md`, `docs/upstream-inventory.md` (the `subprotocols/sumcheck*` row: engine owned from
`f3265b5a`), `docs/verifier.md`, `CLAUDE.md`/`AGENTS.md` (the "both driving upstream
`jolt-sumcheck`'s engine" sentence), `docs/roadmap.md`.

## Execution

1. Copy the modules under `engine/`, compile with the facade still present (two engines, one
   used); port the tests. Gate: `cargo nextest run -p atlas-sumcheck`.
2. Switch the relation core to `engine`, delete the facade, re-point `clippy.toml`. Gate:
   digests, isolation job.
3. Delete the crate; docs; spec status; GPT-2 run recorded.

## References

`a16z/jolt` `crates/jolt-sumcheck` at `f3265b5a`; `atlas-sumcheck/src/upstream_sumcheck.rs`;
`specs/adoption-program.md` stage 4b; `docs/roadmap.md`.
