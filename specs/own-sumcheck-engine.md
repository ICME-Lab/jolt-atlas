# Spec: own-sumcheck-engine — jolt-sumcheck's engine becomes atlas-sumcheck::engine

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-pcs-adapter.md` (phase 2, implemented: `jolt-crypto`, the `committed` feature's only would-be consumer, is gone) |
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

Move the part of `jolt-sumcheck` that atlas reaches (from its `prover`, `verifier`, `batch`,
`claim`, `domain`, `error`, `proof`, `round_proof` and `recorder` modules) into
`atlas-sumcheck/src/engine/`, generic over `atlas_transcript::Transcript` (the trait
`prove_relation_batch` and `verify_relation_batch` already bound on, not the
`ProvingTranscript`/`VerifyingTranscript` subtraits) and `atlas_algebra::field::JoltField`, with
the grind inside the engine's challenge draw: each challenge is `transcript.pow(bits)` then
`transcript.challenge_scalar_optimized::<F>().into()`. Split up and delete `upstream_sumcheck.rs`
and delete `third_party/jolt-sumcheck`, with every round message, challenge and proof byte
unchanged.

Moved: the items atlas calls and whatever they need internally. Not moved: the uniskip prover
(`prove_uniskip_*`, `ProvedUniskip*`), the uncompressed verify and `ClearProof` path, the
integer domains no atlas path reaches, and the `committed`/`r1cs` code. The compiler settles the
exact list: nothing copied is left without a caller.

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
- **Absorption bytes unchanged.** The engine absorbs only through `Transcript::append_bytes`, with
  upstream's legacy encodings: a field element as its `F::NUM_BYTES` canonical bytes, big-endian
  (`jolt_transcript::legacy`'s blanket `AppendToTranscript`); `LabelWithCount` as 32 bytes, the
  label zero-padded to 24 then the count as a big-endian `u64`; the input-claim and round labels
  (`b"sumcheck_claim"`, `SUMCHECK_ROUND_TRANSCRIPT_LABEL`) as today. Never `append_scalar` or
  `append_scalars`, which tag and encode differently (`atlas-transcript/src/sponge.rs`). The codec
  is a private `engine` module until phase 4 owns the transcript. Mechanism: digests.
- **The proof-of-work grind happens exactly where it does today**: before each challenge the
  verifier draws, with `grind_bits_for_loss` from the transcript crate. Mechanism: digests
  (a moved or skipped grind changes every challenge).
- **Upstream's engine tests run.** `tests.rs` and `round_scheduler_tests.rs` never ran in the
  vendored build: they are gated `#[cfg(all(test, feature = "committed"))]` and written against
  BN254 `Fr`, `Blake2bTranscript` and `jolt_crypto`'s Pedersen. The tests that cover the moved
  surface (about 25 of 43) are ported to `Fp128` and `SpongeProver`/`SpongeVerifier` and run
  under `cargo nextest run -p atlas-sumcheck`; the committed, uniskip and integer-domain tests go
  with their code. Mechanism: Test job.
- **The Akita bridge survives.** `atlas-pcs::akita` drives the external `akita-*` batch opening,
  bound on `jolt_transcript::Transcript<Challenge = AkitaField>`, through `FacadeTranscript`
  (`akita/mod.rs`, the two `scope_with` calls). It moves to `atlas-pcs/src/akita/facade.rs` as
  `pub(crate)`, unchanged in behavior, and lives as long as `akita-*` is external; `atlas-pcs`
  keeps its `jolt-transcript` dependency. Mechanism: digests (Akita openings are in every proof).

### Non-Goals

- Changing the batching strategy, round scheduling, the `GruenSplitEqPolynomial` fold path or
  the compressed round proof. This is a move, not a redesign.
- Owning `jolt-poly`'s `UnivariatePoly`/`CompressedPoly`, which the engine's proof type uses:
  the engine keeps importing `jolt_poly` until phase 5.
- The `committed`/`r1cs` features and their code: not copied (atlas commits through the PCS,
  not through the sumcheck recorder; R1CS is the zkVM's). Nor the uniskip prover, the
  uncompressed verify path or unused domains (see Goal).

## Evaluation

### Acceptance Criteria

- [ ] `atlas-sumcheck/src/engine/` holds the moved code; `atlas-sumcheck/src/upstream_sumcheck.rs`
      is deleted; `third_party/jolt-sumcheck` is deleted; no source names `jolt_sumcheck` or
      `upstream_sumcheck` (the CI grep step "No vendored PCS crate or upstream scheme trait named
      in atlas sources" gains both); no `jolt-sumcheck` row in any manifest.
- [ ] No copied item without a caller, and no `#[allow(dead_code)]` in `engine/`.
- [ ] `FacadeTranscript` is `pub(crate)` in `atlas-pcs::akita::facade`.
- [ ] `jolt-atlas-verifier/clippy.toml` has no `jolt_sumcheck` rows and lists the surviving
      engine prover items under `atlas_sumcheck::engine::prover::*`; the uniskip, committed and
      `CommittedSumcheckRecorder` rows and the `allow-invalid` note that existed for them are
      gone; the isolation job passes.
- [ ] `cargo nextest run -p atlas-sumcheck` passes, including the ported engine tests (the
      report states how many were ported and which were dropped with their code).
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

`atlas-sumcheck/src/engine/` with the moved modules (those the kept surface needs) and a
private codec module. The engine's transcript bound changes from upstream's facade `Transcript`
to `atlas_transcript::Transcript`; the grind-before-challenge step moves from
`upstream_sumcheck::FacadeTranscript::challenge` into the engine's challenge draw.
`upstream_sumcheck.rs` splits: `InstanceRounds`, `round_check_failed`, `expect_engine` and the
compressed-poly converters move to `engine`; `FacadeTranscript` moves to
`atlas-pcs/src/akita/facade.rs`. The relation core (`relation/`) and every subprotocol keep
their current imports with the path renamed; `jolt-atlas-core`'s twelve
`use jolt_sumcheck::{ProveRounds, SumcheckError}` sites re-point to `atlas_sumcheck::engine`.
`jolt-atlas-verifier` sees `verify_compressed` through `atlas_sumcheck::sumcheck`, as today.

Manifests: `jolt-sumcheck` rows go from the root (member and workspace dependency) and from
`atlas-algebra`, `atlas-transcript`, `atlas-lookup-tables`, `jolt-atlas-verifier` (unused
edges), `atlas-pcs` and `jolt-atlas-core`; `atlas-sumcheck` gains whatever the engine needs
(`thiserror`, `tracing`, `serde` if still used).

### Alternatives Considered

- **Keep the facade and only move the crate.** Rejected: the facade is the symptom (two
  transcript traits meeting in one adapter) and goes with ownership.
- **Rewrite the engine around our relation core.** Rejected for this phase: the digest gate
  wants a move. A redesign, if wanted, is a later spec with the engine already ours.

## Documentation

`PINS.md`, `docs/upstream-inventory.md` (the `subprotocols/sumcheck*` row: engine owned from
`f3265b5a`), `docs/verifier.md` (the reading order, and the perimeter row on line 21 loses
`jolt-sumcheck`), `CLAUDE.md`/`AGENTS.md` (the "both driving upstream `jolt-sumcheck`'s engine"
sentence and the crate maps), `docs/roadmap.md`.

## Execution

1. Copy the modules under `engine/`, compile with the facade still present (two engines, one
   used); port the tests. Gate: `cargo nextest run -p atlas-sumcheck`.
2. Switch the relation core to `engine`, delete the facade, re-point `clippy.toml`. Gate:
   digests, isolation job.
3. Delete the crate; docs; spec status; GPT-2 run recorded.

## References

`a16z/jolt` `crates/jolt-sumcheck` at `f3265b5a`; `atlas-sumcheck/src/upstream_sumcheck.rs`;
`specs/adoption-program.md` stage 4b; `docs/roadmap.md`.
