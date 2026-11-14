# Spec: own-transcript — jolt-transcript's split surface becomes atlas-transcript

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-sumcheck-engine.md` (phase 3; the engine is then the only in-tree consumer of the transcript besides `atlas-transcript` itself) |
| Supersedes  | `third_party/PINS.md` section `jolt-transcript`; `specs/adoption-program.md` stage 3's vendoring |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`atlas-transcript` is 505 lines over `jolt-transcript`'s 2,719: our `ProvingTranscript` and
`VerifyingTranscript` wrap upstream's spongefish `ProverState`/`VerifierState`, its
`ChallengeCodec`s, `grind_bits_for_loss` and the Blake2b/Keccak sponges. Of upstream's
modules we use `prover`, `verifier`, `setup`, `codec`, `challenge`, `grind_bits`, `pow` and
`digest`; `legacy.rs` (the pre-spongefish transcript) and `poseidon.rs` are dead here. The
Fiat-Shamir transcript is the one component where "we own the code" matters most for an
audit: `docs/verifier.md` lists it in the perimeter, and a reviewer should read one crate,
not a wrapper plus a vendored crate. This phase moves the used modules in and deletes the
wrapper.

## Intent

### Goal

Copy `jolt-transcript`'s `prover`, `verifier`, `setup`, `codec`, `challenge`, `grind_bits`,
`pow` and `digest` modules into `atlas-transcript`, make `ProvingTranscript`/
`VerifyingTranscript` the types those modules define (not wrappers), delete
`third_party/jolt-transcript`, with every absorbed byte, challenge and grind unchanged.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job. The transcript is the
  digest: any change to domain separation, codec, sponge or grind changes every pin.
- **`JoltField::challenge_set_size` and the 125-bit challenge codec keep their definitions**
  (`atlas-algebra` names `jolt_transcript::ChallengeSetSize` and `ChallengeCodec`; they move
  to `atlas-transcript` and `atlas-algebra` imports them from there, or they move into
  `atlas-algebra::field` if that avoids a cycle; the implementer picks and records it).
  Mechanism: the compiler; digests.
- **Prover/verifier split.** `ProvingTranscript`, the proof-of-work grinder and
  `setup::prover_transcript` stay behind the `prover` feature or in a `prover` submodule;
  `clippy.toml`'s `jolt_transcript::*` rows are rewritten. Mechanism: isolation job.
- **spongefish pin unchanged** (`=0.8.0`) and the `blake2 0.11` / `digest 0.11` pair the
  vendored crate brought is reduced to one `blake2` if the sponge code can be written against
  `0.10` without changing output; otherwise both stay and the PR says so. Mechanism: digests
  (a changed sponge changes the pins); `Cargo.lock` review.
- **Upstream's transcript tests run** (in-source, in `challenge.rs`, `grind_bits.rs`, `pow.rs`,
  `codec.rs`). Mechanism: `cargo nextest run -p atlas-transcript`.

### Non-Goals

- Changing the sponge, codec, domain separators or grind parameters.
- The Poseidon transcript and the legacy transcript: not copied.
- The Keccak sponge is kept only if a build uses it (`atlas-algebra` enables
  `transcript-keccak` today; the implementer checks for a consumer and drops the feature if
  none, recorded in the PR).

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-transcript` is deleted; no source names `jolt_transcript`.
- [ ] `atlas-transcript` defines the transcript types directly over spongefish; the
      wrapper-style `sponge.rs` forwarding is gone.
- [ ] `clippy.toml` has no `jolt_transcript` rows; the isolation job passes.
- [ ] `cargo nextest run -p atlas-transcript` passes including the ported tests.
- [ ] Digest pins unchanged; CI green; heavy job green on `main`.
- [ ] `docs/verifier.md` perimeter row drops `jolt-transcript`; `PINS.md`, inventory, crate
      maps updated.

### Testing Strategy

Workspace Test job (digests), isolation job, heavy job; the ported in-source tests. No
proof-format change.

### Performance

Transcript cost is a small fraction of prover time (grinding dominates it and is unchanged).
Sanity check on `models/gpt2-tiny`; no gate beyond "no regression" on the IOP stage, read
from the `--trace-terminal` output.

## Design

### Architecture

`atlas-transcript/src/{prover,verifier,setup,codec,challenge,grind_bits,pow,digest}.rs`
alongside the existing `transcript.rs`/`sponge.rs`, which shrink to the atlas-facing
`Transcript` trait (challenge type, `append_*`, `challenge_*`) and the two state types.
Consumers (`atlas-sumcheck::engine`, `atlas-pcs`, the prover and verifier crates) already go
through `atlas_transcript`; only `atlas-algebra`'s two imports change.

### Alternatives Considered

- **Leave the wrapper and only move the crate.** Rejected: the wrapper is the duplicate.
- **Replace spongefish with an in-house sponge.** Rejected: it is a generic external library
  we do not change (roadmap principle), and replacing it is a protocol change.

## Documentation

`PINS.md`, `docs/upstream-inventory.md` (`transcripts/` row), `docs/verifier.md`,
`CLAUDE.md`/`AGENTS.md`, `docs/roadmap.md`.

## Execution

1. Copy the modules in, compile both paths, port tests. 2. Switch the two state types to the
copied definitions, delete the forwarding, re-point `clippy.toml` and `atlas-algebra`'s
imports. 3. Delete the crate; docs; status.

## References

`a16z/jolt` `crates/jolt-transcript` at `f3265b5a`; `atlas-transcript/src/sponge.rs`;
`specs/adoption-program.md` stage 3; `docs/roadmap.md`.
