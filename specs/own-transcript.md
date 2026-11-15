# Spec: own-transcript — jolt-transcript's split surface becomes atlas-transcript

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | implemented                    |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-sumcheck-engine.md` (phase 3, implemented) |
| Supersedes  | `third_party/PINS.md` section `jolt-transcript`; `specs/adoption-program.md` stage 3's vendoring |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`atlas-transcript` is 505 lines over `jolt-transcript`'s 2,719: our `ProvingTranscript` and
`VerifyingTranscript` wrap upstream's spongefish `ProverState`/`VerifierState`, its
`ChallengeCodec`s, `grind_bits_for_loss` and the Blake2b/Keccak sponges. Of upstream's
modules we use `prover`, `verifier`, `setup`, `codec`, `challenge`, `grind_bits`, `pow` and
`digest`; `poseidon.rs` is dead here. `legacy.rs` (the pre-spongefish facade `Transcript`) is
not: our own Akita adapter absorbs through it at about 50 sites (`atlas-pcs`'s
`akita/backend/{adapters,native_batching,scheme}.rs`, `akita/mod.rs`, `openings/prefix.rs`,
`mock.rs`), bridged by `akita/facade.rs`. The external `akita-*` crates never name
`jolt-transcript` (Akita keeps its own Fiat-Shamir state and receives a session label we
derive). The in-tree consumers are `atlas-algebra` (`ChallengeCodec`, `ChallengeSetSize`),
`atlas-transcript`, `atlas-sumcheck::engine` (`grind_bits_for_loss`) and `atlas-pcs`. The
Fiat-Shamir transcript is the one component where "we own the code" matters most for an
audit: `docs/verifier.md` lists it in the perimeter, and a reviewer should read one crate,
not a wrapper plus a vendored crate. This phase moves the used modules in and deletes the
wrapper.

## Intent

### Goal

Copy `jolt-transcript`'s `prover`, `verifier`, `setup`, `codec`, `challenge`, `grind_bits`,
`pow` and `digest` modules into `atlas-transcript`, with upstream's `ProverTranscript`/
`VerifierTranscript` extension traits on spongefish's states kept intact (a diff base for later
ports); `SpongeProver`/`SpongeVerifier` stay atlas's transcript types on top of them (operation
tags, NARG ownership, test history). `legacy`, `poseidon` and the BN254/Poseidon-gated code are
not copied.

Port the Akita adapter (and `openings/prefix.rs`, `mock.rs`) onto `atlas_transcript::Transcript`.
The deployed-format encodings (`Label`: the label zero-padded to one 32-byte word;
`LabelWithCount`: label padded to 24 bytes then a big-endian `u64`; `U64Word`; field elements
as big-endian canonical bytes; `append_values`) become one public module of `atlas-transcript`,
all through `append_bytes`; the sumcheck engine's private codec moves into it.
`FacadeTranscript` is deleted; Akita's tests move from `Blake2bTranscript` to
`Blake2bProver`/`Blake2bVerifier`. Delete `third_party/jolt-transcript`, with every absorbed
byte, challenge and grind unchanged.

### Invariants

- **Behavior-preserving.** Digest pins unchanged. Mechanism: Test job. The transcript is the
  digest: any change to domain separation, codec, sponge or grind changes every pin.
- **Wire constants unchanged**: `PROTOCOL_ID` (`a16z/jolt-transcript/v1`, padded to 64 bytes)
  and `INSTANCE`; the operation tags (`TAG_*`); the legacy word encodings above; the Akita
  session labels (`jolt-akita/batch`, `jolt-akita/precommitted-group-batch/v4`, …); the bridge
  challenge (`challenge_scalar::<Fp128>()`, as `draw_inner` does today). Mechanism: digests
  (Akita openings are in every proof).
- **`JoltField::challenge_set_size` and the 125-bit challenge codec keep their definitions,
  without a cycle.** `atlas-transcript` depends on `atlas-algebra`, so `ChallengeSetSize` and
  `grind_bits_for_loss` (`grind_bits.rs`, std-only) move to `atlas-algebra::field` and
  `atlas-transcript` re-exports them. The `ChallengeCodecField` bound must be satisfiable
  without `atlas-algebra` depending on `atlas-transcript`: either `ChallengeCodec` and its
  `Fp128` impl move to `atlas-algebra` (taking a `spongefish` dependency only if
  `VerificationError` cannot become a local error) or the bound changes; the implementer picks
  and records it in the PR. Mechanism: the compiler; digests.
- **`jolt-poly` manifest delta.** Its `transcript` feature is declared empty and its optional
  `jolt-transcript` dependency dropped (the precedent was `jolt-sumcheck`'s empty `committed`
  feature); the `cfg` gate in `evaluation_claim.rs` stays valid and must stay off. `atlas-pcs`
  stops enabling it; `PINS.md` records the delta.
- **Prover/verifier split.** `ProvingTranscript`, the proof-of-work grinder and
  `setup::prover_transcript` stay behind the `prover` feature or in a `prover` submodule;
  `clippy.toml`'s `jolt_transcript::*` rows are rewritten. Mechanism: isolation job.
- **spongefish pin unchanged** (`=0.8.0`) and the `blake2 0.11` / `digest 0.11` pair the
  vendored crate brought is reduced to one `blake2` if the sponge code can be written against
  `0.10` without changing output; otherwise both stay and the PR says so. Prover grinding
  (spongefish `yolocrypto`) stays behind `atlas-transcript`'s `prover` feature. Mechanism:
  digests (a changed sponge changes the pins); `Cargo.lock` review.
- **Upstream's transcript tests run** (in-source, in `challenge.rs`, `grind_bits.rs`, `pow.rs`,
  `codec.rs`). Mechanism: `cargo nextest run -p atlas-transcript`.

### Non-Goals

- Changing the sponge, codec, domain separators or grind parameters.
- The Poseidon transcript and the legacy transcript: not copied.
- The Keccak sponge is kept only if a build uses it (only the unused `KeccakProver`/
  `KeccakVerifier` aliases name it today; dropped if none, recorded in the PR).

## Evaluation

### Acceptance Criteria

- [ ] `third_party/jolt-transcript` is deleted; no source names `jolt_transcript` or
      `FacadeTranscript`, and no manifest row names `jolt-transcript`. CI's source grep adds
      `jolt_transcript` and `FacadeTranscript`; its graph check adds `jolt-transcript`.
- [ ] One transcript trait in the tree: the Akita adapter, `openings/prefix.rs` and `mock.rs`
      take `atlas_transcript::Transcript`; the deployed encodings are one public module, used
      by them and by `atlas-sumcheck::engine`.
- [ ] `clippy.toml`'s two `jolt_transcript` rows point at the new prover paths; the isolation
      job passes.
- [ ] `cargo nextest run -p atlas-transcript` passes, including upstream's tests in
      `challenge`, `grind_bits`, `pow` and `codec` that are not BN254/Poseidon-gated.
- [ ] `cargo nextest run -p atlas-pcs` passes (47 tests) on atlas transcripts.
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

`atlas-transcript/src/{prover,verifier,setup,codec,challenge,pow,digest}.rs` alongside
`transcript.rs`/`sponge.rs` (the atlas-facing `Transcript` trait and the two state types) and
the public deployed-encoding module; `grind_bits.rs` in `atlas-algebra::field` (see the cycle
invariant). Consumers that change: `atlas-algebra` (its imports), `atlas-sumcheck::engine` (its
codec moves out), `atlas-pcs` (the Akita port, the facade deleted); seven manifests drop
`jolt-transcript`.

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
