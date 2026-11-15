# Spec: drop-arkworks — own serialization and big integers, remove the fork pin

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | implemented                    |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/own-field.md` (phase 6; after it the only arkworks users are our own crates) |
| Supersedes  | `docs/upstream-inventory.md` "The `akita-*` patch" sentence on the arkworks fork pin (`third_party/PINS.md` is gone since phase 6); the root `Cargo.toml` `[patch.crates-io]` arkworks entries |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

No curve is left in the workspace, but about 37 source files in 7 crates still import
arkworks: the proof object, the opening claims, the commitments and the transcript absorb
through `ark_serialize::CanonicalSerialize`/`CanonicalDeserialize` (15 derive sites and 38
hand-written `CanonicalSerialize`/`CanonicalDeserialize`/`Valid` impls), the field ladder and
accumulators use the fork's `BigInt` and signed big integers (`S64`…`S256`), `fp128` implements
`UniformRand`, and tests use `ark_std::test_rng`. These pull `a16z/arkworks-algebra` at a moving branch (`dev/twist-shout`)
through `[patch.crates-io]`, a fork of a library whose only remaining job here is byte
encoding and a few integer types. A verifier audit should not have to read arkworks. This
phase replaces those uses with an owned `atlas-algebra::serialize` (canonical little-endian
encoding with the same byte layout) and the integer types phase 6 already owns
(`field::Limbs<N>`, `field::signed`), and removes the pin.

## Intent

### Goal

Replace every `ark_serialize`, `ark_ff` and `ark_std` use in atlas crates with owned items
(`atlas_algebra::serialize::{Canonical, Error}` with derive support; `field::Limbs<N>` for
`BigInt<N>` and `field::signed::{S64, S128, S160, S192, S256}` for the fork's signed
integers, with any missing method added there; the field's own `random` for `UniformRand`;
`rand` for test RNGs), keeping the proof wire format byte identical, and remove the arkworks
dependencies and `[patch.crates-io]` entries from the workspace.

**Trait shape.** One encoding, no mode flags: `Canonical` has `serialize(&self, &mut impl
Write)`, `serialized_size(&self)` and `deserialize(&mut impl Read)`, which always validates.
It is byte identical to today's `serialize_compressed`/`deserialize_compressed` (every atlas
call site uses the compressed, validated mode; with no curve points, compressed and
uncompressed write the same bytes). The one unchecked-decode test in `field::fp128` is dropped.

**One type per concept.** No `bigint` module: the fork's types map onto the owned
`Limbs`/`signed` types phase 6 brought from upstream `jolt-field` (itself a port of the same
fork code).

### Invariants

- **Wire format unchanged.** Every serialized proof, commitment and setup is byte identical
  to the arkworks encoding (the compressed mode atlas uses; little-endian field elements,
  `u64` length-prefixed vectors as arkworks writes them). Mechanism: the digest pins (which hash serialized proofs),
  the zoo tests' serialization round-trips, and a one-time fixture test: proofs serialized by
  the arkworks path on `dev` before the branch, checked in under
  `jolt-atlas-core/src/onnx_proof/testing/fixtures/` (written by a test run on `dev` at the
  branch point over the zoo proofs), deserialized and re-serialized by the new path with
  equality asserted. It stays for the switch commits and is deleted in the phase's last
  commit once the digest pins carry the guarantee; that commit's message says so.
- **Transcript absorption unchanged.** `append_*` on the transcript serializes with the same
  bytes. Mechanism: digests.
- **Verifier rejects, never panics, on malformed bytes.** The new deserializer returns `Err`
  on short input, bad length prefixes and non-canonical field elements, exactly where
  arkworks did. Mechanism: `fuzz_regressions_replay` and the `verify` fuzz target
  (`jolt-atlas-core/fuzz/`: `cargo +nightly fuzz run verify -- -max_total_time=3600`, run by
  hand before merge with the result in the PR; not a per-push check); the
  `proof_mutation_sweep` (truncation and bit-flip sites).
- **Field arithmetic unchanged.** The ladder, the folded accumulators and
  `utils/accumulation` compute the same values on `Limbs`/`signed` as on the fork's types.
  Mechanism: the ladder and accumulator tests in `atlas-algebra`; digests.
- **No `ark-*` crate in `Cargo.lock`.** Mechanism: a CI step (the no-BN254 check generalizes
  to `^ark-`).

### Non-Goals

- Changing the proof format (compression, field-element endianness, length encoding). A
  format redesign, if ever, is its own spec with a `ProofId` bump.
- `allocative`: stays (its own pin), unrelated to arkworks except that the fork gated derives
  on it; the moved integer types carry their own `Allocative` impls.
- `spongefish`, `blake2`, `rand`: external, unchanged.

## Evaluation

### Acceptance Criteria

- [ ] No `ark_ff`, `ark_serialize`, `ark_std`, `ark_bn254` or `ark_ec` in any atlas source;
      no `ark-*` in `Cargo.lock`; the `[patch.crates-io]` arkworks entries are gone.
- [ ] `atlas_algebra::serialize` provides the trait pair and a derive (proc-macro crate
      `atlas-serialize-derive` or hand impls, implementer's choice recorded) used at the
      15 derive sites; the 38 hand impls are ported to it.
- [ ] The fixture round-trip test passes; digest pins unchanged; zoo round-trips pass.
- [ ] The `verify` fuzz target run for at least one hour on the new deserializer with no
      panic (command and duration in the PR).
- [ ] CI green; heavy job green on `main`; `docs/verifier.md` perimeter row for external
      crates updated.

### Testing Strategy

Workspace Test job (digests, zoo round-trips, `proof_mutation_sweep`, `fuzz_regressions_replay`),
isolation job (with the `^ark-` step), heavy job; the fixture test; the fuzz run by hand.
No `mutate::Site` changes (same format).

### Performance

Serialization is off the prover's hot path. Sanity check on `models/gpt2-tiny`; the
`ONNXProof::prove` total within noise. Build time: expected decrease (arkworks' `ark-ff`
with `asm`/`parallel` is no longer compiled); recorded with `cargo build --timings`.

## Design

### Architecture

`atlas-algebra/src/serialize.rs` (the trait, `Error`, impls for primitives, `Vec`, arrays,
tuples, `Option`, the field); the ladder (`field::fp128`, `field::folded_accum`,
`JoltField::UnreducedElem`, `UnreducedInteger`) and `utils/{accumulation,small_scalar,math}`
on `field::Limbs`/`field::signed`. Consumers: `jolt-atlas-verifier` (proof object, `proof_serialization`, claims),
`atlas-pcs` (commitments, setups, Akita proof wrappers), `atlas-transcript` (absorption),
`atlas-sumcheck` (proof types), `jolt-atlas-core` (witness and prover types), tests
(`test_rng` to `rand`). The `akita-*` crates resolve no arkworks crate on normal or build edges (at `dfdbd767` only `akita-pcs`'s benches dev-depend on `ark-ff`/`ark-bn254`, which our graph never resolves); the
owned `jolt-field` package needs no arkworks surface.

### Alternatives Considered

- **Depend on crates.io `ark-serialize` 0.5 without the fork.** Rejected: still a foreign
  dependency in the perimeter, and the `BigInt`/signed types would still come from the fork.
- **Switch the wire format to `serde`+`bincode`.** Rejected: a format change breaks every
  pin and every stored proof; out of scope by the Non-Goals.

## Documentation

`docs/upstream-inventory.md` (the arkworks pin sentence and rows removed), `docs/verifier.md`
(perimeter: external crates list), `CLAUDE.md`/`AGENTS.md` (the arkworks sentence removed),
`README.md` if it mentions arkworks, `docs/roadmap.md` (program complete).

## Execution

1. Add `serialize.rs` with tests; move the ladder onto `Limbs`/`signed`; the fixture test against the arkworks path
   (both paths compiled). 2. Switch crates one at a time, verifier first (isolation job per
   step), digests after each. 3. Remove dependencies, pins, patches; CI step; docs.

## References

`docs/upstream-inventory.md` "The `akita-*` patch"; `a16z/arkworks-algebra` `dev/twist-shout`
(`ark-ff/src/biginteger/`, `ark-serialize/src/`); `jolt-atlas-core/fuzz/README.md`;
`docs/roadmap.md`.
