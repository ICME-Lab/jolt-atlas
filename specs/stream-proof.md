# Spec: The proof is a stream

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/verification-schedule.md` (the schedule fixes the order the stream is read in), `specs/one-clamp.md` (no proof part is optional on a claim's value) |
| Supersedes  | `specs/verifier-audit-simplification.md`'s "exact proof shape" mechanisms (`ProofLedger`, the drain check, the orphan mutation sites), which this spec makes unnecessary |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

A proof is a set of keyed containers: a map from `OpeningId` to claim, a map from
`ProofId` to sumcheck proof, a map from node to evaluation-reduction proof, a list of
commitments, an optional reduced opening proof, and a byte string, the NARG, that today
holds only the proof-of-work nonces. The verifier loads every claim before it draws a
challenge, then looks each item up when its turn comes.

A keyed proof can be malformed in ways a transcript cannot. An entry can be missing, present
and never read, read twice, or stored under a key nothing asks for; a claim can sit in the
map without ever being bound to a point; a round message can carry an extra coefficient. Each
of these has a guard: `ProofLedger`, `check_consumed`, `first_unbound_claim`, the degree
check, the strictly-ascending key check in deserialization, the deleted-tag list of
`ProofType`, and a mutation site per failure mode to keep the guards honest. All of it is
checked invariants, and all of it is code the auditor must read and trust.

The schedule already fixes the order in which the verifier needs each item. This spec sends
the proof in that order. Every value the prover sends, a commitment, a claim, a round
message, the opening proof, is written to the NARG as a prover message, and the verifier
reads it at the step that uses it. The proof is one byte string. A missing item is a short
read, a surplus item is trailing bytes, and there is no key to get wrong: the guards above
are deleted, not satisfied.

## Intent

### Goal

Make the proof exactly the transcript's NARG byte string, written by the prover and read by
the verifier in schedule order, with no keyed container and no proof field outside it.

Introduced or modified:

- `atlas-transcript`: `ProvingTranscript` gains `send_scalar`, `send_scalars` and
  `send_serializable`; `VerifyingTranscript` gains `read_scalar`, `read_scalars(n)` and
  `read_serializable::<C>()`. Each writes or reads a spongefish prover message, which the
  sponge absorbs. A scalar read returns `Err` for bytes that are not the canonical encoding
  of a field element, so no two byte strings carry the same proof. A `Canonical` value is framed by its byte length; a read whose length
  exceeds the bytes left is an `Err` before any allocation.
- `atlas-sumcheck`: `VerifierOpeningAccumulator::append_{dense, sparse, virtual}` read the
  claim from the transcript and insert it at its point; the prover's accumulator sends it.
  `engine::verify_compressed` reads each round message as `degree` coefficients;
  `prove_batch` sends them. The evaluation reduction reads and sends `h` the same way. The
  opening reduction's claims are read after its sumcheck. `SumcheckInstanceProof` and
  `EvalReductionProof` are no longer proof fields; the functions that verify take the
  transcript.
- `jolt-atlas-verifier`: `ONNXProof` is a newtype over the NARG bytes, with no type
  parameter. `verify` reads the batch commitment and the commitments first, then runs the
  schedule, then reads the joint opening proof and calls `PCS::verify_rlc`, then requires
  end of stream. `proof.rs` shrinks to the newtype; `proof_serialization.rs` and
  `types.rs` (`ProofId`, `ProofType`, `Claims`) are deleted. `Step::Sumcheck` loses its
  `proof` field.
- `jolt-atlas-core`: the prover sends where it appended; `finalize_proof` is
  `ONNXProof(transcript.into_narg())`. `testing/mutate.rs` enumerates sites from a recorded
  read log.

Deleted: `ProofLedger`, `Verifier::check_consumed`, `populate_accumulator`,
`VerifierOpeningAccumulator::first_unbound_claim` and its `appended` set, `ProofId`,
`ProofType` and its reserved tags, `Claims`, `ReducedOpeningProof`, `ONNXProof::from_parts`
and the `test-util` accessors, `DEFERRED_PROOF_IDX` as a proof key, `ExecutionProofs`.

### Invariants

Relied on, not changed:

1. **The `atlas-trace` behavioral contract**, all of it. *Mechanism:* no diff under
   `atlas-trace/`.
2. **The relations.** Every relation, its expressions, its challenges and the order of
   absorption are what the schedule says today, apart from any append invariant D moves.
   *Mechanism:* the schedule pins of `testing/schedules.rs` are edited only by the commit of
   invariant D, if it moves an append, and by no later commit.
3. **Verifier hygiene.** No panic, Trace-free. *Mechanism:* the crate-root lints, the
   `verifier-isolation` CI job.

Changed:

4. **The wire format and the transcript encoding.** Proof data that was absorbed as a public
   message is absorbed as a prover message, so every challenge changes. *Mechanism:*
   `testing/digests.rs` is re-pinned in its own commit; the heavy job on all three real
   models, since this is a protocol-level change.

New:

- **A. Every proof byte is read exactly once, in schedule order.** *Mechanism:* the
  structure: the verifier has no access to the proof except through the transcript's
  sequential reads; `VerifyingTranscript::finish` rejects trailing bytes.
- **B. Every value read is absorbed before the next challenge.** A prover cannot choose a
  later message after seeing a challenge that should depend on it. *Mechanism:* spongefish:
  a prover message is absorbed by the read itself; no `read_*` returns a value without
  absorbing it. A unit test in `atlas-transcript` that two streams differing in one read
  byte give different next challenges.
- **C. A claim does not exist before it is bound.** The accumulator holds a claim only from
  the step that reads it, at the point that step states, and a second bind of the same id is
  an `Err`. *Mechanism:* the structure (there is no pre-load); the compiler
  (`first_unbound_claim` has nothing to find and is deleted); a test that binds an id twice.
- **D. No read before bind.** No relation reads a claim that a later step appends.
  *Mechanism:* the first commit of this spec, on the current format, makes the accumulator's
  getters return `Err` for a claim not yet appended, and the workspace run passes on it. If
  that commit finds a relation that reads ahead, the append moves to just before its first
  read in the same commit, the PR names the relation and the two steps, and the affected
  digests and schedule pins are re-pinned there: pins are regression checks, and a moved pin
  with a stated reason is acceptable.
- **E. Message sizes are the verifier's.** The number of coefficients in a round message,
  the number of rounds, the number of claims and the number of commitments come from the
  schedule, not from the proof. The only length the proof states is the byte length of a
  framed `Canonical` value, bounded by the bytes left. *Mechanism:* the structure (no count
  is read for scalars); the bounded-allocation test for the framed read.
- **F. Soundness coverage is every read.** Each value the verifier reads is a mutation site.
  *Mechanism:* a recording `VerifyingTranscript` used by the test harness logs each read's
  byte range and kind while verifying the honest proof; `proof_mutation_sweep` mutates one
  site per stratum (every site under `JOLT_SOUNDNESS_EXHAUSTIVE`) and requires an `Err` with
  no tolerance entry, plus truncation at every read boundary and one appended byte.
- **G. The PCS boundary is unchanged.** `CommitmentScheme` keeps its signatures; commitments
  and the joint opening proof are `Canonical` values that happen to arrive through the
  stream. *Mechanism:* the compiler (`atlas-pcs/src/commitment_scheme.rs` has no diff).

### Non-Goals

- Any change to a relation, the schedule, or the order of absorption, including the
  open-around-setup orders that `specs/relations-as-values.md` reproduces.
- The Akita adapter's inner Fiat-Shamir session and its proof bytes.
- Backward compatibility: proofs in the keyed format do not verify, and no reader for them
  is kept.
- Compressing the stream, or omitting values the verifier could recompute.
- Zero-knowledge.
- Changing proof-of-work grinding or its pricing.

## Evaluation

### Acceptance Criteria

- [ ] Invariant D lands first: the accumulator's getters reject a claim not yet appended
      and the workspace run passes. `testing/digests.rs` and `testing/schedules.rs` are
      unmodified on that commit unless it moved an append, in which case the PR names each
      move.
- [ ] `ONNXProof` has one field, the NARG bytes, and no type parameter.
- [ ] `proof_serialization.rs` and `types.rs` do not exist in `jolt-atlas-verifier`;
      `ProofLedger`, `check_consumed`, `populate_accumulator`, `first_unbound_claim`,
      `ProofId`, `ProofType`, `Claims` and `ReducedOpeningProof` do not exist in the
      workspace.
- [ ] `jolt-atlas-verifier/src` contains no `BTreeMap` keyed by a proof or claim identifier
      that is filled from the proof before the schedule runs.
- [ ] Invariant A: a proof with one byte appended, and a proof truncated at each read
      boundary, are rejected with `Err`.
- [ ] Invariant B: the `atlas-transcript` unit test passes.
- [ ] Invariant C: the double-bind test returns `Err`.
- [ ] Invariant E: a framed read whose stated length exceeds the remaining bytes returns
      `Err` without allocating the stated length.
- [ ] A scalar read of a non-canonical encoding returns `Err` (a unit test in
      `atlas-transcript`).
- [ ] Invariant F: `proof_mutation_sweep` takes its sites from the read log, covers every
      read kind (commitment, claim, round coefficient, reduction coefficient, opening
      proof, proof-of-work nonce), and passes with no tolerance entry.
- [ ] `witness_tamper_sweep` and `opening_forgery_sweep` pass; the forgery hook reports a
      hit for every target.
- [ ] `relation_kinds_all_exercised` replaces `proof_types_all_exercised`: every
      `OpRelKind` and every subprotocol relation appears in the schedule of some checked-in
      model.
- [ ] `testing/schedules.rs` is not modified after the commit of invariant D.
- [ ] `testing/digests.rs` is re-pinned in its own commit; the fuzz seed for the `verify`
      target is regenerated there.
- [ ] `fuzz_regressions_replay` passes; the fuzz driver feeds the bytes after the IO header
      straight to `verify`.
- [ ] The `prove` example prints the proof size and the counts it prints today, taken from
      the schedule.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets`, `cargo check -p jolt-atlas-verifier` and `cargo nextest run --workspace`
      pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile; Qwen in the CI heavy
      job.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the zoo (its serialization round trip becomes
"the bytes verify, and the bytes with one more byte do not"); the op-surface table; the
tamper and forgery sweeps; the schedule pins; the lockstep comparison in the test `verify`.

Changed tests: `testing/mutate.rs`. The structural sites (`Claim`, `RoundCoeff`,
`EvalRedCoeff`, `Commitment*`, `BatchCommitment`, `Reduced*`, `Narg*`) become sites over the
read log, one kind per read kind. `ClaimRemoved`, `ProofRemoved`, `EvalRedRemoved`,
`RoundDropped`, `CommitmentDropped` and `ReducedRemoved` become truncations and splices at
read boundaries. `ClaimOrphan`, `ProofOrphan`, `EvalRedOrphan` and `RoundExtraCoeff` have no
counterpart, since the format cannot express them; `NargExtended` covers surplus bytes. The
IO sites are unchanged. `CommitmentSwap`, `CommitmentReplaced` and `ReducedSpliced` (parts of
a second execution's proof) are kept as splices between two read logs.

New tests: the challenge-dependence test of invariant B; the double bind of invariant C; the
bounded framed read of invariant E; truncation at every read boundary.

New `mutate::Site`: `Read { index, kind }` over the read log, replacing the structural
sites as described.

CI coverage: the workspace job; the heavy job on all three real models (required: this is a
protocol-level change); the nightly exhaustive sweep, which now visits every read.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against this
spec's parent commit, on the benchmark machine.

Expected: prove and verify flat; the same bytes are hashed, as prover messages. Proof size
down: the stream carries no `OpeningId` per claim, no `ProofId` per sumcheck, no length
prefix per round message or per container. Peak memory flat.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better; peak
memory within 5%; proof size no larger than the reference on both models.

## Design

### Architecture

```text
before:  proof = { claims: map, sumchecks: map, reductions: map, commitments, opening?, narg }
         verify: load claims -> absorb commitments -> schedule (look up by key) -> drain check
after:   proof = narg
         verify: read commitments -> schedule (read in order) -> read opening proof -> end of stream
```

The stream, in order: the batch commitment and the commitments; then, for each schedule
step, what that step reads (an `Open` reads its claims; a `Reduce` with several claims reads
`h`; a `Sumcheck` reads, per round, the round message and the proof-of-work nonce, then the
produced claims of each member; a `Tie` reads its produced claims); then the opening
reduction's sumcheck and claims; then the joint opening proof. The public inputs are bound
first, as a public message, as today.

Reading replaces absorbing at each site: where the verifier calls
`transcript.append_scalar(&claim)` with a claim taken from a map, it calls
`transcript.read_scalar()`. The labels and counts that separate domains
(`LabelWithCount(b"sumcheck_poly", n)`, the begin and end messages of the reduction) stay
public messages, computed by both sides.

Types whose only reason to exist was to sit in the proof go: a sumcheck is verified by a
function of the transcript and the relations, and returns the challenges. Code that is
generic over `T: Transcript` and runs on one side takes `T: VerifyingTranscript` or
`T: ProvingTranscript`; code shared by both sides (challenge drawing, labels) stays on
`Transcript`.

The test harness's recording transcript wraps the verifier transcript and logs
`(offset, len, kind)` per read. The mutation sweep verifies the honest proof once under it,
then edits byte ranges. No test needs a `&mut` accessor into the proof, so the verifier
crate's `test-util` feature loses its proof accessors.

Commitments and the joint opening proof cross the PCS boundary as values, read with
`read_serializable`. The schedule says how many commitments there are and which
polynomials the joint opening covers; a model that opens no committed polynomial has no
opening proof in the stream.

### Alternatives Considered

- **Keep the keyed proof and its guards.** They work, and the sweep holds them. Rejected:
  they are the largest body of verifier code that checks the proof's shape instead of the
  statement, and each future proof part adds a guard and a site.
- **A flat list of field elements beside the NARG**, read by a cursor, leaving the
  transcript as it is. Rejected: it keeps two proof parts and a separate absorb step per
  read, so invariant B would again be held by each call site instead of by the transcript.
- **Stream the sumchecks and claims but keep commitments and the opening proof as fields.**
  Rejected: the proof would still be a struct with a format of its own, for two fields the
  stream carries as easily.
- **Put the Akita proof's inner messages into the same stream.** Rejected: the adapter runs
  Akita's own Fiat-Shamir session by design, and the PCS choice is open.
- **Normalize transcript order in the same change**, since the digests are re-pinned anyway.
  Rejected: this spec's soundness argument is "same protocol, different carrier", and the
  unedited schedule pins are its evidence.
- **Upstream's shape.** `a16z/jolt` keeps a proof struct of vectors and maps and absorbs
  from it; spongefish's own examples stream. The transcript crate already wraps spongefish's
  prover-message API one level below the traits the protocol uses.

## Documentation

- `docs/verifier.md`: the entry-point step of the reading order; "Proof-shape rules" is
  rewritten (exact consumption and no dead fields hold by construction; the deleted-tag list
  goes); the "Proof object" step names the stream.
- `CLAUDE.md`: the `jolt-atlas-verifier` crate-map entry (the proof object, `from_parts`,
  the ledgers, the drain check); the second bullet of "Prover/Verifier Invariants" (the
  drain check, `ProofType`, `mutate::Site`); the `atlas-transcript` entry (what the NARG
  holds).
- `docs/pull-requests.md`: "a new proof field needs a `mutate::Site`" becomes "a new read is
  a site by construction".
- `README.md`: the proof-size column of the benchmark table.
- `jolt-atlas-core/fuzz/README.md`: the input format.
- Module docs of `atlas-transcript` (`sponge.rs` says the NARG holds only nonces),
  `opening_proof/mod.rs`, `engine/mod.rs`, `proof.rs`.

## Execution

1. Invariant D: getters reject a claim not yet appended; an append that a read precedes
   moves, with the pins it changes.
2. `atlas-transcript`: the send and read methods, the framed `Canonical` read, their tests.
3. The accumulators read and send claims; the claim map leaves the proof.
4. The sumcheck engine and the evaluation reduction read and send round messages; the two
   proof maps, `ProofId`, `ProofType` and the ledgers go.
5. Commitments and the joint opening proof through the stream; `ONNXProof` becomes the
   newtype; delete `proof_serialization.rs`.
6. The recording transcript and `mutate.rs` over the read log; the sweeps.
7. Re-pin the digests and regenerate the fuzz seed, in one commit that changes nothing else.
8. Docs; the heavy job; the gate.

Steps 3 to 5 each change the transcript, so the digests are red between step 3 and step 7;
the zoo, the sweeps and the lockstep comparison are the checks in between.

## References

- `jolt-atlas-verifier/src/{proof.rs, types.rs, proof_serialization.rs}`,
  `onnx_proof/{mod.rs, verifier.rs}`.
- `atlas-transcript/src/{transcript.rs, sponge.rs, prover.rs, verifier.rs, setup.rs,
  codec.rs}` (`prover_message`, `BytesMsg`, `check_eof`).
- `atlas-sumcheck/src/{engine/mod.rs, sumcheck/mod.rs, opening_proof/mod.rs,
  evaluation_reduction/mod.rs, relation/batch.rs}`.
- `jolt-atlas-core/src/onnx_proof/{prover.rs, fuzz.rs}`,
  `testing/{mutate.rs, soundness.rs, digests.rs, harness.rs}`.
- `specs/verifier-audit-simplification.md` (exact proof shape), `specs/own-transcript.md`
  (the spongefish transcript), `specs/verification-schedule.md`.
- spongefish: prover messages and the NARG string.
