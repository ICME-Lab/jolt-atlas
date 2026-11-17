# Spec: bitz-commitment — one bit polynomial under an owned BitZ, in place of Akita

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-17                     |
| Status      | implemented                    |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/one-clamp.md`, `specs/own-pcs-adapter.md` |
| Supersedes  | `docs/prover-plan.md` §3 item 8 and §5 decisions 1 and 3; the `akita-*` rev bump of item 10 |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Commitment and opening are most of the prover. On GPT-2 at 16 tokens (25.0 s on the benchmark
machine) the Akita commit is 6.9 s and the thirteen per-class Akita openings are 8.3 s; one
dense class of 2^28 coefficients costs 5.3 s to commit and 5.3 s to open. Everything the
prover commits is small: base-4 digit columns in the range buckets (93% of committed rows) and
one-hot lookup columns, about 2^28.3 bits in all. Akita prices those per field coefficient and
gives no bound on them, so the prover also runs a digit-range sumcheck to prove the digits are
digits.

BitZ (ePrint 2026/2141) commits bits: the bit vector is packed 128 to an element of
`F_{2^128}`, Reed–Solomon encoded and Merkle-committed through Ligerito, and one evaluation
claim over any ring, a 128-bit prime field included, is proven against it. Every committed
coefficient is a bit by construction. Measured on the same machine at 2^29 bits and a 128-bit
target, its proof-of-concept commits in 53 ms and opens in 387 ms, with a 494 KB opening and a
33 ms verifier (`docs/measurement-plan.md`, "BitZ under the Mosaic layout").

This spec replaces Akita by an owned BitZ and lays every committed column on an aligned
subcube of one bit polynomial (the MosaicPCS layout of Magnetar §2.2), so a proof has one
commitment and one opening. The opening-reduction sumcheck that brings every claim to a
common point stays as it is. The projection for GPT-2 at 16 tokens is about 10 s.

## Intent

### Goal

Commit every witness column as bit planes of one polynomial under an owned BitZ scheme in
`atlas-pcs`, open it once per proof, and delete Akita.

The work is three PRs, with a tuning PR between the first two if the gate below needs it.
This spec's PR is stage 1; the later stages are specified here so the scheme is built for
them, and each is ticked by its own PR. One commitment scheme is in the tree at any time.

- **Stage 1, the scheme.** `atlas-pcs::bitz`: commit a bit vector of 2^n bits, prove and
  verify one claim `⟨eq(z, ·), bits⟩ = µ` with `z ∈ Fp128^n`, at a 128-bit target. Standalone:
  nothing in the prover calls it. Includes the `F_{2^128}` arithmetic, the Reed–Solomon
  encoding and Ligerito opening, the grand-product GKR, BitZ's Round 1 (the projection of an
  Fp128 claim to a transcript-sampled prime, which a 128-bit prime requires), the proof codec
  through `atlas_algebra::serialize::Canonical`, and a benchmark example. It is written from
  the papers (BitZ, Ligerito), on our field, transcript and serialization; no code is taken
  from the BitZ proof-of-concept or from Flock. Stage 1 merges on correctness and records its
  benchmark.
- **Gate before stage 2.** Commit + prove at 2^29 bits, 16 threads, at or below 2.0 s on the
  benchmark machine. If stage 1 is above it, a tuning PR (kernels only, no protocol or format
  change) comes first.
- **Stage 2, the cutover.** The layout (`atlas-pcs::bitz::layout`: column, bit width, aligned
  offset), bit-plane witness generation, the `CommitmentScheme` interface reshaped to one
  commitment and one opening per proof, the prover and verifier on it, proof digests
  re-pinned. The same PR deletes Akita: `atlas-pcs::akita`, `atlas-pcs/schedules/`, the
  `akita-*` git dependency, the root `jolt-field` and `jolt-poly` compatibility packages and
  the `[patch]` that points at them. The last commit with Akita is tagged `akita-last` so it
  can be benchmarked or restored; falling back is a revert.
- **Stage 3, what the bits make redundant.** `atlas_sumcheck::digit_range` and its two
  deferred batches are deleted, as is the Booleanity half of the one-hot checks; the
  Hamming-weight half stays.

Stage 1 introduces:

- `atlas-pcs/src/bitz/`: `gf128` (the binary field in GHASH form, carry-less multiply on
  aarch64 and x86-64 with a portable path), `pack` (bits to `F_{2^128}` elements), `ligerito`
  (encoding, Merkle commitment over BLAKE3, the opening), `gkr` (the batched grand product),
  `round1` (the prime draw and projection), `params` (the shape split `t = ⌈3n/5⌉ − 1`, the
  ladder, the grinding schedule), `prover`, `verifier`, `proof`. Supported sizes are
  `12 ≤ n ≤ 31`; a shorter vector is padded with zeros to 2^12 by the caller. The ladder is
  the unique-decoding one at the 128-bit target.
- `atlas-pcs/examples/bitz_bench.rs`: commits random bits at a given `n` and prints commit,
  prove, verify, proof bytes and peak memory.
- `blake3` as a workspace dependency, for the Merkle tree only. (`sha3` and `keccak` are in
  the graph through the transcript's sponge; the measurements this spec rests on use BLAKE3,
  and the Merkle tree is the commit's hot loop.)

### Invariants

- **Binding and exact range.** A verified opening is an opening of a vector in `{0,1}^{2^n}`:
  no committed coefficient is anything but a bit. Mechanism: BitZ's Theorem 4.7 for the
  parameters `params` selects; `params` rejects a parameter set whose computed bound is below
  the target (a unit test per accepted and per rejected ladder); a tamper test flips one
  committed bit to 2 in the packed message and requires rejection.
- **The claim is over Fp128.** The statement proven is `Σ_x eq(z, x)·bits(x) = µ` in Fp128,
  not in the sampled prime. Mechanism: Round 1's integer lift and projection; a test proves a
  claim whose value differs from the true one by a multiple of a small prime and requires
  rejection; review-held for the bound on the projection error, which the spec's PR states
  with its derivation in the module documentation.
- **128-bit target, same accounting as the transcript.** Soundness is round-by-round with
  proof of work before each challenge, the model `atlas-transcript` already uses. The
  statistical bound without grinding is reported by `params` and printed by the example.
  Mechanism: `params` unit tests on the schedule; review-held for the accounting itself.
- **The verifier rejects, never panics, and consumes the proof exactly.** Stage 1's verifier
  lives under the same lint set as `jolt-atlas-verifier` (no `unwrap`, `expect`, `panic!`,
  indexing outside tests). Decoding rejects trailing bytes, non-canonical field elements and
  lengths that do not match the shape. Mechanism: the crate-level `deny` attributes on
  `atlas-pcs::bitz::verifier` and `proof`; a mutation test over every field of a proof at
  `n = 16`.
- **Prover and verifier modules stay apart.** `atlas-pcs::bitz::prover` is not named by the
  verifier side. Mechanism at stage 1: review-held. The CI path check and `jolt-atlas-verifier/clippy.toml` cover it from
  stage 2, when `jolt-atlas-verifier` first names the scheme: they scan that crate's sources
  only.
- **No new curve or arkworks crate.** Mechanism: the existing CI graph checks on `Cargo.lock`.
- **Transcript.** Every BitZ message is absorbed through `atlas_transcript::Transcript`;
  Ligerito's challenges come from the same transcript, not a second sponge. Mechanism: a test
  that changes one absorbed message and requires a different challenge stream; review.
- **Stage 2 onwards, stated here so stage 1's API allows them.** A column's value is the
  recomposition `Σ_b 2^b·plane_b` with public weights, checked by the verifier from the plane
  evaluations the prover sends; a column of declared width `B` is therefore in `[0, 2^B)`.
  The layout is a function of the model, absorbed before the commitment. The `atlas-trace`
  behavioral contract is not touched by any stage: outputs and golden hashes do not move;
  proof digests move at stage 2 and stage 3.

### Non-Goals

- Zero knowledge. BitZ has none today and this spec adds none.
- Recursion, or a verifier circuit for the scheme.
- A GPU backend.
- Committing lookup addresses as bits and virtualising the one-hot columns from them. One-hot
  columns are committed as their bits (K = 16 adds about 2^26.9 bits on GPT-2, inside 2^29).
- Changing any op relation, the opening-reduction sumcheck, or the tracer.
- Byte compatibility with the BitZ proof-of-concept's proofs.
- Polynomials above 2^31 bits.

## Evaluation

### Acceptance Criteria

Stage 1 (this spec's PR):

- [ ] `atlas-pcs::bitz` commits, proves and verifies a claim over Fp128 at every `n` from 12
      to 24 in the workspace tests, on random bits and on the all-zero and all-one vectors.
- [ ] A claim with a wrong value, a wrong point, a commitment to a different vector and a
      proof for a different `n` are each rejected with an `Err`.
- [ ] The proof mutation test rejects every single-field mutation at `n = 16`; decoding
      rejects trailing bytes and a forged length without allocating it.
- [ ] `params` accepts a ladder at the 128-bit target and rejects `custom:1:4` at it, each
      pinned by a unit test that states the computed bound.
- [ ] `cargo run --release -p atlas-pcs --example bitz_bench -- 29` runs on the benchmark
      machine with the 128-bit ladder, verifies, and its commit, prove, verify, proof size and
      peak memory at `n` = 28, 29, 30 are recorded in `docs/measurement-plan.md` beside the
      proof-of-concept's (0.44 s and 1.6 GB at 29, without Round 1 on a 128-bit prime). Peak
      memory at 29 is at or below 4 GB. Time is recorded, not gated, in this PR.
- [ ] The workspace has no dependency on the BitZ proof-of-concept or on Flock, and no file
      taken from either; `docs/upstream-inventory.md` says so.
- [ ] Round 1 is exercised by every test: all claims are over Fp128. If its error bound at
      `n = 30` cannot be brought to the target by grinding at the cost the paper states, the
      PR implements the paper's Method 2 instead and says so; it does not ship a smaller
      prime.
- [ ] `cargo nextest run --workspace` and the CI graph and path checks pass.

Later stages:

- [x] Before stage 2: `bitz_bench 29` at 16 threads reports commit + prove at or below
      2.0 s (stage 1 as merged, or the tuning PR). 1.79 s after tuning.
- [x] Stage 2 PR: the zoo and the real-model tests prove and verify with one commitment and
      one opening per proof; `bench/gate.sh` on `gpt2 --seq-len 16` against `akita-last`
      shows `commit` at or below 1.5 s and `openings` at or below 4.0 s; no `akita` in
      `Cargo.lock`; `jolt-field/` and `jolt-poly/` are deleted. GPT-2: commit
      0.26 s, openings 3.02 s. The real-model tests (the heavy job) were not run; the gate
      proves and verifies GPT-2 and bge-small with the release binary.
- [x] Stage 3 PR: `atlas_sumcheck::digit_range` and the Booleanity checks are gone; the
      soundness sweeps pass with the range now held by the commitment (the witness-tamper
      sweep's digit sites are rejected at the opening); GPT-2 at 16 tokens proves in at most
      12 s on the benchmark machine. 9.07 s. The
      digit-tamper sites are rejected when the commitments are read, where the verifier
      requires the declared width, not at the opening.

### Testing Strategy

Stage 1 adds tests inside `atlas-pcs` only; nothing in `jolt-atlas-core` changes, so the
op-surface table, the soundness sweeps, the zoo and the golden hashes are unaffected and must
keep passing unchanged, digests included. New tests: the round trips and rejections above;
`gf128` against a bitwise reference multiplication and against the portable path on the
machine's accelerated path; the Reed–Solomon encoder against direct polynomial evaluation at
small sizes; Round 1 against a big-integer reference of the lifted claim at `n = 12`. Ground
truth is the reference computation in each case, never a second copy of the optimized kernel.
The proof-of-concept is not a test dependency; it is run by hand on the benchmark machine for
the comparison table only.

Stage 2 needs a `mutate::Site` for each new proof field (the opening, the plane evaluations)
and removes the sites of the fields that go; the forgery sweep covers plane claims through the
accumulator. Stages 2 and 3 each need the heavy job (GPT-2, bge-small, Qwen).

The `zk` feature is not involved.

### Performance

Benchmark: stage 1, `bitz_bench` at `n` = 28, 29, 30 on the benchmark machine at 16 threads
and at one. From stage 2, `bench/gate.sh` on `gpt2 --seq-len 16` and `bge-small-en-v1.5`
against the commit each stage lands on.

Reference, same machine, `c90a4652e`, GPT-2 at 16 tokens: prove 25.0 s; `commit` 7.1 s;
`openings` 9.9 s (reduction sumcheck 1.6 s, class openings 8.3 s); `iop` 4.1 s, of which the
two bucket batches are 1.65 s; trace execution 3.8 s; verify 1.22 s; proof 1.27 MiB; peak
14.5 GB.

Expected after stage 2, with the scheme at its 2.0 s gate: `commit` 7.1 → about 1 s;
`openings` 9.9 → about 3.5 s (the reduction sumcheck, the plane evaluations, one opening);
verify and peak memory down; proof size about the same (the opening is 0.5 MiB against
thirteen Akita openings). After stage 3: `iop` down by the digit-range batch. That is about
11.5 s on GPT-2, and about 10 s if the scheme reaches the proof-of-concept's speed. Trace
execution is then the largest single item and is outside this spec.

Bounds: stages 2 and 3 may not raise prove time, verify time or peak memory on either model.

## Design

### Architecture

```text
witness columns ──► bit planes ──► one bit vector (aligned subcubes) ──► BitZ commit (one root)
                                                                              │
op relations, lookups ──► opening claims ──► reduction sumcheck ──► claims at one point ρ
                                                                              │
              prover sends plane evaluations at ρ; verifier recomposes each column's claim,
              then combines the planes with eq over their subcube suffixes at a random τ
                                                                              │
                                              one claim ⟨eq(ρ ‖ τ), bits⟩ = µ ──► BitZ open
```

- **`atlas-pcs::bitz`** owns the scheme. It knows nothing about columns or models: its input
  is a bit vector and one claim.
- **`atlas-pcs::bitz::layout`** (stage 2) owns the placement. A column with `ℓ` variables and
  width `B` is `B` planes of `2^ℓ` bits, each on a subcube whose offset is a multiple of
  `2^ℓ`; planes are ordered by size, largest first, so the layout is canonical. It replaces
  `akita::mod`'s class construction and reuses the identity of
  `atlas-pcs/src/openings/prefix.rs` generalised to unequal sizes: a column smaller than the
  common point is claimed at a prefix of it.
- **`CommitmentScheme`** (stage 2) loses `prove_rlc`/`verify_rlc`, `combine_commitments` and
  the per-polynomial commitment, and gains one `open` over the whole batch. `mock` follows.
- **`jolt-atlas-verifier` / `jolt-atlas-core`** (stage 2): `ReducedOpeningProof` carries one
  opening and the plane evaluations; `commit_witness_polynomials` builds planes instead of
  classes. Columns that are signed today are committed with an offset so every plane is a bit.
- **Source.** Written from the papers. The proof-of-concept (`albert-garreta/BitZ-PoC`) and
  Flock's Ligerito are read as references and benchmarked against; no code is taken from
  either.

### Alternatives Considered

- **Stay on Akita and cut volume** (the row check, declared widths, the one-group opening
  layout of item 8). Estimated at about 15 s with every item at its estimate, the largest of
  them unprototyped; one polynomial holding everything is past Akita's 2^28 dense cap, and
  the 2^28 class alone is 10.5 s of commit and opening.
- **The Mosaic layout on Akita.** Not available as one polynomial, for the same cap; its
  Akita form is item 8, which this spec supersedes.
- **Depend on the proof-of-concept, or port its kernels.** Rejected: it is a research artifact
  with competitor benchmarks and 77,000 vendored lines, its field holds at most 113 bits so
  it cannot take our claim, and the program this tree just finished removed every such
  dependency. Porting would keep its measured speed; writing from the paper is chosen
  instead, which is why time is a gate before stage 2 and not a criterion of stage 1.
- **Keep Akita selectable for a release after the cutover.** Rejected: the reshaped interface
  would have to fit two schemes and the digit-range path would have to stay for Akita's sake.
  A tag and a revert are the fallback.
- **BitZ per class, keeping thirteen openings.** Rejected: the opening is 0.4 s at 2^29 and
  sub-linear in practice, so thirteen cost more than one, and the layout is what removes the
  per-class bookkeeping.
- **One cutover PR.** Rejected: the scheme is testable alone against reference computations,
  and a scheme bug found through an end-to-end proof is the expensive way to find it.

### Open points left to the implementation

- **A Johnson ladder at 128 bits.** `custom:1:4` as the proof-of-concept ships it reaches
  118.4 bits; more out-of-domain samples would close it and give a smaller opening than the
  unique-decoding ladder's 494 KB. Not in stage 1; a later change to `params` and the proof
  format.
- **The plane-evaluation pass** (stage 2). The proof-of-concept spends 1.8 s computing a
  2^29-bit evaluation claim outside its timed section. Ours is one parallel pass and is priced
  by stage 2's `openings` bound.

### As built (stage 2)

The cutover is smaller than the Architecture section describes. The opening-reduction sumcheck
already brings every committed polynomial to one point, zero-padded, and hands the scheme the
coefficients of one linear combination. That combination of bit planes is one inner product
of the bit vector with a row `eq` table times a per-column weight, which is the claim BitZ
proves, so the prover sends no plane evaluations and the verifier recomposes nothing.
`CommitmentScheme` keeps its three calls (batch commitment, joint opening, its verification)
and loses the ones nothing called; `jolt-atlas-core` and `jolt-atlas-verifier` change only in
the scheme type they name. Shapes (variables, bits, sign) are declared in the batch
commitment by the prover and bound by the transcript; stage 3 is where the verifier starts to
require a declared width.

### As built (stage 3)

The digit range check also carried the digits' recomposition, so it was not only deleted:
the bucket's value sumcheck now ends on the committed digit columns themselves, whose
recomposition is the packed value. The Booleanity check shared one sumcheck with the
Hamming-weight check; the standalone Hamming-weight relation the small gather already used
replaces both, over the address bits alone. The verifier requires two bits for a digit
column and one for a one-hot polynomial when it reads the commitments
(`CommitmentScheme::coefficient_bits`). Not in the spec: the projection's proof of work is a
keyed BLAKE3 search with its nonce in the BitZ proof, after the sponge search proved to cost
up to 2 s on GPT-2.

## Documentation

- `docs/prover-plan.md`: decision 1 recorded as taken; item 8 and decision 3 marked
  superseded; "Order and gates" updated with the stages.
- `docs/measurement-plan.md`: stage 1's benchmark numbers beside the proof-of-concept's.
- `docs/upstream-inventory.md`: a line stating that `atlas-pcs::bitz` has no upstream source;
  the `akita` and compatibility-package rows removed at stage 2.
- `docs/verifier.md`: the trust perimeter gains the hash-based scheme and loses the lattice
  one (stage 2).
- `CLAUDE.md` / `AGENTS.md`: the crate map's `atlas-pcs` entry (each stage), the
  compatibility packages (stage 2).
- `README.md`: the benchmark table and the scheme description (stage 2).

## Execution

Stage 1, suggested commits: `gf128` with its reference tests; `pack` and the Reed–Solomon
encoder; the Merkle commitment; Ligerito's opening; the grand-product GKR; Round 1; `params`
and the grinding schedule; prover, verifier and codec; the rejection and mutation tests; the
example. Run the example on the benchmark machine before the last commit and record the table
in `docs/measurement-plan.md`.

Build budget: the example is one release build of `atlas-pcs`, not of the prover.

## References

- BitZ: ePrint 2026/2141; proof-of-concept `albert-garreta/BitZ-PoC` at `318ea393d`. Reading
  notes: `research_notes/Paper readings November 2026/bitz_limber.md`.
- Magnetar (Setty, ePrint 2026/2423) §2.2, §5.2, for the layout.
- Ligerito as implemented in Flock.
- `docs/measurement-plan.md`, "BitZ under the Mosaic layout": stages 1 to 3 of the
  measurement this spec rests on.
- `docs/prover-plan.md` §3 item 8, §5 decisions 1 and 3.
- `specs/own-pcs-adapter.md`: the `CommitmentScheme` interface this replaces an
  implementation of.
