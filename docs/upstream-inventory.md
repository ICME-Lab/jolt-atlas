# Upstream divergence inventory

Snapshot of where `joltworks` stands relative to `a16z/jolt`, taken 2026-10-25 against
`upstream/main` `f3265b5a` (the `upstream` git remote; fetch with
`git fetch --filter=blob:none upstream main`). Reference points:

- `joltworks` began as an unrecorded snapshot of the monolithic `jolt-core`; the nearest
  comparable upstream layout is `072998fb~1` (2026-06-23, the commit before upstream stripped
  `jolt-core` into `jolt-prover-legacy`).
- The Akita integration branch vendors `crates/jolt-field` at `0e3faf07` (2026-09-15) under
  `third_party/` with a root Cargo patch — the consumption pattern this inventory assumes for
  every adopted crate. Upstream is 157 commits past that pin.

**The headline:** upstream has fully decomposed into 31 crates (including `jolt-prover` /
`jolt-verifier`, `jolt-spartan-{prover,verifier}`, `jolt-sumcheck`, `jolt-poly`,
`jolt-openings`, `jolt-claims`, `jolt-transcript`, `jolt-field`, `jolt-dory`, `jolt-hyperkzg`,
`jolt-akita`, `jolt-blindfold`), and it has **re-architected**, not just reorganized: a claim
lattice with declarative claim reductions (`jolt-claims`), prover/verifier-split transcripts
with challenge codecs, a per-message bias budget and sponge proof-of-work (`jolt-transcript`),
and a sumcheck round scheduler (`jolt-sumcheck`). Adoption is therefore an API port per
subsystem, not a file swap.

## Module map and buckets

Buckets: **adopt** (take upstream, delta negligible), **adopt+delta** (take upstream, re-port an
atlas-specific delta on top), **rebuild-on-adopt** (our version and upstream's are different
architectures; adopt upstream's design and rebuild our semantics inside it), **atlas-only**
(no upstream counterpart; stays ours).

| `joltworks` module | Upstream counterpart | Bucket | Notes |
|---|---|---|---|
| `field/` (incl. `challenge/`) | `jolt-field` (+ `jolt-transcript` codecs) | adopt | Already vendored on the Akita branch; our `challenge/mont_ark_*` is the pre-codec design. |
| `transcripts/` (blake2b, keccak) | `jolt-transcript` | adopt+delta | Upstream adds codecs, bias budget, PoW, prover/verifier split. Proof-format change; stage 3. |
| `subprotocols/sumcheck{,_prover,_verifier}.rs` | `jolt-sumcheck` | adopt+delta | Heavily diverged both ways (~842 changed lines vs the 2026-06 jolt-core copy); our batching and instance traits vs their scheduler/recorder. |
| `poly/` core (dense, multilinear, eq, split_eq, unipoly, compact, lagrange, …) | `jolt-poly` | adopt+delta | `split_eq_poly` is ~12% diverged from the 2026-06 copy — the closest module we have; most deltas are atlas perf work (fold routines). |
| `poly/opening_proof.rs`, `subprotocols/opening_reduction.rs` | `jolt-openings` + `jolt-claims` | rebuild-on-adopt | Ours is fully diverged (~2,074 changed lines of 1,636): the accumulator is keyed on atlas `OpeningId`/`VirtualPoly` and interleaved with evaluation reduction. Upstream's claim lattice is architecturally what the verifier-simplification arc's claim-ledger/ProofPlan design describes — evaluate adopting it instead of building our own. |
| `subprotocols/evaluation_reduction.rs` | — | atlas-only | Node-output multi-consumer reduction; ONNX-graph-specific. |
| `poly/commitment/{commitment_scheme,dory,hyperkzg,pedersen,mock}` | `jolt-akita` (+ `jolt-openings` traits) | retire (BN254) / adopt (Akita) | `dory` and `hyperkzg` are BN254-only and are deleted at stage 7b of `specs/adoption-program.md`, never vendored; `pedersen` (curve-generic, used by the zk paths) goes with the `zk` deletion at stage 0, or with the curve code at stage 7b if anything non-zk still uses it; until then the Dory perf work (one-hot commit paths, sparse RLC) stays in place. Akita arrives at stage 7a from the `akita-adapter` branch's `AkitaScheme`; `mock` stays as a test PCS. |
| `subprotocols/{shout,booleanity,hamming_booleanity,hamming_weight}` | twist/shout ancestry in upstream crates | adopt+delta | Shared ancestry, diverged; inspect per file during the port. |
| `subprotocols/{ps_shout,ra_virtual,gamma_fold,identity_range_check,mles_product_sum}` | — | atlas-only | Prefix–suffix shout and the atlas one-hot machinery. |
| `subprotocols/univariate_skip.rs` | upstream sumcheck | adopt+delta | Check whether the round scheduler subsumes it. |
| `subprotocols/blindfold/` | — | retire | Deleted with the `zk` feature at stage 0. Upstream `jolt-blindfold` is Dory/BN254-only; zero-knowledge returns as lattice zk, a separate effort. |
| `lookup_tables/` framework (prefix/suffix traits) | `jolt-lookup-tables` | adopt+delta | Framework is shared ancestry; **contents are atlas-only** (relu, clamp, sat_val vs their RISC-V instruction tables). |
| `msm/`, `curve.rs`, `config.rs`, `utils/` | `jolt-utils` + arkworks | adopt+delta | Keep `utils/lookup_bits` alignment in mind; the rest is low-risk. |

Atlas-only by construction (never in scope): everything in `jolt-atlas-core` (ONNX op
protocols, clamp/rescale, einsum engine, range checking) and `atlas-trace` / `atlas-onnx-import`.

## Upstream fixes since the `0e3faf07` pin — triage

37 `fix:` commits in the 157. Most touch zkVM-side code we do not carry (tracer, ELF/program
loading, SDK, host, guest builds, p256, RISC-V semantics): not applicable. The ones that touch
shared-ancestry subsystems, in priority order:

| Commit | Area | Status for us |
|---|---|---|
| `e012da54` reject G1/G2 encodings with trailing bytes | crypto / deserialization | **Inspect now.** Our proofs deserialize commitments; trailing-byte acceptance is proof malleability. Check our ark-serialize paths and Dory/HyperKZG commitment decoding. |
| `521e53f6` verifier: reject trace lengths below the PCS floor | verifier bounds | **Inspect now.** Do we validate minimum sizes before PCS verification? Related to the verifier-panic hardening item. |
| `189baa3e` pick up arkworks MSM caller-pool fix | arkworks dep | **Inspect.** We pin `a16z/arkworks-algebra` `dev/twist-shout` by branch, and `Cargo.lock` currently holds **two different revs of that branch at once** (`b8f0f51`, `76bb3a4`): unify the pin, then check both against the MSM fix. |
| `f75c59c2` defer Dory opening table materialization | Dory | Triage at stage 0 of `specs/adoption-program.md` (applicability verdict); any remediation rides the PCS adoption stage. |
| `39062151` field: state which extension towers are fields | jolt-field | Arrives with the vendored-pin bump on the Akita branch. |
| `89c0c8b5`, `21f1841e` Akita setup sizing / thread-pool | jolt-akita | Arrive with the Akita pin bump. |
| `00223594` blindfold: bound outer row commitment jobs | blindfold | Deferred with zk. |
| `0337eda5` kernels: exceptional equality challenges | eq handling | Triage at stage 0: check whether our eq-polynomial edge cases share the pattern. |

Also not a `fix:` but soundness-relevant to review against our fork: the J2 transcript work
(`7931b223` challenge codecs with a 2⁻¹⁶⁰ per-message bias budget, `f3265b5a` sponge PoW).
Our Blake2b/Keccak transcripts predate this analysis; review our challenge sampling against
upstream's bias budget before, not during, the transcript adoption.

## Adoption sequence

The ordering lives in one place: `specs/adoption-program.md` (nine stages, 0–8). This inventory
is its factual basis — the bucket table above feeds each stage's delta checklist, and the fix
triage above is resolved at its stage 0.

This document is a snapshot; it is regenerated against a newer `upstream/main` at stage 0 of
`specs/adoption-program.md` and at every subsequent pin bump.
