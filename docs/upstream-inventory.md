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
| `field/` (incl. `challenge/`) | `jolt-field` (+ `jolt-transcript` codecs) | adopt (stage 2: vendored) | `jolt-field` is vendored at `0e3faf07`; the BN254 scalar is `jolt_field::Fr` and `JoltField` is implemented for it and for `Fp128`. Since stage 4b's first PR our `JoltField` extends upstream `jolt_field::JoltField` (colliding methods resolve to upstream's), so upstream crates accept every atlas field; `ark_bn254::Fr` is only the private kernel behind `jolt_field::Fr`'s unreduced ladder. The atlas extension (ladder, `MontU128Challenge`, codec field) remains. |
| `transcripts/` (blake2b, keccak) | `jolt-transcript` | adopted (stage 3) | `jolt-transcript` is vendored; `joltworks::transcripts` keeps our `Transcript` trait as a thin layer over upstream's split surface (`ProverState`/`VerifierState`, codecs, `grind`). The remaining delta is that trait itself, which the stage-4 ports replace with upstream's. |
| `subprotocols/sumcheck{,_prover,_verifier}.rs` | `jolt-sumcheck` | adopt+delta (stage 4b) | Heavily diverged both ways (~842 changed lines vs the 2026-06 jolt-core copy); our batching and instance traits vs their scheduler/recorder. Upstream takes the facade `T: Transcript<Challenge = F>` (no proof-of-work hook, full-field challenges); our rounds grind (stage 3) and bind with the 125-bit `MontU128Challenge`. Adoption carries a recorded delta for both. |
| `poly/` core (dense, multilinear, eq, split_eq, unipoly, compact, lagrange, …) | `jolt-poly` | adopt+delta (stage 4b) | `split_eq_poly` is ~12% diverged from the 2026-06 copy — the closest module we have; most deltas are atlas perf work (fold routines). `jolt-poly` bounds on upstream `jolt_field::JoltField`, which our generic `F: joltworks::field::JoltField` cannot carry (method names collide with upstream `Ring`/`Field`, and the `ark_bn254::Fr` kernel cannot implement a foreign trait), so the field-trait unification is 4b's first step. |
| `poly/opening_proof.rs`, `subprotocols/opening_reduction.rs` | `jolt-openings` + `jolt-claims` | rebuild-on-adopt | Ours is fully diverged (~2,074 changed lines of 1,636): the accumulator is keyed on atlas `OpeningId`/`VirtualPoly` and interleaved with evaluation reduction. Upstream's claim lattice is architecturally what the verifier-simplification arc's claim-ledger/ProofPlan design describes — evaluate adopting it instead of building our own. |
| `subprotocols/evaluation_reduction.rs` | — | atlas-only | Node-output multi-consumer reduction; ONNX-graph-specific. |
| `poly/commitment/{commitment_scheme,dory,hyperkzg,pedersen,mock}` | `jolt-akita` (+ `jolt-openings` traits) | retire (BN254) / adopt (Akita) | `dory` and `hyperkzg` are BN254-only and are deleted at stage 7b of `specs/adoption-program.md`, never vendored; `pedersen` was deleted with `zk` at stage 0 (nothing else used it); until then the Dory perf work (one-hot commit paths, sparse RLC) stays in place. Akita arrives at stage 7a from the `akita-adapter` branch's `AkitaScheme`; `mock` stays as a test PCS. |
| `subprotocols/{shout,booleanity,hamming_booleanity,hamming_weight}` | twist/shout ancestry in upstream crates | adopt+delta (stage 4b) | Shared ancestry, diverged. They are sumcheck instances, so they move with the engine port (per op family), not before it. |
| `subprotocols/{ps_shout,ra_virtual,gamma_fold,identity_range_check,mles_product_sum}` | — | atlas-only | Prefix–suffix shout and the atlas one-hot machinery. |
| `subprotocols/univariate_skip.rs` | `jolt-sumcheck` (`prove_uniskip_*`, `CenteredIntegerDomain`) | retire on adopt (stage 4b) | Upstream ships the uni-skip first round in `jolt-sumcheck`'s prover and domain abstraction; ours is replaced when its callers port. |
| `subprotocols/blindfold/` | — | retired (stage 0) | Deleted with the `zk` feature at stage 0. Upstream `jolt-blindfold` is Dory/BN254-only; zero-knowledge returns as lattice zk, a separate effort. |
| `lookup_tables/` framework (prefix/suffix traits), `utils/lookup_bits.rs`, `poly/prefix_suffix.rs` | `jolt-lookup-tables` (`LookupBits`, prefix/suffix) | adopt+delta (with the shout port, 4b) | Framework is shared ancestry; **contents are atlas-only** (relu, clamp, sat_val vs their RISC-V instruction tables) and stay in our crates. |
| `utils/thread.rs` (`unsafe_allocate_zero_vec`) | `jolt-utils` | adopted (stage 4a) | Re-exported from `jolt-utils`; `drop_in_background_thread` stays atlas-only. |
| `utils/math.rs` (`Math`) | `jolt-utils::Math` | atlas-only delta | Ours rejects `log_2` of a non-power of two and adds `ceil_log_2`; upstream's `log_2` silently rounds up. Kept as a guard. |
| `utils/{accumulation,small_scalar}.rs` | — (upstream accumulators live in `jolt-field`) | atlas-only until 4b | The small/medium/wide signed accumulators feed our sumcheck inner loops; they move or retire with the engine port, measured against `jolt_field`'s `WithAccumulator`. |
| `utils/{counters,profiling,monitor}.rs` | — | atlas-only | Test/benchmark instrumentation (`TrackedFr` counters, flamegraphs, `monitor` feature). |
| `utils/{errors,expanding_table,gaussian_elimination}.rs` | — | atlas-only until 4b | No upstream counterpart at `f3265b5a`; `expanding_table` serves the one-hot binding and moves with it. |
| `msm/` | arkworks MSMs | retire (stage 7b) | A wrapper over arkworks' variable-base MSM for the BN254 commitment schemes; deleted with HyperKZG/Dory. |
| `config.rs` | `jolt-claims` one-hot geometry | atlas-only until stage 5 | One-hot parameter derivation (`OneHotParams`); upstream's equivalent lives with the claim layer, so it is revisited at stage 5. |
| `curve.rs` | — | retired (stage 0) | `JoltCurve` was deleted with `zk`; only Pedersen and BlindFold used it. |

Atlas-only by construction (never in scope): everything in `jolt-atlas-core` (ONNX op
protocols, clamp/rescale, einsum engine, range checking) and `atlas-trace` / `atlas-onnx-import`.

## Upstream fixes since the `0e3faf07` pin — triage

37 `fix:` commits in the 157. Most touch zkVM-side code we do not carry (tracer, ELF/program
loading, SDK, host, guest builds, p256, RISC-V semantics): not applicable. The ones that touch
shared-ancestry subsystems, in priority order:

| Commit | Area | Verdict (stage 0) | Remediation |
|---|---|---|---|
| `e012da54` reject G1/G2 encodings with trailing bytes | crypto / deserialization | **Applies, robustness.** The upstream pattern (serde decoding a point from a `Vec<u8>` and ignoring the tail) does not exist here: commitments and points stream through one ark reader. The same byte-level malleability exists one level up: `deserialize_proof` accepts `proof ‖ junk`, and the map decoders (`deserialize_btreemap`, `Claims`) accept unsorted and duplicate keys. No false acceptance: the verifier checks the decoded object, and the transcript absorbs re-serialized values, never raw bytes. | Done at stage 3: `deserialize_proof` rejects leftover bytes and map decoding requires strictly increasing keys. |
| `521e53f6` verifier: reject trace lengths below the PCS floor | verifier bounds | **Does not apply.** `ONNXProof` carries no trace length; every size reaching PCS verification comes from the verifier's model and accumulator (round counts, claim counts and commitment counts are all checked against them). | None. |
| `189baa3e` pick up arkworks MSM caller-pool fix | arkworks dep | **Resolved at stage 0.** `ark-ec` already resolved to `b8f0f51`, which contains the fix (a liveness issue: a thread pool built per MSM call, panicking where threads cannot spawn). The lockfile split is unified on `b8f0f51`; `76bb3a4` is its ancestor and differs only in `ark-ec`. | Done. |
| `f75c59c2` defer Dory opening table materialization | Dory | **Does not apply.** A memory/perf change for upstream's GPU kernel seam over zkVM program tables; we have neither. | None. |
| `0337eda5` kernels: exceptional equality challenges | eq handling | **Applies, robustness.** Same prover-side divisions: `split_eq_poly.rs` (`/ eq_eval_1` in the degree-2/3 Gruen recoveries), `mles_product_sum.rs` (`/ (1 − r)`), `opening_reduction.rs` (`/ eq_r_address_claim`). An honest prover panics when a 125-bit challenge coordinate is exactly 0 or 1 (≈2⁻¹²⁵ per coordinate). Verifier unaffected. | Stage 4b: arrives with upstream `jolt-poly`'s `split_eq`, whose fallback needs each caller to supply a direct endpoint evaluation from its witness tables, so it lands per op family as callers port; the same fallback goes to `mles_product_sum.rs` and `opening_reduction.rs`. |
| `39062151` field: state which extension towers are fields | jolt-field | Arrives with the vendored-pin bump on the Akita branch. | Stage 2. |
| `89c0c8b5`, `21f1841e` Akita setup sizing / thread-pool | jolt-akita | Arrive with the Akita pin bump. | Stage 7a. |
| `00223594` blindfold: bound outer row commitment jobs | blindfold | Moot: BlindFold deleted at stage 0. | None. |

**J2 transcript review** (`7931b223` challenge codecs with a 2⁻¹⁶⁰ per-message bias budget,
`f3265b5a` sponge PoW). Conclusion: **no reduction bias, reduced challenge sets; hardening, not a
break.** Our Blake2b/Keccak transcripts draw `challenge_scalar*` from 16 hash bytes (a value
below 2¹²⁸ < p, so no reduction) and the `_optimized` challenges as 125-bit `MontU128Challenge`s
(top 3 bits masked). Each is exactly uniform on its declared set, so upstream's 2⁻¹⁶⁰ TV rule is
met trivially — the mod-p bias J2 addresses does not arise. The cost is the set size: per-round
soundness is about d/2¹²⁵ (≈2⁻¹²² for degree ≤ 8) rather than d/2²⁵⁴, about 122-bit security
against grinding, below a 128-bit target. Upstream's sponge PoW (`grind` before a challenge,
difficulty b with |C|·2ᵇ ≥ L·2^λ) exists to cover exactly this gap, and the margin shrinks again
under Fp128. Remediation: stage 3 adopts the codecs and the `ChallengeSetSize`-priced `grind`
before the 125-bit sumcheck challenges. (The `mont_ark_u128` module doc says the two low bits are
zeroed; the code masks the top three. Fixed with the transcript adoption.)

*Resolved at stage 3.* The transcript is upstream `jolt-transcript`'s split surface: full-width
challenges decode through the `ChallengeCodec` (within the 2⁻¹⁶⁰ budget), and every sumcheck
round challenge (including the univariate-skip round) is preceded by a `grind` priced by
`grind_bits_for_loss(degree, |C| = 2¹²⁵, λ = 128)`, 3–6 bits for degree ≤ 8, so each round
reaches 2⁻¹²⁸ per hash query. Measured cost on GPT-2 (Dory, M3, median of 3): prove 27.00 s →
27.30 s (+1.1%), verify unchanged, proof +7,810 bytes (+0.31%, the nonce NARG string).

**Further robustness findings from the triage** (none lets a false proof verify):

- *Verifier memory exhaustion on crafted bytes*: ark's `Vec<T>` decoder and dory-pcs's proof
  decoder pre-allocate from a prover-supplied length (`Vec::with_capacity(len)`), so a few bytes
  can request a huge allocation. Our own decoders (maps, claims, the NARG string) never
  pre-allocate from a decoded length. The remaining sites are inside third-party decoders of
  types later stages replace, so they are fixed with them: the sumcheck and polynomial proof
  types at stage 4b, the PCS proofs at stage 7 (capped decoding in the adopted types, or a
  `MAX_PROOF_BYTES` bound with length checks against it).
- *Verifier panic on an empty opening batch*: a proof carrying `Some(reduced_opening_proof)` with
  zero claims for a model with no committed openings reaches `.max().unwrap()` on an empty
  instance list in `BatchedSumcheck::verify`. Stage 5 (the opening/claim system is replaced):
  reject a reduced proof when the accumulator holds no sumchecks.
- *Dory proof shape not pinned*: `DoryScheme::verify` accepts any `(nu, sigma)` the dory-pcs
  bounds allow instead of the split the prover must use, a union-bound loss of about log n, not a
  forgery. Accepted until Dory's deletion at stage 7b.

## Adoption sequence

The ordering lives in one place: `specs/adoption-program.md` (nine stages, 0–8). This inventory
is its factual basis — the bucket table above feeds each stage's delta checklist, and the fix
triage above is resolved at its stage 0.

This document is a snapshot against `f3265b5a`, the upstream rev the program pins; stage 0
resolved its fix triage above. It is regenerated at every subsequent pin bump.
