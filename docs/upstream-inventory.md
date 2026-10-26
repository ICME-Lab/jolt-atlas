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
| `field/` (incl. `challenge/`) | `jolt-field` (+ `jolt-transcript` codecs) | adopt (stage 2: vendored) | `jolt-field` is vendored at `0e3faf07`. Since stage 4b's first PR our `JoltField` extends upstream `jolt_field::JoltField` (colliding methods resolve to upstream's), so upstream crates accept every atlas field. Since stage 7b `Fp128` is the only implementation: the BN254 impl, its arkworks kernel, `TrackedFr` and the `challenge/` types (`MontU128Challenge`, `Mont254BitChallenge`) are deleted. The atlas extension (ladder, codec field) remains. |
| `transcripts/` (blake2b, keccak) | `jolt-transcript` | adopted (stage 3) | `jolt-transcript` is vendored; `joltworks::transcripts` keeps our `Transcript` trait as a thin layer over upstream's split surface (`ProverState`/`VerifierState`, codecs, `grind`). The remaining delta is that trait itself, which the stage-4 ports replace with upstream's. |
| `subprotocols/sumcheck{,_prover,_verifier}.rs` | `jolt-sumcheck` | adopt+delta (stage 4b) | Heavily diverged both ways (~842 changed lines vs the 2026-06 jolt-core copy); our batching and instance traits vs their scheduler/recorder. Upstream takes the facade `T: Transcript<Challenge = F>` (no proof-of-work hook, full-field challenges); our rounds grind (stage 3) and bind with the 125-bit `MontU128Challenge`. Both are met without a vendored delta (`subprotocols/upstream_sumcheck.rs`: a facade view that grinds before each challenge, and `JoltField::challenge_from_field`). The batched driver runs on upstream's engine, and `SumcheckInstanceProver` extends upstream's `ProveRounds`: every instance proves its own rounds, and the trait keeps only the claim lifecycle (degree, input claim, cached openings) that stage 5 replaces. |
| `poly/` core (dense, multilinear, eq, split_eq, unipoly, compact, lagrange, …) | `jolt-poly` | adopt+delta (stage 4b) | Round kernels run on upstream's `GruenSplitEqPolynomial` and round messages are upstream `UnivariatePoly`s; ours is retired, and `poly/split_eq_poly.rs` keeps only atlas extensions over upstream's (the delayed-reduction fold, Gruen round helpers with endpoint closures). The operand polynomials stay ours (`MultilinearPolynomial`, `RaPolynomial`): they bind with the 125-bit challenge and keep the compact small-scalar layouts, about a quarter faster per bind than `jolt-poly`'s full-field dense binding (2²² evaluations: ~11 ms vs ~15 ms). `UniPoly`/`CompressedUniPoly` remain as the proof's wire form. |
| `poly/opening_proof.rs`, `subprotocols/opening_reduction.rs` | `jolt-openings` + `jolt-claims` | relation layer adopted, store atlas-only (stage 5 spike) | Ours is fully diverged (~2,074 changed lines of 1,636): the accumulator is keyed on atlas `OpeningId`/`VirtualPoly` and interleaved with evaluation reduction. The stage 5 spike found no claim store to adopt at `f3265b5a`: `jolt-claims` is transcript-free (the `Expr`/`SymbolicSumcheck` layer plus typed claim structs) and `jolt-openings` is stateless PCS traits; the store, transcript wiring and `ConcreteSumcheck` driver live in `jolt-verifier`, bound to the zkVM's stage graph. Atlas relations (`onnx_proof/relation.rs`) are `SymbolicSumcheck`s over atlas openings, run by an atlas engine on the atlas accumulator. The claim-struct traits (`InputClaims`/`OutputClaims` and their derives) are not used: they assume a static opening vocabulary, and atlas opening ids are graph-indexed. |
| `subprotocols/evaluation_reduction.rs` | — | atlas-only | Node-output multi-consumer reduction; ONNX-graph-specific. |
| `poly/commitment/{commitment_scheme,dory,hyperkzg,pedersen,mock}` | `jolt-akita` (+ `jolt-openings` traits) | retired (BN254, stage 7b) / adopted (Akita, stage 7a) | `dory` and `hyperkzg` were BN254-only and are deleted at stage 7b, never vendored; `pedersen` was deleted with `zk` at stage 0 (nothing else used it). Akita arrived at stage 7a from the `akita-adapter` branch's `AkitaScheme`; `mock` stays as a test PCS. |
| `subprotocols/{shout,booleanity,hamming_booleanity,hamming_weight}` | twist/shout ancestry in upstream crates | adopted onto the engine (stage 4b) | Shared ancestry, diverged. Each implements upstream's `ProveRounds` over upstream's split eq; the relations themselves stay ours. |
| `subprotocols/{ps_shout,ra_virtual,gamma_fold,identity_range_check,mles_product_sum}` | — | atlas-only | Prefix–suffix shout and the atlas one-hot machinery. |
| `subprotocols/univariate_skip.rs` | `jolt-sumcheck` (`prove_uniskip_*`, `CenteredIntegerDomain`) | retired (stage 4b) | It had no callers and is deleted; a protocol that needs a uni-skip first round uses upstream's. |
| `subprotocols/blindfold/` | — | retired (stage 0) | Deleted with the `zk` feature at stage 0. Upstream `jolt-blindfold` is Dory/BN254-only; zero-knowledge returns as lattice zk, a separate effort. |
| `lookup_tables/` framework (prefix/suffix traits), `utils/lookup_bits.rs`, `poly/prefix_suffix.rs` | `jolt-lookup-tables` (`LookupBits`, prefix/suffix) | atlas-only (decided at stage 4b) | Framework is shared ancestry; **contents are atlas-only** (relu, clamp, sat_val vs their RISC-V instruction tables). At `f3265b5a` the upstream crate depends on `jolt-riscv` (and `tracer` for its tests), so adopting the framework would vendor the RISC-V ISA into a zkML prover; the Shout provers moved onto the engine (stage 4b) and keep our framework. Revisit if upstream splits the framework from the instruction set. |
| `utils/thread.rs` (`unsafe_allocate_zero_vec`) | `jolt-utils` | adopted (stage 4a) | Re-exported from `jolt-utils`; `drop_in_background_thread` stays atlas-only. |
| `utils/math.rs` (`Math`) | `jolt-utils::Math` | atlas-only delta | Ours rejects `log_2` of a non-power of two and adds `ceil_log_2`; upstream's `log_2` silently rounds up. Kept as a guard. |
| `utils/{accumulation,small_scalar}.rs` | — (upstream accumulators live in `jolt-field`) | atlas-only (decided at stage 4b) | The small/medium/wide signed accumulators and small-scalar multiplication serve our compact operand polynomials, which stay ours at stage 4b (their 125-bit-challenge binding beats `jolt-poly`'s; see the `poly/` row), and the round kernels' delayed reduction runs on our `JoltField` ladder. They go if the operand polynomials move to `jolt-poly`. |
| `utils/{profiling,monitor}.rs` | — | atlas-only | Benchmark instrumentation (flamegraphs, `monitor` feature). `utils/counters.rs` (the `TrackedFr` operation counters) was retired with BN254 at stage 7b. |
| `utils/{errors,expanding_table,gaussian_elimination,serialization}.rs` | — | atlas-only (decided at stage 4b) | No upstream counterpart at `f3265b5a`. `expanding_table` serves the Booleanity and one-hot opening provers, which bind with it on the engine; `serialization` holds the capped proof-vector decoder. |
| `msm/` | arkworks MSMs | retired (stage 7b) | A wrapper over arkworks' variable-base MSM for the BN254 commitment schemes; deleted with HyperKZG/Dory. |
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
| `0337eda5` kernels: exceptional equality challenges | eq handling | **Applies, robustness.** Same prover-side divisions: `split_eq_poly.rs` (`/ eq_eval_1` in the degree-2/3 Gruen recoveries), `mles_product_sum.rs` (`/ (1 − r)`), `opening_reduction.rs` (`/ eq_r_address_claim`). An honest prover panics when a 125-bit challenge coordinate is exactly 0 or 1 (≈2⁻¹²⁵ per coordinate). Verifier unaffected. | Done at stage 4b: every Gruen round runs on upstream `jolt-poly`'s `split_eq` with an endpoint closure from the caller's witness tables, `mles_product_sum.rs` finishes through upstream's `gruen_poly_from_evals`, and a vanishing `eq_r_address_claim` makes the one-hot cycle message the zero round. `testing/rounds.rs` drives members at vanishing eq points and checks every endpoint closure against the division on the probe models. |
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
  can request a huge allocation (a length of 2⁶⁴ − 1 panics with a capacity overflow). Our own
  decoders (maps, claims) never pre-allocate from a decoded length. *Done at stage 4b* for
  everything outside the PCS: `joltworks::utils::serialization::deserialize_vec` reads ark's
  encoding but grows as elements decode, and the sumcheck proof, `UniPoly`/`CompressedUniPoly`,
  the reduced opening claims, the commitment list and the NARG string decode through it. The PCS
  proofs' internal decoders follow at stage 7 (capped decoding in the adopted types, or a
  `MAX_PROOF_BYTES` bound with length checks against it).
- *Verifier panic on an empty opening batch*: a proof carrying `Some(reduced_opening_proof)` with
  zero claims for a model with no committed openings reaches `.max().unwrap()` on an empty
  instance list in `BatchedSumcheck::verify`. Stage 5 (the opening/claim system is replaced):
  reject a reduced proof when the accumulator holds no sumchecks.
- *Dory proof shape not pinned*: `DoryScheme::verify` accepts any `(nu, sigma)` the dory-pcs
  bounds allow instead of the split the prover must use, a union-bound loss of about log n, not a
  forgery. Closed by Dory's deletion at stage 7b.

## Op dispositions at the close of stage 6

The test `disposition` table in `onnx_proof/testing/relations.rs` is the source of truth; this
records it as stage 6 closed.

- **Relation-defined** (verifier derived from a `SymbolicSumcheck` in `onnx_proof/relation.rs`):
  Input, Constant, Identity, Neg, IsNan, Broadcast, MoveAxis (ties); Add and Sub (accumulation
  ties, non-scalar nodes); Mul, And, Square, Cube, Rsqrt, Div, Iff, ScalarConstDiv (eq-weighted
  engine relations); Reshape, Slice, Concat, Sum (dense engine relations); Einsum and
  MeanOfSquares (relations with specialized provers, checked by `relation_prover_equiv_*`).
- **Lookup family, moved as code at stage 8**: ReLU, Clamp, Tanh, Sigmoid, Erf, Sin, Cos,
  GatherSmall, GatherLarge, and the clamp lookups, remainder range checks and scalar clamp checks
  that relation-defined ops still run beside their relations.
- **Deferred to stage 8 as code**: SoftmaxLastAxis. The #218 aux-vector rework had not merged
  when stage 6 closed, so by the spec's rule softmax moves as code; migrating its stages to
  relations is a named follow-up once #218 lands.

Existing gaps the relations preserve (stage 6 is behavior-preserving), each recorded in its
relation's S ⊆ R note: ScalarConstDiv never range-checks its committed remainder, so the
quotient is not pinned to the floor (soundness); Neg cannot prove negating `i32::MIN`, which
saturates (completeness).

## Akita as the PCS (stage 7a)

`jolt-akita` is vendored at `f3265b5a` (`third_party/PINS.md`) and `joltworks::poly::commitment::
akita::AkitaScheme` is the default PCS, over Fp128. Atlas deltas against upstream's own use of
Akita, all in the adapter rather than the vendored crate:

- **Joint opening without a homomorphism.** `CommitmentScheme` gained a batch commitment
  (`BatchCommitment`, `batch_commit_all`) and `verify_rlc`; a polynomial's commitment is a slot
  of a class, and the verifier rebuilds the joint claim from the classes' slot evaluations,
  checking each opened polynomial's arity against its class (from the reduction instances that
  open it), before verifying each class natively.
- **Classes.** Dense classes are prefix-packed per `(encoding, num_vars)` with upstream's
  `PrefixPackedLayout` under the dense-bounded flavor; signed (`i32`/`i64`) classes are committed
  shifted by a public offset into `[0, 2^64)`. K = 16 one-hot polynomials of 16+ variables are
  committed sparsely as 64-column trace one-hot groups (unused columns empty), opened as one
  group at a shared point. Smaller one-hot polynomials and other K go through the dense path.
- **Catalogs.** Upstream's packaged catalogs (dense profile `w8r2`, one-hot `w1r0`) cover the
  checked-in zoo, GPT-2 and BGE; no atlas-specific catalog is generated. Setups are derived on first
  use and cached per arity.
- **Transcript.** Akita runs its own Fiat-Shamir, bridged through full-width challenges of the
  facade transcript (`FacadeChallenge`), without proof of work.

Cost against Dory on GPT-2 (release `prove` example, median of 3, M3): prove 39.9 s vs 26.7 s,
verify 2.17 s vs 3.42 s, proof 2.42 vs 2.37 MiB, peak memory 6.2 vs 6.6 GB. The commitment phase
dominates (24.4 s), and two of the six 2^24-variable one-hot groups take about 10 s each of it.

## BN254 removal (stage 7b)

Dory, HyperKZG, the BN254 `JoltField` impl (with its arkworks kernel, `TrackedFr` and the
125/254-bit challenge types), the MSM wrapper, the curve-point transcript methods and the
256-bit folded accumulators are deleted. Tests that ran over BN254 run over `Fp128`; the
opening-reduction unit tests that opened through HyperKZG check the reduced claim against the
materialized RLC's evaluation instead. The `prove` example drops `--pcs` and `--srs-vars`.

The gate is the resolved dependency graph: CI resolves every workspace member (vendored upstream
crates with default features off, since they keep optional `bn254` features) and fails on
`ark-bn254` or `ark-ec`. The `a16z/arkworks-algebra` fork pin stays for `ark-ff` (`BigInt`, the
signed big integers and their `Allocative` derives, used by the Fp128 ladder and the small-scalar
polynomials) and `ark-serialize`; `third_party/PINS.md` records it.

## Adoption sequence

The ordering lives in one place: `specs/adoption-program.md` (ten stages, 0–9; stage 9 moves every atlas-only module below out of `joltworks` and deletes it). This inventory
is its factual basis — the bucket table above feeds each stage's delta checklist, and the fix
triage above is resolved at its stage 0.

This document is a snapshot against `f3265b5a`, the upstream rev the program pins; stage 0
resolved its fix triage above. It is regenerated at every subsequent pin bump.
