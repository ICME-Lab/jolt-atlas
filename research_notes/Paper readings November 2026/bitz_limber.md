# BitZ (ePrint 2026/2141) and Limber (ePrint 2026/1635), read in full, against research-directions option C

Read 9 November 2026 from the local PDFs (`papers/2026-2141.pdf`, 85 pages, version dated 4 October 2026; `papers/2026-1635.pdf`, 52 numbered pages, version dated 19 October 2026). Page numbers below are the papers' printed page numbers. Two passes: a full read, then a targeted re-read of the constructions (BitZ §2, §4, §A; Limber §3, §5), the theorems (BitZ Thm 4.7, 4.15, Lemma 6.3; Limber Thm 5.1, 5.5, Cor 5.8), the parameter tables (BitZ Tables 19–20; Limber Tables 3–4) and the evaluations (BitZ §7, Tables 9–18; Limber §7, Tables 1, 5, Fig. 3). Limber's Appendix A parameter script was transcribed and run at a 128-bit base field (it reproduces the paper's Table 3 rows, so the 128-bit rows are trustworthy).

Throughout: "paper states" is cited to a page; everything under "Inferences" is mine. jolt-atlas figures (40.6 s GPT-2 proof on an M3, 25.2 s witness and commitment, 5.3 s IOP, 6.7 s opening reduction and openings, about 2^28 committed one-hot digits, about 90 ns per nonzero) are taken from `docs/optimization-roadmap.md` §1 and the task brief, not re-measured.

Headline: the two docs treat BitZ and Limber as one option. They are opposites. BitZ commits bits over F_{2^128} with a hash-based PCS and proves the linear (evaluation) claim over any finitely generated ring, including a fixed 128-bit prime; its range guarantee is exact per coefficient and its measured cost is about 0.5 ns per committed bit at 10 threads. Limber is a compiler that turns a field PCS into an integer PCS by limb-splitting; it gives no range check (it consumes one) and, over a 128-bit base field, its own formulas put the extra commitment at 3x or more. Limber is irrelevant to our bottleneck; BitZ attacks it directly.

## Q1. What is committed, over which field or ring, and at what cost per bit

### Takeaway
BitZ commits the bit decomposition of the witness, packed 128 bits per element of F_{2^128}, Reed–Solomon encoded (rate 1/2 or 1/8) and Merkle-committed through Ligerito. Measured cost for committing ℓ bits and proving one evaluation claim: 561 ms at 2^30 bits on 10 threads of a MacBook Air M5 (2.12 s single-threaded), 4.0 GB peak, 373 KB proof, 11 ms verifier. Limber commits a limb-split integer polynomial in whatever field its base PCS uses (BLS12-381 scalar field or Tom256 in the paper), plus remainder/quotient oracles, plus a range check.

### Cited Findings
BitZ:
- Commitment object: "the prover creates a vector bits(f) ∈ {0,1}^ℓ ⊆ Z^ℓ of integer bits ... packs the vector π_2(bits(f)) as a vector from F_{2^ν} ... encodes the packed vector with a Reed-Solomon code C (or any other code) over F_{2^ν}, and sends a Merkle root" — [BitZ p. 12](https://eprint.iacr.org/2026/2141), Step 1 of §2.1.
- Packing: every ν consecutive bits form one field element in an F_2-basis (Definition of Pack_ν) — BitZ p. 33.
- Instantiation: "Our BitZ-PCS is instantiated with Reed-Solomon codes over F_{2^ν} = F_{2^128} of rate 1/2 or 1/8. As backend of BitZ we use the Ligerito implementation from Flock [BRW26] (commit e636760, with local modifications) with folding arity 4." — BitZ p. 62. Table 19 (p. 79) lists ν = 128, proximity parameter at the unique decoding radius or up to the Johnson bound, ℓ_1 ∈ [ℓ^0.5, ℓ^0.6].
- Cost table for the core relation (commit ℓ bits, prove one claim ⟨v, f⟩ = µ over F_q with |F_q| < (2^128 − 1)/ℓ_1, 100 bits round-by-round soundness, MacBook Air M5 24 GB, CPU only), BitZ Table 10 p. 64:

  | ℓ (bits) | Commit (10 thr) | Grand product (10 thr) | Ring switch | Ligerito | Prover 1 thr | Prover 10 thr | Verifier 1 / 10 thr | Proof | Peak mem |
  |---|---|---|---|---|---|---|---|---|---|
  | 2^28 | 14.8 ms | 112 ms | 9.4 ms | 12.5 ms | 537 ms | 153 ms | 6.0 / 6.6 ms | 271 KB | 1.0 GB |
  | 2^30 | 61.3 ms | 412 ms | 39.7 ms | 38.2 ms | 2121 ms | 561 ms | 11.1 / 11.3 ms | 373 KB | 4.0 GB |
  | 2^31 | 133 ms | 829 ms | 74.6 ms | 72.8 ms | 4271 ms | 1128 ms | 21.0 / 20.9 ms | 388 KB | 8.1 GB |

- The grand product (a batched GKR over F_{2^128}) is "by far the most expensive step of BitZ" — BitZ p. 13; it is 73% of the 10-thread prover at 2^30 (my division of Table 10).
- "Pay-per-bit and algebraic independence. Its performance depends essentially exclusively on the bit-size of the witness f. In particular, the choice of ring S (and R and ψ if it applies) has a small impact on its performance." — BitZ p. 4.
- End-to-end SNARK throughput: 2^23 multiplications of 32-bit integers (2^30 witness bits) in 678 ms at 10 threads, rate 1/2, 375 KB proof, 3.68 GB — BitZ Table 15 p. 71; the abstract's "9 million proved 32-bit integer multiplications per second" is this experiment at rate 1/8 (865 ms).
- Verifier is O(√ℓ): "the verifier handles only vectors of size ℓ_1 and ℓ_2 ... We leave as an open problem making these asymptotically polylogarithmic" — BitZ Remark 2.2 p. 14; Table 7 p. 53 gives verifier work O(ℓ_1 + ℓ_2 + log² ℓ) field operations.

Limber:
- "To commit to f, the prover first limb-splits it into a polynomial f_ℓ whose coefficients are bounded by T, casts f_ℓ into F, and commits to it with PCS. A range check certifies that the committed field polynomial encodes small limbs." — [Limber p. 23](https://eprint.iacr.org/2026/1635), §5.1; Protocol 5.2 p. 24.
- Extra oracles: remainder and quotient polynomials a_j^(i), b_j^(i) for s primes and t iterations, total length 2sM/(2^k − 1), size sM log q/(2^k − 1) bits — Limber Theorem 5.5 p. 26.
- Implementation parameters: q ~ 2^256, T = 2^64, k = 9, s = 15, commitment overhead ~0.13 — Limber Table 4 p. 43. Base PCS: Hyrax over Tom256 or Brakedown over the same scalar field — p. 42.
- Measured: RSA accumulator benchmark, 12,796 integer constraints, 2.11 s (Hyrax) / 2.07 s (Brakedown), verifier 63 / 59 ms, proof 273 KB / 8.2 MB, single-threaded M4 Pro — Limber Table 1 p. 3. Poseidon2 mixed-field: 438 / 334 ms — Table 5 p. 44.
- Overhead over native Spartan at equal constraint count: 5.4x–9.4x prover, 1.7x–3.2x verifier; explained by "each 256-bit witness value splits into 4 limbs, and each limb is committed as four 16-bit chunks in order to share the commitment with the 16-bit range check, so the committed polynomial has 16× as many (small) values as plain Spartan's" — Limber p. 44 and Fig. 3 p. 45.
- BitZ's head-to-head: 2^19 32-bit multiplications, Limber (Brakedown) 4436 ms single-thread / 1359 ms 10 threads, 12.3 MB proof, against BitZ 181 / 56 ms, 212 KB — BitZ Table 15 p. 71.

### Inferences
- Per committed bit, BitZ costs 0.52 ns at 10 threads and 2.0 ns single-threaded at 2^30 (Table 10 divided by ℓ). Per 4-bit digit that is 2.1 ns and 7.9 ns.
- The Table 10 configuration (Round 1 not triggered) is not quite ours; see Q2 for what a 128-bit prime target adds. The addition is small.
- Limber's measured numbers are single-threaded on an M4 Pro and are for a 2^14-constraint circuit; they say nothing directly about 2^30-scale commitments.

### Gaps
- No BitZ number above ℓ = 2^31 (8 GB on a 24 GB laptop); memory grows linearly with ℓ.
- BitZ does not separate witness generation ("Witgen") from the PCS in Table 10; Table 15 shows witgen at 2.7 ms for 2^23 multiplications, so it is negligible there.

## Q2. Target ring: can evaluation claims be proven over a fixed 128-bit prime field, and what does the cross-ring step cost

### Takeaway
Yes for BitZ: R may be any finitely generated ring, a fixed prime field included. For q ≈ 2^128 the condition that skips the projection step fails, so BitZ's Round 1 runs: the prover lifts the claim to Z, the verifier samples a random prime q′ < (2^128 − 1)/ℓ_1 and the claim is projected to F_{q′}. The error of that step is about 2^-105 for ℓ = 2^30 (paper's Table 20), closable to 128 bits with 23 bits of grinding (26 ms), a larger binary field, or the paper's Method 2. Prover cost of the step is O(ℓ) integer additions. Limber's IntEval proves the integer evaluation exactly, so reducing it modulo our fixed prime is trivially possible; its constraint is on the base PCS field, not on p.

### Cited Findings
BitZ:
- Generality: "Given another arbitrary ring R and a ring homomorphism ψ : S → R, BitZ then proves evaluation claims over R for the polynomial ψ(f)" — abstract p. 1; all rings are "finitely generated as rings", "satisfied by, for example: 1. All finite fields and the rings Z_n; 2. Z ..." — p. 28–29.
- Skip condition: "This step can be skipped (i.e. prover and verifier move to Round 2 directly) if R is a prime field F_p and p < (|F_{2^ν}| − 1)/ℓ_1" — Construction 4.4 Round 1, p. 36. Also Remark 4.5 (p. 37): skippable when R = Z and the entries of v^(1) are nonnegative and below (2^ν − 1)/ℓ_1.
- Otherwise, Round 1: "The prover computes and sends µ′ ∈ Z[X_1,...,X_k] such that ⟨π_canon^{-1}(v^(1)) ⊗ π_canon^{-1}(v^(2)), f⟩ = µ′ ... The verifier checks that µ′ has total degree at most 2·D and ‖µ′‖_∞ ≤ T·ℓ·N_R², then it checks that π_canon(µ′) = µ, and replies with a random prime q ∈ P and a random morphism π_rand" — p. 36–37. For a prime field, k = 0, D = 0, T = 1 and π_canon^{-1} is the canonical lift to [0, q−1] (Example 4.2, p. 34).
- Error of the step (Theorem 4.7, p. 40, with Lemma 4.3): κ_0 = 2D/q_min + log(2·T·ℓ·N_R²)/(|P|·⌊log q_min⌋); for a prime field this is log(2·ℓ·(q−1)²)/(⌊log q_min⌋·|P|) (eq. (28), p. 26).
- Prover cost of Round 1: O(ℓ·(D+1)) additions of log(ℓ_1·N_R)-bit integers, O(ℓ_2·(D+1)²) multiplications of log(ℓ·N_R²)-bit integers, O((ℓ_1+ℓ_2)(D+1)) reductions mod q — Table 5 p. 39. Proof bits added by Round 1: (2D+1)·log((D+1)·ℓ·N_R²), i.e. one ~290-bit integer for a prime field at ℓ = 2^30. Round 2 adds ℓ_2 integers of log(q·ℓ_1) bits each (Table 5).
- Concrete numbers for a 128-bit R (Table 20, p. 82, "log|F| = 128, d = 1"): log ℓ = 28, τ = 0.5: κ_0 ≤ 2^-106.3, bit deficit 22, grinding 11.9 ms; log ℓ = 30, τ = 0.5: κ_0 ≤ 2^-105.3, deficit 23, 25.9 ms; log ℓ = 32, τ = 0.5: κ_0 ≤ 2^-104.3, deficit 24, 54.9 ms; with τ = 0.6 the deficits are 25 and grinding 98 ms.
- Alternatives to grinding at large ℓ (Remark A.1, p. 82): "Method 1: Use a larger binary field F_{2^ν}, i.e. simply take F_{2^ν} with |F_{2^ν}| comfortably larger than 2^128. Method 2: ... relax the requirement max P < (|F_{2^ν}|−1)/ℓ_1 to max P < |F_{2^ν}|−1 ... The cost is an extra batch of low entropy grand products." The paper also says (p. 79) that F_{2^128} in GHASH form "is crucial for efficiency, since the cost of the GKR batched grand product, the most expensive step of BitZ, is dominated by arithmetic in F_{2^ν}".
- Strategy 1 (p. 80): if the PIOP runs over a 112–113-bit prime field (with 18–19 bits of grinding per PIOP round), Round 1 is skipped as long as ℓ_1 ⪅ 2^15–2^16, i.e. ℓ ⪅ 2^30–2^32 at ℓ_1 ≈ ℓ^0.5.
- The claim form BitZ consumes is ⟨ψ(f), v⟩ = µ with v tensor-decomposable; "This is typically the case in practice since v is usually an 'eq' vector, i.e. v = (eq(b, r))" — p. 12.

Limber:
- The mod-PCS evaluation relation takes "p ∈ Primes(2^{λ−1}, 2^λ), r_f ∈ Z_p^n, v ∈ Z_p" and proves f(r_f) ≡_p v — Definition 4.12, p. 17. The IntEval relation proves the exact integer evaluation g(r) = y with r ∈ Z_p^m — Definition 5.4, p. 25; Protocol 5.3 step 2: "Prover sends the integer evaluation y ∈ Z, and Verifier directly checks y′ = y mod p" — p. 24.
- Soundness of IntEval does not involve p's randomness: ε_IntEval = (32λm/P)^s + sm²/(k|F|) under ‖r‖ < λ bits — Theorem 5.5, p. 26.
- Field requirement on the base PCS: q ≥ 2^{k+1} P^k max(T, P) + P (p. 13), equivalently q = Ω(λ²m²) at k = 1 (Remark 5.9, p. 32), and |F| ≥ 2^λ for the Schwartz–Zippel checks (p. 32).
- "Note that the random fingerprinting prime p is distinct from, and sampled independently of, the characteristic q of the PCS field F" — Limber p. 7. Remark 6.5 (p. 40): "Using this randomly sampled prime p will likely slow down prover performance as p is not a nicely structured prime like a Solinas prime."

### Inferences
- With R = our Fp128 (p ≈ 2^128, Solinas) the skip condition p < (2^128 − 1)/ℓ_1 fails by a factor ℓ_1 ≈ 2^15–2^18, so Round 1 runs. The Solinas structure of p is irrelevant to BitZ; BitZ only lifts the eq-vector entries to integers in [0, p−1] and reduces them modulo a random ~106–113-bit prime q′. Our PIOP keeps its Fp128 arithmetic untouched.
- Per opening, the added prover cost is one pass of ℓ ≈ 2^30 additions of ~146-bit integers plus 2^12 big-integer multiplications (ℓ_2 ≈ 2^12 at ℓ_1 ≈ ℓ^0.6). I estimate tens of milliseconds; the paper gives the operation counts, not timings, for this step.
- Per claim: BitZ handles one claim per run. We have many claims on many polynomials; see Q4 for how they collapse to one tensor-structured claim.
- Limber over a fixed p would work, but its p is sampled by the verifier only because its own PIOP (integer R1CS fingerprinting, Protocol 6.1) needs it; a sumcheck PIOP that is already over Fp128 does not.

### Gaps
- BitZ gives no measured end-to-end run where R is a 128-bit field and Round 1 triggers; the Table 20 grinding times are the only measured component. The prover cost of Round 1 is tabulated in operations only.

## Q3. Range checks: exact or approximate

### Takeaway
BitZ: exact, per coefficient, for the declared width. The extracted witness lies in S_{≤B}^{<D}, which for S = Z or a finite field is "simply a range check" to [0, 2^B − 1]. It is a consequence of committing the bits and recomposing the value with public weights (1, 2, ..., 2^{B−1}); no slack, no aggregate norm. A rounding remainder in exactly [0, 2^s) is the case B = s. Bounds that are not powers of two are not directly covered. Limber provides no range check; it requires one.

### Cited Findings
- "Note that R_LinBitsRings(S, R, ψ) includes, in its definition, a restriction of f into a finite subset of S, namely S_{≤B}^{<D}. This means that a proof for R_LinBitsRings(S, R, ψ) ensures that f belongs to such a subset. ... When S is, say, a finite field or the integers, membership to S_{≤B}^{<D} is simply a range check (the < D is vacuous in this case)." — BitZ Remark 2.4, p. 16–17.
- Mechanism: f = L · bits(f)_S with L = Id ⊗ (1, 2, ..., 2^{B−1})_S ⊗ (γ_1, ..., γ_d), and the requirement "L · g_S ∈ (S_{≤B}^{<D_S})^{ℓ_0} for all g ∈ {0,1}^ℓ", footnote 13: "This condition is required to guarantee that extractors obtain witnesses in the specific subset S_{≤B}^{<D_S} of S" — Definition 4.9, p. 44–45.
- Theorem 4.15 (p. 48): Construction 4.13 is a perfectly complete IOP for R_LinBitsRings(S_{≤B}^{<D_S}, ψ, R, C, L) with round-by-round knowledge soundness with relaxation (R^δ_LinBitsRings, ⊥); the relaxation concerns only the oracle being δ-close to the encoding (Definition 4.10, p. 45), not the value range.
- Conditions: proximity parameter δ within the unique decoding radius of C, or up to the Johnson bound with an out-of-domain sample (Round 0) and the extra error κ_OOD ≤ (L_δ choose 2)·(ℓ/ν − 1)/|F_{2^ν}| — Theorem B.1, p. 85; Remark A.2 p. 83 gives L_δ ≤ 71 (rate 1/8) or 36 (rate 1/2) and κ_OOD ≤ 2^{log ℓ − 123.7}.
- Heterogeneous widths: "it is possible to add other heterogeneous subset membership checks by using a non-standard bitification matrix, see Remark 4.11. One, however, needs to make sure that L admits sufficient tensor structure" — p. 17; Remark 4.11 p. 46.
- Soundness of the in-exponent step that makes the recomposition exact: with g a generator of F*_{2^ν}, "⟨f_{*j}, γ⟩ = µ_j mod ord(g) ... since γ_i ∈ [0, q−1], µ_j ∈ [0, (q−1)·ℓ_1], and f_{*j} contains bits, we have |⟨f_{*j}, γ⟩ − µ_j| ≤ ℓ_1·(q−1) ... ord(g) = |F_{2^ν}| − 1 > ℓ_1 · max P" hence equality over Z — proof of Theorem 4.7, p. 43.
- Limber: Commit "run a Range Check argument (Range Check PIOP compiled with PCS) to prove that |f̂_ℓ| < T" — Protocol 5.2 p. 24; Theorem 5.1 (p. 23) assumes "any perfectly complete batch range check argument with soundness error ε_rc"; Section 8 (p. 45): "range proofs ... often account for a significant portion of the prover time in our experiments; faster range proofs would directly improve our construction".
- Limber Remark 5.11 (p. 35): negative numbers handled by adding T_f as a public offset.

### Inferences
- A signed clamp range [−2^31, 2^31) is the width-32 check on f + 2^31 with the shift applied publicly in the PIOP (same device as Limber Remark 5.11). A Euclidean remainder in [0, 2^scale) is exactly B = scale.
- A bound [0, D) with D not a power of two needs two width checks (R and D − 1 − R), as FlashProve does and as roadmap B1 already describes. All of jolt-atlas's scales are powers of two, so this does not arise for remainders.
- Mixing widths in one commitment: with the standard L every entry gets the same B. The clean way around it is to commit bit-planes as separate polynomials and make the value virtual in the PIOP (f(r) = Σ_k 2^k g_k(r)); then the core relation with B = 1 applies and no custom L is needed. The paper does not say this; it follows from Lemma 4.12 and the virtualization in §2.4.

### Gaps
- None material. BitZ's statement is exact and proved.

## Q4. Composition with a sumcheck PIOP over Fp128 and soundness accounting

### Takeaway
A BitZ opening is precisely the claim ⟨eq(r), ψ(f)⟩ = µ in R, so a sumcheck's final evaluation claim at r ∈ Fp128^n is the native input. Many claims on many committed vectors reduce to one tensor-structured claim by the standard random-combination sumcheck (which jolt-atlas already runs as its opening reduction). The error budget adds κ_0 ≈ 2^-105 (random-prime projection), the GKR's per-round 3/2^128, and Ligerito's error; round-by-round knowledge soundness is proved, so Fiat–Shamir applies.

### Cited Findings
- The SNARK recipe: "a PIOP is run over F, reducing (51) to a linear claim over F or, if F is too small, to a field extension F_ext of F. The claim is of the form ⟨v, π(f)⟩ = µ in F_ext ... We assume v admits a tensor decomposition ... this is typically the case since v is usually an 'eq' vector" — BitZ §5 Step 3, p. 49–50; Step 4: "The claim (52), together with the commitment ... is an instance of the relation R_LinBitsRings ... prover and verifier prove it with the BitZ PCS" — p. 50.
- Extension-field claims are supported (k = 1 case, Example 4.2 p. 34): "The PIOP may output its claim over a field extension of F_q, which BitZ can also prove" — p. 5.
- Non-tensor u: "if the ring R supports it (e.g. if R is a finite field or a ring with a large exceptional set), one can run a sumcheck to reduce the claim ⟨bits(f)_R, u⟩ = µ to an MLE evaluation claim over R, thereby 'replacing' the vector u by an 'eq vector'" — Remark 4.14, p. 47.
- Virtualization: BitZ can prove constraints on f "without committing to f" when bits(f) is a public F_2-linear image of the committed vector g — §2.4, p. 22; the R1CS-bits-virtual relation p. 22–23.
- Errors: Theorem 4.7 (p. 40): errors κ_0, then (κ_red,i), then (κ_iopp,i). Lemma 6.3 (p. 53): GKR "error r_2/|K| in the first round, 3/|K| in every sumcheck round, and 1/|K| in every unification step". Appendix A (p. 80): "with |F_{2^ν}| = 2^128, all these errors are already below 2^-100. To achieve λ = 128 it suffices to grind 2 bits at each sumcheck and unification round". Ligerito is configured per its own analysis [NA25, ACFY24/25].
- Round-by-round framework with relaxation, composition Lemma 3.7 (p. 31); "Grinding" definition p. 31 and Table 3 p. 32 (BLAKE3 grinding costs: 18 bits 0.8 ms, 22 bits 10.3 ms, 25 bits 90 ms, 10 threads).
- Limber: Theorem 5.1 (p. 23): knowledge error ε_PCS + ε_rc + 2l/p + ε_IntEvalArg; round-by-round soundness argued in §6.4 p. 41–42.

### Inferences
- Our flow would be: PIOP over Fp128 as today, with every committed small-valued polynomial replaced by its bit-planes and recomposed virtually; opening reduction to one point as today (now over the bit-plane polynomials); one random-combination step across polynomials gives a single claim on the concatenated bit vector at point (ρ, r), whose eq vector is eq(ρ) ⊗ eq(r), tensor-decomposable; then one BitZ run with Round 1 triggered. The paper supports each piece (§5, Remark 4.14, §2.4); the assembly is mine.
- Soundness accounting for GPT-2 scale (ℓ = 2^30): our PIOP's Fp128 error (unchanged) + κ_0 ≈ 2^-105 (Table 20) + GKR ≈ 3/2^128 per round, ~400 rounds (r_1·r_2 + r_1(r_1+1)/2 with r_1 = 18, r_2 = 12 is 387) + Ligerito at its configured level. Without grinding the system sits near 100 bits; with ~23 bits of grinding at Round 1 and 2 bits per GKR round, 128.
- What the PIOP side loses: nothing in the matmul (Thaler sumcheck, virtual intermediates). What it gains: Booleanity and Hamming-weight checks on range digits disappear, read-raf on range tables disappears.
- Lookups (nonlinearities): BitZ says nothing about them. With bits committed, a Shout one-hot ra(k, j) = Π_i eq(k_i, b_i(j)) becomes a virtual product of committed bit polynomials: the same shape as upstream's "four-factor virtual selectors" (roadmap Group E) with the factors being bits. Degree per 4-bit chunk is 4 instead of 1. This is a redesign of the lookup layer, not a drop-in.

### Gaps
- No worked example in BitZ of a multi-polynomial, multi-point PIOP batched into one BitZ claim; the paper's own SNARK has one witness vector.
- The cost of our opening reduction over bit-planes (2^30 cells instead of 2^28 digits' one-hot cells) is unmeasured.

## Q5. Zero knowledge, post-quantum status, assumptions, transparency

### Takeaway
BitZ: transparent, hash-based, "plausibly post-quantum secure", not zero-knowledge today. Limber: inherits all three from its base PCS; its own protocol sends the integer evaluation in the clear and the paper does not claim ZK for the construction.

### Cited Findings
- "a proof system using a PCS that is not plausibly post-quantum secure could replace its PCS with BitZ (which is plausibly post-quantum secure)" — BitZ p. 4.
- "BitZ currently does not support zero-knowledge" — BitZ p. 65; also p. 63 ("we do not support the latter at the moment of writing").
- Assumptions: Reed–Solomon proximity (unique decoding radius, or Johnson bound with OOD sampling) and Merkle commitments through Ligerito/WHIR — BitZ p. 35, 62, 79, 83; BCS/Fiat–Shamir on round-by-round knowledge soundness — p. 31.
- Limber: "compatible with arbitrary field polynomial commitment schemes, which can be instantiated from a variety of assumptions" — p. 1; Theorem 4.20 (p. 20) notes compilation preserves ZK "if PCS is hiding and Eval is honest-verifier zero-knowledge" but the Limber Eval is not shown HVZK; the Zaratan/DARK baseline is called "not quantum secure" (p. 4), Brakedown is hash-based.

### Inferences
- Adopting BitZ means a hash-based commitment inside a program whose recorded end state is "Fp128 + Akita only" (memory `lattice-only-end-state`). The PIOP field would stay Fp128; the commitment would not stay lattice. That is a strategic, not technical, conflict and should be decided explicitly.
- ZK would have to come from the PIOP side or from a future BitZ version; upstream's Whiteout path is Akita-specific.

### Gaps
- Neither paper discusses recursion. Verifying Ligerito plus a GKR over F_{2^128} inside an Fp128 guest is non-native; upstream's Akita verifier guest (a16z/jolt #1911) would not cover it.

## Q6. Implementation status, benchmarks, stated limitations

### Takeaway
BitZ has a proof-of-concept repository that produced every published number, and a separate work-in-progress production repository; the PoC prover was AI-optimized. Limber has a 9,000-line Rust implementation. Both report their own benchmarks on Apple laptops; no third party has reproduced either.

### Cited Findings
- "Our PoC repository can be found in https://github.com/albert-garreta/BitZ-PoC. Reported benchmark numbers correspond to this PoC implementation. Our WIP production-grade repository is at https://github.com/worldfnd/BitZ." — BitZ p. 63. "the PoC implementation was obtained from the production-grade one, after adding a few missing steps (grinding, out of domain sampling, machinery around random primes), and then asking an AI to optimize the prover without changing the verifier. We then reviewed and refactored the results." — p. 64.
- "Our implementation may not match 100% the optimizations described in this section." — BitZ Remark 6.1, p. 50.
- Stated limitations: verifier and proof O(√ℓ), polylog left open (Remark 2.2, p. 14); 128-bit security at large ℓ needs grinding whose cost "scale[s] exponentially in the number of grinding bits" or a larger binary field (Remark A.1, p. 82); not competitive with Flock/Binius64 for pure hashing (p. 63); no ZK (p. 65); peak memory 4–8 GB at 2^30–2^31 bits (Table 10).
- Experiments at 100 bits of round-by-round knowledge soundness except the Limber comparison at 114 bits; footnote 17: Limber's implementation "only achieves 114 bits of security ... because the random prime sampled to perform the fingerprint is of 128 bits, and no other measures (like, say, grinding) are taken" — BitZ p. 65.
- Limber: "We implement Limber ... in 9000 lines of Rust [Imp]" with [Imp] = https://github.com/lucasxia01/limber-impl — p. 42, 48; all numbers single-threaded medians over 10 runs on a MacBook Pro M4 Pro 24 GB — p. 43.
- Limber's open problems: remove the requirement on the PCS field characteristic ("which would in particular allow compiling binary-field PCSs to integer mod-PCSs") and faster range proofs — p. 45.

### Inferences
- The BitZ grinding, OOD and random-prime machinery exist in the PoC, so a run with R a 128-bit prime should be possible without writing protocol code.

### Gaps
- Licences of both repositories are not stated in the papers.
- No independent reproduction found; this reading did not go to the repositories.

## Q7. Limber's "o(1) overhead over any field PCS": what it means, which base PCS, and whether it could sit over Akita or a hash PCS

### Takeaway
"o(1) multiplicative commitment overhead" means the extra committed data (remainder and quotient oracles) is a vanishing fraction of the witness: ratio s·log q/((2^k − 1)·log T), which is 0.03–0.76 at a 256-bit base field and 0.13 in their implementation. The requirement is a base field of characteristic q ≥ 2^{k+1}P^{k+1}·max(T, P) with |F| ≥ 2^λ. At a 128-bit base field (Akita) the paper's own script gives a minimum overhead of 3.1x; at 64 bits, 16x. Limber over Akita would therefore commit at least 4x the witness and still need a separate range check on every limb. It does not help us.

### Cited Findings
- Definition: "the extra commitment overhead is (2^{n−k} s log q)/(2^n log T_f) = s log q/(2^k log T_f). By setting k appropriately, e.g. k = Ω(log λ), we achieve o(1) multiplicative commitment overhead" — Limber p. 12.
- Table 3 (p. 33), λ = 128, m = 25: log q = 256, log T = 32: k = 6 → 0.762, k = 8 → 0.314, k = 13 → 0.033; log q = 64, log T = 32: k = 1, s = 8 → 16.
- "our protocol supports field characteristic q that is much smaller than 2λ bits, at the cost of increased prover commitment cost ... q ≥ 2^{k+1}P^{k+1} = 4P² = Ω(λ²m²)" — Remark 5.9, p. 32.
- "the first (almost) fully PCS-agnostic integer mod-PCS compiler" with footnote: "Defined over a field F with characteristic char(F) = Ω(λ²·n²)" — p. 2.
- Instantiations in the paper: Hyrax over Tom256 and Brakedown over its scalar field, 256-bit — p. 42.
- The committed limbs are range-checked with LogUp-GKR; the only extra committed object for that is "a table-sized multiplicity polynomial" — Remark 5.10, p. 34–35.
- Transcription of Appendix A (p. 51) run at λ = 128, m = 30: log q = 128, log T = 32 gives overheads 12.0 (k=1), 6.67, 4.57, 3.73, 3.10 (k=5, log P = 18, s = 24), 5.46; log q = 128, log T = 16: minimum 3.68 (k=6); log q = 128, log T = 8: minimum 6.55 (k=7). The script reproduces the paper's 256-bit rows (0.762 at k=6, 0.314 at k=8) and the 64-bit row (16.0), so these values follow the paper's own bounds.

### Inferences
- Over Akita (Fp128): a 32-bit witness value would be committed as one dense cell plus 3.1 cells of remainder/quotient data plus whatever the range check commits. Our problem is the range check; Limber adds commitment volume and leaves the range check to us.
- Over a hash PCS in a small field (Goldilocks, 31-bit): the paper's open problem is exactly this; at log q = 64 the overhead is 16x and below that the bounds fail. Over a 256-bit hash PCS it would work but we have no such PCS and no reason to want 256-bit field arithmetic.
- Limber's value is non-native modular arithmetic for cryptographic workloads; a zkML prover whose PIOP field equals its PCS field has no non-native arithmetic to remove.

### Gaps
- None needed; the conclusion follows from the paper's own formulas.

## Q8. Honest comparison at GPT-2 volume (about 2^28 committed digits)

### Takeaway
Encoding today's 2^28 4-bit one-hot digits as 2^30 committed bits, BitZ's measured cost is 561 ms at 10 threads (2.12 s single-threaded) for commitment plus one opening, against about 25 s of commitment and 6.7 s of opening on our stack. That is 15–55x on the commitment stage before hardware normalisation, and the comparison stays above 5x under a generous 2x hardware discount. Limber is not a candidate.

### Cited Findings (the paper's inputs)
- BitZ Table 10 p. 64: ℓ = 2^30 bits: prover 2121 ms (1 thread) / 561 ms (10 threads), verifier 11 ms, proof 373 KB, 4.0 GB; ℓ = 2^28: 537 / 153 ms, 271 KB, 1.0 GB; ℓ = 2^31: 4271 / 1128 ms, 8.1 GB. Hardware: MacBook Air M5 24 GB, CPU only (p. 64).
- BitZ Table 20 p. 82: grinding to reach 128 bits with a 128-bit R at ℓ = 2^30: 23 bits, 25.9 ms (τ = 0.5).
- BitZ Table 5 p. 39: Round 1 prover cost O(ℓ) integer additions of ~log(ℓ_1·N_R) bits plus O(ℓ_2) big-integer multiplications; Round 2 O(ℓ) additions.
- Limber Fig. 3 p. 45: 5.4–9.4x prover overhead over native Spartan; Table 1 p. 3: 2.1 s for 12,796 constraints single-threaded.

### Our inputs (not the paper's)
- 2^28 committed one-hot digits at 4 bits each (LOG_K_CHUNK = 4) → 2^30 bits. Committing values' bits directly (32-bit quotient + s-bit remainder) gives the same count as 8 + s/4 digits, so the bit volume is the digit volume times four either way.
- About 90 ns per nonzero on Akita over Fp128 on an M3, from 25.2 s / 2^28; that figure includes witness generation, so it is an upper bound on commitment alone. The 6.7 s opening stage is separate.
- The M5 Air at 10 threads versus the M3 used for our runs: I assume at most 2x in BitZ's favour; the paper gives no cross-machine data.

### Inferences
- Per 4-bit digit equivalent: BitZ 2.1 ns (10 threads) or 7.9 ns (1 thread) versus about 90 ns. Per element of 32 bits: BitZ 17 ns at 10 threads, versus the 28–35 ns per element at 32 threads that research-directions §4 quotes for hash-based dense commitments, and versus 8 × 90 = 720 ns for eight one-hot digits.
- GPT-2 stage estimate if the IOP were unchanged: 25.2 s + 6.7 s → (witness generation, unknown share of 25.2 s) + ≈0.6–2.2 s BitZ + tens of ms Round 1 + 26 ms grinding + our opening reduction (currently inside the 6.7 s). Total proof time in the 8–12 s range instead of 40.6 s is the order of magnitude, not a prediction.
- The IOP would also shrink (no Booleanity, Hamming-weight or read-raf on range digits), and grow elsewhere if lookups move to virtual products of bits. Net unmeasured.
- Proof size rises by about 0.4 MB (one BitZ proof) and verifier time is comparable to Akita's (11 ms versus 10.5–12.7 ms).
- Memory: 4 GB at 2^30 bits is well within the GPT-2 run's footprint; 2^32 bits (a 4x larger model or longer context) extrapolates to ~16 GB and ~2.3 s at 10 threads.

### Gaps
- No measurement of BitZ on our hardware or with Round 1 triggered; no measurement of Akita commitment time separated from witness generation. Both are the experiment below.

## What this changes in the docs

Each entry quotes the doc, states the correction, and cites the page that supports it.

### research-directions.md

1. Reading-depth table, "Abstract only: BitZ, Limber, ...". Now read in full. Move both to the left column.

2. §4 option C: "BitZ (ePrint 2026/2141) and Limber (2026/1635) claim constructions, with range checks 'essentially for free' and about 9M 32-bit multiplications per second on a laptop." Two schemes are conflated. The free range check and the 9M/s figure are BitZ's alone (abstract p. 1; Remark 2.4 p. 16–17; Table 15 p. 71). Limber provides no range check and consumes one as an input (Protocol 5.2 p. 24, Theorem 5.1 p. 23) and names range proofs as a major cost (p. 45). Rewrite C as BitZ only and move Limber to "not applicable".

3. §4 option C: "Commit integer bits in a binary field, prove over a prime field". Correct for BitZ (p. 12, p. 36). Wrong for Limber, which commits limbs in the base PCS's own field (p. 23–24) and whose base field must have characteristic Ω(λ²m²) (p. 2, p. 32); it cannot sit over a binary field (open problem, p. 45).

4. §4 option C: "whether the target ring can be our fixed 128-bit prime is unknown". Known: any finitely generated ring, a fixed prime field included (p. 1, p. 28–29). For p ≈ 2^128 the projection step (Round 1) runs with error κ_0 ≈ 2^-105 at ℓ = 2^30 and 23 bits of grinding for 128-bit security (Construction 4.4 p. 36–37; Table 20 p. 82), or a larger binary field (Remark A.1 p. 82). Prover cost of the step is O(ℓ) integer additions (Table 5 p. 39).

5. §6: "If BitZ's range guarantee is exact over a fixed prime and its commitment is an order of magnitude cheaper per bit, C replaces one-hot range digits while the sumcheck layer stays." Both conditions are met on the paper's evidence: exactness by Remark 2.4 and Theorem 4.15 (p. 16–17, 48); cost 0.52 ns per bit at 10 threads at 2^30 (Table 10 p. 64) against about 22 ns per bit on our stack (90 ns per 4-bit digit, our figure), a factor above 10 even after a 2x hardware discount. The doc's own trigger for C has fired.

6. §6 recommendation "B now, C as the next question, A declined" and C "rests on abstracts". C now rests on a full read and should move from "next question" to "funded experiment in parallel with B". B's items that only reduce range-digit counts (D1, D3, parts of B1 and B2) are the ones BitZ makes moot; B's fusions (C1–C4) survive since they reduce bits as well as digits.

7. §1 thesis: "The commitment scheme certifies nothing about ranges. Akita gives no bound on committed values, so every range check is ours to build (section 4)." True of Akita, and no longer a property of every available commitment: BitZ is a commitment whose relation includes the range (Remark 2.4 p. 16–17). Strengthens the case that the bottleneck is a choice of commitment, not a law.

8. §1 thesis: "The bottleneck is one-hot digits over a 128-bit field, not lattices as such." Strengthened: BitZ's per-bit cost is independent of the target ring (p. 4), so proving over Fp128 is not what makes range checks expensive; committing them as one-hot digits in a 128-bit prime-field PCS is.

9. §3.5 and §4 option D: "Requires leaving Fp128 for a 64-bit or 31-bit field" (for a hash-based commitment of small integers). Weakened: BitZ is a hash-based commitment of small integers that keeps the PIOP over Fp128 (p. 36–37, p. 49–50). Leaving Fp128 is a choice, not a requirement, for going hash-based.

10. §4 option D: "CPU throughput is about 28–35 ns per element on 32 threads by published figures". Add BitZ's measured 0.52 ns per bit, i.e. about 17 ns per 32-bit element at 10 threads (Table 10 p. 64, my division).

11. §3.5: "Not advantages, despite earlier drafts of this document: verifier time (hash-based verifiers run 0.8–3.3 ms against Akita's 10.5–12.7 ms ...)". BitZ's verifier is 6–21 ms at 2^28–2^31 bits (Table 10), so verifier time stays a non-advantage either way; its proof is 271–388 KB (Table 10), well under our 2.4 MiB.

12. §4: "On lookup arguments: a LogUp or grand-product argument loses about log₂N bits of soundness when challenges come from a 128-bit field." Nuance: BitZ's grand product over F_{2^128} is analysed round-by-round with 3/2^128 per sumcheck round (Lemma 6.3 p. 53) and the authors reach 128 bits with 2 bits of grinding per round (p. 80). The summed loss is a feature of the standard soundness accounting, not of the argument family. The doc's point about LogUp under Fiat–Shamir without grinding stands.

13. §8 "Reads still owed", item 1 "BitZ in full: the target ring, the cost per opening, whether its range guarantee is exact". Done; answers: any f.g. ring with Round 1 for a 128-bit prime; about 0.5 ns per bit plus one O(√ℓ) verifier and 270–390 KB per opening; exact.

14. References: Limber's authors are Chen, Xia, Nguyen, Bünz (NYU/Microsoft), dated 19 October 2026 (p. 1); BitZ's are Bloemen, Garreta, Kostrzewa, Londhe, Soukhanov, Wu (World Foundation, Nethermind, Reilabs, Ivy), dated 4 October 2026 (p. 1).

### optimization-roadmap.md

15. §1 cost model: "What we commit is one-hot digit polynomials ... So the number to drive down is committed one-hot digits per tensor element." Under a BitZ commitment the number to drive down is committed bits per tensor element, and the one-hot structure is needed only where a Shout read requires it, not for range checks (Remark 2.4 p. 16–17). The digit-count targets in the table remain valid as bit counts times four.

16. §6 D3: "Dense small-digit columns for exact range digits ... Gated on a microbenchmark of dense narrow cells against one-hot nonzeros." Add a third arm to the microbenchmark: BitZ bits at the paper's 0.52 ns per bit (Table 10). If Akita dense cells do not come within about 10x of that, D3 is dominated.

17. §6 D1: the approximate row check "proves 'no overflow' with slack, not a tight bound, so it needs D2 or a semantics change" and "Completeness becomes imperfect". BitZ's exact width check has neither drawback and costs 32 bits per clamped value instead of 8 digits; the paragraph should say D1 is the option if the commitment stays Akita, and is unnecessary if it does not.

18. §6 D4: "Over a 128-bit challenge field a LogUp or grand-product argument loses about log₂N bits". Same nuance as item 12; Limber's implementation uses LogUp-GKR for its range checks (p. 42) and reaches only 114 bits for a different reason (BitZ footnote 17 p. 65).

19. §7 Group E, "Four-factor virtual selectors ... Measured upstream (a16z/jolt #2188)". If bits are committed, a 4-bit chunk's one-hot selector is a product of four committed bit polynomials; this item becomes the lookup design under option C rather than an optimisation of the current one (my inference from BitZ §2.4 p. 22 on virtual F_2-linear transforms).

20. §9 "Not pursuing: ... range bounds from the Akita commitment". Still correct for Akita; a commitment with a built-in range bound now exists elsewhere.

## Ranked view and the one experiment

Pursue, not park. Specifically: promote option C from "next question" to an experiment run in parallel with group A and B1/B2, and restate it as "BitZ-style bit commitment under the existing Fp128 sumcheck layer", dropping Limber from it.

Why pursue: every condition the docs set for C is met on the paper's own evidence (exact range, fixed-prime target, more than an order of magnitude cheaper per bit), and the architecture change is bounded: the matmul sumcheck, the trace semantics and the tables do not move; what changes is the commitment and the range-check and lookup layers of the IOP.

Why not adopt outright: (a) it is hash-based and conflicts with the recorded lattice-only end state and with the upstream Akita recursion path; (b) no ZK; (c) the published numbers come from an AI-optimized PoC on an M5 with Round 1 not triggered; (d) the lookup layer needs a redesign the paper does not provide; (e) peak memory scales to ~16 GB at 2^32 bits.

The one experiment that settles it: build the BitZ PoC (`albert-garreta/BitZ-PoC`) and run its Construction 4.4 harness (Table 10) on our M3 at ℓ = 2^30 bits with R a 128-bit prime, so Round 1 triggers, at both 100 and 128 bits; on the same machine, time an Akita commitment of 2^28 one-hot digits separated from witness generation. If BitZ is under 3 s single-threaded (the paper's 2.1 s with a 1.5x hardware allowance) and more than 5x below the Akita figure, write the option-C spec. The digit census in `docs/measurement-plan.md` is still needed, but as a bit census.

Further reads, in order: LaBinius (ePrint 2026/2103) and upstream #2040, to learn whether a lattice commitment can offer the same per-bit cost and range guarantee and so keep the end state; Fields-Witch (Soukhanov, github.com/morgana-proofs/fields-witch) only if a 2^127−1 field were ever acceptable, which it is not for us (BitZ p. 11, 22); Flock (ePrint 2026/1329) for the Ligerito backend BitZ reuses.
