# LaBinius, Neo/SuperNeo and Mikros, read in full, against our docs

Read 2026-11-09. All three PDFs were read cover to cover including appendices (LaBinius 49 pp.,
Neo/SuperNeo 60 pp., Mikros 63 pp.), then a second targeted pass over constructions, theorems,
parameters and evaluation tables. Page numbers below are the printed page numbers of each PDF.
"Paper states" means a direct reading; "inference" is mine and is marked as such. Nothing here is
a measurement by us.

Sources:
- [LaBinius] M. Osadnik, G. Seiler. LaBinius: Fast Lattice-Based Binary Polynomial Commitment
  Scheme. ePrint 2026/2103. https://eprint.iacr.org/2026/2103 (local: papers/2026-2103.pdf)
- [Neo] W. Nguyen, S. Setty. Neo and SuperNeo: Post-quantum folding with pay-per-bit costs over
  small fields. ePrint 2026/242, to appear CRYPTO 2026. https://eprint.iacr.org/2026/242
  (local: papers/2026-242.pdf)
- [Mikros] M. Cong. Mikros: Bitwidth-Aware SNARKs over Extensions of Small Fields. ePrint
  2026/2646. https://eprint.iacr.org/2026/2646 (local: papers/2026-2646.pdf)

One-paragraph verdict before the detail: none of the three papers gives what section 4 option C
of `docs/research-directions.md` hoped for. LaBinius's "no bitness check" is a change of
semantics (the committed object *means* its reduction mod 2, and every element of F_2 is a bit),
not a range proof, and it does not transfer to a 128-bit prime. SuperNeo's range guarantee is
exact, but it is produced by a product-of-(z−a) zero-check inside the sumcheck, i.e. it is our
own roadmap item D3 done at digit base b = 2, and the paper has no implementation. Mikros is a
PIOP packing trick whose gain is the ratio log|E|/log|F|, which is 1 for Fp128. The docs' option
B / D3 is strengthened; option C via LaBinius should be dropped.

## Q1. LaBinius: what is committed, over which ring, how binary claims arise, and whether a 128-bit prime could replace F_2

### Takeaway
LaBinius commits a *binary* witness as integer coefficients of a cyclotomic ring modulo a composite
of 14–15-bit NTT-friendly primes, and reads the committed object only through its reduction mod 2,
which lands in the field F_2^162. "No bitness check" is a consequence of F_2 = {0,1}; nothing is
proved about the integer size of what was committed (the extractor returns a *rational* vector
over the number field). The same decoupling can in principle use another prime p, but the
evaluation field becomes R_p (a 162-fold extension, or a SIMD product if p splits), and for a
128-bit p the mod-p residue of the extracted witness is an arbitrary field element, so no range
guarantee of any kind results.

### Cited Findings
- Setting: R = Z[x]/(Φ_f) with f odd so that 2 is inert; "R = Z[x]/(x^162 + x^81 + 1) gives
  R_2 ≅ F_2^162"; the implementation uses f = 3^5 = 243, φ = 162, and the superring
  S ≅ Z[x]/(x^648 − x^324 + 1) of conductor 2^3·3^5, packing four F elements per S element —
  [LaBinius p.8, §1.3.1; p.8 §1.3.2; p.24–25 §5.1.1–5.1.2]
- Commitment: Y = F W mod q, F uniformly random over R_q (S_q in the implementation); q is a
  product of two or three of the primes 3889, 9721, 17497, 19441 (fully splitting) or 2917, 4861,
  12637 (quadratic factors), "fit in 14 bits"/"15 bits"; "we obtain the target SIS security
  level" by choosing how many — [LaBinius p.8 §1.3.2; p.25 §5.1.2]
- What is committed: Com(F, f) computes W with vec(W) = lift(ϕ(f)) where lift maps F_2
  coordinates to {0,1} power-basis coefficients; "an honest W has binary coefficients, so
  ‖W‖ ≤ sqrt(φ m)" — [LaBinius Def. 13 p.14; Def. 24 and Thm. 1 proof p.22]
- How binary claims arise: "the arithmetisation and sumcheck [run] modulo a different prime p";
  the consistency check u^T c = α^T v is "checked modulo p over R_p"; extraction divides by the
  challenge difference "over the number field K", the extracted w_j^* is "a proper rational
  vector", and the two proved properties are (i) binding w.r.t. extracted rational vectors and
  (ii) "the reduction modulo the prime p ... is well-defined" when challenge differences are
  coprime to p — [LaBinius p.6–8 §1.3.1]
- The key sentence on bitness: "The extracted witness over K need not itself have binary
  coefficients: its reduction modulo 2 defines the binary coefficients to which the arithmetised
  statement refers. We therefore avoid a separate proof that the committed witness is binary."
  — [LaBinius p.8]
- Relaxed opening actually extracted: (W, s) with F W = Y diag(s) mod q, ‖W‖ ≤ 2β′,
  ‖s_j‖_op ≤ 2γ_C, s_j ∉ 2R, and the committed message is read as
  (W mod 2) diag(s mod 2)^{-1} — [LaBinius Defs. 20–23 p.17; Cor. 1 p.21; Thm. 1 p.22]
- Generality statement: "In this work, we focus solely on p = 2 ... The same extraction strategy
  applies to other choices of p and R satisfying these conditions" (p inert so R_p is a field,
  challenge set injective mod p); also "negative valuations will only occur at prime ideals of
  small norm" — [LaBinius p.8]
- Field switch: Binius/Flock produce claims over T = F_2^128 (GHASH basis); LaBinius packs the 128
  coordinates into an F_2-subspace of F_2^162 (ι = 7, 128 partial evaluations sent) and runs a
  sumcheck over F_2^162 — [LaBinius §2.7 p.13–14; Fig. 2 p.18; §5.1.1 p.24]
- Costs of the opening: prover O(2^ν) multiplications in T and F, O(2^ν) ring-by-challenge
  multiplications; prover sends 2^ι elements of T, 3ν + r elements of F_2, and m ring elements
  of norm ≤ β′; verifier does O(nm) multiplications in R_q — [LaBinius Lemma 11 p.22–23]
- Folding is "three to four times faster than commitment" because the short (binary) witness is
  folded over a single fully-splitting prime — [LaBinius p.8 §1.3.2]

### Inferences
- The absence of a bitness check is *not* a range proof. The scheme binds the prover to a vector
  over F_p (p = 2) obtained by reducing a rational extracted witness; for p = 2 every residue is a
  bit, so "bitness" is vacuous. Nothing bounds the integer coefficients the prover actually
  committed beyond the SIS norm bound 2β′ needed for binding, which is a loose aggregate bound.
- For b-bit integers (b > 1) the mechanism gives nothing: the residue mod p of the extracted
  witness is some element of F_p; no statement "entry ∈ [0, 2^b)" follows. This is the same
  situation the docs already established for Akita (weak opening, no range).
- Replacing F_2 by our 128-bit prime p: (a) with p inert, R_p = F_{p^162}, a ~20,000-bit field, so
  the arithmetisation would have to live there, which is useless for us; (b) with p fully
  splitting, R_p ≅ F_p^162 is a SIMD product and challenge-difference invertibility holds only
  statistically per component (the LatticeFold-style regime the paper is trying to escape);
  (c) in either case the "committed value" is W mod p, an arbitrary field element, so option C's
  "range checks included" does not materialise. The decoupling trick could lower the *commitment
  modulus* for an Fp128 statement (small NTT-friendly q instead of a 128-bit q), which is a
  commitment-speed idea, not a range-check idea.
- The claim on p.2 that Akita "support[s] only arithmetisations over fixed finite fields whose
  characteristics have bit lengths typically between 30 and 50" is inconsistent with our use of
  Akita over Fp128; treat LaBinius's characterisation of Akita as loose.

### Gaps
- The paper never states how the decoupled commitment would perform for a non-binary, b-bit
  witness (number of RNS limbs needed as β grows), so "pay-per-bit" for LaBinius is not a paper
  statement; it only measures binary witnesses.
- No statement about upstream Jolt adopting LaBinius techniques (the docs' "#2040" claim is from
  the PR, not the paper). The paper cites Akita/Jolt only as a comparison point (p.2, p.4–5).

## Q1 (cont.). The "within 1.2x" claim, proof sizes with and without LaBRADOR, relation to Akita

### Takeaway
"At most 1.2x" is measured against the Binius codebase's own hash-based prover (BaseFold-style
opening) on Keccak-256, SHA-256 and BLAKE3 circuits at the largest sizes run, single core of an
AMD EPYC 9R05, 100-bit security; against Flock the lattice prover is faster (0.92–0.93x).
Non-recursive proofs are 0.3–5.4 MiB and grow as sqrt(N) (the commitment alone is 0.28–2.4 MiB);
with LaBRADOR the proof is 73–100 KiB at a 1.3–2x prover cost and a verifier of 0.1–0.7 s. There
is no author or code overlap with Akita.

### Cited Findings
- Hardware and security: "Every number below is a single thread pinned to one core of an AMD EPYC
  9R05 on an AWS m8azn.metal-12xl instance with 192 GiB of RAM"; SIS instances estimated with the
  Lattice Estimator at "λ = 100 bits of security"; PCS table rows "all target 100 bits"; Flock
  modified "to aim for 100 bits of security at rate 1/2" and switched to SHA-256 Merkle trees;
  AVX-512 kernels, Rust — [LaBinius p.27 §5.2; p.27 §5.1.8; p.26 §5.1.5; p.24 §5.1]
- Against whom: "For Binius, at the largest parameters ... our non-recursive prover costs at most
  1.2 times the original one and our non-recursive verifier less than 1.1 times, our recursive
  prover less than 1.5 times"; "For Flock ... our non-recursive prover takes about 0.9 of the time
  of the original one ... recursive prover costs less than 1.5 times"; recursive verifier "26 to
  27 times the Flock verifier" — [LaBinius p.28–29 §5.4]
- Hash table numbers (ms, KiB). Keccak-256 under Binius, 55,924 permutations: Binius P 10,821.3,
  V 6,250.7, |π| 471.3; LaBinius 12,166.0 / 6,266.6 / 5,452.5; +LaBRADOR 13,828.4 / 6,969.7 /
  97.6. SHA-256 under Binius, 72,628 compressions: 4,853.6 / 2,315.2 / 471.4 vs 5,753.1 / 2,488.6
  / 5,380.1 vs 6,923.1 / 3,191.2 / 97.7. BLAKE3 under Binius, 131,624 compressions: 6,544.8 /
  3,240.4 / 471.4 vs 7,334.4 / 3,399.9 / 5,417.8 vs 8,485.8 / 4,091.2 / 97.6 — [LaBinius
  Tables 2–4 p.29–30]
- BLAKE3 under Flock, 131,072 compressions: Flock 2,121.8 / 27.3 / 490.0; LaBinius 1,953.9 /
  38.6 / 5,130.3; +LaBRADOR 3,072.4 / 729.0 / 96.4. SHA-256 under Flock, 65,536: 2,184.1 / 27.8 /
  490.1 vs 2,036.8 / 38.7 / 4,848.6 vs 3,172.3 / 730.7 / 96.5 (Flock times include witness
  generation; Binius times exclude it) — [LaBinius Tables 5–6 p.31; p.28 §5.4]
- "over 67,000 BLAKE3 compressions per second" and "4.3 MB of data per second" non-recursive,
  "2.7 MB per second in the recursive mode" — [LaBinius abstract p.1; p.9; p.29]
- PCS-only table (ms, KiB), 2^24 elements: BaseFold 1/2 Comm 468.7, P 175.4, V 0.6, |π| 478.5;
  WHIR 1/4 1,069.4 / 594.5 / 1.1 / 300.9; Ligerito 1/4 963.5 / 917.9 / 0.6 / 249.2; Brakedown
  0.581 978.3 / 86.4 / 27.1 / 10,705.6; LaBinius 636.0 / 167.3 / 6.7 / |C| 2,430.0 / |π| 2,927.3;
  LaBinius + LaBRADOR 642.9 / 1,333.8 / 713.1 / |C| 4.5 / |π| 82.9. At 2^18: LaBinius 7.0 / 2.8 /
  0.3 / 283.5 / 307.5; +LaBRADOR 9.8 / 121.7 / 76.8 / 4.1 / 75.3 — [LaBinius Table 1 p.28]
- Paper's own reading of Table 1: non-recursive commit+prove "within 25% of Brakedown and
  BaseFold, and between 1.6 and 2.3 times faster than WHIR and Ligerito"; recursive proof "about
  three times smaller than the best of the others at the largest size"; "In terms of
  verification time, our scheme in the recursive mode is not competitive" — [LaBinius p.27–28 §5.3]
- Relation to Akita: authors are Osadnik (Aalto) and Seiler (IBM Research Europe); Akita is cited
  as [DBK+26] (Dao, Bodaghi, Khajehpour, Vitto, Badakhshan, Georghiades, Liu, Zhang, Thaler).
  Code: "https://github.com/osdnk/labinius"; recursion links "the C implementation of LaBRADOR
  ... as a static library" (LaZer). Akita's Keccak throughput via Jolt is *estimated* by the
  authors at "600 hashes/s on a MacBook CPU and 3,000 hashes/s with GPU acceleration" — [LaBinius
  p.1; p.24 fn.3; p.27 §5.1.7; p.5 and fn.1]
- Binius and Flock codebases: "at commit 81cdfff of 14 September 2026 and Flock at commit b684b12
  of 11 September 2026"; "We could not integrate BinarySpartan [Set26], as its code was not
  available" — [LaBinius p.26 fn.4; p.26 §5.1.5]

### Inferences
- Ratios from the tables: Binius Keccak 1.124x, SHA-256 1.185x, BLAKE3 1.121x (non-recursive);
  recursive 1.28x, 1.43x, 1.30x. Flock BLAKE3 0.921x, SHA-256 0.933x; recursive 1.45x. The
  abstract's "at most 1.2 times" is accurate for the non-recursive Binius integration only.
- 131,072 / 1.9539 s = 67,082 compressions/s, so the headline number is the Flock BLAKE3 row.
- Per-element cost at 2^24 F_2^128 elements, single core: commit 636 ms → 37.9 ns per 128-bit
  element ≈ 0.30 ns per bit; commit + open 803 ms → 47.9 ns per element ≈ 0.37 ns per bit.
  BaseFold 1/2 on the same machine: 38.4 ns per element. These are hash-circuit witnesses (bits);
  they are not directly a Fp128 figure.
- The non-recursive commitment |C| is O(sqrt N) and reaches 2.4 MiB at 2^24; for us that would be
  part of the proof unless wrapped, which costs 1.3 s and a 0.7 s verifier at 2^24.

### Gaps
- No multi-threaded numbers anywhere in the paper, so no comparison with the 32-thread hash
  figures quoted in our docs is possible on equal footing.
- No Fp-type (odd prime) instantiation is implemented or parameterised.

## Q2. Neo/SuperNeo: what "pay-per-bit" means, fields, exactness of the norm guarantee, folding costs, composability with a non-folding Fp128 PIOP, one-hot vs dense

### Takeaway
"Pay-per-bit" means the Ajtai commitment cost scales with the *norm* of the committed
coefficients because, without an NTT (d ≤ 64, |F| ≤ 64 bits), ring multiplication is a sum of
rotations scaled by coefficients, and a tiny coefficient costs almost nothing. The witness is
embedded coefficient-wise (norm-preserving), so a b-bit value is committed as b-ish tiny digits
after decomposition; every concrete parameter set uses norm bound b = 2, i.e. ternary digits. The
range guarantee is *exact* (ℓ∞ < b) and comes from a product-of-(z−a) zero-check batched into the
single sumcheck over K; it is not a property of the commitment. The construction is generic over
a prime q but only parameterised for ≤ 64-bit fields, has no implementation or benchmarks in the
paper, and is a folding scheme (no PCS evaluation proof). The embedding plus in-sumcheck range
check is exactly our roadmap D3; it composes with Fp128 in principle but the opening proof would
still have to be Akita's.

### Cited Findings
- Six desiderata, PB defined: "committing to a vector of b-bit values should be roughly
  log|F|/b times cheaper than committing to values that span the full field"; LatticeFold loses it
  because "the NTT is not a norm-preserving map" — [Neo p.4 §1.1]
- Mechanism of pay-per-bit: "When both the degree d ≤ 64 and field bit size ⌈log2|F|⌉ ≤ 64 are
  small, a ring operation a·b ∈ R_F is most efficiently implemented (using AVX-512) with a
  rotation matrix product ... The dominating costs are scaling b_i · rot(a)_i ∈ F^d, which scales
  linearly with the norm of b_i. When the b_i's are small (such as bits), the cost to compute the
  ring operation is essentially adding the rotations rot(a)_i ∈ F^d for which b_i is non-zero."
  Also: the ring "cannot be perfectly splitting ... so using the NTT transform is a more
  inefficient method" — [Neo p.15–16 "Pay-per-bit costs"]
- Embeddings: Neo places d field vectors in the d coefficient slots (SIMD); SuperNeo packs a
  single vector z ∈ F^{dn} into n ring elements, "norm-preserving ... since the underlying field
  vector z ∈ F^{dn} is exactly the coefficients of the committed ring vector"; evaluation recovered
  as the constant term via the inner-product trick Trans_F — [Neo p.8–10; Thm. 8–10 p.24–25]
- Norm check and where it lives: Π_CCS runs one sumcheck over K on
  Q(X) = Eval_K + γ^{kd} Eval_A + γ^{kd(t+1)} eq(X, α)(F(X) + γ^K NC(X)) with
  NC(X) = Σ_i γ^{i−1} Π_{a=−b+1}^{b−1} (z̃_i^{♭,pad}(X) − a); Lemma 9: the sum identity holds iff
  the CCS constraints hold, "‖z_i^♭‖_∞ < b for all i", and the evaluation claims hold — [Neo
  §7.3 p.30–31; Lemma 9 p.42–44]
- The relation requires it: CCS(b, L) and CE(b, L) include "‖z^♭‖_∞ < b" — [Neo Defs. 20–21 p.29]
- Why the exact norm is needed: binding is only "(2B, C)-relaxed binding"; the extracted
  openings are relaxed Δ·c = A z, and including "sum-check constraints which ensure that the
  extracted openings are low norm" is what makes the composition knowledge sound — [Neo p.16–17
  §2.4; Def. 7 p.20; Def. 22 p.29]
- Decomposition: Π_DEC splits z^♭ by signed base-b digits into k vectors with ‖z_i^♭‖_∞ < b, and
  the verifier checks c = Σ b^{i−1} c_i and y = Σ b^{i−1} y_i homomorphically (no sumcheck) —
  [Neo §7.5 p.33]
- Concrete parameters (all with b = 2): Almost Goldilocks q = 2^64 − 2^32 + 1 − 32, η = 128,
  Φ = X^64 + 1, d = 64, κ = 15, m = n_L = n_F = 2^33, n_R = 2^27, k = 13, K ∈ [50], B = 2^13,
  |C| = 2^128, |K| ≈ 2^128, "≈ 129 bits of security"; Goldilocks: η = 81, Φ = X^54 + X^27 + 1,
  d = 54, κ = 18, m = 2^30, n_L = 2^30 − 46, n_R = 19,884,107, k = 14, K ∈ [61], B = 2^14,
  T = 216, |C| ≈ 2^125; Mersenne 61: η = 81, d = 54, κ = 18, m = 2^28, n_R = 4,971,026, k = 14,
  B = 2^14, |C| ≈ 2^125, |K| ≈ 2^122, ≈ 129 bits. "This section provides three efficient
  parameterizations over ≤ 64-bit fields"; "τ = 1, so L = F, and take K := F_{q^2}" — [Neo §8
  p.34; lattice-estimator script p.58–59]
- Field generality: Def. 1 takes "Base field: F := F_q a prime field of order q", with L a
  degree-τ extension for the relation and K a degree-ν extension for sumcheck, A the ambient
  field; cyclotomic Φ of arbitrary index η (not only X^d + 1), which is what allows Goldilocks
  — [Neo Def. 1 p.18; p.11–12]
- Folding costs stated qualitatively: "the prover's work is dominated by the cost to commit to
  a witness"; the verifier work is "roughly taking the weighted sum of the commitments"; recursion
  goal "logarithmic recursion overhead (with similar constants) analogous to HyperNova" — [Neo
  p.3–5]
- Soundness terms: ε_SC ≤ max(u, 2b, 2)·log m/|K|, ε_SZ = (kd(t+1) + 2K + k − 1 + log m)/|K| —
  [Neo p.47]
- Related work the paper positions against: Hachi "is the first lattice-based polynomial
  commitment scheme supporting extension-field evaluation of field polynomials"; "Neo (prior to
  Hachi) is the first work to check norms of committed ring vectors via a sum-check purely over
  the extension field"; Cyclo is "equivalent to Neo" and observes the norm check "can be removed
  for efficiency" when soundness is needed only for a bounded number of steps — [Neo p.12–13]
- No implementation, benchmark or evaluation section exists in the paper; the only benchmark
  mentioned is Cyclo's "initial benchmark" of Neo — [Neo p.12; table of contents p.2]
- The paper does not discuss one-hot encodings anywhere.

### Inferences
- "Pay-per-bit" in SuperNeo is literally: decompose values into ternary digits (b = 2), commit
  the digits, and the rotation-sum commitment cost is proportional to the number of nonzero
  digits. A b-bit value therefore costs about b digit-slots; a 4-bit digit would cost 4 slots or
  a degree-7 (signed) zero-check if committed as one coefficient with b = 16 → degree 31. The
  exponent of the norm check, 2b − 1, is why the paper picks b = 2.
- On "exact or approximate": exact, ℓ∞ < b, enforced on every coefficient by the zero-check.
  The only approximate object is the *binding* (relaxed, factor 2B/T slack), which is why the
  exact in-PIOP check is mandatory. This is the mirror image of Akita: the commitment certifies
  nothing about ranges; the PIOP does.
- Composition with a non-folding sumcheck over Fp128: the SuperNeo embedding (coefficient
  packing + Trans) and the NC(X) zero-check are field-generic and would sit inside our relation
  engine as "one more term of the batched sumcheck"; but SuperNeo supplies no evaluation proof
  for the Ajtai commitment, so Akita's opening would still be needed, and the Ajtai commitment
  over Fp128 (q = 128-bit modulus) is Akita's regime, not the d ≤ 64, |F| ≤ 64 regime in which
  the rotation-sum "pay-per-bit" argument is made. Whether Akita's NTT-based commitment has any
  pay-per-norm behaviour is not addressed by this paper; the docs' upstream measurement (K = 16
  vs K = 256 at 2^29) suggests magnitude does matter for Akita, but that is a one-hot measurement.
- One-hot vs dense: SuperNeo never considers one-hot. Its design is "dense tiny digits +
  zero-check", i.e. a published, security-proved instance of our option B / roadmap D3 with
  digit base 2. It does not settle the microbenchmark the roadmap asks for (dense narrow cells vs
  one-hot nonzeros), because no numbers exist.

### Gaps
- No prover-time, proof-size or verifier-time measurements at all; the folding cost is stated
  only as "dominated by commitment".
- No 128-bit-field parameter set; no statement about M-SIS parameters when q is 128 bits.
- Zero knowledge is not discussed; transparency follows from A being uniformly random but is not
  claimed as a feature.

## Q3. Mikros: what bitwidth-aware packing is, fields, cost, range guarantee, applicability to Fp128

### Takeaway
Mikros packs a length-N vector over a small field F into a length-N/κ vector over a degree-κ
extension E (κ ≈ λ/log2 q), using a fixed F-basis for linear relations and formal polynomials
(Lagrange / Laurent) for bilinear ones, so that existing extension-field sumcheck protocols cost
O(Nκ) base-field operations instead of O(Nκ²). It is purely a PIOP transformation on top of any
multilinear PCS over E. It gives no range guarantee beyond "entries are elements of F"; its gain
is the ratio κ = [E : F], which is 1 for a 128-bit prime field, so it does not apply to Fp128.

### Cited Findings
- Core: "We pack vectors of length N = mκ over a base field F into length-m vectors over a
  degree-κ extension field E, which turns length-N relations over F into length-m relations over
  E"; "the prover cost is proportional to the effective bitwidth, the combined bitwidth of the
  witness entries and the wiring-position encodings" — [Mikros abstract p.1]
- Fields: "Let F be a small finite field of order q, and let E = GF(q^κ) ... κ ≃ λ/log2 q; for
  fixed λ, κ scales inversely with the base-field bitwidth log2 q" — [Mikros p.2; p.10 §3.1]
- Two routes: fixed-basis packing ā = Aβ̄ for linear maps (requires L with base-field entries);
  polynomial packing g_A(X) = Aφ(X) for bilinear relations, evaluated at a random x̄ ∈ E after
  auxiliary polynomials h are fixed — [Mikros §2.1 p.7–8]
- Costs: packed inner product O(Nκ + κ³) bb, memory O(N + κ²) F (Prop. 1 p.15); packed Hadamard
  O(Nκ + κ³) bb (Prop. 2 p.17); packed permutation/grand product O(Nκ log2 κ + κ⁴) bb (Prop. 3
  p.20); packed PCS opening O(κN) bb on top of an extension-field PCS over E^m (§4.4 p.21, Prop. 4
  p.43); single-instance SNARK prover O(Nκ log_q N log2 κ) bb, multi-instance amortised O(Nκ) bb
  per instance (Thms. 1–2 p.23, 25)
- Table 2 comparison of Hadamard provers: classical sumcheck O(Nκ²); SFF O(Nκ(3/2)^t_skip + ...);
  Gruen O(N t_skip + Nκ + Nκ²/2^t_skip); Flock O(N 2^t_skip + Nκ + ...), binary only; Mikros O(Nκ),
  "General extensions" — [Mikros Table 2 p.5]
- Lookup: "our bitwidth-aware lookup argument can prove nonlinear activation functions on
  low-bitwidth inputs in zkML protocols"; indexed lookup via packed permutation with
  value–position–counter encoding, cost O((M+N) d κ log2 κ′) bb — [Mikros p.7; App. B.5 p.49–51]
- Small-field caveat: when q < 2κ − 1 the interpolation nodes must come from an intermediate
  field K = F_{q^t}, with precomputed tables of size O(t q^{2t}) — [Mikros §4.2 p.16–17]
- PCS: "Solution 1: From a PCS over the Extension Field" (any multilinear PCS over E^m; packing
  is a bijection so binding carries over, Prop. 4 p.43); "Solution 2: From a Linear Code over the
  Base Field" via code switching — [Mikros App. B.2 p.41–44]
- No cryptographic assumption of its own; soundness errors are O(N/|E|) + ε_pc; security level
  stated only as |E| = q^κ ≥ 2^λ (κ = 128 over F_2 in experiments) — [Mikros Thm. 1–2 p.23, 25;
  p.10; p.26]
- Zero knowledge is not discussed (the word appears only in reference titles).

### Inferences
- Range/bitness: the only "range" Mikros provides is membership in F. For F = F_2 that is
  bitness by type; for F = GF(2^8) it would be "8-bit by type". There is no sub-field range
  check; a b-bit range inside a larger F needs the lookup argument of B.5, at O((M+N)dκ log κ′)
  base-field cost.
- Applicability to Fp128: F must be *small* with κ = [E:F] ≥ 2. With F = Fp128 the only
  extensions are F_{p^κ} with κ ≥ 2, i.e. a 256-bit-plus challenge field with no gain (κ = 2 saves
  nothing; the paper's regime is κ ≈ 128). Mikros therefore requires leaving Fp128 for a field
  with log2 q ≈ 8–64 and an extension of degree 2–16+. It is irrelevant to the lattice stack.
- The idea that *does* transfer conceptually is already in the docs as "small-value sumcheck"
  (Group E), and Mikros's Table 2 confirms that the gain is the κ = log|E|/log|F| ratio, which for
  our two-limb field over small integers the docs already estimate at ~9x per multiplication.

### Gaps
- No lattice or Fp-prime instantiation; no statement about an Ajtai/Akita PCS as the underlying
  "PCS over E".

## Q4. Across the three: does any construction give commitment cost ∝ b *and* an exact [0, 2^b) proof at low cost with claims usable at a sumcheck's random point?

### Takeaway
No construction in these papers gives all three over Fp128. SuperNeo comes closest in structure
(coefficient-packed Ajtai commitment, exact ℓ∞ < b range by zero-check, evaluation claims
y = M̃z(r) at the sumcheck point) but at b = 2, over ≤ 64-bit fields, as a folding scheme without an
evaluation proof, and unimplemented; it is our D3 by another name. LaBinius gives cost ∝ bits and
claims at a random point, but only mod-2 semantics (no integer range) and only over binary
extension fields. Mikros gives cost ∝ base-field bitwidth and claims at a random point but no
range beyond field membership and nothing for a 128-bit prime.

### Cited Findings
- SuperNeo satisfies (a): commitment cost ∝ coefficient norm via rotation sums [Neo p.15]; (b):
  exact ‖z^♭‖_∞ < b via NC(X) in Π_CCS [Neo p.30–31; Lemma 9 p.43]; (c): the sumcheck ends with
  claims y′_i = h̃_i(r′) whose constant terms are z̃^{♭,pad}(r′) and M̃_j z_i(r′) [Neo step 4,
  p.31; Thm. 10 p.25]. Field: F = F_q small prime with K = F_{q^2}, parameters only for ≤ 64-bit
  q [Neo §8 p.34].
- LaBinius satisfies (a) for bits (0.30 ns/bit commit single-core, inference from Table 1 p.28)
  and (c) (claim MLE[f](r) = t over T = F_2^128 after field switch, Thm. 1 p.22); (b) fails: the
  committed message is W mod 2, with W extracted as a rational vector [LaBinius p.7–8].
- Mikros satisfies (a) in the sense "prover cost ∝ log2 q" [Mikros p.2] and (c) (terminal MLE
  claims at ȳ ∈ E^{log m}, Fact 1 p.12; Algorithm 4 p.38); (b) only as field membership; requires
  [E:F] = κ ≫ 1 [Mikros p.10].

### Inferences
- If one leaves Fp128: the closest fit to "(a)+(b)+(c)" is SuperNeo's design over Goldilocks or
  M61 with K = F_{q^2} (≈ 128-bit soundness), committing ternary digits and range-checking with a
  degree-3 term in the existing sumcheck; the commitment would be an Ajtai commitment over a
  64-bit modulus with d = 54–64, and a compressing SNARK (Spartan + FRI) would replace Akita's
  opening. This is a whole-stack change comparable to option A/D, not an add-on.
- Staying on Fp128: the transferable content is only the shape of D3 — dense tiny digits plus an
  exact product zero-check plus homomorphic recomposition — and the engineering question remains
  Akita's per-coefficient commitment cost for dense small values versus one-hot nonzeros, which
  none of the three papers measures.

### Gaps
- No paper reports commitment cost for dense b-bit integers (b in 4–8) on any lattice scheme.

## Q5. Concrete numbers, hardware and security, for comparison with Akita (~90 ns per one-hot nonzero on M3) and hash-based (~28–35 ns per element on 32 threads)

### Takeaway
LaBinius is the only one of the three with a lattice commitment benchmark: 37.9 ns per 128-bit
element (0.30 ns per bit) to commit and 47.9 ns per element to commit and open, single core,
AMD EPYC 9R05, 100-bit security, on bit-valued hash witnesses. Mikros reports PIOP-only times over
F_2 (0.64 ns per Hadamard entry at 2^30) on a Xeon Gold 6342, single thread, excluding
commitments. Neo/SuperNeo has no measurements.

### Cited Findings
- LaBinius hardware/security: AMD EPYC 9R05, one core, AWS m8azn.metal-12xl, 192 GiB, AVX-512,
  λ = 100 bits for SIS and 100 bits for all compared PCS — [LaBinius p.27 §5.2, §5.1.8, §5.1.6]
- LaBinius PCS at 2^24 F elements: Comm 636.0 ms, P 167.3 ms, V 6.7 ms, |C| 2,430.0 KiB, |π|
  2,927.3 KiB; recursive: 642.9 / 1,333.8 / 713.1 / 4.5 / 82.9. At 2^22: 164.7 / 44.8 / 2.7 /
  1,174.5 / 1,394.9 and recursive 174.2 / 603.4 / 370.4 / 4.1 / 86.1. At 2^20: 42.7 / 10.8 / 1.2 /
  587.2 / 658.1 and recursive 48.1 / 300.9 / 192.8 / 4.1 / 72.6. At 2^18: 7.0 / 2.8 / 0.3 / 283.5 /
  307.5 and recursive 9.8 / 121.7 / 76.8 / 4.1 / 75.3 — [LaBinius Table 1 p.28]
- Hash-based on the same machine at 2^24: BaseFold 1/2 468.7 + 175.4 ms, V 0.6 ms, 478.5 KiB;
  BaseFold 1/4 907.0 + 243.8, 0.4, 345.3; WHIR 1/2 649.9 + 909.9, 2.0, 352.7; Ligerito 1/2
  482.9 + 837.8, 1.0, 469.5; Brakedown 0.581 978.3 + 86.4, 27.1, 10,705.6 — [LaBinius Table 1 p.28]
- LaBinius end-to-end (see Q1 cont.): e.g. Flock BLAKE3 131,072 compressions in 1,953.9 ms
  non-recursive (verifier 38.6 ms, 5,130 KiB) or 3,072.4 ms recursive (729.0 ms, 96.4 KiB) —
  [LaBinius Table 5 p.31]
- Mikros hardware: "dual-socket server with two Intel Xeon Gold 6342 processors (24 cores each,
  nominally 2.80 GHz), 384 GiB of RAM, Ubuntu 22.04.5", single thread, GFNI/VPCLMULQDQ/AVX-512,
  Ligerito PCS shared with Flock; κ = 128 over F_2 — [Mikros §6 p.26]
- Mikros Table 3, binary Hadamard N = 2^30, PIOP only: SFF (Binius) 106,331.53 ms, Flock 995.23,
  Mikros 684.48 (1.45x); verifier 39 / 396 / 34 µs; proof 1,504 / 4,912 / 4,888 B; peak RSS
  12.38 / 0.76 / 0.89 GiB — [Mikros p.26]
- Mikros Table 4, batches of 256 hashes, end-to-end single thread: SHA-256 Flock 174.27 ms vs
  Mikros 79.54 (2.19x), verifier 37.47 vs 38.34 ms, proof 306.59 vs 367.71 KiB; Keccak 255.27 vs
  100.15 (2.55x), 7.62 vs 19.30 ms, 289.85 vs 380.31 KiB; BLAKE3 103.96 vs 56.13 (1.85x), 35.14
  vs 35.74 ms, 279.86 vs 355.64 KiB; RSS equal — [Mikros p.27]
- Mikros Table 5, single binary circuit N = 2^25 gates: Flock 10.12 s vs Mikros 86.56 s (6.72 s
  excluding the permutation argument); verifier 4,316.57 ms vs 24.43 ms; proof 426.49 vs
  1,169.23 KiB — [Mikros p.27]
- Neo/SuperNeo: no measurements; security ≈ 129 bits M-SIS by lattice estimator for all three
  parameter sets — [Neo §8 p.34]

### Inferences
- Normalisations (single core, different hardware from our M3, bit-valued witnesses): LaBinius
  commit 0.30 ns/bit, commit + open 0.37 ns/bit. Our Akita figure of ~90 ns per one-hot nonzero
  carries 4 bits of information (K = 16), i.e. ~22 ns per useful bit. The ~60x gap mixes
  hardware, scheme, field (F_2^162 vs Fp128) and encoding (dense bits vs one-hot), so it is a
  motivation for the D3 microbenchmark, not a result.
- Against the docs' "28–35 ns per element on 32 threads" hash figure: LaBinius's single-core
  BaseFold number of 38.4 ns per 128-bit element on an EPYC is in the same range *single-threaded*,
  so the docs' hash figure may be conservative or refer to smaller (31/64-bit) elements; worth
  re-sourcing before it is used in a decision.
- Mikros's 0.64 ns per Hadamard entry (2^30, single thread, F_2) is a PIOP number with
  commitments excluded and does not bear on our commitment-dominated profile.

### Gaps
- No multi-threaded data in any paper; no M-series data; no Fp128 data.

## Q6. Assumptions, post-quantum status, zero knowledge, transparency, implementation, stated limitations

### Takeaway
LaBinius: SIS over a cyclotomic (super)ring with composite small modulus, plus LaBRADOR for
recursion; plausibly PQ; transparent; no ZK (listed as future work); open Rust/AVX-512 code.
SuperNeo: Module-SIS (ℓ∞), relaxed binding with strong sampling sets; plausibly PQ; transparent;
ZK not discussed; no implementation. Mikros: no assumption of its own (information-theoretic
reductions over any knowledge-sound PCS; hash-based Ligerito in experiments); ZK not discussed;
open Rust code; permutation argument is its stated weak spot.

### Cited Findings
- LaBinius: SIS_{R,q,m,n,β} hardness (Def. 4–5 p.11); Setup samples F uniformly (Def. 24 p.22,
  transparent); recursion assumes "knowledge soundness of LaBRADOR" (Thm. 2 p.48); "A second
  direction is to extend our construction to zero knowledge" (p.9–10); code at
  github.com/osdnk/labinius (p.24 fn.3); limitations stated: recursive verifier "not competitive"
  (p.27–28), sqrt-size commitment and proof non-recursively (p.27), BinarySpartan not integrated
  (p.26), p = 2 only (p.8), and the module-BKZ/subfield attack accounting "obtained over Q(ζ_3)"
  (p.27). Acknowledgement: generative AI used for text and code (p.29).
- SuperNeo: MSIS^{∞,κ,q}_{m,B} (Def. 5 p.19); Ajtai commitment B-binding and (B, C)-relaxed
  binding (Thm. 6 p.20); "plausible post-quantum security" (abstract p.1); A random (Def. 8 p.20);
  interactive-reduction framework for modular proofs (§6 p.27–28); AI disclaimer (GitHub Copilot)
  App. A p.40; no implementation section.
- Mikros: "Combining proof protocols with a knowledge-sound PCS and applying the Fiat–Shamir
  transform yields a non-interactive argument of knowledge" (p.11); Lemma 3 composes PIOP
  soundness with PCS knowledge soundness (p.33–34); experiments use Ligerito (p.26); code
  "git@github.com:mirandaprivate/mikros.git" (p.26 fn.1); limitations: "proving the wiring
  matrix—the permutation argument of Section 4.3—is by far the most expensive part" (p.27);
  larger proofs and slower verifier than Flock in batches (p.27); intermediate-field tables of
  size O(t q^{2t}) for q < 2κ − 1 (p.17); AI tools acknowledged (p.29).

### Inferences
- None of the three offers zero knowledge today, so none changes the docs' ZK line
  ("Upstream's Whiteout stack, when it becomes selectable").
- Mikros inherits PQ status from its PCS; with Ligerito it is hash-based PQ. It is not a lattice
  paper and should not be grouped with LaBinius in the docs.

### Gaps
- Whether LaBinius's 100-bit SIS target with module-BKZ accounting is comparable to Akita's
  security claim is not derivable from this paper alone.

## Q7. What this changes in the docs

### Takeaway
Two statements in `docs/research-directions.md` are wrong as written (option C's description of
LaBinius and the "commit once, range checks included" characterisation of C), one owed read
(SuperNeo) resolves into support for option B / roadmap D3, and the reading-depth table must
move three papers from "abstract only" to "read in full". `docs/optimization-roadmap.md` D3 is
strengthened and gains two design constraints (digit base vs zero-check degree; homomorphic
recomposition). Nothing in the three papers supports leaving Fp128 for their constructions.

### Cited Findings
Each item quotes the doc, then the correction with the page that justifies it.

1. research-directions.md line 15, reading table: "Abstract only | BitZ, Limber, LaBinius,
   Mikros, Celer, SuperNeo, ...". Correction: move LaBinius, SuperNeo and Mikros to "Read in
   full" (this note).

2. research-directions.md line 175, option C: "Commit integer bits in a binary field, prove over
   a prime field ... LaBinius (2026/2103) does the lattice version within 1.2x of hash-based
   provers ... whether the target ring can be our fixed 128-bit prime is unknown."
   Contradicted. LaBinius commits bits as *integer coefficients in a cyclotomic ring modulo a
   composite of 14–15-bit primes* and proves over the *binary* field F_2^162 (reduction mod 2),
   not "in a binary field, prove over a prime field" [LaBinius p.6–8, p.24–25]. "Within 1.2x" is
   against the Binius codebase's own BaseFold-style prover on Keccak/SHA-256/BLAKE3, single core
   EPYC 9R05, 100-bit security; against Flock it is 0.92x [LaBinius p.28–31]. The target field
   can be another prime p in principle [p.8], but then the evaluation field is R_p (a degree-162
   extension or a SIMD product), and for a 128-bit p the committed meaning W mod p is an arbitrary
   field element, so no range follows (inference from the mechanism on p.7–8). Replace with:
   "LaBinius decouples the commitment modulus from the evaluation prime; it only instantiates
   p = 2, where bitness is vacuous. It gives no integer range guarantee and does not target
   prime fields."

3. research-directions.md line 229–230, strategy C: "A bit-committed hybrid through BitZ or
   LaBinius. The only route found to 'commit once, derive everything, range checks included'."
   Weakened: the LaBinius leg provides no range checks [LaBinius p.8: "its reduction modulo 2
   defines the binary coefficients"]; drop LaBinius from C. The BitZ leg is outside this reading.

4. research-directions.md line 232 and 236–237, "C as the next question ... If BitZ's range
   guarantee is exact over a fixed prime and its commitment is an order of magnitude cheaper per
   bit, C replaces one-hot range digits". Unchanged for BitZ; the LaBinius alternative is gone.
   SuperNeo supplies the structural answer instead: the exact range guarantee comes from a
   zero-check in the PIOP, not from the commitment [Neo p.30–31, Lemma 9 p.43], which is option
   B / D3, so "C as the next question" collapses into "D3 with the digit base as the variable".

5. research-directions.md line 174, option B: "Dense digit columns with our own zero-check ...
   Needs a microbenchmark of dense narrow cells against one-hot nonzeros; no source has that
   number." Strengthened: SuperNeo is a published, security-proved lattice scheme built on
   exactly this (coefficient-packed Ajtai commitment, NC(X) = Π_{a=−b+1}^{b−1}(z̃ − a) inside the
   one sumcheck, homomorphic recomposition in Π_DEC) [Neo §7.3 p.30–31, §7.5 p.33]. Still no
   number: the paper has no implementation. Keep the microbenchmark gate.

6. research-directions.md line 260, reads owed: "SuperNeo, below BitZ in priority." Done; see Q2.
   Result: supports B/D3; no measurements; b = 2 in every parameter set; ≤ 64-bit fields only.

7. research-directions.md line 29, thesis: "The bottleneck is one-hot digits over a 128-bit
   field, not lattices as such." Strengthened by LaBinius: a lattice commitment reaches
   0.30 ns per committed bit single-core on bit-valued witnesses (inference from Table 1 p.28),
   so lattices as such are not the ceiling; what is committed is.

8. research-directions.md line 159–161: "verifier time (hash-based verifiers run 0.8–3.3 ms
   against Akita's 10.5–12.7 ms ...)". Consistent with LaBinius Table 1 (lattice non-recursive
   0.3–6.7 ms growing as sqrt N; hash 0.2–2.2 ms flat; lattice recursive 77–713 ms) [p.28]. Add
   that LaBRADOR-compressed lattice proofs cost a 0.1–0.7 s verifier, relevant to the "sub-100 KB
   by recursion" line (line 153).

9. research-directions.md line 176, option D: "CPU throughput is about 28–35 ns per element on
   32 threads by published figures." Weakened as a comparison basis: LaBinius measures BaseFold
   at 38.4 ns per 128-bit element on *one* core [Table 1 p.28]. Re-source the 32-thread figure
   (element size, field) before using it in a decision.

10. research-directions.md line 189–191 (LogUp/grand-product soundness loss): consistent with
    Mikros's permutation bound ε_Perm ≤ min{1, (N−1)/|E|} + 2ε_GP + ε_PCS [Mikros Prop. 3 p.20].
    No change.

11. optimization-roadmap.md D3 (line 262–266): "a dense digit column with an application-level
    zero-check and a recomposition identity, replacing one-hot digits for remainders: 16x fewer
    committed cells per 4-bit digit and no Booleanity." Strengthened by SuperNeo, with two
    constraints the item should state: (i) the exact zero-check for a digit in (−b, b) has degree
    2b − 1 (16 for [0,16), 31 signed), which is why SuperNeo uses b = 2 [Neo p.30, §8 p.34]; so
    "4-bit dense digit" trades 4x fewer cells against a degree-16+ relation versus "bit digits" at
    degree 2–3; (ii) recomposition can be done homomorphically on commitments and evaluation
    claims (c = Σ b^{i−1} c_i, y = Σ b^{i−1} y_i) rather than as a sumcheck identity [Neo §7.5
    p.33]. The "16x fewer cells" count is per cell; the per-cell commitment cost for dense
    small values on Akita remains unmeasured.

12. optimization-roadmap.md line 276–279, dropped items "Range bounds from the Akita
    commitment": confirmed in spirit by SuperNeo, whose whole security argument relies on the
    PIOP proving the norm because binding is only relaxed [Neo §2.4 p.16–17, Def. 7 p.20].

13. optimization-roadmap.md Group E line 296–298, small-value sumcheck "about 9x on our two-limb
    field": Mikros Table 2 [p.5] frames the same gain as κ = log|E|/log|F| and shows it is the
    extension degree that is saved, i.e. zero for a prime field acting as its own challenge
    field. No change to the estimate; Mikros is not a route for us.

14. Both docs' references (research-directions.md line 273): keep LaBinius and Mikros, but
    annotate "binary fields only" and "small-field PIOP packing, inapplicable to Fp128".

### Inferences
- The docs' Direction section should read: B now; D3 as the first commitment-side experiment,
  with digit base b ∈ {2, 4, 16} as the variable; C survives only as "read BitZ/Limber".

### Gaps
- Whether upstream's #2040 "binary-field track" actually uses LaBinius's decoupling is not
  verifiable from the paper; it must be read from the PR.

## Q8. Ranked view: what jolt-atlas should do about each paper

### Takeaway
Pursue SuperNeo's idea (as D3), park LaBinius, drop Mikros.

### Cited Findings
- SuperNeo: the embedding, exact norm zero-check and homomorphic decomposition are stated on
  [Neo p.8–11, p.30–31, p.33]; no implementation [Neo, no §]; parameters for ≤ 64-bit fields only
  [Neo p.34].
- LaBinius: binary-only, p = 2 [LaBinius p.8]; 0.30 ns/bit single-core commitment [Table 1 p.28];
  "same extraction strategy applies to other choices of p" [p.8]; sqrt-size commitments and slow
  recursive verifier [p.27–28].
- Mikros: gain is κ = [E:F] [Mikros p.2, p.10]; binary experiments only [p.26–27]; PIOP-only
  contribution [p.41–44].

### Inferences
1. **SuperNeo — pursue, as roadmap D3 (rank 1).** It is the published form of "dense tiny digits
   + exact zero-check + homomorphic recomposition", with a security proof structure (strong/weak
   interactive reductions) we can borrow for the relation-engine argument. The one experiment that
   settles it is the D3 microbenchmark the roadmap already names, run with three digit bases:
   commit N dense cells with values in {0,1} (b = 2), [0,4) and [0,16) through Akita and compare
   per-cell cost with one one-hot nonzero of K = 16; then add the cost of the degree-(2b−1)
   zero-check term in the batched sumcheck. If bit-digits at b = 2 beat one-hot per *useful bit*,
   D3 proceeds with b = 2 and the Booleanity/HammingWeight machinery is replaced by the product
   check. A further read: Cyclo (EUROCRYPT 2026) for its "initial benchmark" of Neo, the only
   measurement of this design that the paper points to [Neo p.12].

2. **LaBinius — park (rank 2).** It does not give range checks and does not target prime fields;
   its decoupling of commitment modulus from evaluation prime is a commitment-speed idea whose
   Fp128 variant would land in R_p = F_{p^162} or a SIMD product. Revisit only if (a) upstream's
   binary track (#2040) shows the decoupling applied to a prime p, or (b) a Hachi/RoKoko-style
   extension-field evaluation over a small-modulus Ajtai commitment appears for odd p. The one
   further read: a16z/jolt #2040's description, to see whether "its techniques" means the
   composite-modulus commitment or only the binary field. No experiment on our side.

3. **Mikros — drop (rank 3).** κ = 1 for Fp128; the technique needs a small base field with a
   high-degree extension, which is option A/D territory, and even there it reports larger proofs
   and slower verifiers than Flock and an 8.5x slower single-instance prover. Nothing to run.
   If the project ever leaves Fp128 for a small field, re-read §4.3 and B.5 (bitwidth-aware
   grand product and lookup) alongside the LogUp soundness line in the docs.

### Gaps
- None of the three answers the roadmap's central unknown, the per-cell cost of dense small
  values on Akita versus one-hot nonzeros; that remains our measurement to make.
