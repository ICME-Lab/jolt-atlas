# Sumcheck proving (ePrint 2026/587), BinarySpartan (2026/1656) and Jagged PCS (2025/917): what they mean for jolt-atlas

Read in full from the PDFs in `papers/` (2026/587: 51 pages, dated March 24, 2026; 2026/1656: 13 pages;
2025/917: 26 pages, dated April 10, 2026), then re-read for algorithms, cost lemmas and benchmark
tables. Page numbers are the printed page numbers, which coincide with PDF page numbers for all
three. Links point at the ePrint entry; the page cited is in the local PDF.

Labels: **[paper]** is a statement of the paper, quoted or closely paraphrased with a page.
**[mine]** is my inference or arithmetic. Our field numbers (Fp128 on M3: field multiplication
4–5 ns, field-by-u64 3.4 ns, addition 1.6 ns) come from the task brief, not from any paper.

Two repository facts used below, read from source and not run: `atlas-pcs/src/akita.rs` (header
comment) says Akita has no additive homomorphism, that dense polynomials of one `(encoding,
num_vars)` are prefix-packed into one physical polynomial per class, that up to 64 one-hot
polynomials of one arity form one one-hot class, and that every class with an opened member is
opened natively at its own point. `jolt-atlas-core/src/onnx_proof/ops/einsum/dot.rs` (header)
says every einsum sumcheck runs over operands folded at the output randomness, `Σ_h L(h)·R(h)·EQ(h)`.

---

## Q1. The sumcheck paper's techniques and cost model (2026/587)

### Takeaway

The paper has four prover-only techniques (small-by-big multiplication with delayed reduction, a
divide-and-conquer product of d multilinears costing Θ(d log d) big multiplications instead of
Θ(d²), round-batched "small-value" proving, and a split handling of the eq factor) plus one
protocol change (univariate skip adapted to large prime fields). Its cost model counts big-by-big
(bb) multiplications and defines κ as the bb-to-small-small (ss) cost ratio; it models κ ≈ 2N²+N
for an N-limb Montgomery field, so κ ≈ 10 for a two-limb field, close to our 9. All measurements
are on BN254 (N = 4, κ measured ≈ 29); it gives no 128-bit measurement.

### Cited Findings

Techniques and what is new in the extended version:

- Three techniques are headlined: a product-of-many-multilinears algorithm, a "small-value
  sum-check prover" (also a streaming prover), and near-free handling of an equality-polynomial
  factor. [paper, abstract p.1] — [2026/587](https://eprint.iacr.org/2026/587)
- "The present extended version reorganizes and expands that presentation, and directly combines
  and improves upon results from prior work [8, 23]" ([8] = ePrint 2024/1046 small characteristic,
  [23] = Dao–Thaler 2024/1210). The earlier version is [7] = ePrint 2025/1117. [paper, p.2]
- Section 7 (univariate skip over large prime fields) and Appendix D (optimized extrapolation) are
  presented as new relative to [7]: "In prior work [7], we described essentially the same
  small-value proving algorithm (called Algorithm 4 and Algorithm 6 in that paper). In this
  version, we opt for an alternative explanation ... This algebraic framing abstracts away the
  complex, round-by-round 'accumulator' state machines and Toom-Cook index mappings (e.g., the
  idx4 function) used in that prior work [7]." [paper, p.15]
- All techniques "except for one (univariate skip (Section 7)), leave the protocol, verifier, and
  soundness unchanged." [paper, p.2]

Cost model and κ:

- Montgomery multiplication of N-limb elements costs "2N² + N native multiplications"; a
  small-by-big (sb) product costs "2N + 1" (N for the bignum-by-u64 product, N + 1 for a single
  Barrett pass), "which in practice is about a 2-3x speedup for a ~256-bit field." [paper, p.9]
- Delayed reduction for a dot product `S = Σ c_i·a_i` with 64-bit `c_i`: "k small-big (integer)
  multiplications (at N native multiplications each) plus one final reduction (at N + 1 native
  multiplications), for a total of k·N + (N + 1) native multiplications." Requires `Σ c_i < 2^64`
  or one extra limb; negative coefficients are handled by "splitting the sum into positive and
  negative terms ... taking the difference ... and finally performing a Barrett reduction."
  [paper, p.10]
- Application to sumcheck: "For each summand, one can first compute h(x') := Π_{i<d} p_i(...),
  then form the final product h(x')·p_d(...) in an unreduced representation, and defer reduction
  until after many summands have been accumulated ... The effect is most pronounced for low
  degrees, especially d ∈ {2, 3}." [paper, p.11]
- Rough cost model for 256-bit fields: ss = 1, bb addition 5, sb 8, bb 32, sb linear combination
  of n variables 5n + 3. [paper, p.11]
- Measured on BN254 (M4 Max, single thread): ss multiply 0.36 ns, bb add 1.99 ns, sb multiply
  4.65 ns (2.02 bignum × u64 + 3.67 Barrett), bb multiply 10.50 ns (4.48 bignum + 6.67 Montgomery
  reduce). [paper, Table 3, p.21]
- κ definition: "Let κ denote the ratio of costs between bb and ss multiplications." Footnote 11:
  "For large prime fields with Montgomery multiplication, κ ≈ 2N² + N, where N is the number of
  limbs, so κ = Θ(N²); for tower extension fields with Karatsuba, κ ≈ N^{log₂3}." [paper, p.16]
- LinearTime baseline: "exactly d²(M − 1) field multiplications and Θ(d²·M) field
  additions/subtractions" for a product of d multilinears over M = 2^ℓ points, i.e. d²·2^{ℓ−i}
  in round i (d(d−1) for evaluations, d for binding). [paper, Lemma 9, p.35; p.34]
- High-degree product (Procedure 1): bb count per hypercube point satisfies
  a(d) = a(⌊d/2⌋) + a(⌈d/2⌉) + (d + 1), closed form a(d) = d⌈log₂d⌉ + 2d − 2^{⌈log₂d⌉} − 1, versus
  (d − 1)(d + 1) naive. "At d = 4, the number of bb multiplications is reduced from 12M to 10M. By
  d = 32, the number of bb multiplications is reduced from 992M to 190M." Lower-order cost is
  Θ(d²) sb/additions per point (v = 1), Θ(d^{v+1}) for a v-round window. [paper, p.12, Lemma 4
  p.13, Lemma 15 p.49]
- Table 2 ratios (naive/optimized bb count, v = 1): d=3 8/7, d=4 15/11, d=5 24/16, d=8 63/31,
  d=9 80/37, d=16 255/79, d=32 1023/191. [paper, Table 2, p.14]
- Small-value prover (Lemma 5 / Lemma 12): at the optimal early window v* = log_{d+1}(d²κ), the
  bb-equivalent runtime is Θ(M·d^{2−2/δ}·κ^{−1/δ}) with δ = log₂(d+1), i.e. speedup
  Θ((d²κ)^{1/δ}) over LinearTime, with space Θ(d²κ). The window grid costs Θ(M·((d+1)/2)^v) ss
  multiplications; "every delayed-reduction accumulator must stay within the limb bounds."
  [paper, Lemma 5 p.16, p.15, Lemma 12 p.41]
- "In the first round, it is possible to perform ss and sb multiplications ... However, from the
  second round onwards, the prover exclusively performs bb multiplications." Round batching
  delays binding so the first v rounds are answered from a (d+1)^v grid computed by Procedure 1
  from small values only. [paper, p.14]
- Eq factor (Section 6, Lemma 7 / Lemma 14): appending `eq(w, X)` to a product of d multilinears
  adds "O(2^{ℓ−i−v+1}) field operations for suffix weights" and "(d+1)^v small-by-big
  multiplications per window", while "the dominant multiproduct evaluation cost remains Θ((d+1)^v)
  rather than increasing to Θ((d+2)^v)." The eq factor is split as a past-rounds constant α, a
  product of v linear factors in the window, and suffix weights over future rounds. The authors
  note this is distinct from and complementary to Gruen's eq optimization. [paper, p.17–18, p.44]
- Delayed reduction measured in a degree-2 sumcheck `Σ p·q` on BN254: 1.31–1.41x for 14–26
  variables; with a split-eq factor `Σ eq·p·q`: 1.06–1.11x. [paper, Tables 6–7, p.22–23]
- Spartan outer (eq·(A·B − C), small values) ablation on BN254, steady state: baseline → split-eq
  1.26–1.47x; → delayed reduction 1.09–1.31x; → round batching 4.10–6.01x; → univariate skip
  1.11–1.24x. Memory-bound regime: round batching 8.81–14.95x. Cumulative at 8192 SHA-2
  iterations: 57.9x (univariate skip), 45.9x (round batching), with peak RSS 46.6–50.1 GB for
  dense variants versus 11.5–15.0 GB. [paper, p.23–24, Tables 8–9]
- "All experiments are run over the BN254 scalar field (the field currently used in Jolt)."
  [paper, p.20] The only statement about 128-bit fields: "our techniques remain applicable to
  future versions of Jolt and other zkVMs that avoid elliptic curves and hence use 128-bit fields."
  [paper, p.25] No numbers are given.

### Inferences

- **κ for Fp128.** The paper's model gives κ ≈ 2·2² + 2 = 10 for two limbs; using the paper's ss
  time (0.36 ns, M4) against our bb (4–5 ns, M3) gives κ ≈ 11–14. Our "about 9" is the right order
  and slightly low. [mine] The more consequential difference is the bb-to-addition ratio: 10.50/1.99
  ≈ 5.3 on BN254 versus 4.5/1.6 ≈ 2.8 on Fp128, so every technique that trades bb multiplications
  for additions (the high-degree product, the small-value grid) pays relatively twice as much for
  its additions on our field. [mine]
- **Small-value gain on Fp128.** By Lemma 5, speedup ∝ (d²κ)^{1/δ}. For d = 2 (δ = 1.585):
  (4·29)^{0.631} ≈ 20 at BN254's κ against (4·12)^{0.631} ≈ 11.5 at ours, a ratio of 1.75. Scaling
  the measured BN254 round-batching gain (4.1–6.0x) by that ratio gives roughly 2.3–3.4x; the
  dearer additions pull it down, so "roughly 2–3x on affected rounds" is my estimate. The
  asymptotic hides constants, so this is a scaling argument, not a prediction. [mine]
- **High-degree product gain on Fp128.** The bb saving is exact (63 → 31 bb per point at d = 8,
  255 → 79 at d = 16), but the Θ(d²) additions per point cost 0.36 bb each on Fp128 against
  0.19 bb on BN254. A crude count for d = 8 gives ~1.2x for us where BN254 measures 1.7–1.8x; for
  d = 16 perhaps 1.3–1.6x. [mine]
- **Delayed reduction on Fp128.** On BN254 the reduction is 64% of a bb multiply (6.67/10.50 ns),
  which is what delayed reduction removes. Our Fp128 is a Solinas prime, not Montgomery form; the
  paper's sb routine (bignum × u64 then one Barrett pass) does not transfer literally, but the
  principle (accumulate the 256-bit products, reduce once) does. The share of our 4–5 ns that is
  reduction is unmeasured; if it is similar, the paper's 1.3–1.4x on clean degree-2 sumchecks is a
  fair expectation. [mine]

### Gaps

- No measurement on any 128-bit prime field; all tables are BN254 on an M4 Max.
- The paper does not model additions in Lemma 5 and does not report the add-to-bb ratio's effect.
- The paper's sb routine is Montgomery-specific; nothing is said about Solinas or non-Montgomery
  representations.

---

## Q2. Where each technique applies to our relations; the paper's numbers for Shout and Spartan

### Takeaway

Small-value and univariate skip apply only where the sumcheck's tables are the raw small witness:
our `eq·small·small` relations, Booleanity on 0/1 values, and the first rounds of one-hot chunk
products; neither applies to the folded degree-2 matmul. The high-degree product algorithm applies
to the degree-9/17 ra-virtualization products. Delayed reduction and the small-by-big fold apply
to the matmul, including after folding. The paper measures 1.70–2.18x on Jolt's RA virtualization
(degrees 5–33) and 4.1–6.0x from round batching on Spartan, all on BN254.

### Cited Findings

- Shout RA virtualization in Jolt: the address of κ = 128 bits is split into N blocks of d chunks
  of b = 4 bits, d = (κ/N)/b, sumcheck degree d + 1: "N = 8: d = 4, deg = 5; N = 4: d = 8, deg = 9;
  N = 2: d = 16, deg = 17; N = 1: d = 32, deg = 33." The identity proved is
  `Σ_i γ^i α_i = Σ_j eq(r_cyc, j) · Σ_i γ^i Π_t ra_{i,t}(r_{i,t}, j)`. [paper, Appendix E.2,
  p.50] — [2026/587](https://eprint.iacr.org/2026/587)
- Table 12 (scale 22, multi-threaded, BN254): degree 5: 1,602 → 921 ms (1.74x); degree 9: 2,135 →
  1,247 ms (1.71x); degree 17: 3,837 → 2,254 ms (1.70x); degree 33: 6,501 → 2,979 ms (2.18x). RA
  share of total prove time fell from 13.0/17.5/26.0/39.7% to 8.4/11.2/17.1/22.8%. [paper,
  Table 12, p.26]
- "The RA prover can evaluate this product either with a naive O(d²) interpolation/extrapolation
  path or with the optimized high-degree product kernel ... This is exactly the comparison
  reported in Table 12." [paper, p.50]
- Isolated kernel (Table 5, single thread, product of d multilinears in v variables): d = 8:
  1.27–2.01x; d = 16: 1.33–2.04x; d = 32: 1.48–2.48x, for v = 12–20. "Speedups grow with d and v,
  reaching about 2.5x at (d, v) = (32, 18) and (32, 20)." [paper, Table 5 p.22, p.20]
- End-to-end Jolt (SHA2-chain, BN254): prove time 9.76 s (scale 22) to 51.09 s (scale 25); RA
  virtualization share 8.0–11.8%, Spartan outer share 3.0–4.2%, after optimization. [paper,
  Table 13, p.26]
- Spartan: see Q1 (4.10–6.01x from round batching; 1.11–1.24x more from univariate skip). The
  Spartan outer polynomial is `eq(w, X)·(Ā(X)·B̄(X) − C̄(X))`; "many coordinates are zero/one and,
  more generally, small signed integers. This is exactly the structure we exploit." [paper, E.1,
  p.49]
- Small-value applicability condition: "the small-value specialization is valid whenever the
  chosen window satisfies the bit-width preconditions of the underlying arithmetic routines:
  extrapolated values must remain within the machine-word budget used by the ss/sb kernels."
  [paper, p.15]
- When all but one polynomial is small (the eq factor): "we must convert d·M/2 multiplications in
  the computation of s_1(u) from ss ... to sb multiplications." [paper, p.35]
- Linear combination with small coefficients (the fold primitive), n terms: baseline bb 55.10 ns
  (n = 2) to 873.31 ns (n = 32); per-term sb 11.49 to 167.60 ns; one-reduction 5.26 to 36.84 ns;
  speedup over baseline 10.48–23.71x. [paper, Table 4, p.21]

### Inferences

- **Degree-2 matmul over folded tables.** After the fold, L and R are full field elements, so ss
  and small-value do not apply and the paper's product algorithm offers nothing at d = 2 (a(2) =
  3 = naive). What does apply: (i) delayed reduction inside the round loop (paper measures 1.31–
  1.41x at d = 2 on BN254, Table 6); (ii) the fold itself is the paper's Section 3.3 primitive,
  `Σ_i eq(r, i)·A[i, k]` with 32-bit integer `A` and field `eq`, i.e. sb products with one reduction
  per output, which the paper measures at 2.2–4.5x over per-term sb and 10–24x over naive bb
  (Table 4). Our roadmap A1/A2 already plan both. The doc's "none apply" is wrong only in that
  delayed reduction and the sb fold are this paper's techniques. [mine]
- **eq·small·small (Mul, Square, Cube, softmax products).** This is the Spartan shape. Both
  round-batched small-value proving and univariate skip apply directly; with the eq factor
  handled by Section 6 the product has d = 2 small factors. Expected gain roughly 2–3x on those
  rounds (Q1). These relations are a small share of our 5.3 s IOP; the absolute saving is bounded
  by that share. [mine]
- **Degree-3 Booleanity `eq·(ra² − ra)`.** Also Spartan-shaped with 0/1 values; small-value and
  univariate skip apply. Upstream's pattern kernels (#1923, #2257 in our roadmap) exploit the 0/1
  structure more directly; the paper does not compare against them. [mine]
- **Degree-9 and degree-17 chunk products.** The paper's Table 12 rows for degree 9 and 17 are our
  case (b = 4, d = 8 or 16). The exact bb saving per point is 63 → 31 and 255 → 79; realized gain
  on Fp128 likely 1.2–1.6x (Q1). The first round of these products also has all-0/1 inputs, so the
  first-round message is ss-only; later rounds are bb. [mine]
- **Batched attention (degree 3 with eq).** Because `dot.rs` folds both operands first, the eq
  factor multiplies two full-field tables; Section 6's split-eq applies (A1 already plans it) and
  the paper's Table 7 says delayed reduction adds only ~1.09x after split-eq. [mine]

### Gaps

- The paper gives no number for the high-degree kernel on a two-limb field or for the kernel's
  addition count at d = 8 and 16 (only the Θ(d²) bound).
- No measurement isolates the Booleanity-shaped sumcheck; Spartan outer is the closest analog.

---

## Q3. What the paper says about univariate skip for small integers in prime fields

### Takeaway

The extended version says univariate skip *does* apply to small values inside a large prime field,
gives the construction (a symmetric integer interpolation domain, quadratic interpolation made
cheap by sb arithmetic and delayed reduction), implements it in Jolt's Spartan outer sumcheck and
reports it as the fastest linear-space variant and Jolt's current default. The one cost named is a
higher-degree verifier interpolation step.

### Cited Findings

- "Univariate skip, introduced by Gruen [33], is a generic way to reduce prover work by 'fusing'
  several Boolean variables into a single higher-degree variable, in a setting where we start out
  with base field values but sumcheck is over the extension field. In this section, we show that
  univariate skip can also apply in the setting of small values inside a large prime field, which
  allows us to apply the technique to applications such as Spartan-in-Jolt." [paper, Section 7,
  p.18] — [2026/587](https://eprint.iacr.org/2026/587)
- Construction: pack m components as `f_pack(Y, ·) = Σ_j L_j(Y)·f_j(·)` over a domain of size m;
  "Instead of a multiplicative subgroup, we take a small symmetric integer domain around 0:
  D_m := {−⌊m/2⌋, ..., −⌊m/2⌋ + m − 1}." Benefits: interpolation points stay small, and symmetry
  "improves conditioning of integer-lifted intermediate expressions ... reducing delayed-reduction
  pressure." [paper, p.19]
- "With this domain, we use the straightforward O(|D_m|²) interpolation/evaluation routine. This is
  practical because |D_m| = m is chosen to be small in our target regime and the arithmetic in the
  interpolation loop is dominated by multiplying large field elements by small coefficients."
  [paper, p.19]
- Packed-width bound: with |v_i| ≤ B and Λ the maximal Lagrange-coefficient sum, need BΛ < 2^127
  with a one-extra-limb budget; for D = {−K, ..., K} and a degree-2 target window, Λ_14 ≈ 2^62.6,
  Λ_15 ≈ 2^67.3, Λ_16 ≈ 2^72.0, "So under worst-case full-width 64-bit values in this degree-2
  symmetric setting, K ≤ 14 is the conservative safe choice." [paper, p.19]
- Tradeoff: "larger m gives more prover amortization but increases univariate degree and
  interpolation cost" on the verifier side. [paper, p.20] "univariate skip instead trades a faster
  prover for a higher-degree verifier interpolation step." [paper, p.25]
- In Jolt: of 19 uniform constraints, 10 are proved "via one univariate-skip round over a size-10
  domain" with first-message degree ≤ 3d = 27 (d = 9), the rest in ordinary rounds; the split
  places in C_small "the constraints whose arithmetic profile is most 'small-value friendly'".
  [paper, E.1, p.49–50]
- Measured: round batching → univariate skip 1.11–1.24x (steady state), 1.23–1.26x (memory-bound);
  "yielding our fastest linear-space Spartan outer variant (and is the current default in Jolt)."
  [paper, p.22–23]
- Related-work framing: Gruen's "univariate-skip technique itself ... targets the same small-value
  regime as ours but modifies the protocol and increases the degree of the first-round prover
  polynomial; we revisit this tradeoff in Section 7." [paper, p.5]

### Inferences

- Our values are i32 (B = 2^31), so the BΛ < 2^127 budget allows Λ < 2^96 and hence a wider domain
  than K = 14 before needing a second extra limb; the exact Λ must be recomputed for the chosen
  window as the paper says. [mine]
- The natural place for us is the first round of a deferred batch of identical element-wise
  relation instances (many nodes, same shape): pack the instance index, as Jolt packs the
  constraint index. It does not apply to the folded matmul. The verifier-side cost is one
  degree-≤3d interpolation per batched instance, which matters for our verifier and for recursion
  through upstream's Akita guest. [mine]
- The earlier version (2025/1117) is not in `papers/`, so whether it "argues against" univariate
  skip cannot be checked here; the extended version's Section 7 supersedes it either way. [mine]

### Gaps

- 2025/1117 not read; the exact wording the roadmap relied on is unverified.

---

## Q4. Implementation: where, and whether forward differences / balanced trees are the recommendation

### Takeaway

The implementation is a Jolt branch (`quangvdao/jolt`, `zkproof-submission`) extending arkworks
for delayed reduction, with Spartan-outer variants and the high-degree RA kernel. The paper's
recommended product algorithm is a balanced divide-and-conquer (split the d factors in half,
recursively) with extrapolation by a shifted-evaluation recurrence whose integer coefficients are
binomial, i.e. forward differences in substance, but it uses neither phrase; it explicitly drops
the earlier version's Toom-Cook index mappings.

### Cited Findings

- "See https://github.com/quangvdao/jolt/tree/zkproof-submission." [paper, footnote 5, p.4] —
  [2026/587](https://eprint.iacr.org/2026/587)
- "For field arithmetic, we extended the arkworks [17] library to support the delayed reduction
  technique in Section 3. Our Jolt integration ... delayed reduction across sum-check-heavy
  components ... Spartan outer sum-check prover ... split-eq, delayed reduction, round-batching
  and univariate skip ... streaming variants ... high-degree optimization to the instruction RA
  virtualization sum-check prover ... effective degree is configurable in the range {4, 8, 16, 32}."
  [paper, Section 8, p.20]
- Procedure 1 (MultiProductEval): "m ← ⌊d/2⌋; q_L ← MultiProductEval(p_1..p_m); q_R ←
  MultiProductEval(p_{m+1}..p_d); extrapolate both to U_d; pointwise product." [paper, p.12]
- On Hyperplonk's FFT-based divide-and-conquer: "there are large constants hidden in the O-notation
  which make this algorithm significantly slower than the naive approach for common values of d."
  [paper, p.11]
- Extrapolation: the naive Vandermonde route needs (k+1) sb per extrapolated value, "concretely
  worse than the naive approach unless sb operations were cheaper than bb by a factor of at least
  log d." Instead, "computing p(k + c) can be expressed as the problem of computing q_c(k) where
  q_c(k) := p(k + c)", so each new value is obtained from the previous k + 1 consecutive values by a
  fixed small-integer row such as [24, −1, 4, −6, 4] for k = 4, and "the low-level optimizations
  here require finding minimal multi-addition chains for the small integer coefficients. We
  provide explicit, optimized procedures for this computation in our implementation." [paper,
  Appendix D.1, p.46–47]
- Earlier version's Toom-Cook machinery abandoned: see the p.15 quote in Q1.
- The Toom-Cook reference is kept only for the streaming prover: "allowing speedups using
  techniques similar to the Toom-Cook multiplication algorithm" [paper, p.3]; the one-way extension
  to Jolt's degree-{5,9,17,33} kernel is the Procedure 1 path.

### Inferences

- The row [−1, 4, −6, 4] is the k-th forward-difference extrapolation of a degree-4 polynomial on
  consecutive integers (the alternating binomial coefficients), with the leading-coefficient term
  (24 = 4!·p(∞)) supplied separately because U_d includes the point at infinity. So "forward
  differences and a balanced tree" is a fair description of Appendix D plus Procedure 1, although
  the paper's words are "shifted evaluations", "minimal multi-addition chains" and "divide the list
  of d polynomials in half". [mine]
- Whether a16z/jolt's mainline (as opposed to the `zkproof-submission` branch) carries exactly this
  code under PR #1966 is not stated in the paper; the roadmap's attribution to #1966 comes from
  upstream PR descriptions, not from here. [mine]

### Gaps

- The paper names no a16z/jolt PR number or crate; only the fork branch.

---

## Q5. BinarySpartan (2026/1656): protocol, cost per constraint, packing, optimizations used, benchmarks, transferability

### Takeaway

BinarySpartan is a 13-page systems paper: Spartan over F_{2^128} with Ligerito plus Diamond–Posen
ring-switching (128 witness bits per field element), using only pre-existing optimizations (SIMD
R1CS, SuperSpartan's `next` extension, Gruen's univariate skip and eq factorization, Dao–Thaler eq
tables, Bagad et al. round computation, Binius64 byte tables). It gives hash throughput numbers
and constraint counts but no per-constraint cost model and no protocol-level detail beyond the
list. Everything transferable is already known from Spartan-over-prime-field practice; the
characteristic-2-only parts are the bit packing and the byte-lookup tables.

### Cited Findings

- "We instantiate Spartan over a binary field, using Ligerito as the commitment scheme along with
  the ring-switching technique of Diamond and Posen." Transparent, polylog proofs, hash-based.
  [paper, abstract, p.1] — [2026/1656](https://eprint.iacr.org/2026/1656)
- Optimizations applied (verbatim list): "the SIMD R1CS of Phalanx [41] to efficiently prove many
  R1CS instances defined over the same R1CS matrices; the `next` multilinear extension of
  SuperSpartan [39] to check input and output consistency across neighboring SIMD instances;
  Gruen's univariate skip and factorization of the equality polynomial eq [14]; optimizations to
  the sum-check round work: the sublinear-sized decomposed eq tables of Dao and Thaler [10] and
  the round-computation improvements of Bagad, Dao, Domb, and Thaler [2]; and position-specific
  byte lookup tables ... following Binius64: for the bit-valued linear map, at each byte position
  we precompute the 256 subset sums of its basis images and XOR one selected entry per byte. All
  but the last are not specific to binary fields." [paper, p.3]
- Why Spartan works over F_2 (answering Flock and Binius64's objections): "Gruen's univariate skip
  binds the first rounds as a single univariate message whose construction weights eq by bits, a
  selection rather than a multiplication, so extension arithmetic begins only on a domain smaller
  than the skip factor; and the sublinear-sized eq tables avoid materializing the linear-sized
  one." Inner sumcheck: "Phalanx's SIMD R1CS shares one set of matrices across every instance in a
  batch, so the entry count scales with the block and not the batch. Sparse polynomial evaluation
  proofs do not arise at all: the shared matrices are small enough for the verifier to evaluate
  directly, so BinarySpartan never invokes Spark." [paper, Section 2, p.4]
- Implementation: "Arithmetic is in F_{2^128} modulo x^128 + x^7 + x^2 + x + 1. The witness is a
  vector of bits: ring-switching packs 2^7 = 128 of them into one field element, so committing to
  128 witness bits is equivalent to committing to a single field element. We apply Gruen's
  univariate skip over the first 4 rounds of Spartan's outer sum-check." [paper, p.5]
- Security: 100 bits; Ligerito in unique-decoding regime, no grinding; "soundness error of at most
  2^{-102.6} at the sizes evaluated here, and the bound stays below 2^{-100} up to 2^28 committed
  coefficients." [paper, p.5]
- Circuits: SHA-256 30,376 constraints per compression; BLAKE3 14,000; Keccak-f 38,400 per
  permutation (three packed per instance). [paper, p.6]
- Throughput (M4 Max, 12 performance cores, witness generation included): BLAKE3 871,796/s,
  SHA-256 400,925/s, Keccak-f 287,464/s; versus Flock 802,899 / 400,928 / 288,315; Binius64 55,488 /
  45,474 / 45,786; Plonky3 53,634 / — / 7,604; Hashcaster — / — / 72,535. [paper, Table 1, p.7]
- Proof size 519 KiB at 2^14 BLAKE3 hashes (Flock 357 KiB); verifier 2.7–4.1 ms. Peak memory 3.92 GB
  (BLAKE3), 1.25 GB (Keccak), 7.56 GB (SHA-256). [paper, p.8]
- Single SHA-256 latency: 3.47 ms (128 B) to 6.24 ms (2 KiB), "4.2–7.0x faster than Flock, at
  least 6.4x faster than either Vega configuration, and 9.9–14.2x faster than Binius64."
  [paper, Table 2, p.10]
- On SIMD R1CS and `next`: antecedents in Ben-Sasson et al. and Holmgren–Rothblum, "where it links
  adjacent steps of a computation transcript." [paper, p.5]

### Inferences

- **Cost per constraint** is not given. Dividing the BLAKE3 throughput by its constraint count
  gives about 1.2·10^10 constraints/s on 12 cores including witness generation
  (871,796 × 14,000), roughly 1 ns per bit-constraint per core; SHA-256 gives the same order
  (400,925 × 30,376 ≈ 1.2·10^10). This is my arithmetic on Table 1 and p.6, not a figure in the
  paper. [mine]
- **Transferable to a prime-field prover:** (i) the SIMD-R1CS stance that a relation shared by many
  instances has its "matrices" evaluated by the verifier directly, so no sparse-polynomial
  argument is needed, which is already how our fixed per-op relations work; (ii) univariate skip
  over the first rounds (also in 2026/587 Section 7 for prime fields); (iii) Dao–Thaler decomposed
  eq tables and Bagad et al. round computation (the latter is 2026/587). (iv) `next`-style chaining
  of instance i's output to instance i+1's input is the analog of our producer-to-consumer opening
  ties, already virtual. [mine]
- **Characteristic-2 only:** the 128-bits-per-element ring-switching packing and the XOR byte
  tables (the paper says so for the latter). The packing is why a committed bit costs 1/128 of an
  element and needs no Booleanity check, which is the Magnetar point our research-directions.md
  already makes. [mine]
- The paper says nothing about keeping witness segments virtual; the witness is one committed bit
  vector. The brief's "the way witness segments are kept virtual" has no source in this paper.
  [mine]

### Gaps

- No per-constraint prover cost, no stage breakdown (commit vs sumcheck vs witness), no protocol
  description beyond the optimization list; the Phalanx SIMD inner-sumcheck mechanics are cited
  ([41]), not described.
- Nothing on zkML, lookups, or integer arithmetic.

---

## Q6. Jagged PCS (2025/917): construction, adapter cost, soundness, versus aligned subcubes, PCS requirements, fit over Akita

### Takeaway

Jagged PCS stacks columns of arbitrary heights into one dense multilinear by a height-indexed
bijection, sends the cumulative heights in the clear, and reduces an evaluation claim on the
stacked (sparse) polynomial to one claim on the dense one by a product-of-two-multilinears sumcheck
whose prover cost is at most 5·2^m + 2^n + 2^k field multiplications for 2^m total cells; the
verifier is an arithmetic circuit of size O(m·2^k) (O(m·#tables) with "fancy jagged", or one
structured evaluation plus O(m·k) with the "jagged assist"); soundness error 2m/|F|. It needs any
multilinear PCS and no homomorphism, so it fits over Akita. Against an aligned-subcube layout it
buys arbitrary heights and a height-independent verifier at the price of the adapter sumcheck;
with heights already powers of two, the aligned layout needs no adapter.

### Cited Findings

- "This scheme allows a prover and verifier to 'bootstrap' any existing multilinear PCS into an
  extremely efficient sparse multilinear PCS for a particular type of sparsity." [paper, p.3] —
  [2025/917](https://eprint.iacr.org/2025/917)
- Definition 1.1: a jagged function on a 2^n × 2^k table with heights h_y is zero at (x, y) for
  x ≥ h_y. m is the smallest integer with Σ h_y ≤ 2^m − 1. Footnote 4: "By treating the non-zero
  part as roughly a power of two, we may be introducing a 2x overhead (in the worst-case). This
  overhead can be avoided if the commitment scheme ... supports 'free padding' ... Schemes based on
  interleaving such as Ligero, Brakedown and Blaze can easily be made to support such free
  padding." [paper, p.5]
- Commitment: cumulative heights t_y = h_0 + ... + h_y (m bits each) define a bijection
  i ↦ (row_t(i), col_t(i)) between {0,1}^m and the non-zero part; the dense q(i) := p(row_t(i),
  col_t(i)). "The commitment to p̂ is the multilinear oracle q̂ and the cumulative heights
  (t_y)_y which are sent in the clear." [paper, Fact 1.3 and following, p.6]
- Theorem 1.4 (Basic Jagged): m-round protocol; completeness; "Soundness: ... with probability at
  least 1 − 2m/|F| the verifier either rejects or outputs a point z' and value v' such that
  q̂(z') ≠ v'"; "Efficiency: The prover performs O(2^m) field operations, out of which at most
  5·2^m + 2^n + 2^k are multiplications, and the verifier is implemented as an arithmetic circuit
  of size O(m·2^k)." [paper, p.6–7]
- Mechanism (Section 3): p̂(z_r, z_c) = Σ_i q̂(i)·f̂_t(z_r, z_c, i) with f̂_t(z_r, z_c, i) =
  eq(row_t(i), z_r)·eq(col_t(i), z_c) on Boolean i (Eq. 3–4); a sumcheck for a product of two
  multilinears (Lemma 2.3: prover 4·2^m multiplications) reduces this to one claim on q̂ and one on
  f̂_t; generating the f̂_t table costs 2^m + 2^n + 2^k multiplications. [paper, p.12–13]
- Verifier's f̂_t evaluation: Claim 3.2.1, f̂_t(z_r, z_c, i) = Σ_{y∈{0,1}^k} eq(z_c, y)·ĝ(z_r, i,
  t_{y−1}, t_y) with g(a, b, c, d) = 1 iff b < d and b = a + c; g is "computable by a width-4
  read-once branching program" (Claim 3.2.2), so by Lemma 4.2 (Holmgren–Rothblum) ĝ costs
  n·(2w² + 2^b) multiplications; total O(m·2^k) (Proposition 3.2). [paper, p.13–15]
- Jagged assist (Theorem 1.5 / Lemma 5.1): K evaluations of a multilinear ĥ are batched into one
  with soundness (2m+1)/|F|; "The prover needs to evaluate h on O(m·K) points and additionally does
  O(m·K) arithmetic operations. The verifier needs to do only a single evaluation of ĥ and in
  addition does O(m·K) arithmetic operations"; for a width-w ROBP the prover is
  O(m·w²·(K + w)). [paper, p.8–9, p.19–20]
- Fancy jagged (Section 6): columns grouped into tables of power-of-two width 2^{c_y}; "verifier
  work is merely proportional to the number of tables"; branching program width 6, "(6/4)² = 2.25
  larger" cost; non-power-of-two widths decomposed into power-of-two tables, verifier cost
  Σ_T log₂(#col(T)). [paper, p.9, p.20–21]
- Committing to heights: "prepending the height data to the dense representation" and delegating
  the verifier's height computation via GKR, "reduces the verifier's cost to poly(k, m), while only
  increasing the prover cost by an additional O(2^k·m) multiplications." [paper, p.9]
- Versus general sparse PCS: "In Spark and Surge the prover needs to commit to at least 7
  polynomials of size M ... In contrast, in the jagged PCS, the prover does not commit to any
  additional polynomials." Spark++ costs "(d² + 3d + 2)·M multiplications". [paper, p.10]
- Prior packing with fixed heights: "[RRR21, Proposition 3.8] and [Irr24], but focusing on the
  case that the column heights are fixed in advance and not dynamic ... the columns are assumed to
  be of height that is a power-of-two, which requires substantial overhead in padding." [paper,
  p.9–10]
- Remark 1.2: small-base-field packing and sumcheck optimizations "are compatible (and compounding)
  with the jagged PCS". [paper, p.6]
- Concrete (SP1 Hypercube): trace with w = 2^12 columns, up to 2^21 rows per table, area 2^29;
  batch-FRI verifier touches q·w = 2^19 elements, post-jagged t·q = 2^15, "a 16x improvement over
  the cost of batch FRI verification". Prover: "this cost is dominated by 5 finite multiplications
  per field element in the trace, which is very small compared to other costs (such as Reed-Solomon
  encoding, Merkle hashing, and the zero check and lookup protocols)." "With jagged we can support
  arbitrary column heights." [paper, Section 7, p.22–23]
- The verifier "is implemented as an arithmetic circuit ... all information about the column
  heights is processed purely via arithmetic over the field (rather than any conditional branching
  logic). This fact is important in the context of recursive proof composition." [paper, p.7]

### Inferences

- **Versus MosaicPCS / aligned subcubes.** Placing a 2^{m_j}-size polynomial at an offset that is a
  multiple of 2^{m_j} in a 2^m polynomial Q makes P_j(z) = Q(z, prefix_j) with the high variables
  fixed to Boolean bits, so a claim lifts with no sumcheck and the verifier's extra work is zero;
  this is the fixed-heights packing the paper attributes to [RRR21, Irr24] and says needs
  power-of-two padding. Jagged removes the padding and the height-dependent verifier at the cost
  of the adapter (≤5 multiplications per dense cell, m rounds, m challenges, one extra opening
  claim on f̂_t evaluated by the verifier). For a system that already pads every dimension to a
  power of two at compile time, as ours does, the adapter buys nothing on the prover side. [mine]
- **PCS requirements.** Only a multilinear PCS for q̂; no homomorphism, no extra oracles. The
  adapter's output is one point on q̂, which is exactly what a non-homomorphic scheme like Akita
  wants, since it cannot verify an RLC of commitments and today opens every class natively
  (`akita.rs` header). [mine]
- **Fit over Akita.** Two obstacles, neither in the paper: (i) Akita's one-hot flavor commits
  sparse columns at cost proportional to nonzeros; a generic dense q̂ of all one-hot groups would be
  93.75% zeros (K = 16), so the stacked object must stay a one-hot-shaped trace group (heights
  summed, width 16), which is exactly fancy jagged's "tables of power-of-two width" and also what
  an aligned layout gives; (ii) Akita setups are exact-shape per arity (`akita.rs` header), so a
  stacked object needs a catalog entry at the summed arity (the one-hot catalog goes to 2^35, the
  dense to 2^34). [mine]
- **Soundness over Fp128.** 2m/|F| with m ≈ 30 is about 2^{−122}; the jagged assist adds 1/|F|.
  Not a constraint. [mine]

### Gaps

- The PDF does not mention a venue; the research-directions.md attribution "EUROCRYPT 2026" is
  unverified from the paper.
- No prover wall-clock for the adapter is reported, only the 5-multiplications-per-cell bound and
  the statement that it is small relative to encoding and hashing.
- Whether Akita's commitment cost for one stacked object equals the sum of per-class costs is an
  Akita question the paper cannot answer.

---

## Q7. What Jagged or an aligned layout would save on our 6.7 s opening stage

### Takeaway

The 6.72 s is "opening reduction sumcheck + Akita class openings" in one span (README), and the
split is not published, so no paper expression can be applied without that split. Structurally:
the opening-reduction sumcheck's prover work is Σ over opened polynomials of their size times a
small constant and does not shrink by relayout; what a single stacked object removes is the
per-class Akita opening, whose count is the number of `(kind, arity)` classes with an opened member.
Jagged's adapter would add work of its own (≤5 multiplications per dense cell); an aligned layout
adds none. All of this section is mine except where a paper is cited.

### Cited Findings

- Jagged adapter prover cost: ≤ 5·2^m + 2^n + 2^k multiplications for 2^m dense cells (Theorem 1.4,
  p.7); batch-evaluation of K points (Lemma 5.1, p.19) is the RLC-of-eq sumcheck with prover
  4·2^m multiplications plus O(m·K). [paper] — [2025/917](https://eprint.iacr.org/2025/917)
- Our stage numbers: witness + commitment 25.2 s, IOP 5.28 s, "Opening reduction sumcheck + Akita
  class openings (`ONNXProof::prove_reduced_openings`) 6.72 s", total 40.6 s traced, 39.9 s
  untraced; proof 2.42 MiB. [README.md, GPT-2 table]
- `docs/upstream-inventory.md` line 156–158 (quoted in `reports/Op proving efficiency survey.md`):
  the commitment phase "dominates (24.4 s), and two of the six 2^24-variable one-hot groups take
  about 10 s each of it."
- Akita classes and native per-class opening: `atlas-pcs/src/akita.rs` header (quoted at the top).

### Inferences

- **What relayout cannot save.** The reduction of many (polynomial, point) claims to one joint
  claim is a batch-evaluation sumcheck; its prover work is, per Lemma 5.1's shape, proportional to
  the total size of the opened polynomials, whether they are separate objects or subcubes of one.
  So the sumcheck part of the 6.72 s stays. Only the per-class Akita openings and their
  transcript/serialization are in play. [mine]
- **Jagged adapter cost for GPT-2.** Taking six one-hot groups of 2^24 cells plus the rest as
  ~1.5·10^8 dense cells, 5 multiplications per cell at 4.5 ns is about 3.4 s single-threaded,
  roughly 0.4–0.6 s on 8 performance cores, unless the product sumcheck exploits one-hot sparsity
  (not in the paper). That is a real cost against an unknown saving. [mine]
- **Aligned-subcube extension.** `akita.rs` already prefix-packs within a class (one arity). The
  across-arity version places smaller classes at aligned offsets of a larger object; the joint
  point gains fixed prefix bits per member (today the code handles different sizes by "index-0
  overlap of smaller polynomials", i.e. all-zero prefixes, so the generalization is in the point
  bookkeeping, not in new cryptography). Adapter cost zero; what is new is one Akita object of
  summed arity per kind instead of one per `(kind, arity)`. [mine]
- **Bound on the saving.** At most 6.72 s minus the reduction sumcheck time minus one or two
  openings' time. If the per-class openings are, say, half the stage and the class count is in the
  tens, the saving is on the order of 2–3 s; if the sumcheck dominates, under 1 s. I cannot narrow
  this without the span split. [mine]
- **Verifier and recursion.** The jagged verifier is an O(m·#tables) arithmetic circuit independent
  of heights (p.7, p.21); the aligned layout's verifier is simpler still (fixed bits). Either
  reduces the verifier's per-class Akita verifications, which matters for the 2.2 s verify time and
  for a future recursion through upstream's Akita guest. [mine]

### Gaps

- The split of 6.72 s into reduction sumcheck versus Akita openings, and the number of classes
  opened on GPT-2, are unmeasured. This is the one measurement that decides the item.
- Akita's cost for one object of summed arity versus the sum of per-class costs is unknown.

---

## What this changes in the docs

Each entry quotes the doc, then gives the correction and the page it rests on.

### `docs/optimization-roadmap.md`

1. **Section 7 intro:** "These apply to the generic relation engine and the lookup subprotocols.
   They do not apply to the einsum contraction, whose tables are full field elements after the
   fold and whose degree is 2." — **Weakened.** Small-value, univariate skip and the high-degree
   product indeed do not apply to the folded contraction (a(2) = 3 = naive, p.12; small-value needs
   small evaluations, p.14). But two of the paper's techniques do: delayed reduction inside a
   degree-2 round loop (1.31–1.41x on BN254, Table 6 p.22; 1.06–1.11x after split-eq, Table 7
   p.23) and the small-by-big fold with one reduction per output (Table 4 p.21: 2.2–4.5x over
   per-term sb, 10–24x over naive). These are A1's "accumulate unreduced" and A2's
   `mul_u64_unreduced` items, so the sentence should say which techniques are excluded rather than
   all of Group E. The paper also says the degree-2 gain shrinks after split-eq, so A1's
   unreduced-accumulation sub-item should expect ~1.1x on the batched layouts, not more.

2. **Section 7:** "Balanced product trees for the high-degree ra-virtualization checks. Upstream
   replaced Toom interpolation with forward differences and a balanced tree (#1966); ePrint
   2026/587 claims 1.7–2.2x on Shout's high-degree sumcheck." — **Supported in substance,
   weakened in magnitude for us.** The paper's Procedure 1 splits the factor list in half
   recursively (p.12) and its extrapolation is a shifted-evaluation recurrence with binomial
   integer coefficients (p.46–47), which is forward differences, though the paper never uses either
   phrase; it explicitly drops the earlier version's Toom-Cook index mappings (p.15). The 1.7–2.2x
   is Table 12 (p.26): 1.74/1.71/1.70/2.18x at degrees 5/9/17/33, BN254, multi-threaded. The gain
   trades bb multiplications for additions (exact bb count 63 → 31 at degree 9, 255 → 79 at degree
   17, Table 2 p.14), and bb/addition is 5.3 on BN254 (Table 3 p.21) against about 2.8 on Fp128,
   so expect less than the paper's figure; my estimate is 1.2–1.6x.

3. **Section 7:** "Small-value sumcheck for eq · small · small relations. Lower priority than first
   thought: the gain scales with the cost ratio of a full field multiplication to a small one,
   about 30x on BN254 and about 9x on our two-limb field (our estimate), so under 2x on the
   affected rounds." — **κ confirmed, conclusion overturned.** The paper measures 29x on BN254
   (10.50/0.36 ns, Table 3 p.21) and models κ ≈ 2N²+N (footnote 11 p.16), i.e. 10 for two limbs;
   our 9 is fine. But the gain does not scale linearly with κ: Lemma 5/12 (p.16, p.41) gives
   Θ((d²κ)^{1/δ}), δ = log₂(d+1), so for d = 2 the κ = 10–12 figure is about 0.57 of the κ = 29
   figure, and the measured BN254 round-batching gain of 4.10–6.01x (p.23) scales to roughly
   2.3–3.4x. Our dearer additions pull that down; "roughly 2–3x on the affected rounds, unmeasured"
   is the defensible statement. The absolute saving is still bounded by those relations' share of
   the 5.3 s IOP.

4. **Section 7:** "Univariate skip. Parked. ePrint 2025/1117 itself argues against it for small
   integers in prime fields." — **Contradicted by the extended version.** 2026/587 Section 7
   (p.18–20) is titled "Univariate Skip over Large Prime Fields" and states "we show that
   univariate skip can also apply in the setting of small values inside a large prime field"; it
   gives the symmetric integer domain, the O(m²) interpolation made cheap by sb arithmetic, and the
   K ≤ 14 width bound for 64-bit values. In Jolt it is "our fastest linear-space Spartan outer
   variant (and is the current default in Jolt)" (p.22), 1.11–1.24x over round batching (p.23). The
   cost is a higher-degree verifier interpolation (first message degree ≤ 3d = 27 for 10 packed
   constraints, p.49). Replace "Parked" with "applies to the first rounds of deferred element-wise
   batches and Booleanity; not to the folded contraction; verifier cost to be weighed".

5. **Section 9:** "Small-value sumcheck, univariate skip or Toom-Cook inside the einsum
   contraction." — **Supported.** Folded tables are full field elements (`dot.rs`), and the paper
   itself moved off Toom-Cook for the product kernel (p.15).

6. **Section 3, A2:** "Accumulate `field × i64` products unreduced (`mul_u64_unreduced`, two
   accumulators for sign) and reduce once per output coefficient." — **Strengthened.** This is the
   paper's Section 3.3 primitive including the two-accumulator sign handling (p.10), with Table 4's
   numbers (p.21). The paper's Barrett pass is Montgomery-specific; our Solinas reduction differs,
   so the 2.2–4.5x over per-term sb is a BN254 figure.

7. **Section 7:** "Four-factor virtual selectors ... (a16z/jolt #2188)." — **Consistent.** The
   paper's E.2 (p.50) states N·d = κ/b is fixed, so fewer factors per selector means more
   selectors; it reports no measurement of that trade, so the 38–44% remains an upstream PR
   figure.

### `docs/research-directions.md`

8. **Section 2:** "One commitment, one opening. ... Upstream Jolt has a one-native-group layout
   (a16z/jolt #2012), and JaggedPCS (EUROCRYPT 2026) is the published form. This targets the 6.7 s
   opening stage." — **Corrected.** JaggedPCS is not the published form of the aligned-subcube
   layout; it is a different construction that handles arbitrary heights through an adapter
   sumcheck (prover ≤ 5·2^m + 2^n + 2^k multiplications, verifier an O(m·2^k) arithmetic circuit,
   soundness 2m/|F|, Theorem 1.4 p.6–7), and it describes the fixed-height power-of-two packing as
   prior work needing padding (p.9–10). The PDF gives no venue; "EUROCRYPT 2026" is unverified from
   it. With our compile-time power-of-two padding, the aligned layout (what `akita.rs` already does
   within a class) is the right target and needs no adapter; Jagged would add ~0.4–0.6 s of adapter
   work on GPT-2 by my count. The saving is bounded by the Akita per-class opening share of the
   6.72 s, which is unmeasured.

9. **Section 2, Magnetar:** "128 bits pack into one element of F₂¹²⁸, so commitment cost is per bit
   and a committed bit needs no Booleanity check." — **Strengthened.** BinarySpartan uses the same
   Diamond–Posen ring-switching: "packs 2^7 = 128 of them into one field element, so committing to
   128 witness bits is equivalent to committing to a single field element" (p.5).

10. **Section 2:** "We cannot adopt its field. Characteristic 2 cannot count ... Magnetar proves
    RV64I only, with no multiply instruction." — **Consistent.** BinarySpartan proves bit-level
    R1CS for hash functions only (p.6) and says nothing about integer arithmetic.

11. **Reading-depth table and reference list.** 2026/587, 2026/1656 and 2025/917 move to "read in
    full". The table's claim that "ePrint refused PDF downloads" no longer applies to these.

12. **Section 6, strategy B ceiling "perhaps 1.5–3x on GPT-2 (our estimate)":** — **Unchanged.**
    The sumcheck-engine items act on the 5.3 s IOP and part of the 6.7 s; nothing in these papers
    touches the 25 s commitment phase.

### `docs/optimization-roadmap.md` Section 8 (order)

13. "Bump the Jolt and Akita pins ... Upstream's commit and field kernels since the pin are the
    cheapest speedup available" — **Weakened as a source.** The paper's implementation is the fork
    branch `quangvdao/jolt/tree/zkproof-submission` (footnote 5 p.4), not necessarily what a pin
    bump brings; which parts landed on a16z/jolt mainline is an upstream question the paper does
    not answer.

---

## Ranked: what to adopt, and the one measurement that settles each

1. **Round-batched small-value prover (or univariate skip) for the first rounds of deferred
   element-wise batches and Booleanity.** Expected roughly 2–3x on those rounds by the paper's
   scaling (Lemma 5 p.16; Spartan 4.1–6.0x on BN254 p.23). Measurement: one Fp128 microbenchmark
   of `Σ eq·a·b` over 2^20 i32 values, LinearTime against a v = 2–4 early window, counting the
   add-heavy grid honestly. Univariate skip adds a verifier-degree question; measure the prover
   first.
2. **Delayed reduction and the one-reduction sb fold in the einsum engine (A1/A2 as written).**
   Supported by Table 4 and Table 6 (p.21–22). Measurement: Fp128 versions of Table 4 (n = 16
   linear combination with u64 coefficients, per-term reduce vs one reduce) and Table 6 (degree-2
   sumcheck at 2^20, reduced vs unreduced accumulators). If the reduction share of our 4–5 ns is
   under a third, this item is small.
3. **Aligned-subcube layout across `(kind, arity)` classes, not the Jagged adapter.** Zero adapter
   cost; the Jagged paper's framework (Theorem 1.4) is what to use only if we stop padding.
   Measurement: one `--trace-terminal` GPT-2 run with the `prove_reduced_openings` span split into
   the reduction sumcheck and the per-class Akita openings, plus the class count; the saving is
   bounded by the second number.
4. **High-degree product kernel for the remaining degree-9/17 ra-virtualization instances.** Exact
   bb savings (63 → 31, 255 → 79 per point, Table 2 p.14), realized gain on Fp128 uncertain because
   it buys bb with additions. Measurement: Fp128 microbenchmark of the product of 8 and 16 linear
   polynomials in 16 variables, naive (d−1)(d+1) against Procedure 1 with the Appendix D
   recurrence; adopt if above 1.3x, since Groups B and C remove most instances anyway.
5. **Jagged PCS proper.** Only if a future design drops compile-time power-of-two padding (footnote
   4 p.5: up to 2x padding saved) or wants a height-independent verifier circuit for recursion
   (p.7). Measurement: the digit census's padding fraction; if under 20%, skip.
6. **BinarySpartan.** Nothing to adopt beyond what items 1 and the SIMD-batching stance already
   cover; it is confirmation that a bit-packed binary field is a different product (hashes), not a
   path for integer matmul. No measurement.
