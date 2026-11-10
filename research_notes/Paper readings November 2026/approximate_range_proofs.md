# Approximate range proofs by random projection: soundness and cost for a jolt-atlas row check

Papers read in full (first pass) and then by targeted lemma (second pass). Page numbers are PDF
page numbers, which coincide with the printed page numbers in all four files.

- LNS = Lyubashevsky, Nguyen, Seiler, "Practical Lattice-Based Zero-Knowledge Proofs for Integer
  Relations", ePrint 2020/1448, 41 pp. — https://eprint.iacr.org/2020/1448
- GHL = Gentry, Halevi, Lyubashevsky, "Practical Non-Interactive Publicly Verifiable Secret
  Sharing with Thousands of Parties", ePrint 2021/1397, 46 pp. — https://eprint.iacr.org/2021/1397
- TTRP = Geng, Plançon, "Succinct Shortness Check Under a Few Kilobytes via Tensor Train Random
  Projections", ePrint 2026/2146, 29 pp. — https://eprint.iacr.org/2026/2146
- PikkuFold = Osadnik, "PikkuFold: Efficient Folding in a Few Kilobytes", ePrint 2026/1809, 57 pp.
  — https://eprint.iacr.org/2026/1809

Notation used throughout: v ∈ Z^k is one row of width k (the papers' s, w, x); R ∈ {−1,0,1}^{n×k}
is the projection with n rows (the papers' B, R, J, M; FlashProve n = 128); D = the distribution
0 w.p. 1/2, ±1 w.p. 1/4 each (TTRP calls it D_ghl, PikkuFold χ, LNS C with p = 1/2); T is the
verifier's per-coordinate threshold on the projection (what the committed digit width encodes);
2^w is the honest magnitude bound of entries of v. "Papers' statement" and "my derivation" are
marked in every section: everything under "Cited Findings" is the paper's, everything under
"Inferences" is mine.

## Q1. The core lemma in each paper: bounds, completeness, soundness, slack, norms

### Takeaway
The soundness side of the check is one exact, assumption-free lemma (LNS Lemma 2.5) about the
ℓ∞ norm: if some coordinate of the fixed vector v has |v_i| ≥ 2T, then each projection coordinate
is below T with probability at most 1/2, independently over the n projections, so all n are below T
with probability ≤ 2^−n — a slack factor of exactly 2 that does not depend on n or k. Everything
else (the ℓ2 Johnson–Lindenstrauss constants, the heuristic Gaussian tails) belongs to the
completeness side, which sets how large T must be for an honest prover: T ≈ t·‖v‖₂/√2 ≤
t·√(k/2)·2^w with t ≈ 13.8 for a 2^−141 per-coordinate failure, giving an overall honest-to-proven
slack of 2T/2^w ≈ 19.5·√k. A deterministic ℓ1 threshold T = k·2^w makes completeness perfect at a
slack of 2k.

### Cited Findings
- LNS defines the distribution C on {−1,0,1} with Pr[0] = p, Pr[±1] = (1−p)/2 and states
  **Lemma 2.5**: "Let s ∈ Z_q^m and y ∈ Z_q^n. Then Pr[‖Bs + y‖∞ < ½‖s‖∞ : B ← C^{n×m}] ≤
  max{p, 1−p}^n." Norms are of balanced representatives mod q; no condition on ‖s‖ relative to q
  other than q odd. — [LNS §2.7, p. 8](https://eprint.iacr.org/2020/1448); proof as Lemma A.1,
  [App. A, pp. 23–24](https://eprint.iacr.org/2020/1448).
- The proof (Lemma A.1) is a two-case conditioning argument: pick the coordinate i with |s_i| =
  ‖s‖∞, write ⟨c, s⟩ + y = s_i·c_i + r with r fixed by the other coordinates; if |r| ≥ ½‖s‖∞ the
  sum can only be small when c_i = ±1 (probability 1−p), if |r| < ½‖s‖∞ only when c_i = 0
  (probability p); the wrap-around case |r + b·s_i ± q| < |s_i|/2 is shown impossible because it
  would force q < 2|s_i| ≤ q. "This lemma can be then easily generalised to the matrix setting."
  — [LNS App. A, pp. 23–24](https://eprint.iacr.org/2020/1448).
- LNS motivation: "Baum and Lyubashevsky [BL17] showed that if Bs has small coefficients, for a
  vector s over Z_q and uniformly random binary matrix B, then with high probability s must have
  small coefficients as well. ... We extend the lemma for the case when B is sampled from a
  distribution centered at 0. The main advantage of this approach is that the infinity norm of Bs
  decreases significantly, which is essential for the rejection sampling." — [LNS §2.7, p. 8](https://eprint.iacr.org/2020/1448).
- LNS use (Appendix E, ℓ2-norm proof): with τ·d projection rows the verifier checks
  ‖z‖∞ < ½√(q/(2ℓd)) and concludes ‖v‖∞ < √(q/(2ℓd)) "with probability 1 − max{p,1−p}^{τd}";
  parameters (d, l, τ, p) = (128, 128, 1, 0.5), i.e. 128 projections at p = 1/2. — [LNS App. E
  Step 2, p. 33; Fig. 9, p. 39](https://eprint.iacr.org/2020/1448).
- GHL restates the lemma as **Lemma 3.4** ([42], Lemma 2.5): "Fix q, d ∈ Z and any two vectors
  y ∈ [±q/2]^128 and w ∈ [±q/2]^d. Let D[0] = 1/2 and D[±1] = 1/4, then choosing R ← D^{d×128}
  we have Pr_R[‖wR + y mod q‖∞ < ½‖w‖∞] < 2^−128." — [GHL §3.4, p. 15](https://eprint.iacr.org/2021/1397).
- GHL's completeness side is heuristic: **Fact 3.1** (continuous Gaussian) Pr[|⟨w, r/√2⟩| >
  9.75‖w‖] < 2^−141; Pr[‖wR/√2‖² < 30‖w‖²] < 2^−128 and Pr[‖wR/√2‖² > 337‖w‖²] < 2^−128 for
  R ∈ N^{d×256}; **Corollary 3.2 [heuristic]** transfers these to D: Pr[|⟨w,r⟩| > 9.75‖w‖] ⪅
  2^−141, Pr[‖wR‖² < 30‖w‖²] ⪅ 2^−128, Pr[‖wR‖² > 337‖w‖²] ⪅ 2^−128 (R ← D^{d×256}), "where ⪅
  denotes a heuristic bound." — [GHL §3.2, p. 14](https://eprint.iacr.org/2021/1397).
- GHL **Corollary 3.3** (modular ℓ2): for b ≤ P/45d and ‖w‖ ≥ b, Pr[‖wR mod P‖ < b√30] < 2^−128;
  proof case 1 (‖w‖∞ ≥ P/4d) is the LNS conditioning argument ("at most one of the three values
  {0,±1} yields |⟨w, r⟩ mod P| < P/8d"), case 2 has no wrap-around and uses Corollary 3.2.
  — [GHL §3.3, p. 15](https://eprint.iacr.org/2021/1397).
- GHL **Lemma 3.5**: the ℓ∞ approximate proof of smallness (λ = 128 projections, uniform masking
  vector y, restart if ‖wR‖∞ > b/2λ or ‖z‖∞ > b/2, verifier accepts if ‖z‖∞ ≤ b/2) "is an
  approximate proof-of-smallness for the l∞ norm, with size gap γ < 2500√d", where γ =
  2·9.75·λ·√d; honest ‖u‖∞ ≤ 9.75√d‖w‖∞ except with probability 2^7·2^−141 = 2^−134 (union bound
  over 128 coordinates); soundness: a prover with ‖w‖∞ > b passes with probability ≤ 2^−128 "regardless
  of y". — [GHL §3.4, pp. 15–16](https://eprint.iacr.org/2021/1397).
- GHL on the gap: "This proof system has a fairly large gap between the l∞ norm of the vector used
  by honest provers and what the prover can prove. But this gap will not show up in the rest of our
  scheme, because these proofs are only used to show that there is no wraparound modulo P." — [GHL
  §3.4, p. 15](https://eprint.iacr.org/2021/1397).
- GHL Appendix D shows the Gaussian heuristic is *false* in the extreme lower tail: for a spike
  vector (all weight on one coordinate) Pr[‖wR‖² < α‖w‖²] ≥ 2^−k for every α > 0 since the row
  is zero w.p. 2^−k, and more precisely equals Σ_{i≤⌊α⌋} C(k,i)·2^−k (eq. 19), which for k = 256 is
  ≈ 2^−126 > 2^−128. — [GHL App. D, pp. 34–37](https://eprint.iacr.org/2021/1397).
- TTRP states the dense-row ℓ∞ lemma as **Lemma 2**: "For any fixed vector x ∈ [−q/2, q/2]^m, if
  ‖x‖∞ ≥ B, for any B < q/4 then Pr_{m ← D_ghl^m}[|⟨m, x⟩ mod q| < B/2] ≤ 1/2", proof: "After
  choosing all but the i-th entry in m, at most one of the three values {−1,0,1} yields |⟨m,x⟩
  mod q| < B/2." — [TTRP §2.1, p. 8](https://eprint.iacr.org/2026/2146). TTRP describes its own
  analysis as "assumption-free bounds for TTRP with entries following the distribution D. Schemes
  that rely on JL projections with entries following D adopt the heuristic assumption that D is a
  Gaussian distribution". — [TTRP §1.1, p. 4](https://eprint.iacr.org/2026/2146).
- TTRP's statement of the standard dense JL slack: "For practical proof systems [LNP22,BS23], the
  concrete slack factor is √(128/30) ≃ 2, yielding a tight ℓ2-norm proof." and LaBRADOR's modular
  lemma: for ‖x‖ ≥ B, B ≤ q/125, Pr[‖Mx mod q‖ < √30·B] ⪅ 2^−128 with M ← D^{256×D}. — [TTRP p. 3
  and §5.1, p. 17](https://eprint.iacr.org/2026/2146).
- PikkuFold **Theorem 2** (rigorous, 2λ rows): for χ = D, every m and every non-zero w ∈ R^m,
  J ∈ {−1,0,1}^{2λ×m} with rows u_j: Pr[|⟨w,u_j⟩| > γ1‖w‖] ≤ 2^−(λ+1)/(2λ); Pr[‖Jw‖² > β‖w‖²] ≤
  2^−(λ+1); Pr[‖Jw‖² ≤ α‖w‖²] ≤ 2^−(λ+1); consequently Pr[‖Jw‖/‖w‖ ∉ [√α, √β]] ≤ 2^−λ; and for
  every q ≥ 2, 0 < θ ≤ q/b and w ∈ [±q/2]^m with ‖w‖ ≥ θ: Pr[‖Jw mod q‖ ≤ √α·θ] ≤ 2^−λ. **Table 1**
  (λ, 2λ, γ1, α, β, b): (64, 128, 6.97, 13.53, 171.8, 73); (96, 192, 8.43, 20.45, 257.5, 100);
  (128, 256, 9.66, 27.37, 343.2, 126); (192, 384, 11.75, 41.21, 514.6, 176); (256, 512, 13.51,
  55.05, 686.0, 224). — [PikkuFold §4, pp. 26–27](https://eprint.iacr.org/2026/1809).
- PikkuFold **Remark 4** (why 2λ rows for ℓ2): "For the spike w = e1, the event Jw = 0 already has
  probability 2^−N with N rows, so N = λ consumes the entire failure budget. More quantitatively, our
  transform bound gives a positive α only once N ln(Ψ^−1) > (λ+1) ln 2. Asymptotically, this
  requires N > 1.1623λ, while N = 2λ yields the constants in Table 1." **Remark 5**: α(λ) ≈ 0.216λ −
  0.300, β(λ) ~ 2.68λ, γ1(λ) ~ √(λ ln 2), b(λ) = Θ(λ). — [PikkuFold p. 27](https://eprint.iacr.org/2026/1809).
- PikkuFold's single-row bound is proved via Fact 1 (a ternary entry is u = (ε+ε')/2 for
  independent signs, so ⟨w,u⟩ is a Rademacher sum with coefficient norm ‖w‖/√2) and the
  Bentkus–Dzindzileta Gaussian comparison with optimal constant κ_BD = 3.178..., giving
  γ1(λ) = erfc^{-1}(2^−(λ+1)/(2λκ_BD)). — [PikkuFold §4.1, pp. 27–28](https://eprint.iacr.org/2026/1809).

### Inferences
- The LNS lemma already *is* the per-coordinate ℓ∞ guarantee that a no-wrap-around check needs:
  it bounds ‖v‖∞ directly, never passes through ℓ2, and holds for any v ∈ Z_q^k with no condition
  such as B < q/4 (TTRP) or b ≤ P/45d (GHL), because the proof handles wrap-around for |s_i| ≤ q/2.
  Reading it with y = 0 and threshold T: **if the verifier learns |(Rv mod q)_j| < T for all
  j ∈ [n], then ‖v‖∞ < 2T except with probability ≤ 2^−n over R** (for v fixed before R).
- Soundness-side slack is exactly 2 and is independent of n and k. n controls only the error 2^−n.
- Completeness-side threshold. Each honest coordinate y_j = Σ_i R_ji v_i has mean 0 and variance
  Σ_i v_i²·Var(D) = ‖v‖₂²/2, so σ = ‖v‖₂/√2 ≤ √(k/2)·2^w. For a per-coordinate failure 2^−c one needs
  T ≥ t_c·σ with t_c the two-sided Gaussian quantile (rigorous for D by PikkuFold Fact 1 + Lemma 12
  up to the factor κ_BD): t_40 ≈ 7.15, t_80 ≈ 10.3, t_141 ≈ 13.8 (= GHL's 9.75·√2). GHL's 9.75‖w‖₂
  and PikkuFold's 9.66‖w‖₂ are the same quantity (13.8σ resp. 13.7σ). Overall honest-to-proven
  ℓ∞ slack: 2T/2^w = t_c·√(2k): at t = 13.8, 19.5√k (≈ 2^9.3 at k = 1024, 2^10.3 at k = 4096).
- Deterministic alternative: |y_j| ≤ ‖v‖₁ ≤ k·2^w always, so T = k·2^w gives *perfect* completeness
  with slack 2k. Compared with the probabilistic threshold this costs log2(2k) − log2(t√(2k)) =
  ½log2(k) − log2(t) + ½ bits, i.e. 1.7 bits at k = 1024 and 2.2 bits at k = 4096 for t = 13.8 — at
  4-bit digit granularity, zero or one extra digit (see Q6 table).
- The ℓ2 JL statements (GHL Cor. 3.2, PikkuFold Thm 2) are *not* needed for a no-wrap-around check
  and would cost 256 rows instead of 128: PikkuFold Remark 4 shows the ℓ2 lower tail is vacuous at
  128 rows and 2^−128. FlashProve's choice of exactly 128 projections identifies its test as the LNS
  ℓ∞ lemma (GHL §3.4 with λ = 128), not an ℓ2 JL projection.
- Consistency check against the FlashProve numbers in our docs (honest < 2^7, proven < 2^18, so
  T = 2^17): 2^17 ≥ t·√(k/2)·2^7 means t√(k/2) ≤ 1024; for k = 5120 (Llama-4 hidden width) that is
  t ≤ 20, for k = 8192 t ≤ 16 — a per-coordinate threshold far beyond 2^−141, which is consistent
  with a 2^−40 *total* completeness error only if the budget is dominated by something else (e.g.
  the number of checked coordinates ≈ 2^30 at t ≈ 9.7, or other probabilistic steps). Not resolvable
  from the papers.

### Gaps
- No paper proves a rigorous version of the completeness tail for the *ℓ∞ of a single projection
  coordinate with fewer than 2λ rows*: PikkuFold Theorem 2's single-row bound is stated with the
  2^−(λ+1)/(2λ) budget for 2λ rows, but Lemma 12 (Bentkus–Dzindzileta) applies to any single row, so
  the per-coordinate statement is rigorous; the 128-row union bound is my derivation.
- FlashProve's own completeness accounting (how 2^−40 arises) is not in any of the four papers.

## Q2. The distribution {0 w.p. 1/2, ±1 w.p. 1/4}: origin, rationale, constants

### Takeaway
LNS 2020/1448 §2.7 introduces the zero-centred ternary projection distribution (with a free zero
probability p, instantiated at p = 1/2), as the generalisation of Baum–Lyubashevsky's uniform
binary matrix; GHL fixes D(0) = 1/2, D(±1) = 1/4 and names it, and later lattice work (LaBRADOR,
TTRP's D_ghl, PikkuFold's χ) adopts it. p = 1/2 is the unique minimiser of the LNS soundness
bound max{p,1−p}^n; the zero-centring halves the variance (½ per entry) which shrinks the projected
values and hence the masking/digit width, and the sparsity halves the prover's work. PikkuFold
certifies rigorous constants (γ1, α, β, b) = (9.66, 27.37, 343.2, 126) at λ = 128 with 256 rows,
replacing LaBRADOR/GHL's heuristic (√30, √337, b = 125) at a cost of "a few percent".

### Cited Findings
- LNS: "we define the probability distribution C : {−1,0,1} → [0,1] such that Pr[c = 0] = p and
  Pr[c = 1] = Pr[c = −1] = (1−p)/2 for some p ∈ [0,1]"; advantage over the uniform binary matrix of
  [BL17]: "the infinity norm of Bs decreases significantly, which is essential for the rejection
  sampling." Parameter choice p = 0.5 in Fig. 9. — [LNS §2.7, p. 8; Fig. 9, p. 39](https://eprint.iacr.org/2020/1448).
- LNS bounds the operator norm of B ← C^{τd×ℓd}: the entries are "0-subgaussian with parameter
  s = √(2π)" and by [MP12, Lemma 2.9] s1(B) ≤ √(τd) + √(ℓd) + 6 except with probability 2^−163;
  they set the masking standard deviation s0 = 11·T′ with T′ = (√(τd) + √(ℓd) + 6)·a. — [LNS App.
  E.3, p. 38](https://eprint.iacr.org/2020/1448).
- GHL: "the distribution D that we use for the entries of R has D(0) = 1/2 and D(±1) = 1/4" and the
  justification "comes from the analysis of Achlioptas [1], who proved that for an arbitrary vector
  w and R ← D^{n×k}, all the moments of the induced distribution over ‖wR‖² are bounded by the
  corresponding moments of the distribution ‖wR′‖² where the entries of R′ are chosen from the
  corresponding zero-mean continuous Normal distribution." — [GHL §3.2, p. 14](https://eprint.iacr.org/2021/1397);
  intro: "When the coefficients of R are instead chosen from a discrete distribution over {0, ±1}
  where the probability of 0 is 1/2, one can heuristically verify that these confidence intervals
  are bounded by the continuous ones. If we would like to be in a 1 − 2^−128 interval, then R can
  have around 256 columns and then the ratio between the smallest value of ‖vR‖ and the largest is
  under 4." — [GHL §1, p. 7](https://eprint.iacr.org/2021/1397).
- TTRP: "The distribution D_ghl is defined as the Bin(2, ½) − 1, i.e., it is 0 with probability 1/2
  and 1 (resp. −1) with probability 1/4." Moments m1 = m3 = 0, m2 = m4 = 1/2; Lemma 7: for a ←
  D_ghl^m and fixed b, E[⟨a,b⟩⁴] ≤ ¾‖b‖₂⁴. — [TTRP p. 8; App. A.3, pp. 27–28](https://eprint.iacr.org/2026/2146).
  TTRP replaces the Gaussian/Rademacher cores of the ML literature by D_ghl and drops the 1/√k
  normalisation (two modifications, §3.1). — [TTRP §3.1, p. 10](https://eprint.iacr.org/2026/2146).
- PikkuFold, Remark 2: "For the sparse ternary distribution with probabilities 1/4, 1/2, and 1/4
  on −1, 0, and 1, respectively, the 128-bit parameters of [BS23, Lemmas 4.1 and 4.2] are n_rp =
  256, α_rp = √30, β_rp = √337, and b = 125, chosen heuristically. Theorem 2 gives a self-contained
  parameterised analysis and concrete values without relying on heuristics." — [PikkuFold §2.6,
  p. 11](https://eprint.iacr.org/2026/1809). Contribution (C2): "it does so at a cost of only a few
  percent in the resulting norm-preservation constants." — [PikkuFold §1.2, p. 4](https://eprint.iacr.org/2026/1809).
- PikkuFold: "The ternary distribution makes the projection efficient to evaluate." The four
  certified estimates are the single-row bound |⟨w,u_j⟩| ≤ γ1‖w‖, the upper and lower tails
  α‖w‖² < ‖Jw‖² ≤ β‖w‖², and the modular lower tail ‖Jw mod q‖ > √α·θ when ‖w‖ ≥ θ ≤ q/b.
  Techniques: upper tail by moment domination by N(0, ½) (Lemma 13: E[u^{2j}] = 1/2 ≤ (2j−1)!!·2^−j);
  lower tail by a Laplace-transform/Fourier identity L_w(s) = E_ξ[Π_j cos²(w_j ξ/2)], an envelope
  env(s) = max{(1+e^−s)/2, (1+s)^−1/2} meeting at s× ≈ 2.31057, env(s×) ≈ 0.549602, and a
  branch-and-bound finite certificate over 2–4 "large" coordinates (|w_j| > 0.45) processing 645,
  9,241 and 33,687 boxes, giving sup_{‖w‖=1} L_w(s×) ≤ 0.5508 (Prop. 2) and α(λ) = −((λ+1) +
  2λ log2 0.5508)/(s× log2 e); the modular bound follows LaBRADOR's three-regime proof with
  auxiliary constants (T_λ, D_λ, r_λ) = (30, 10.08, 6.53) at λ = 128. — [PikkuFold §4.2–4.4, pp.
  28–34](https://eprint.iacr.org/2026/1809). Sage notebook:
  github.com/osdnk/pikku/blob/main/ternary_jl_parameters.ipynb (footnote 1, p. 25).

### Inferences
- Why p = 1/2 rather than uniform ±1 (Rademacher) or p = 2/3 (Achlioptas's sparse variant): the LNS
  bound is max{p,1−p}^n, minimised at p = 1/2 (2^−n); p = 2/3 would need 1.71× more rows for the
  same error, and the Rademacher matrix (p = 0) gives the trivial bound 1 from the lemma, since the
  "c_i = 0 case" has no counterpart (its actual ℓ∞ soundness is a separate, worse analysis in which
  the sign-only coordinate can cancel against r). Zero-centring also halves the variance relative
  to Rademacher (½ instead of 1), shrinking every projected value by √2 and so the masking vector
  (LNS/GHL) or the committed digit width (FlashProve/us) by half a bit, and halves prover work
  since half the entries are zero.
- Why not Gaussian: not an integer matrix (cannot be derived exactly from a transcript, nor
  evaluated as an MLE by additions), and its tails are what the ternary analysis is compared
  against anyway (GHL heuristic; PikkuFold Lemma 13 makes the upper tail rigorous, the lower tail
  needs the separate spike analysis).
- Constants to carry for jolt-atlas: per-entry variance 1/2; fourth moment 1/2; E[⟨a,b⟩⁴] ≤ ¾‖b‖⁴;
  single-row quantile γ1 = 9.66·‖v‖₂ at failure 2^−137 (PikkuFold) or 9.75‖v‖₂ at 2^−141 (GHL
  heuristic, which the rigorous value confirms within 1%); ℓ2 slack √(343.2/27.37) = 3.54 (rigorous)
  vs √(337/30) = 3.35 (heuristic); modular margin b = 126 (ℓ2 only). For the ℓ∞ check none of α, β,
  b is used.

### Gaps
- None of the papers gives a rigorous *single-coordinate* constant for fewer than 2λ rows at a
  2^−λ total budget as an explicit table entry; I derived it from the Gaussian quantile (exact for
  the κ_BD-adjusted Rademacher comparison).

## Q3. Soundness accounting at 128 bits over a 128-bit field; sharing R across rows

### Takeaway
Per fixed vector, n = 128 projections give error 2^−128 (LNS/GHL) with slack 2. Sharing one R
across all rows and all matrices is what every paper does for block-structured projections
(I_r ⊗ J in PikkuFold Lemma 9, RoK-and-Roll's structured projection in TTRP §5.1) and costs a union
bound: error ≤ (total number of rows)·2^−n. Strict 2^−128 over 2^20 rows therefore needs n = 148;
otherwise accept ≈ 2^−(128 − log2 rows). No wrap-around condition on q is needed for the ℓ∞ lemma;
the field only has to be large enough that the proven bound 2T, times the downstream fan-in, stays
below q/2 (trivial at 128 bits).

### Cited Findings
- LNS Lemma 2.5: error max{p,1−p}^n for one fixed s; GHL Lemma 3.4: 2^−128 for 128 columns.
  — [LNS p. 8](https://eprint.iacr.org/2020/1448); [GHL p. 15](https://eprint.iacr.org/2021/1397).
- GHL uses *one* R and *one* exact-smallness proof when the same LWE secret appears in several
  relations: "the prover will only send a single commitment to that LWE secret s, the verifier will
  only send a single challenge matrix R, and the parties will only run a single exact-smallness
  proof". — [GHL §3.6, p. 20](https://eprint.iacr.org/2021/1397). For m amortised vectors the
  projected dimension and slack grow with √m: Lemma 3.6 requires b < √P/(3536(d+4)√m); Lemma 3.7:
  size gap γ = √(337/30)·(1 + 20/√b*) ≤ 3.36·(1 + 20/√b*). — [GHL §3.5–3.6, pp. 17–19](https://eprint.iacr.org/2021/1397).
- PikkuFold **Lemma 9** (layered/block projections): for P = (I_{r_{d−1}} ⊗ J_{d−1})···(I_{r_0} ⊗
  J_0) with J_i ← χ^{n_i×m_i} independently, for every non-zero w: Pr[‖Pw‖/‖w‖ ∉ [Π ℓ_i, Π u_i]] ≤
  Σ_i r_i κ_i, and the modular statement Pr[‖Pw mod q‖ ≤ θ Π ℓ_i] ≤ Σ_i r_i κ_i when θ Π_{j<i} ℓ_j ≤
  q/b_i; proof: "A union bound over the r_i blocks shows that the ith layer preserves the norm of its
  input within [ℓ_i, u_i] except with probability at most r_i κ_i." "In particular, setting κ_i :=
  κ/(d r_i) gives failure probability at most κ." — [PikkuFold §3.1, pp. 15–16](https://eprint.iacr.org/2026/1809).
- PikkuFold Theorem 1 knowledge error (how the projection error composes with sumcheck and
  batching): κ = k/|C| + kε_C + κ_proj + (log2 N_d/q)^μ + 1/q^e + (2Σ_j ν_j + 1)/q^a — the projection
  term κ_proj = Σ_j η_j κ_j is simply added to the Schwartz–Zippel terms. — [PikkuFold §3.2, p. 19](https://eprint.iacr.org/2026/1809).
  Its extraction argument: "The vector w*_all is fixed before P is sampled, and the verifier
  accepted only if ‖v_tr‖ ≤ ω. The modular lower-bound part of Lemma 9, applied with threshold
  β′_in, now gives ‖w*_all‖ ≤ β′_in except with probability at most κ_proj." — [PikkuFold p. 22](https://eprint.iacr.org/2026/1809).
- TTRP: RoKoKo's "coarse projection" is the block-diagonal V = (I_{m/m_rp} ⊗ M)·W with the same M in
  each block; TTRP Theorem 4 knowledge error δ0 = p + q^−k′ where p is the projection failure
  probability (Theorem 3) and q^−k′ the batching error; Corollary 1 adds the sumcheck's 2ν/|C|.
  — [TTRP §5.1, p. 17; Thm 4, p. 13; Cor. 1, pp. 15–16](https://eprint.iacr.org/2026/2146).
- TTRP, Remark 1: batching k projection constraints over Z_q "via the challenge matrix Γ requires a
  row dimension k′ = Ω(λ/log q)". — [TTRP p. 14](https://eprint.iacr.org/2026/2146).
- TTRP parameter search fixes "the projection error p ≈ 2^−90" and k ≤ 300 rows. — [TTRP §7.2,
  p. 21](https://eprint.iacr.org/2026/2146).

### Inferences
- Soundness error of the whole row check for a proof with M_rows rows in total (over all matrices
  and all ops that use it): ε_proj ≤ M_rows·2^−n, plus the Schwartz–Zippel errors of the linking
  sumchecks (Q5), which are of order (log k + log m)·deg/|F| ≈ 2^−115 for a 128-bit field and are
  already of the same order as the rest of jolt-atlas's sumchecks. GPT-2 at 16 tokens: about
  16 × 12 × 6 ≈ 2^10 einsum output rows → 2^−118 at n = 128; at 1024 tokens 2^16 rows → 2^−112;
  n = 128 + ⌈log2 M_rows⌉ restores 2^−128 (n = 144–148). The cost of extra projections is linear
  (Q6), so this is a 12–16 % cost knob, not a design problem.
- The union bound is tight only when the cheating rows are different bad vectors; identical bad
  rows are perfectly correlated and give 2^−n. Nothing better than the union bound is available in
  the papers, and nothing better is needed.
- Sharing R across rows of *different* matrices (same width) is equally fine, provided every
  commitment any of those matrices depends on precedes R in the transcript (Q5). One R per distinct
  width, drawn once after the witness commitment batch, is the natural jolt-atlas instantiation.
- Field size: the ℓ∞ lemma needs only q odd; 2T ≈ 2^20 ≪ 2^127. The downstream no-wrap-around
  argument needs fan-in × 2T × |weight| < q/2: 2^13 · 2^20 · 2^16 = 2^49 ≪ 2^127. No constraint
  binds.

### Gaps
- Fiat–Shamir grinding on R: none of the papers analyses an adversary who re-randomises an earlier
  commitment to re-draw R; the standard FS argument (each attempt costs a hash and succeeds w.p.
  M_rows·2^−n) applies but is not written in these papers.

## Q4. Verifier cost and the tensor-train (TTRP) succinct variant

### Takeaway
TTRP replaces each dense ternary projection row by a tensor train of μ small ternary cores of
internal rank c, so the matrix is described by k·d·((μ−2)c² + 2c) ternary entries instead of k·m,
and the verifier evaluates its MLE by block-wise contraction in O(k·μ1·c²·d) ring operations. The
price: per-row ℓ∞ failure probability rises from 1/2 to 1/2 + (μ−1)/2^{c+1}, the ℓ∞ slack rises
from 2 to 2^μ, the ℓ2 slack to ~2^7–2^10 (vs 3.5 dense), the proof grows by about 2 KB, and the
protocol has completeness error 1/2 (Markov) unless tightened. It is designed for whole-witness
dimension m = 2^20–2^24; at jolt-atlas's row widths k ≤ 4096 the dense 128 × k matrix is already
2^17–2^19 ternary entries and the TT version is *more* expensive for the verifier, so it buys
nothing for a per-row check unless the verifier is itself a circuit.

### Cited Findings
- Abstract: "For a witness of dimension m and a security parameter λ, the JL projection vector is
  O(λ), but the projection matrix is O(λm). Structured JL ... produce large projection vectors that
  need to be sent in costly committed form ... TTRP + sum-check as a drop-in replacement for existing
  range proof techniques yields an approximate shortness check with O(λ log m (log log m)²) verifier
  complexity and proof size around 4 KB." — [TTRP p. 1](https://eprint.iacr.org/2026/2146).
- Table 1: JL (LaBRADOR) at mφ = 2^27 / 2^30: proof 1.84 / 1.89 KB, slack √(128/30), matrix
  representation 6,492 / 51,936 MB; TTRP: 3.90 / 3.98 KB, slack ~2^10, representation 0.2 / 0.6
  MB. "~90,000× improvement in verifier complexity at the cost of a ~2 KB increase in proof size and
  a ~2^9× increase in the slack factor". — [TTRP pp. 4–5](https://eprint.iacr.org/2026/2146).
- **Definition 4**: m = d^μ; f^k_TT(μ,c)(x)^{(j)} = Mat(M_1^{(j)} ⊗ ··· ⊗ M_μ^{(j)})·x for j ∈ [k],
  with M_1 ∈ Z^{1×d×c}, M_i ∈ Z^{c×d×c}, M_μ ∈ Z^{c×d×1}, i.i.d. entries from D_ghl; the 1/√k
  normalisation is dropped. — [TTRP §3.1, p. 10](https://eprint.iacr.org/2026/2146).
- **Lemma 3**: E[y²] = (c^{μ−1}/2^μ)‖x‖₂², E[y⁴] ≤ ¾(c(c+2)/4)^{μ−1}‖x‖₂⁴. — [TTRP p. 11](https://eprint.iacr.org/2026/2146).
- **Lemma 4 / Theorem 2** (ℓ∞, modular): for fixed x with ‖x‖∞ ≥ B_∞ < q/4, one TT row satisfies
  Pr[|f_TT(x) mod q| < B_∞/2^μ] ≤ 1/2 + (μ−1)/2^{c+1}; k rows: Pr[‖y‖∞ < B_∞/2^μ] ≤ (1/2 +
  (μ−1)/2^{c+1})^k; if ‖x‖₂ ≥ B < q√m/4 then Pr[‖y‖₂ < B/(√m 2^μ)] ≤ same; with c ≥ ⌈log2(μ−1)⌉ + δ
  and k ≥ λ/(−log2(1/2 + 2^−δ−1)) the probability is ≤ 2^−λ. The proof peels one core per layer,
  losing a factor 2 of threshold per layer and a factor (1 − 2^−c) of probability per layer.
  — [TTRP §3.2, pp. 11–12](https://eprint.iacr.org/2026/2146).
- **Lemma 5 / Theorem 3** (ℓ2 via Cantelli): Pr[‖y‖₂ ≤ √θ‖x‖₂] ≤ (1 − (η2−θ)²/(η4 − 2η2θ + θ²))^k
  with η2 = c^{μ−1}/2^μ, η4 = ¾(c(c+2)/4)^{μ−1}; the moment ratio η4/η2² = 3(1 + 2/c)^{μ−1} must be
  bounded, so c = Ω(μ) = O(log m). "The slack is dominated by the divergence between the second and
  fourth moments of the TTRP distribution, which scales exponentially with the tensor order μ"
  (Remark 2). — [TTRP §3.3, pp. 12–13; §7.1, p. 20; Remark 2, p. 14](https://eprint.iacr.org/2026/2146).
- **Protocol Π⁰ (Fig. 1) and Theorem 4**: verifier sends the cores; prover sends y₀ = ct(M·vec(W))
  in the clear (k elements of Z_q); verifier checks ‖y₀‖₂ ≤ B̂ := √(k(c/2)^{μ−1})·B; a random Γ ∈
  Z_q^{k′×k} batches the k constraints; the k′ claims become sumcheck claims Σ_z mle[m*(i)](z)·
  mle[vec(W)](z) − y′_i·(mr)^−1 = 0 and are linearised by one degree-2 sumcheck ending in the
  verifier evaluating mle[m*(i)](r) itself and receiving mle[vec(W)](r). "Π⁰ is correct with a
  correctness error of 1/2" (Markov) and knowledge sound with error p + q^−k′, extracting
  ‖W‖ ≤ B̃ = √(k c^{μ−1}/(θ 2^{μ−1}))·B. — [TTRP §4.1–4.2, pp. 13–15; App. A.1, pp. 25–26](https://eprint.iacr.org/2026/2146).
- **Lemma 6** (costs): prover O(k c m φ) over Z_q (the projection, "dominated by the initial
  layers") plus O(mr) ring ops for the sumcheck; verifier samples O(k c² d μ) elements of D_ghl,
  evaluates the MLE "via block-wise tensor contraction" in O(k μ1 c² d) ring ops plus O(k c φ)
  additions, forms aggregated claims in O(k′k); communication k elements of Z_q, k′+1 of R_q and
  2ν of F_{q^e}. The verifier chunks the challenge point into μ1 blocks of length ℓ = log d,
  evaluates each spatial core's MLE at its chunk, and chains the 1×c boundary vector through the
  cores. — [TTRP §4.2, pp. 16–17](https://eprint.iacr.org/2026/2146).
- **Table 4** (concrete, p ≈ 2^−90): m = 2^20: slack 2^7 with (k, μ, c) = (296, 4, 11), repr. 2^22.13;
  slack 2^10 with (292, 4, 10), repr. 2^21.83; m = 2^21: (296, 3, 7) → 2^22.41 and (278, 3, 7) →
  2^22.32; m = 2^24: (296, 4, 11) → 2^23.13 and (292, 4, 10) → 2^22.83; "Repr." = k d((μ−2)c² + 2c).
  — [TTRP §7.2, p. 22](https://eprint.iacr.org/2026/2146).
- Asymptotics: verifier O(λ log(mr)(log log(mrφ))²); extraction slack B̃/B = O(√(λ (log log
  (mrφ))^{log(mrφ)})·(mr)^{3/2}) ∈ poly(mr). — [TTRP §7.1, p. 20](https://eprint.iacr.org/2026/2146).
- Related structured projections: RoK-and-Roll's block projection I_{m/ℓ} ⊗ J with ℓ = O(λ²) "making
  the overall verifier complexity at least O(λ³) per recursion" and requiring commitment to the
  projection image; naive tensor-product rows need Ω(λ log m) repetitions. — [TTRP §1, p. 3; §1.2,
  p. 5](https://eprint.iacr.org/2026/2146). PikkuFold's layered random projection shrinks the
  dimension layer by layer with block matrices I_{r_i} ⊗ J_i, so that only the final short image is
  sent; the verifier evaluates MLE[I_r ⊗ M_j]((x0,x1),(y0,y1)) = eq(x0,y0)·MLE[M_j](x1,y1) and
  "scans each unexpanded matrix description once", cost O(S_JL + μN_d + k log m + kn) ring ops
  where S_JL = Σ n_j m_j + φ n_{d−1} m_{d−1} is the number of ternary entries. — [PikkuFold §3,
  p. 14; Lemma 11 and proof, pp. 23–24](https://eprint.iacr.org/2026/1809). PikkuFold's folding
  verifier at (k, m, φ) = (2, 2^20, 128) runs in 5.39 ms with three 256-row layers. — [PikkuFold
  Table 6, p. 42](https://eprint.iacr.org/2026/1809).

### Inferences
- The jolt-atlas row check with one R shared across rows is exactly the block projection I_m ⊗ R.
  By PikkuFold's identity the verifier's terminal evaluation is eq(ρ_x, r_x)·R̃(ρ_u, r_z): the
  O(n·k) term is independent of the number of rows m. Concretely: build the eq(r_z, ·) table (k
  multiplications), accumulate Σ_i R_ji·eq_i for each j (n·k signed additions, no multiplications,
  since R is ternary), then Σ_j eq(ρ_u, j)·(…) (n multiplications). At k = 1024, n = 128: ~1.2k
  multiplications + 131k additions ≈ tens of microseconds; at k = 4096: 524k additions. Done once
  per distinct width if the linking sumchecks for same-width matrices are batched with shared
  challenges (as jolt-atlas's relation batching already does).
- TTRP at our widths: k = 1024 = 32², μ = 2, c = 7: per-row failure 1/2 + 2^−8, so 130 rows for
  2^−128 (vs 128), ℓ∞ slack 4 instead of 2 (one more digit bit), representation 32·14 = 448 cores
  entries per row vs 1024 dense, verifier O(k_rows·μ1 c² d) ≈ 130·2·49·32 ≈ 408k ring ops vs 133k
  additions dense. TTRP loses on every axis here; its regime is m ≥ 2^20.
- What TTRP does show that transfers: (i) the linearisation shape (verifier evaluates the
  projection's MLE at the sumcheck's terminal point; prover supplies the witness evaluation) is the
  same as ours; (ii) a structured R can be substituted later for recursion without changing the
  protocol, at the stated slack/probability cost; (iii) the completeness error 1/2 of Π⁰ is an
  artefact of sending y₀ in the clear with a Markov threshold, not of the projection — committing
  digits with an ℓ1 or 13.8σ threshold has no such issue.

### Gaps
- TTRP gives no concrete parameters below m = 2^20, so the k ≤ 4096 comparison above is my
  extrapolation from Theorem 2 and Lemma 6.

## Q5. Protocol for a VIRTUAL vector known only through v(r); soundness; transcript order

### Takeaway
All four papers project a *committed* vector and use the commitment for exactly one thing in the
soundness proof: fixing the vector before R is sampled (TTRP A.1: "Since M_Z ... is independent of
W"; PikkuFold p. 22: "The vector w*_all is fixed before P is sampled"). Nothing in them requires the
vector itself to be committed. In jolt-atlas every virtual tensor is a deterministic field-valued
function of public data and of the committed digit polynomials, so it is fixed once the witness
commitment batch is absorbed; drawing R after that batch, committing the projection digits in a
second batch, and linking them to v's evaluation claim with one eq-weighted degree-2 sumcheck gives
a sound check. The one hard constraint the papers surface is the ordering: if any commitment v
depends on came after R, the prover could steer v into the kernel of R (k ≫ 128, so the kernel is
large) and the lemma gives nothing.

### Cited Findings
- GHL Lemma 3.4 is stated for "any two vectors y ∈ [±q/2]^128 and w ∈ [±q/2]^d" fixed before "choosing
  R ← D^{d×128}", and GHL's protocol commits w (Pedersen) before R and proves z = wR + y (mod P) by a
  linear ZKPOK. — [GHL §3.4, pp. 15–16](https://eprint.iacr.org/2021/1397).
- LNS commits (BDLOP) the vector and the masking vector, receives B, sends z = y + Bv, and proves
  the linear relation z = y + Bv inside the LNS framework; soundness extracts v*, u*, y*, w* as weak
  openings and then applies Lemma 2.5 to v*. — [LNS App. E.1–E.2, pp. 33–36](https://eprint.iacr.org/2020/1448).
- TTRP extractor E_TT: "If ‖W‖₂ ≤ B̃ ... concludes ... As before, if W ≠ W′ mod q, we extract a SIS
  solution ... Assuming W = W′ mod q, the verifier's ℓ2 norm check ensures that ‖y₀‖₂ ≤ B̂. Since M_Z
  (and its embedding M) is independent of W, by Theorem 3 and the established equivalence y₀ =
  M_Z·cf(vec(W)), we have ‖W‖₂ ≤ B̂/√θ ... unless with a probability of at most p." The sumcheck
  linearisation of the projection is the degree-2 claim Σ_z mle[m*(i)](z)·mle[vec(W)](z) = y′_i·2^−ν
  and the terminal check v = Σ_i γ_i(mle[m*(i)](r)·w_r − y′_i (mr)^−1) with the verifier
  computing mle[m*(i)](r) locally. — [TTRP App. A.1, p. 26; §4.2, pp. 14–15](https://eprint.iacr.org/2026/2146).
- PikkuFold: the witness evaluation claims and the projection claims are batched into one sumcheck
  so that a single terminal evaluation of the witness MLE survives: "We first rewrite them as the
  sumcheck identities t = Σ_b MLE[w](b) eq(b, s) ... We then batch these identities with the
  sumcheck for the JL projection. Since all batched claims use the same sumcheck challenges, the
  only witness-dependent claims left at the terminal point are t′ = MLE[w](r), t′_acc = MLE[w_acc](r)
  at the same random point r." — [PikkuFold §3, p. 15](https://eprint.iacr.org/2026/1809). Protocol
  Π^fold (Fig. 2): verifier sends M; prover sends v_tr = Tr(M w_all) in the clear; verifier checks
  ‖v_tr‖ ≤ ω; trace-batching points c_i; batched claims SC^proj, SC^eval; RingSC; terminal
  evaluations t′_i; folding challenge z last. — [PikkuFold Fig. 2, p. 18](https://eprint.iacr.org/2026/1809).
- PikkuFold soundness of the projection step: "Outside this event, Lemma 10 gives P coeff(w*_all) =
  Tr(M w*_all) = v_tr mod q. The vector w*_all is fixed before P is sampled, and the verifier
  accepted only if ‖v_tr‖ ≤ ω. The modular lower-bound part of Lemma 9, applied with threshold
  β′_in, now gives ‖w*_all‖ ≤ β′_in except with probability at most κ_proj." — [PikkuFold p. 22](https://eprint.iacr.org/2026/1809).
- PikkuFold's principal relation Ξ^lin carries the witness as "a short vector w committed under F"
  together with "a multilinear-evaluation claim on w" MLE[w](s) = t; the folding acts on the
  evaluation claims, the commitment supplies binding. — [PikkuFold Def. 22, p. 13](https://eprint.iacr.org/2026/1809).
- GHL on why the proof only needs the no-wrap-around property: after the approximate ℓ∞ proof,
  "proving statement (3) implies that for all the v_i's we have ‖v_i‖∞ < √(P/2(d+4)), and therefore
  ‖v_i‖² = Σ v²_{i,j} < P/2. This implies that statement (2) holds over the integers and not just
  modulo P". — [GHL Lemma 3.6 proof, p. 17](https://eprint.iacr.org/2021/1397).

### Inferences
Protocol (my construction, in jolt-atlas terms). Let V ∈ F^{m×k} be a virtual tensor, m = 2^a rows,
k = 2^b columns, with MLE Ṽ(x, z), x ∈ F^a, z ∈ F^b. Honest entries satisfy |V_ri| < 2^w. The
verifier holds (or the op relation will produce) an evaluation claim Ṽ(ρ′_x, ρ′_z) = c.

1. **Commitment batch 1.** All one-hot digit polynomials of the trace (remainders, table
   addresses, …) are committed and absorbed; V is now a fixed function of public data and these
   commitments (e.g. for an einsum output V = (acc − rem)/D in F, with acc determined by inputs,
   weights and earlier V's).
2. **Draw R.** For each distinct width k in use, squeeze R_k ∈ {−1,0,1}^{n×k} from the transcript
   (2 bits per entry: 00/01 → 0, 10 → +1, 11 → −1), n = 128 or 128 + ⌈log2 M_rows⌉.
3. **Project and commit (batch 2).** Prover computes Y_rj = Σ_i R_ji V_ri ∈ Z for every row r and
   j ∈ [n] (m·n·k signed additions), shifts by T (Y + T ∈ [0, 2T)) and commits the D_y = ⌈(1 +
   log2 2T)/4⌉ one-hot 4-bit digit polynomials of Y (or dense digits, roadmap D3). Define the virtual
   Ỹ(x, u) := Σ_d 2^{4d}·digit_d(x, u) − T, u ∈ F^{log n}.
4. **Range-check the digits** with the existing one-hot machinery (Booleanity + Hamming weight +
   recomposition). This fixes Y_rj ∈ [−T, T) as integers.
5. **Linking relation** (one degree-2 sumcheck over a + b variables, batched with the existing
   claim on V): verifier draws ρ_x ∈ F^a, ρ_u ∈ F^{log n}, λ ∈ F;
   claim  Ỹ(ρ_x, ρ_u) + λ·c = Σ_{x∈{0,1}^a, z∈{0,1}^b} Ṽ(x,z)·[ eq(ρ_x, x)·R̃(ρ_u, z) + λ·eq((ρ′_x,ρ′_z),(x,z)) ].
   The left side is a committed-digit evaluation (Ỹ at (ρ_x, ρ_u) opens through the digits) plus
   the known c. At the terminal point (r_x, r_z) the verifier evaluates eq(ρ_x, r_x), R̃(ρ_u, r_z)
   (cost O(n·k) additions, Q4) and eq((ρ′_x,ρ′_z),(r_x,r_z)) itself, and receives the single claim
   Ṽ(r_x, r_z), which replaces the old claim c and flows into V's producing op exactly as today.
   This is PikkuFold's §3 batching, and it is the natural shape of a jolt-atlas `Relation`.
6. **Order of transcript operations:** batch-1 commitments → R → batch-2 commitments (Y digits) →
   range-check challenges → (ρ_x, ρ_u, λ) → sumcheck rounds → opening reduction over both batches.

Soundness (my argument, using the papers' lemmas). Fix the prover's batch-1 commitments; by Akita
binding they determine unique field vectors for every committed digit polynomial (a weak/relaxed
opening suffices: uniqueness as a field vector is all that is used; no range property of the
commitment is needed, consistent with our §4 note on Akita). Together with the public data these
determine a unique V* ∈ F^{m×k} (the field value of each virtual tensor, e.g. (acc* − rem*)/D with
division in F). R is sampled after this, so V* is independent of R. Batch 2 determines Y*. If the
verifier accepts: (a) the digit range check gives Y*_rj ∈ [−T, T) ⊂ Z for all r, j; (b) the
linking sumcheck gives, except with probability ≤ (2(a+b) + 1)/|F|, the polynomial identity
Ỹ*(x,u) = Σ_z R̃(u,z)Ṽ*(x,z) on the hypercube, i.e. Y*_rj ≡ Σ_i R_ji V*_ri (mod q) for all r, j, and
Ṽ*(ρ′_x, ρ′_z) = c; (c) for each row r, applying LNS Lemma 2.5 with y = 0, n projections, p = 1/2 to
the fixed vector V*_r ∈ Z_q^k: if ‖V*_r‖∞ ≥ 2T (balanced representatives) then all n coordinates
of R V*_r mod q lie below T with probability ≤ 2^−n; union over m rows. Hence except with
probability m·2^−n + (2(a+b)+1)/|F|, every row of V* has ‖V*_r‖∞ < 2T as a balanced integer. With
2T·(fan-in)·(max weight) < q/2 for every consumer, the field identities of the consumers hold over
the integers (GHL Lemma 3.6's argument), which is the no-wrap-around property the check exists to
provide; combined with rem* ∈ [0, D) from the exact remainder digits, D·V*_r + rem*_r = acc*_r over
Z makes V* the Euclidean quotient, so the output is identical to today's whenever the honest clamp
never engages (roadmap D2).

What the papers forbid or complicate:
- *The vector must be fixed before R* — the only role of the commitment in all four proofs. If
  jolt-atlas ever committed part of the witness after R (staged commitments), a prover could choose
  the later part to move V* into ker(R) mod q: R is n × k with k ≫ n, so the kernel has dimension
  ≥ k − n and contains vectors of every magnitude. Example: for V = (acc − rem)/D, choosing rem′ ≠
  rem per entry changes V by (rem − rem′)/D, a large field element; with D^k choices of rem and
  only q^n kernel constraints (2^{16·1024} ≫ 2^{128·128}) kernel-hitting choices exist. Hence the
  rule "R after every commitment V depends on" is not a convenience but the soundness hinge.
- *The prover does choose Y after R* — that is fine and intended; Y is bound to R·V* by the
  sumcheck, and Lemma 2.5 holds "regardless of y" (GHL) for any adversarial additive term.
- *Second commitment round.* GHL/LNS commit the masking vector before R and the response is sent in
  the clear; TTRP/PikkuFold send the (short) projection in the clear. Committing Y's digits is
  FlashProve's choice and ours; it forces a second Akita commitment batch and a second set of
  polynomials in the opening reduction. No paper has this shape; nothing in them argues against it.
- *Knowledge soundness vs soundness.* The papers prove knowledge soundness by extraction of the
  committed vector. Our statement is plain soundness of a deterministic-witness relation (the trace
  is a function of inputs plus the committed digits), which is what jolt-atlas proves today; the
  LNS lemma is used on the extracted field vector either way.
- *Masking and rejection sampling* (LNS Rej0/Rej2, GHL Step 2/4) exist only because z is sent in the
  clear; with committed digits they are unnecessary, and with them goes the factor λ in GHL's gap
  γ = 2·9.75·λ·√d, leaving 2·9.75·√d.

### Gaps
- No paper treats a projection of a vector that is *only* defined through evaluation claims; the
  reduction above is mine. The one place it leans on jolt-atlas structure rather than on the
  papers is the claim that each virtual tensor is a unique field-valued function of the batch-1
  commitments; that is true of the current pipeline by construction (commit, then prove) but must
  be re-checked if D1 is combined with any staged-commitment change.

## Q6. Cost model: digits per row, break-even width, savings at k = 1024 and 4096

### Takeaway
Per row the check commits n·D_y one-hot 4-bit digits, where D_y = ⌈(1 + ⌈log2 T⌉)/4⌉ and T is set by
the completeness rule (T = 13.8·√(k/2)·2^w) or the deterministic ℓ1 rule (T = k·2^w). For honest
values below 2^7 (FlashProve's regime) D_y = 5 at every k ≤ 4096 under either rule, so at n = 128 the
row costs 640 digits: break-even k = 80, 12.8× fewer clamp digits at k = 1024 and 51× at k = 4096.
For honest values up to 2^16, D_y = 7: break-even 112, 9.1× and 37×. The docs' figures (176, 6×,
20–25×) are reproduced exactly by D_y = 11, i.e. by assuming honest values fill the whole i32 range
(Y of 41–44 bits); that is the pessimistic end. Taking n = 148 for a strict 2^−128 over 2^20 rows
scales every cost by 1.16.

### Cited Findings
- The number of projections required for error 2^−λ per vector in the ℓ∞ lemma is n = λ (128) —
  [LNS Lemma 2.5, p. 8](https://eprint.iacr.org/2020/1448); [GHL Lemma 3.4, p. 15](https://eprint.iacr.org/2021/1397);
  for the ℓ2 variants 256 rows — [GHL Cor. 3.2, p. 14](https://eprint.iacr.org/2021/1397); [PikkuFold
  Table 1, p. 27](https://eprint.iacr.org/2026/1809); ~290 rows at p ≈ 2^−90 for TTRP — [TTRP
  Table 4, p. 22](https://eprint.iacr.org/2026/2146).
- Per-row union bound factor r_i for shared blocks — [PikkuFold Lemma 9, pp. 15–16](https://eprint.iacr.org/2026/1809).
- Prover cost of the dense projection is the matrix–vector product, O(k·m) per vector; for TTRP
  O(k c m φ) — [TTRP Lemma 6, p. 16](https://eprint.iacr.org/2026/2146).

### Inferences
Derivation. Today: 8 one-hot digits per element for the overflow/clamp check (roadmap §1), i.e. 8k
per row. New: n·D_y per row. Threshold rules: probabilistic T_p = 13.8·√(k/2)·2^w (per-coordinate
failure 2^−141); deterministic T_ℓ1 = k·2^w. Bits = 1 (sign) + ⌈log2 T⌉; D_y = ⌈bits/4⌉.

| honest bound 2^w | k | T_p (bits, digits) | T_ℓ1 (bits, digits) |
|---|---|---|---|
| 2^7 | 768 | 2^15.1 (17, 5) | 2^16.6 (18, 5) |
| 2^7 | 1024 | 2^15.3 (17, 5) | 2^17 (18, 5) |
| 2^7 | 3072 | 2^16.1 (18, 5) | 2^18.6 (20, 5) |
| 2^7 | 4096 | 2^16.3 (18, 5) | 2^19 (20, 5) |
| 2^16 | 1024 | 2^24.3 (26, 7) | 2^26 (27, 7) |
| 2^16 | 4096 | 2^25.3 (27, 7) | 2^28 (29, 8) |
| 2^31 | 1024 | 2^39.3 (41, 11) | 2^41 (42, 11) |
| 2^31 | 4096 | 2^40.3 (42, 11) | 2^43 (44, 11) |

| n | D_y | digits/row | break-even k | saving at 1024 | saving at 4096 | total digits/element at 1024 (with 4 remainder digits) |
|---|---|---|---|---|---|---|
| 128 | 5 | 640 | 80 | 12.8× | 51.2× | 4.62 |
| 128 | 7 | 896 | 112 | 9.1× | 36.6× | 4.88 |
| 128 | 8 | 1024 | 128 | 8.0× | 32.0× | 5.00 |
| 128 | 11 | 1408 | 176 | 5.8× | 23.3× | 5.38 |
| 148 | 5 | 740 | 92 | 11.1× | 44.3× | 4.72 |
| 148 | 7 | 1036 | 130 | 7.9× | 31.6× | 5.01 |
| 148 | 11 | 1628 | 204 | 5.0× | 20.1× | 5.59 |

- "Saving" is on the 8 clamp digits only; the 4 exact remainder digits stay, so the per-element
  total goes from 12 to 4.6–5.4 at k = 1024 (2.2–2.6× overall on einsum outputs), matching the
  roadmap's "4 exact remainder digits, plus about 1.4 amortised" only in the D_y = 11 case.
- Perfect completeness (ℓ1 rule) costs 0 extra digits at w = 7 for all k ≤ 4096, and 1 extra digit
  (of 7) at w = 16 and k ≥ 3072. FlashProve's 18-bit choice is tight because it packs 10- and 8-bit
  digits; with 4-bit digits we round up anyway.
- The value of w is the unknown that moves the answer by 2×: it is roadmap §2 item 4 ("value
  histograms from the tracer: einsum output magnitudes"). The honest bound can also be set per op
  from the compile-time weight-ℓ1 bound of D2, which would make T a public per-op constant.
- Width applicability in GPT-2: hidden 768 and MLP 3072 rows qualify; the attention-score matrix
  has row width = sequence length (16 in the test fixtures), below break-even, so it keeps exact
  digits until long contexts.
- Additional prover work beyond commitment: the projection (m·n·k additions; for m = 2^10, k = 2^10,
  n = 128: 2^27 additions, negligible) and one degree-2 sumcheck over a + b variables per batched
  set of same-width matrices (prover O(m·k) field ops per matrix, the same order as the existing op
  relations). Verifier: O(n·k) additions per width (Q4) plus the sumcheck rounds.

### Gaps
- The docs' break-even of 176 could not be traced to a written derivation; the table shows which
  assumption (11 digits per projection) reproduces it.
- No measurement exists of the per-digit commitment cost of a second Akita batch (Y digits) versus
  the same number of digits in the first batch; assumed equal.

## Q7. Completeness failure, retry, and leakage

### Takeaway
In LNS and GHL the projection response is sent in the clear, so the honest prover masks it and uses
rejection sampling (LNS repetition rate M′ ≈ 3; GHL restart probability ≈ 1 − 1/e); an aborted
attempt is never sent, and conditioned on the honest-norm event the accepted response leaks
nothing. Completeness failure of the *projection itself* (honest ‖Rv‖∞ exceeding the threshold) is
set to be negligible (GHL: 2^−134) or constant and retried (TTRP: 1/2 by Markov; PikkuFold: κ_proj,
summed per block). With committed digits there is no in-the-clear response to leak, so the masking
machinery disappears, and with the ℓ1 threshold the failure event disappears too.

### Cited Findings
- LNS: prover samples y ← D^{τd}_{s0}, computes z = y + Bv, "If Rej0(z, Bv, s0) = 1 abort"; Theorem E.1:
  "correctness error 1 − 1/(M′·2M), where M′ is the repetition rate for the first rejection sampling
  Rej0"; parameters give "repetition rates for both rejection samplings are M′ ≈ 3 and 2M ≈ 3.3";
  s0 = 11·T′ with T′ the operator-norm bound on ‖Bv‖. — [LNS Fig. 8, p. 35; Thm E.1, p. 36; App.
  E.3, pp. 38–39](https://eprint.iacr.org/2020/1448). Appendix F: bimodal Gaussians z = y + (−1)^b Bs
  with Rej2 reduce s0 "by a factor of around 12 at the cost of committing to 2τ more vectors";
  soundness survives because "the matrix (−1)^{b*}·B is still randomly generated from C^{d×ℓd} for
  a fixed bit b*". — [LNS App. F, pp. 40–41](https://eprint.iacr.org/2020/1448).
- GHL: masking y ← [±⌈(b/2)(1 + 1/λ)⌉]^λ, "restarts the protocol from Step 2 if either ‖u‖∞ > b/2λ
  or ‖z‖∞ > b/2"; "A restart due to this check [‖u‖∞] therefore only happens with negligible
  probability [2^−134]. Conditioned on ‖u‖∞ ≤ b/2λ, the rejection sampling check for ‖u + y‖∞ ≤ b/2
  leaks nothing about u (or w), by [39]. Furthermore, using the analysis from [40, Section 5.2], the
  probability of the prover restarting due to this check is about 1 − 1/e ≈ 0.63. Hence the
  expected number of repetitions is constant." — [GHL §3.4, p. 16](https://eprint.iacr.org/2021/1397).
  In the LWE protocol: "If ‖u_i‖ > √30 b_i/γ2 or ‖v_i‖ > √30 b′_i/γ2 then the prover aborts" and
  "by Corollary 3.2 the prover only aborts in Step 3 with negligible probability". — [GHL §3.6,
  pp. 19–20](https://eprint.iacr.org/2021/1397). The side remark that revealing δ_i = ±1 would be
  "a one-bit leakage on each ‖w_i‖²" ("We do not use this option") shows GHL's leakage accounting
  standard. — [GHL p. 17](https://eprint.iacr.org/2021/1397).
- TTRP: "Π⁰ is correct with a correctness error of 1/2" via Markov on E‖y₀‖²; Remark 2 proposes a
  Cantelli variance bound to tighten it; footnote 10: LaBRADOR "instantiates β to match the expected
  squared norm. The completeness error p_β is bounded inherently by the right-skewness of the
  underlying (heuristic) chi-squared distribution ... The prover satisfies this condition via
  rejection sampling. Alternatively, selecting a larger β yields an abort-free prover at the cost of
  a looser bound." The projection y₀ is sent in the clear; no zero knowledge is claimed. — [TTRP
  Thm 4, p. 13; Remark 2, p. 14; fn. 10, p. 17](https://eprint.iacr.org/2026/2146).
- PikkuFold: correctness error κ_proj = Σ_j η_j κ_j (Theorem 1), parameters chosen for "failure
  target κ = 2^−100"; "the presented protocol is not zero knowledge, and hiding the witness
  throughout sumcheck without losing its communication and prover efficiency requires additional
  techniques, for example those developed in Libra". — [PikkuFold Thm 1, p. 19; §6.1, p. 40; §1.3,
  p. 5](https://eprint.iacr.org/2026/1809).

### Inferences
- In a Fiat–Shamir, committed-digits instantiation an honest prover that finds some |Y_rj| ≥ T
  cannot "retry": R is a function of the transcript and the trace is deterministic, so there is
  nothing to re-randomise (unless hiding commitments with fresh randomness are introduced, as in a
  future zk build). Failure then means "no proof", exactly the partial-function semantics of D2;
  its probability is ≤ (total projected coordinates)·2^−141 under the 13.8σ rule, or zero under the
  ℓ1 rule. Nothing is published on failure, so nothing leaks.
- The leakage FlashProve worries about ("a rejected honest proof may leak") can only arise when a
  failed attempt is observable, e.g. a service that visibly refuses to answer. The papers' answer is
  structural (don't publish aborted transcripts) rather than a lemma.
- For a later zk jolt-atlas, the committed Y digits are ordinary witness polynomials; GHL's masking
  and rejection sampling are not needed, and the ℓ1 threshold removes even the abort-channel.

### Gaps
- No paper quantifies what an observable abort leaks about v beyond GHL's qualitative "leaks
  nothing conditioned on the honest-norm event"; FlashProve's planned retry mechanism is not in
  these sources.

## What this changes in the docs

### Takeaway
The docs have the attribution and the architecture right and the constants partly wrong: the
soundness slack is a factor 2 (not a property of the 2^11 FlashProve gap, which is mostly the
completeness threshold), the imperfect completeness is a FlashProve packing choice that our 4-bit
digits can avoid for free, the break-even and savings are 1.6–2.2× better than stated unless honest
values fill the full i32 range, the 2021/1397 citation is Gentry–Halevi–Lyubashevsky rather than
LNS, and the virtual-tensor soundness proof that was "still owed" is now written (Q5) with its one
precondition identified.

### Cited Findings
Each entry quotes the doc, then the correction with the page that supports it.

- research-directions.md §3.3: "This is the Lyubashevsky–Nguyen–Seiler approximate range proof
  (ePrint 2020/1448, 2021/1397)." → 2020/1448 is LNS (Lemma 2.5, [p. 8](https://eprint.iacr.org/2020/1448));
  2021/1397 is Gentry–Halevi–Lyubashevsky, whose §3.4 ([pp. 15–16](https://eprint.iacr.org/2021/1397))
  instantiates the LNS lemma with λ = 128 projections and whose ℓ2 JL (Cor. 3.2, 256 columns,
  [p. 14](https://eprint.iacr.org/2021/1397)) is heuristic. The check FlashProve uses is the ℓ∞
  lemma, not the ℓ2 JL projection (128 rows cannot give an ℓ2 lower tail at 2^−128: [PikkuFold
  Remark 4, p. 27](https://eprint.iacr.org/2026/1809)).
- §3.3: "It proves a loose bound. Honest cells are below 2^7; the check proves they are below
  2^18. It rules out field wrap-around and nothing tighter." → Weakened: the lemma's own slack is
  exactly 2 (proven bound = twice the digit range, [LNS p. 8](https://eprint.iacr.org/2020/1448)); the
  remaining 2^10 is the completeness threshold t·√(k/2) for the honest Gaussian-like tail
  ([GHL Fact 3.1, p. 14](https://eprint.iacr.org/2021/1397); [PikkuFold Thm 2, p. 26](https://eprint.iacr.org/2026/1809)),
  so the proven bound is ≈ 19.5√k·(honest bound), adjustable by digit width. "Rules out field
  wrap-around and nothing tighter" is correct and is also GHL's own description ([p. 15](https://eprint.iacr.org/2021/1397)).
- §3.3: "Completeness is imperfect. At most 2^−40 overall in FlashProve. A rejected honest proof
  may leak, which they plan to fix by retrying." → True of FlashProve; not inherent. The ℓ1 bound
  |Y_j| ≤ k·2^w makes completeness perfect at ≤ 1 extra 4-bit digit (Q6 table). On leakage, the
  papers' position is that aborted attempts are never sent and the accepted response leaks nothing
  conditioned on the honest-norm event ([GHL p. 16](https://eprint.iacr.org/2021/1397)); the
  rejection-sampling apparatus exists only for in-the-clear responses.
- §3.3: "The verifier is not succinct in the random matrix. It does work proportional to 128 times
  the row width, which is cheap because the matrix is shared across rows. A succinct alternative
  exists on paper (tensor-train projections, ePrint 2026/2146)." → Strengthened on the first
  sentence (the block structure I_m ⊗ R gives verifier cost eq·R̃ independent of m: [PikkuFold
  p. 24](https://eprint.iacr.org/2026/1809)); the second sentence should say the alternative targets
  whole-witness dimension 2^20–2^24 with slack 2^7–2^10, ~290 rows, +2 KB and completeness 1/2
  ([TTRP Table 1 p. 5, Table 4 p. 22, Thm 4 p. 13](https://eprint.iacr.org/2026/2146)) and is more
  expensive than the dense matrix at k ≤ 4096 (Q4 inference); it is relevant only for a recursive
  verifier.
- §3.3: "For our digit counts the break-even width is about 176, with about 6x fewer digits at
  1,024 and 20–25x at 4,096." and roadmap D1 (same numbers) → Reproduced only with 11 digits per
  projection (honest values up to 2^31). With FlashProve-like 2^7 values: 80 / 12.8× / 51×; with
  2^16 values: 112 / 9.1× / 37× (Q6 tables, my derivation from [LNS p. 8](https://eprint.iacr.org/2020/1448)
  and [PikkuFold p. 26](https://eprint.iacr.org/2026/1809)). Replace the fixed figures by the
  formula n·D_y with D_y from the measured value histogram (roadmap §2 item 4).
- §3.3: "That the check works when the tensor is virtual is our analysis, and the written soundness
  proof is still owed." and roadmap D1 "needs a written proof" → Written in Q5 above. The papers'
  proofs use the commitment only to fix the vector before R ([TTRP p. 26](https://eprint.iacr.org/2026/2146);
  [PikkuFold p. 22](https://eprint.iacr.org/2026/1809)); the precondition for us is that R is
  squeezed after the entire batch-1 commitment, and the Y digits form a second batch.
- §3.3: "FlashProve's own proof of the test's error is in a non-public report; the round's error
  bound is machine-checked." → The test's soundness error is a public, exact, two-case lemma with
  no heuristic ([LNS App. A, pp. 23–24](https://eprint.iacr.org/2020/1448)); only the completeness
  side had a heuristic ([GHL Cor. 3.2, p. 14](https://eprint.iacr.org/2021/1397)), now rigorous
  ([PikkuFold Thm 2, pp. 26–27](https://eprint.iacr.org/2026/1809)) or avoidable (ℓ1 threshold).
- roadmap D1: "The relation is one degree-2 sumcheck over the column axis linking the committed
  combination digits to the tensor's evaluation claim." → Refine: sum over rows and columns with
  eq(ρ_x, ·) weighting, batched with the existing evaluation claim so that one terminal evaluation
  of V survives ([PikkuFold §3, p. 15](https://eprint.iacr.org/2026/1809)); the verifier's own work
  is R̃(ρ_u, r_z), O(n·k) additions per width.
- roadmap D1: "Completeness becomes imperfect (FlashProve: at most 2^−40)." → Optional, see above.
- roadmap D1 (add): soundness error M_rows·2^−n; n = 128 + ⌈log2 M_rows⌉ for a strict 2^−128
  ([PikkuFold Lemma 9, pp. 15–16](https://eprint.iacr.org/2026/1809)); R must be drawn after batch 1;
  the Y digits are a second commitment batch; one R per distinct width.
- research-directions §3.3 "The same combinations are shared by every row." → Justified: block
  projections with a shared J and a union bound over blocks ([PikkuFold Lemma 9, p. 15](https://eprint.iacr.org/2026/1809);
  [TTRP §5.1, p. 17](https://eprint.iacr.org/2026/2146)).
- Unchanged and confirmed: "It does not make rounding exact" (remainder digits stay; Q6), "Linearly
  consumed values get no exact check at all ... valid only because the model is defined as a partial
  function" (the papers prove only ‖v‖∞ < 2T; GHL [p. 15](https://eprint.iacr.org/2021/1397)), and
  D2's necessity.

### Inferences
- The docs' only substantive overstatement is the cost pessimism; the only substantive
  understatement is treating imperfect completeness as a property of the method.

### Gaps
- The FlashProve-specific claims (2^−40, the 10+8-bit split, the retry plan) are outside the four
  papers and were not re-checked here.

## Go/no-go and the minimal prototype

### Takeaway
**Go, conditional on D2.** The soundness side rests on one exact lemma (LNS 2.5) with a clean
reduction in our virtual-tensor setting whose single precondition (R after the witness batch) the
current commit-then-prove pipeline already satisfies; the cost side is 9–13× fewer clamp digits on
768–1024-wide rows and 37–51× on 4096-wide rows at honest magnitudes below 2^16, with perfect
completeness available at ≤ 1 extra digit; the verifier pays O(128·k) additions per distinct width.
What the check cannot do is reproduce saturating-clamp semantics, so it is output-preserving only
where the clamp never engages (D2). Nothing in the papers argues against the design; the TTRP
variant is not worth adopting at our widths.

### Cited Findings
- Soundness lemma: [LNS Lemma 2.5, p. 8; proof pp. 23–24](https://eprint.iacr.org/2020/1448).
- The completeness constant: [PikkuFold Thm 2 / Table 1, pp. 26–27](https://eprint.iacr.org/2026/1809).
- The linearisation and claim-batching shape: [TTRP Fig. 1, p. 14](https://eprint.iacr.org/2026/2146);
  [PikkuFold §3 p. 15, Fig. 2 p. 18](https://eprint.iacr.org/2026/1809).
- The block-projection verifier identity and union bound: [PikkuFold Lemma 9 pp. 15–16, p. 24](https://eprint.iacr.org/2026/1809).

### Inferences
Minimal prototype that would validate it (no cargo commands were run for this note):

1. **Measure first** (roadmap §2 item 4): einsum output magnitude histogram per op on GPT-2 and
   Qwen from the tracer, to fix w per op and hence D_y; and a census of rows by width to compute
   M_rows and the n needed.
2. **Standalone relation test** in `jolt-atlas-core/src/onnx_proof/testing/`: take one virtual
   einsum output V (GPT-2 MLP: m = 16 tokens, k = 3072, or a synthetic m = 64 × k = 1024), squeeze
   R from the transcript after the witness batch, compute Y, commit its digits through the existing
   one-hot range-check path as a second batch, and implement the Q5 relation as a `Relation` with
   `type Spec` bound to the op (degree 2, eq(ρ_x,x)·R̃(ρ_u,z) weight, batched with the op's own
   output claim). Verifier side: R̃(ρ_u, r_z) by the add-only evaluation.
3. **Soundness sweeps**: a `witness_tamper_sweep` site that sets one entry of V to 2^20 and to
   q − 1 (both must be rejected); a `mutate::Site` for the Y digits; an opening forgery at the Y
   accumulator; a negative test that draws R *before* the remainder commitment and shows the
   kernel-steering attack exists (documenting the ordering precondition as a test).
4. **Numbers to collect**: committed digits before/after on the chosen op; commit time of the Y
   batch vs the removed 8 overflow digits; verifier time for R̃ at k = 768, 3072, 4096; proof size
   delta; and the completeness margin (max |Y| / T over the zoo) under the probabilistic and ℓ1
   thresholds.
5. **Decision rule**: adopt if the measured clamp-digit commitment time on wide rows drops by the
   predicted ≥ 9× and the sumcheck overhead is below the saved commitment time; keep exact digits
   on rows narrower than n·D_y/8.

Risks that the prototype would surface but the papers cannot answer: the real cost of a second
Akita commitment batch in the opening reduction, and whether D2's compile-time bounds hold on the
real models (if the honest clamp engages anywhere, that op must keep exact digits).

### Gaps
- No prover-time measurement exists on Fp128 + Akita for either the current clamp digits or a
  second commitment batch; all cost statements here are digit counts.
