# Celer (ePrint 2026/1453) and "A Separation Principle for Lookup-Based zkML" (ePrint 2026/1390): full readings and what they change in our docs

Both PDFs were read cover to cover (1390: 39 pages incl. Appendix A and references; 1453: 25 pages), then
re-read at the constructions, theorems, cost tables and experiments. Page numbers are the PDF's printed page
numbers. Local files: `/Users/kg/swe/icme/official/jolt-atlas/papers/2026-1390.pdf`,
`/Users/kg/swe/icme/official/jolt-atlas/papers/2026-1453.pdf`. Statements marked **[paper]** are the paper's;
**[inference]** is mine; **[repo]** is what the jolt-atlas git history / tree shows.

Headline findings, before the detail:

1. The separation paper did not measure the backend we have now. Every Jolt Atlas number in it is from the
   **July 2026 HyperKZG/BN254 prover on Apple Silicon** (its PR #258 was merged 2026-07-07, before Fp128 + Akita
   and before the atlas-trace rewrite). Its *theorem* (Prop. 5) transfers; its *constants* do not.
2. Its own falsification experiments **contradict our "committed digits are the cost" reading of it**: removing
   an entire one-hot family (33 → 17 commitments on the rsqrt node) moved the per-lookup slope only to 0.90×, and a
   1,400× collapse in address entropy moved it ~6% (p. 21). On that backend the per-lookup cost was dominated by
   ∝ m sumcheck work, not by commitment. Whether that still holds on Akita is exactly what our per-op profile must
   decide; the roadmap's digit-count savings table assumes it does not.
3. Celer is a **grand-product (Quark) lookup with a Pippenger-style multiplicity side**, not a LogUp. It takes the
   query as an *oracle* (a single evaluation claim), so it **does work on a virtual key**. Its soundness error is
   **O(m·n/|F|)** (Thm. 2, p. 18), i.e. log₂(m·n) bits lost: **about 84 bits at m = 2²⁸, n = 2¹⁶ over a 128-bit
   field**, worse than LogUp's ≈ log₂(m+n). The PIOP is PCS-agnostic; Dory is chosen only to open the committed
   query sublinearly, which a virtual key makes unnecessary. No ZK construction and no public code are given.
4. The upstreamed change is real: **jolt-atlas PR #258, commit `f5214237e`, merged 2026-07-07** (scale-aware
   SatDiff range check, −5 one-hot polynomials per softmax node as measured, −4 as merged; prover −2.6 % to −6.0 %
   on four models).

---

## Q1. The separation principle: exact statement, proof, scope

### Takeaway
Proposition 5 says the Shout prover's *per-lookup* cost is a function of the access multiset alone, never of
table values; Corollary 3 turns that into an Ω(m) commitment floor. The only per-lookup lever it leaves is the
committed address width d = ⌈bits / log₂ k_chunk⌉. It is stated for the one-hot read-address object with d digit
chunks and covers prefix-suffix tables through the per-proof term; it is explicitly "architecture-relative (Shout
one-hot; Twist's read–write accounting unanalyzed)".

### Cited findings
- **Statement [paper, p. 20, Prop. 5]:** "In a Shout-style lookup argument the prover's per-lookup cost is a
  function of the access multiset {aᵢ}ᵢ∈[m] alone, independent of the table values {T[a] = f(a)}. Consequently two
  activations f ≠ g evaluated on the same quantized inputs incur identical per-lookup cost, and any transformation
  of f that keeps the lookup acts only on the per-proof term a(r), where it already receives maximal credit (one f̃
  evaluation amortized over all m lookups). It has zero leverage on the slope b."
- **Proof [paper, p. 20]:** "The committed read-address object is the one-hot polynomial family ra whose ∼d·m
  nonzeros are determined by {aᵢ} alone; the values vᵢ never appear in it. They enter only through the read-check
  sumcheck Σₐ f̃(a) ra(i,a) = vᵢ, which references f̃, a per-proof object." Empirical premise check: GELU-like vs
  1/√ on identical inputs "produce identical address arrays (27,347 distinct addresses, f-independent)."
- **Node cost kernel [paper, p. 7 and p. 18]:** K̃(r) = κ_addr·d·m + κ·c·2^{r/c}, "dense is c = 1, 2^r";
  prefix-suffix is c·2^{r/c}, degree 2; "with no grand-product / offline-memory-checking tax (absent from the
  source)". Written on p. 18 as K(f, r) = a(r) [per-proof table term κ c 2^{r/c}] + b·m [per-lookup address term
  κ_addr d m].
- **Corollary 3 [paper, p. 20, proof p. 37]:** "Proving m activation evaluations requires Ω(m) commitment work
  (m·d one-hot nonzeros in Shout). Function structure amortizes the shared table, never the m per-operation
  commitments; a sub-Ω(m) prover requires either fewer operations (a model change) or a commitment scheme that
  exploits access-pattern compressibility (a different substrate)."
- **The one remaining dial [paper, p. 21]:** "the address width d = ⌈bits/log₂ k_chunk⌉, not the table's values but
  the committed range of its address space, which must cover the values the computation actually reaches."
- **Multiple digit chunks:** the kernel is written with d chunks explicitly (∼d·m nonzeros, p. 7, p. 20, p. 21);
  the SatDiff win is "d drops 8 → 3" at 4-bit chunks (p. 22).
- **Prefix-suffix tables [paper, p. 12, EXP-4d]:** "Jolt uses two orthogonal lookup pathways: a dense path
  (neural-teleport tables for sigmoid/tanh/erf) whose table-MLE is materialized, costing O(2^r) with measured
  per-bit slope 1.0 (c ≈ 1, matching halo2) up to a compute wall near r ≈ 28; and a prefix-suffix path (ReLU/range)
  whose cost is polynomial in the log-table-size: a logical 2⁶⁴-entry table proves in 82 ms, where the dense path
  already needs 8 s at 2²⁸." m-sweep: "at fixed r = 24, growing m by 1024× grows the dense-table cost gap by only
  1.12× (per-lookup would demand 1024×)".
- **Falsification channels [paper, p. 21]:** EXP-B3 (commitment-count): "collapsing the 1/√ range checks 2 → 1
  removes an entire one-hot family (33 → 17 commitments) and 39% of the per-proof constant, yet the slope moves
  only to 0.90×." EXP-B4 (pure sharing): batching the two range-check sumchecks, "33 → 33; b(batched)/
  b(independent) = 0.993 ... mechanically, each round still folds every instance's length-m polynomial before
  combining messages, so the ∝ m work is untouched." Entropy: "driving the address multiset degenerate (distinct
  addresses 11,499 → 8, a > 1400× collapse) moves b by only ∼6%". EXP-B5 (substitution): "replacing the lookup
  itself by the holonomic Horner circuit (leaving Shout, not retabulating) does move the constant, b(poly)/
  b(dense) = 0.40×, while the order Θ(m) stands".
- **Scope [paper, p. 21, p. 24]:** "the principle is architecture-relative (Shout one-hot; Twist's read–write
  accounting unanalyzed), and the EXP-B4 slope ratio is a point estimate (n = 4 sizes; the formal CI is wide)".
- **Cost = bipartition rank [paper, p. 19–20, Props. 3–4]:** the per-proof table cost of a node is Θ̃(ρ·2^{r/2})
  for a balanced two-way cut, ρ = rank of the rearranged value table; "an addition theorem gives an outer product,
  ρ = 1 (checked: the r=16 exp table has σ₂/σ₁ = 1.6×10⁻¹⁵; this is the deployed hi/lo split)".

### Inferences
- The proposition is about *values*, not *width*. Our docs compress it to "cost depends on the address width
  only", which drops two things the paper itself keeps: the per-proof table term κ c 2^{r/c} (negligible for
  prefix-suffix and for 2¹⁸ dense tables, seconds at 2²⁸), and the ∝ m sumcheck work that EXP-B3/B4 show dominated
  the slope on the July backend.
- The "∼d·m" accounting lumps Booleanity, Hamming-weight and ra-virtualization sumchecks into κ_addr. EXP-B3 is
  the strongest evidence in the paper on how those split: removing half the committed families of the rsqrt node
  (which kept its 64-bit `UnsignedLessThan` read-raf and degree-17 ra-virtualization) saved only 10 % of per-lookup
  time. That is consistent with our roadmap B1 (the read-raf, not the digits, is rsqrt's cost), and inconsistent
  with reading our 25 s / 40 s stage split as "digits are the cost" without a per-op profile.
- Corollary 3's second exit, "a commitment scheme that exploits access-pattern compressibility", is exactly what a
  multiplicity-based lookup (Celer, LogUp-GKR) is: it commits O(n) multiplicities instead of d·m addresses. The
  paper's own entropy measurement ("only attention access patterns are strongly compressible, effective keys
  5.8 ≪ 512") says where that pays most: attention, i.e. our C3.

### Gaps
- The paper never states κ_addr or κ numerically for Jolt Atlas, nor the measured split between commit and
  sumcheck per lookup; only the slope ratios above.
- Whether EXP-B3's 0.90× holds on Akita is unknown; Akita's commit cost per nonzero (our ≈ 90 ns figure) is a
  different regime from HyperKZG MSMs.

---

## Q2. What was measured on Jolt Atlas and EZKL

### Takeaway
Jolt Atlas was built from source in **July 2026** (HyperKZG over BN254, Apple Silicon, `--release`, K = 3
medians) and measured on: a depth sweep of ReLU MLPs, a dense-table r×m sweep (sigmoid/tanh/erf), the
prefix-suffix range path, the softmax hi/lo exp node (N and r sweeps), rsqrt range checks (the three falsification
channels), an attention-only export at N_seq ∈ [8,128], and four GPT models for the SatDiff PR. EZKL 23.0.5/halo2
(KZG) was measured on the same ONNX MLPs and a sigmoid table sweep. Einsum clamps and rescale remainders were not
analysed as separate protocols.

### Cited findings
- **Environment [paper, p. 9]:** "PyTorch 2.12 (MPS), EZKL 23.0.5, Jolt Atlas (Rust, stable toolchain)";
  "gemma-2-2b ran at a third of the token budget on CPU" (p. 11). **[repo, PR #258 body]:** "Measurements (Apple
  Silicon, `--release`, K=3 medians, same-session build pairs; pre-fused main @ `8fcd92c`" and "5 fewer HyperKZG
  commitments in the proof", so the PCS was HyperKZG/BN254.
- **Depth sweep [paper, p. 11, Table in §8.3]:** ReLU MLPs of depth L ∈ {2, 4, 8, 16, 32}. Jolt Atlas (Twist &
  Shout): prove_ms ∼ L^0.95, affine 13.98 L + 6.67 (R² = 0.999), proof size ∼ O(L), 18 → 252 KB. EZKL 23.0.5
  (halo2 KZG): circuit num_rows ∼ L^1.05, proof ∼ O(1) (20.1 KB), "bits → cost: scale 2–12 flat".
  Caveats listed by the paper: "num_lookups is not directly instrumented in Jolt Atlas, so prover time is the cost
  proxy".
- **Dense table kernel [paper, p. 12, EXP-4a, EXP-4d]:** halo2 sigmoid: log₂(prove_s) = 1.024 r − 16.5
  (R² = 0.9976), build wall at logrows ≈ 24–26 on 48 GB. Jolt dense path: "8 s at 2²⁸", slope 1.0 bit/bit;
  prefix-suffix 2⁶⁴ logical table in 82 ms; m-sweep 1024× → 1.12×.
- **Softmax [paper, p. 12–13, EXP-5]:** "implemented by Jolt not as one dense table but as a hi/lo 2-way split with
  an explicit degree-3 multiplication"; 36 points; "the 2-way split makes the table term 2·2^{r/2} so small that
  total cost rises only 0.05 bits per bit of r, buried under a fixed ∼50 ms overhead. The sequence axis N carries a
  per-element term with slope b ≈ 0.021 ms/element, r-independent and linear through N = 2¹⁶ with no knee ...
  Since 0.021 ms is ≈1.4–1.5× a single tanh lookup". So a tanh lookup ≈ 14–15 µs per element on that setup
  **[inference from the ratio]**.
- **Rsqrt range checks [paper, p. 21, EXP-B3/B4/B5; p. 19]:** 33 → 17 commitments when collapsing 2 range checks to
  1; "native 1/√ is 1.28× more expensive (its two range checks carry 89% of the cost), flipping to 0.81–0.96×
  once the redundant check is collapsed 2 → 1".
- **Attention block [paper, p. 24, EXP-B16]:** "attention-only export, N_seq ∈ [8,128], K=3: prove-time log-log
  slope rises monotonically 0.30 → 1.37 (large-N_seq OLS 1.16) toward the ∝ N²_seq commitment-length axis, while
  the committed one-hot count stays N_seq-invariant (96 polynomials, softmax 32 = 33 % by count) ... verifier
  sub-linear (26 → 98 ms)".
- **SatDiff PR models [paper, p. 22; repo PR #258 table]:** microgpt (1 softmax node) 395.3 → 371.7 ms (−6.0 %),
  minigpt (2) 916.9 → 872.3 ms (−4.9 %), nanoGPT (4) 2391.2 → 2328.8 ms (−2.6 %), GPT-2 (12 softmax nodes,
  `test_gpt2` workload) 10855.5 → 10351.2 ms (−4.6 %); one-hot commitments 121 → 116 / 209 → 199 / 433 → 413 /
  1235 → 1175.
- **Protocols analysed:** Shout one-hot read-check with per-proof table MLE (dense), prefix-suffix range/ReLU
  path, softmax hi/lo exp split with its five one-hot families incl. SatDiff, rsqrt's two range checks. Not
  analysed: einsum rescale remainders and output clamps (the PR body mentions "fused i64 arithmetic from #256" only
  as a rebase dependency).
- **The paper cites jolt-atlas itself as** "Benno, Centelles, Douchet, Gibran. Jolt atlas: Verifiable inference via
  lookup arguments in zero knowledge. arXiv:2602.17452 (ICME-Lab / NovaNet)" (p. 38, ref. [7]).

### Inferences
- The July GPT-2 workload proved in ≈ 10.4 s on Apple Silicon with HyperKZG, versus the 40.6 s GPT-2 figure in our
  roadmap (M3, Akita). The workloads are almost certainly different (sequence length, model variant), so no
  conclusion about Akita vs HyperKZG follows; but the two numbers should be reconciled on one workload before
  either is quoted as "GPT-2".
- The paper's per-element figures (tanh ≈ 14–15 µs, softmax 21 µs) give a prior for our census: at ≈ 90 ns per
  committed nonzero, 13 one-hot digits for a standalone tanh cost ≈ 1.2 µs of commitment, under a tenth of the
  measured per-element cost on that backend. Either the sumcheck side dominated per lookup in July, or the Akita
  switch inverted the split. Only the per-op profile distinguishes these.

### Gaps
- No hardware model (which Apple chip, thread count) is given in the paper; the PR says only "Apple Silicon".
- No per-stage (witness / commit / sumcheck / opening) breakdown is reported for Jolt Atlas.

---

## Q3. The "address-width reduction upstreamed": PR #258

### Takeaway
It is jolt-atlas PR #258 (commit `f5214237e`, merged 2026-07-07, author `pdezk`): the attention-mask sentinel was
moved from ≈ −2³⁰ to a scale-indexed −C(ν)·2^ν, and the softmax SatDiff range-check width from a global 32 bits to
log₂(scale) + 8 (the paper measured the +4 form). Measured −5 one-hot polynomials per softmax node, prover −2.6 % to
−6.0 % on four models, outputs bit-identical.

### Cited findings
- **[paper, p. 21–22]:** "the saturation-gap family SatDiff was range-checked at a global 32 bits, while the
  honest saturation gap measures 833 (10 bits) across the shipped models; the width existed solely to cover an
  attention-mask sentinel (−∞ quantized to ≈ −2³⁰) ... Re-deriving the sentinel one float unit above the exp
  round-to-zero score bound C(ν) = ⌈(ν+1) ln 2⌉+1 ... the resulting sentinel value is −2048 and the range-check
  width ν + 4 bits at ν=8 ... d drops 8 → 3, removing 5 of the 16 committed one-hot polynomials per softmax node on
  the ∝ N²_seq axis."
- **[paper, p. 22, footnote 2]:** "https://github.com/ICME-Lab/jolt-atlas/pull/258 (with the measurement record;
  merged 2026-07-07). The merged form widens the margin to ν+8 (d 8 → 4, −4 polynomials per node); all
  measurements reported here are at the as-submitted ν+4 form (d 8 → 3, −5 per node)".
- **[paper, p. 22]:** soundness argument: "the degree-3 complementary-slackness sumcheck that pins the unique
  decomposition never references the range-check upper bound (the check load-bears only non-negativity/
  index-validity), and the completeness boundary is exact; across 12 prover runs in one build, verification passed
  iff the witness fit the width (4095/4096 accept/reject at ν=8)".
- **[repo]** `git log`: `f5214237e perf(softmax): scale-aware SatDiff range check — shrink from 32 bits to
  log₂(scale)+4 (#258)`. `gh pr view 258`: mergedAt 2026-07-07T12:54:24Z, author `pdezk`, files
  `atlas-onnx-tracer/src/utils/quantize.rs`, `jolt-atlas-core/src/onnx_proof/ops/softmax_last_axis/{mod.rs,rc.rs}`,
  `jolt-atlas-core/src/onnx_proof/witness.rs`, `jolt-atlas-core/src/onnx_proof/zk.rs`. Body: "Update 2026-07-07:
  rebased onto fused-i64 main (post #256 / #265 / #266), budget widened to `log_scale + 8`" because "re-measured
  max sat_diff at scale 8 is 13,773 (whole-model GPT-2 e2e) / 13,825 (BGE-small e2e), past the original
  `log_scale + 4` (4096) budget".
- **[repo, today's tree]:** the sentinel survives as `mask_sentinel_magnitude(scale)` in
  `/Users/kg/swe/icme/official/jolt-atlas/atlas-onnx-import/src/quantize.rs` (pinned tests: 8.0 at scale 8, 11.0 at
  scale 12). No `SatDiff`/`sat_diff` identifier exists anywhere in `jolt-atlas-core`, `jolt-atlas-verifier` or
  `atlas-trace` after the tracer rewrite (#382); the softmax op directory now holds `significance_clamp.rs`,
  `exponentiation/`, `exp_sum.rs`, `max.rs`, `recip_mult.rs`, `rc.rs`.

### Inferences
- The roadmap's D2 sentence "A scale-derived width already landed in jolt-atlas #258" is correct. The paper's
  lesson is more general than the PR: every one-hot family's encoded width should be audited against the value
  range the computation reaches, which is the roadmap's §2 item 4 (value histograms) extended to range-check
  widths.
- Because the SatDiff family was restructured in the rewrite, the −4/−5 polynomial saving is not necessarily
  present in today's softmax layout in the same form; the digit census should confirm the current softmax digit
  count (roadmap table says 22) against the paper's 16 (July) and 11–12 (post-PR).

### Gaps
- Whether today's `significance_clamp` width is scale-aware in the same way was not checked (out of scope; no
  cargo runs).

---

## Q4. The paper's recommendations; approximate range checks, dense vs one-hot digits, table fusion, field size

### Takeaway
Its recommendations: (i) audit and narrow committed address widths to the reachable range; (ii) do not expect
activation structure to cut per-lookup cost under Shout; (iii) the only ways below Ω(m) are fewer operations or a
commitment that exploits access-pattern compressibility; (iv) attention is Θ(N²_seq) for any sound prover of
standard softmax, with only constant-factor room. It says nothing about approximate range checks, dense digit
columns, or field size beyond Prop. 4's remark; on table fusion it only observes that the hi/lo exp split's table
term is "essentially free".

### Cited findings
- **Address width [paper, p. 21]:** quoted in Q3; "this is the separation principle's positive contrapositive".
- **Structure moves constants only [paper, p. 19, p. 21–22]:** "at deployed fidelity these constructions sit
  within an O(1) factor of the dense table, on both sides of unity: native 1/√ is 1.28× more expensive ... flipping
  to 0.81–0.96× once the redundant check is collapsed 2 → 1 (EXP-B3); holonomic erf is ∼0.56× (EXP-B2), and
  0.27–0.33× with the odd-symmetry rearrangement, halving its commitment count 19 → 11 (EXP-B5)". And: "structure
  buys feasibility and constants, the floor buys the allocation, and nothing on the function axis moves the Θ(m)
  address term."
- **Holonomic route described as [paper, p. 19, Prop. 2]:** "a differentially-algebraic activation satisfying a
  polynomial ODE (tanh, sigmoid, SiLU ...) admits an ODE-certified degree-p(ε) polynomial evaluation, table-free and
  range-check-free, Θ(p·m)"; "the holonomic route for erf is table-free with zero range checks and ε(p) certified
  by the ODE (EXP-B2)".
- **Exits below Ω(m) [paper, p. 20, Cor. 3]:** "fewer operations (a model change) or a commitment scheme that
  exploits access-pattern compressibility (a different substrate)."
- **Attention [paper, p. 14–17, Thm. 3, Cor. 2, §8.5]:** "C cannot compress honest-input attention below
  o(N²_seq)"; content-selected top-k "incurring a prover-work obligation: certifying that the N_seq − k omitted
  scores per query are not larger touches all keys ... Ω(N²_seq) certified comparisons in total"; "The only escape
  is approximate, economically-sound selection with fixed-dimension preprocessing (LSH/ANN)". Also p. 17: "a sound
  protocol can certify entry bounds first and route to a certified sub-quadratic approximation only when the input
  is provably on the easy side".
- **Compressibility of attention addresses [paper, p. 21]:** "only attention access patterns are strongly
  compressible, effective keys 5.8 ≪ 512 ... a floor no one-hot scheme realizes."
- **Table fusion [paper, p. 13, EXP-5]:** the hi/lo split "makes the table term 2·2^{r/2} so small that total
  cost rises only 0.05 bits per bit of r".
- **Field size [paper, p. 20, p. 36]:** only "over a cryptographic field |F| ∼ 2²⁵⁶ it is overwhelming" in the
  generic-table rank argument. Nothing on challenge-field soundness.
- **Approximate range checks, dense digit columns:** not mentioned anywhere in the paper.

### Inferences
- The "range-check-free" holonomic erf claim is the paper's; it does not explain how the polynomial output is
  requantized to a unique integer, which our behavioural contract requires. Treat its 0.56× / 0.27–0.33× as a
  measurement of *their* harness, not as a drop-in for our exact-semantics activations.
- Its attention result weakens our research-directions §7 hope for sub-quadratic attention via "most pairs are
  exactly zero after the exp cutoff": proving a pair is below the cutoff is itself a comparison per pair, so the
  pair count stays Θ(N²); the room is the per-pair constant (no exp lookup for cut pairs), which is C3's territory.

### Gaps
- The paper has no view on FlashProve-style approximate (LNS) range checks or on Akita; both post-date it.

---

## Q5. Celer: construction, argument family, degree/rounds, verifier cost, soundness

### Takeaway
Celer proves the lookup a ⊆ T by checking the polynomial identity h(X) = ∏ᵢ(X − aᵢ) = g(X) = ∏ᵢ(X − Tᵢ)^{eᵢ} at
one random point r. h(r) is a Quark grand product over the m query values (commits 2m/log³m packed elements, the
leaves r − aᵢ are never committed, and the protocol ends in one claim on ã(r′)). g(r) is a product of powers proved
through a Pippenger decomposition of the multiplicities into w base-s digits, committing O(w·(n + s)) elements, with a
small LogUp only inside the "bucket boundary" step. Prover: m/log³m + O(m^{1/w} + n) committed elements and
10m + O(m^{1/w} + n) field ops; verifier O(log m · log log m) field ops plus O(log m) pairings; proof
O(log m · log log m). Soundness error O(m·n/|F|); completeness error O((m^{1/w} + n)/|F|).

### Cited findings
- **Relation [paper, p. 2, eq. (1)]:** R_T = {(x = [[a]], w = a) : ∀x ∈ {0,1}^{log m}, a(x) ∈ T}, "where [[a]]
  represents oracle of multilinear polynomial a(·)". Footnote 3 (p. 3): "Existing lookup arguments all treat the
  commitment of the query vector as a public input. Consequently, their analyses exclude the cost of committing to
  the queries. We follow this convention in this paper."
- **Core idea [paper, p. 8, §3; p. 12, §4]:** "Celer avoids this overhead by directly proving the original
  polynomial equivalence between g(X) and h(X), which reduces to checking their evaluations at a random point r."
  LogUp's cost "stems from the need to commit to m reciprocals of the form 1/(X − aᵢ)" (p. 8).
- **Grand product used [paper, p. 7–8, §2.3]:** Quark [SL20]: packs the last log m − 3 log log m layers of the
  binary product tree into one committed F, checks the layer recursion with one sumcheck
  Σ_y ẽq(β, y)(F(1, y) − F(y, 0)·F(y, 1)) = 0, then peels the remaining 3 log log m layers by GKR-style sumchecks
  until "reaching a claim on Ṽ₀(·), which can be verified by opening the commitment to K̃". Cost: "m/log³m G + O(m)
  F, a verifier cost of O(log m log log m) F, and a proof size of O(log m log log m)".
- **Multiplicity side [paper, p. 9–15, Alg. 1, Fig. 1–2]:** eᵢ = Σⱼ s^{j−1}·E_{j,i}, s = ⌈(m+1)^{1/w}⌉; for each
  digit j the prover commits U_j (table T re-ordered by bucket, length n) and p_j (bucket start indices, length
  s − 1) **before** r, then D_j (suffix products of r − U_j, length n) and S_j (length s − 1) after r. Relations:
  (1) D_{j,i} = ∏_{ℓ≥i}(r − U_{j,ℓ}) via the shifted-vector identity Σᵢ αⁱ D_{j,i} = Σ_t B_t D_{j,t}(r − U′_{j,t})
  proved by sumcheck over 2n with the verifier evaluating B̃ in O(log n); (2) S_{j,v} = D_{j,p_{j,v}} via an
  indexed LogUp of query size s − 1 against table size n; (3) R_j = ∏_v S_{j,v} and "U_j is a permutation of
  {T₁,…,Tₙ}" by grand products. Then g(r) = ∏ⱼ R_j^{s^{j−1}} (verifier checks). Implementation uses w = 2 (p. 17).
- **Theorem 1 [paper, p. 17]:** "prover work: committing to m/log³m + O(m^{1/w} + n) field elements, and performing
  (10m + O(m^{1/w} + n)) F; proof size: O(log m log log m); verifier time: O(log m log log m)". p. 12 adds the
  verifier's "O(log m) P" (pairings) and that proving h(r) alone "requires committing to m/log³m field elements and
  O(m) inexpensive field operations". Footnote 5: "a sumcheck protocol with degree = 3 on a 2^{log m}-size hypercube
  consumes 10m field operations".
- **Theorem 2 [paper, p. 18]:** "Celer achieves completeness error O((m^{1/w} + n)/|F|) and soundness error
  O(m·n/|F|)." Proof sketch: "The soundness of Celer hinges on ensuring that each R_j is computed as the product of
  s − 1 entries from the suffix product vector S_j, which bounds the degree of g(r) by O(m·n). Since U_j and p_j are
  committed before the challenge r is sampled, the polynomial g(·) is fixed in advance. Soundness then follows from
  the Schwartz–Zippel lemma. Notably, although Celer does not enforce the well-formedness of p_j, the resulting
  soundness error remains negligible." Full proof (p. 19): e′ᵢ ≤ s^w − 1 ≤ m + Θ(m^{1−1/w}), "the degree of
  polynomial g(X) = ∏(X − Tᵢ)^{e′ᵢ} is at most O(m·n)"; union-bounded with the sumcheck terms O(log n/|F|) and the
  grand-product terms O(log m log log m/|F|).
- **Table 1 [paper, p. 5]:** committed elements / field ops: plookup 5m / O(m log m); LogUp m + 2n / 12m + 12n;
  logUp-GKR n / 43m + 43n; Lasso 3m + m/log³m + n / 20m; Celer m/log³m + O(m^{1/w} + n) / 10m + O(m^{1/w} + n).

### Inferences
- Soundness loss in bits is log₂(m) + log₂(n) (+ small): at m = 2²⁸, n = 2¹⁶ over a 128-bit field that is
  ≈ 128 − 44 = **84 bits**; LogUp's rational identity has degree ≈ m + n, losing ≈ 28 bits (≈ 100 left), which is
  what our roadmap D4 states. So Celer is *less* tight than LogUp per challenge, by log₂ n bits, because a cheating
  prover's digit sums are not constrained to Σ e′ᵢ = m. Enforcing Σᵢ e′ᵢ = m (or committing E_j and checking the
  digit recomposition) would recover O((m + n)/|F|); the paper does not do this.
- Rounds: the Quark part runs log m − 3 log log m variables in the packed sumcheck (degree 3) plus 3 log log m
  layered sumchecks of decreasing size; the multiplicity side adds O(w) sumchecks of size O(n) and the small LogUp.
  Degree 3 throughout; nothing of our degree-9/17 ra-virtualization kind.
- "Sublinear in m" means the *committed* object is m/log³m: at m = 2²⁸ that is ≈ 2²⁸/21,952 ≈ 12k packed field
  elements (2m/log³m ≈ 24k); at m = 2²⁰ it is ≈ 130–260 elements. The prover's time is then the 10m field
  operations, not commitment.

### Gaps
- The paper does not give the exact constant hidden in O(m·n); the degree bound n·(m + Θ(m^{1−1/w})) is the
  concrete statement.
- No explicit round count is tabulated.

---

## Q6. The Llama benchmark: m = 2²⁸, n = 2¹⁶

### Takeaway
Single-threaded, BN254 with Dory, Intel Xeon Platinum 8460Y+: Celer 257 s vs logUp-GKR 604 s, Lasso 986 s,
LogUp 2402 s, plookup 6655 s; verifier ≤ 0.11 s, proof ≤ 248 KB. That is ≈ 0.96 µs per lookup single-threaded,
with the query commitment excluded.

### Cited findings
- **Setup [paper, p. 20, §5.1]:** "Our implementation uses the BN254 elliptic curve, which provides 128-bit
  security. We develop our lookup protocols in C++ language using the mcl library." "All the experiments are
  conducted on a server with 256 GB memory and Intel(R) Xeon(R) Platinum 8460Y+ 64 CPU cores. We always use a single
  thread to execute the experiments." Baselines: Lasso's official implementation (a16z/jolt `lasso` branch, Hyrax
  PCS); plookup, LogUp, logUp-GKR implemented by the authors.
- **Workload motivation [paper, p. 3]:** "zkLLM (CCS'24) shows that a single inference of LLaMA-7B on a 512-token
  input evaluates GeLU more than 2×10⁸ times, yielding m ≈ 2²⁸ lookups—about three orders of magnitude larger than
  the table (n = 2¹⁶)".
- **Numbers [paper, p. 21]:** "under m = 2²⁸ and n = 2¹⁶, Celer achieves a prover time of only 257 seconds, while
  logUp-GKR requires 604 seconds, Lasso requires 986 seconds, LogUp requires 2402 seconds, plookup requires 6655
  seconds. This means that Celer outperforms all of the baselines by 2.4 ∼ 25.9× on prover time." "the prover
  time of Celer grows almost linearly with m ... bottlenecked by the grand product check protocol plus the opening
  of Dory PCS, costing O(m) F." Verifier "≤ 0.11 seconds" (pairing-dominated); proofs "at most 248 KB"; Lasso's
  Hyrax "incurs O(√m) overhead in both metrics".
- **Abstract/intro ratios [paper, p. 1, p. 4]:** 25.9× over plookup, 9.4× over LogUp, 3.8× over Lasso, 2.4× over
  logUp-GKR.

### Inferences
- Per lookup: 257 s / 2²⁸ ≈ 957 ns single-threaded on a server core, BN254 scalar field (≈ 20–30 ns per
  multiplication). On Fp128 (two-limb Solinas, several times cheaper per multiplication) and on 8–12 threads the
  same protocol would plausibly land in the tens of ns per lookup; this is an extrapolation, not a measurement.
- Our one-hot lookup on a 16-bit key costs 4 committed digits ≈ 4 × 90 ns ≈ 360 ns of Akita commitment per lookup
  *plus* Booleanity, Hamming-weight/ra-virtualization and read-check sumchecks over those 4 digit polynomials, and
  the per-proof table term. Celer replaces all of that with ≈ 10 field multiplications per lookup, an O(n) side
  computation and a commitment of 2m/log³m + O(n + √m) full-width field elements. The paper's own LogUp-GKR
  baseline (n committed multiplicities, 43m field ops) is the simpler cousin at 2.4× the field work.
- The comparison most relevant to us is not the 2²⁸ headline but the small-m regime of a short-context GPT-2,
  where m per table is 2¹⁶–2²⁴: Celer's Pippenger trick then buys little over committing n multiplicities
  directly, and the choice between Celer and logUp-GKR is a constant factor on 10m vs 43m field operations.

### Gaps
- No multi-threaded numbers, no per-component (grand product vs Dory opening) breakdown, no memory figures.
- The query vector's commitment is excluded by convention; for a committed key this is a real extra cost on any
  PCS (irrelevant for a virtual key).

---

## Q7. Virtual keys; soundness at 2²⁸ over 128 bits; 192-bit challenges

### Takeaway
Yes: Celer's input is oracle access to ã, the h(r) grand product reduces to a single claim ã(r′), and the leaf
values r − aᵢ are never committed. A key defined by a linear combination of existing evaluation claims (FlashProve's
low-16-bit remainder, our fused-rescale remainder acc − q·2^s) fits exactly, provided the key is fixed by
commitments made before r. At m = 2²⁸, n = 2¹⁶ the paper's bound leaves ≈ 84 bits over a 128-bit field; a 192-bit
extension challenge would restore ≈ 148.

### Cited findings
- **Oracle input [paper, p. 16, Protocol 1]:** "Input: Oracle access to ã : F^{log m} → F, the multilinear extension
  of query vector a". "Run grand product check protocol to check h(r) = ∏ᵢ(r − aᵢ). The protocol reduces to prover's
  claim on ã(r′), which is verified by V's oracle access to ã."
- **Fixing before r [paper, p. 19]:** "Note that both e′ and a are fixed by their commitment before the verifier
  samples r. So if a ⊄ T, the two polynomials ... are distinct polynomials, and by the Schwartz–Zippel lemma, they
  can agree at most with probability O(m·n/|F|)."
- **Soundness bound [paper, p. 18, Thm. 2]:** O(m·n/|F|). Completeness error O((m^{1/w} + n)/|F|) from the inner
  LogUp.
- **Our doc's FlashProve row [research-directions §3.2]:** "Nonlinearities: Grand-product lookup through GKR
  product trees; keys never committed, multiplicities committed as bit-planes" — the same family Celer builds on.

### Inferences
- Virtual key: the Quark tree's leaf layer is V₀(i) = r − aᵢ; the final reduction is to Ṽ₀(r′) = r − ã(r′), linear
  in the key's MLE. If ã(r′) is itself derivable from other claims (e.g. rem(r′) = acc(r′) − 2^s·q(r′) with acc a
  matmul sumcheck output and q committed), no commitment to the key is needed and the Dory-specific requirement
  ("efficiently open the committed vector a", p. 12) vanishes. The "fixed before r" condition is satisfied when
  every constituent of the virtual key is committed or public before r is drawn; in our pipeline that is the
  einsum operands and the committed quotient digits.
- The range-check reading: a Celer lookup on rem against T = [0, 2¹⁶) proves rem ∈ [0, 2¹⁶) but, like any lookup,
  not that rem is *the* Euclidean remainder; the tie acc = q·D + rem with 0 ≤ rem < D is the pair (lookup on rem,
  lookup on D − 1 − rem) or a divisor-aligned table, exactly FlashProve's "one 2¹⁶ lookup plus one small lookup".
- Soundness over Fp128: ≈ 84 bits at 2²⁸ × 2¹⁶ by Celer's stated bound; ≈ 100 bits for a LogUp/logUp-GKR identity
  of degree m + n. If a 100-bit floor is the requirement, Celer as published needs either the Σ e′ᵢ = m fix
  (inference above) or an extension-field challenge; FlashProve's 192-bit choice is the published precedent. Over
  192-bit challenges Celer would retain ≈ 148 bits.

### Gaps
- Celer gives no treatment of virtual (uncommitted) queries explicitly; the oracle formulation implies it but the
  paper's cost accounting assumes a Dory-committed ã.

---

## Q8. Commitment requirements, Akita, zero knowledge, implementation

### Takeaway
Celer is a PIOP compiled with any multilinear PCS; Dory is used because opening a committed length-m query needs
sublinear group operations. Nothing in the PIOP needs homomorphism. It could sit on Akita (committed vectors:
2m/log³m packed products, and w copies of U_j, p_j, D_j, S_j; D_j and S_j are full-width field elements). The paper
proves completeness and soundness only — no zero-knowledge construction — and names no public repository.

### Cited findings
- **PCS choice [paper, p. 12, p. 16]:** "The missing piece in the Celer lookup argument is to compile the PIOP
  using a multilinear PCS. The main bottleneck in this compilation is to efficiently prove the final random
  evaluation ã(r′). To this end, we need a multilinear PCS whose opening group cost is sublinear in m, while
  maintaining proof size and verifier time polylogarithmic in m. Fortunately, Dory PCS [Lee21] precisely provides
  these properties." Table 1 note: "Verifier work and proof size depend on the underlying PCS and are therefore not
  reported here."
- **Security theorem [paper, p. 18, Thm. 2]:** completeness and soundness only. The word "zero-knowledge" appears
  only in the framing ("zkSNARKs", p. 1) and in "ZK-unfriendly" memory operations (p. 10); no ZK compiler or
  masking is described.
- **Implementation [paper, p. 20]:** "We develop our lookup protocols in C++ language using the mcl library." No
  URL or artifact link anywhere in the paper.

### Inferences
- On Akita the cost profile changes shape: Akita's per-element cost depends on coefficient width, and D_j, S_j and
  the packed Quark F are full-width field elements, while U_j (table values) and p_j (indices) are small. At
  m = 2²⁸, n = 2¹⁶, w = 2 that is ≈ 2·(2¹⁶ + 2¹⁴) + 2·2¹⁶ + 24k ≈ 300k committed elements, versus 2³⁰ one-hot
  nonzeros for 4-digit addresses. With a virtual key, Celer on Akita commits nothing of size m.
- Zero knowledge would have to come from the compiler (masked sumchecks / hiding PCS), as upstream Jolt's Whiteout
  stack does; nothing in Celer prevents it, nothing in Celer provides it.
- No public implementation means a D4 prototype is written from the paper; Protocol 1 (p. 16) is complete enough
  to do so.

### Gaps
- Whether the authors have since released code was not checked (no web access used; the task was the PDFs).

---

## What this changes in the docs

Each entry: the quoted statement, the correction, the page.

### `docs/optimization-roadmap.md`

1. **§1:** "Table contents do not matter: in a one-hot lookup the cost depends on the address width only."
   — *Weaken and replace.* Prop. 5 (p. 20) says the per-lookup cost is independent of table **values**; it does not
   say it depends on width *only*. The paper's kernel (p. 7, p. 18) is K̃(r) = κ_addr·d·m + κ·c·2^{r/c}: a
   per-lookup address term ∝ d·m plus a per-proof table term (c = 1 dense, c > 1 prefix-suffix). And its
   falsification runs (p. 21) show the per-lookup slope b was only 10 % sensitive to halving committed families
   (EXP-B3) and ∼6 % sensitive to a 1,400× entropy collapse, so on the measured backend b was dominated by the
   ∝ m sumcheck work per instance, not by committed nonzeros. Corrected cost model below.
2. **§1:** "So the number to drive down is committed one-hot digits per tensor element. Sumcheck degree and
   instance count come second." — *Weaken until profiled.* EXP-B3/B4 (p. 21) on the July backend: removing a
   family (its commitment *and* its sumcheck instance) gave 0.90×; batching instances without removing any gave
   0.993×. The driver was the remaining instances' per-round folding of length-m polynomials (rsqrt's 64-bit
   read-raf and ra-virtualization). This *strengthens B1* (remove the read-rafs) and makes the digit-count savings
   table in §1 an upper bound on time saved, not an estimate.
3. **§1 table, "Softmax alone: 22 today":** the paper counted "16 polynomials at scale 2⁸" in five families per
   softmax node in July (p. 21), 11 after the measured PR, 12 after the merged form (p. 22 fn. 2). Our 22 is a
   different (post-rewrite) accounting; the census should reconcile them.
4. **§2:** "No per-op timing exists on the current backend." — *Still true*, but per-op timings exist on the July
   backend and should be recorded as priors: tanh lookup ≈ 14–15 µs/element, softmax 21 µs/element with a ∼50 ms
   fixed overhead, 2⁶⁴ prefix-suffix in 82 ms, dense 2²⁸ table-MLE 8 s, ReLU-MLP 13.98 ms per layer, attention-only
   export slope 0.30 → 1.37 in N_seq (p. 11–13, p. 24).
5. **§4 B3:** "This drops 10 of 22 digits and 10 of 19 relation instances." — *Strengthened on mechanism:* EXP-5
   (p. 13) measured the hi/lo split's table term as "essentially free (0.05 bits per bit of r)", so B3's gain, if
   any, is entirely from the removed digits, the degree-3 multiplication and the remainder — not from table size.
   Consistent with our note that C3 supersedes B3.
6. **§5 C3, open question "How wide z_acc is on real models":** the PR #258 record gives data: honest saturation
   gap 833 (10 bits) on shipped models; post-fused-i64 max sat_diff 13,773 (GPT-2) / 13,825 (BGE) at scale 8;
   masked gaps 429 (GPT-2) and 557 (Qwen2-0.5B) at scale 8 (PR body; paper p. 22).
7. **§6 D2:** "A scale-derived width already landed in jolt-atlas #258." — *Confirmed and made precise:* commit
   `f5214237e`, merged 2026-07-07; sentinel −2³⁰ → −C(ν)·2^ν with C(ν) = ⌈(ν+1) ln 2⌉ + 1; width 32 → log₂scale + 8
   (paper measured the +4 form); −5/−4 one-hot polynomials per softmax node; prover −2.6 % to −6.0 % (p. 22).
8. **§6 D4:** "Over a 128-bit challenge field a LogUp or grand-product argument loses about log₂N bits of soundness
   (roughly 100 bits left at 2²⁸) ... Celer (ePrint 2026/1453) targets this regime." — *Correct for LogUp,
   corrected for Celer:* Celer's stated error is O(m·n/|F|) (Thm. 2, p. 18), ≈ 84 bits left at 2²⁸ × 2¹⁶, because
   a cheating prover's multiplicity digits are not constrained to sum to m. Also *strengthened:* Celer takes the
   query as an oracle (Protocol 1, p. 16), so **a virtual key works**, and with a virtual key its only commitments
   are 2m/log³m packed products plus O(n + m^{1/w}) multiplicity-side vectors; the PIOP is PCS-agnostic (p. 12,
   Table 1 note), so it can sit on Akita. Per-lookup field work is ≈ 10 multiplications (p. 17). D4 can move from
   "research, not scheduled" to "microbenchmark candidate" (see the experiment below).
9. **§6, Dropped: "8-bit digits":** consistent with the paper's d = ⌈bits / log₂ k_chunk⌉ accounting; nothing to
   change.
10. **§9:** "Piecewise-polynomial activations, symmetric half-tables, ... none reduces committed digits" —
    *Contradicted as a blanket statement, with caveats.* The paper measured (p. 19, p. 21): holonomic (ODE-
    certified polynomial) erf at 0.56× the dense lookup, 0.27–0.33× with the odd-symmetry rearrangement, "halving
    its commitment count 19 → 11", and b(poly)/b(dense) = 0.40× (EXP-B5). Caveats: that is a *substitution* that
    leaves Shout; the paper calls it "range-check-free" without saying how the output is requantized to a unique
    integer, which our contract requires; and the win is a constant (Cor. 3). Keep the item in "not pursuing" only
    with the reason restated as exactness/uniqueness, not "does not reduce digits".

### `docs/research-directions.md`

11. **Reading-depth table:** Celer and "A Separation Principle" move from "Abstract only" to "Read in full".
12. **§1 Thesis:** "The cost of proving inference is the cost of proving values are small. On GPT-2, 25 s of the
    40 s proof is committing one-hot digits" — *Qualify.* The paper's theorem supports Ω(m) commitment work
    (Cor. 3, p. 20) but its measurements on the July backend (p. 21) found per-lookup time insensitive to committed
    family count. "25 s witness and commitment" needs splitting into witness generation vs Akita commit vs the
    lookup sumchecks before "committing digits" is named as the cost.
13. **§3.2 table, "Overflow checks: 8 exact digits per value":** the separation paper's audit shows a cheaper first
    step than any protocol change: narrowing the *encoded* width of a range check to the reachable range
    (p. 21–22). Our D1/D2 planning should include a width audit per one-hot family before the approximate check.
14. **§4:** "On lookup arguments: a LogUp or grand-product argument loses about log₂N bits of soundness when
    challenges come from a 128-bit field. FlashProve uses one anyway and escapes the loss with 192-bit challenges.
    The constraint is the challenge field, not the argument family." — *Partly contradicted.* The family sets the
    constant: LogUp ≈ log₂(m + n) bits; Celer ≈ log₂(m·n) bits (Thm. 2, p. 18). With 192-bit challenges Celer
    keeps ≈ 148 bits at 2²⁸ × 2¹⁶; over Fp128 alone ≈ 84.
15. **§6 Strategy B ceiling "perhaps 1.5–3x on GPT-2 (our estimate)":** *Weakened as an estimate from digit
    counts* by EXP-B3 (p. 21); it may be right, but the mechanism has to be the sumcheck instances removed (B1, B2,
    C2) rather than digits alone. Not changed in direction.
16. **§7 Open directions, "Sub-quadratic attention. No 2026 system has it. Most pairs contribute exactly zero after
    the exp cutoff."** — *Weakened.* Thm. 3 / Cor. 2 (p. 14–16): content-agnostic compression is blocked by a rank
    gap; content selection (which "pairs are zero" is) needs Ω(N²_seq) certified comparisons for exact soundness
    (p. 16); only approximate, economically-sound selection escapes. The honest version of our item is "lower the
    per-pair constant for cut pairs", which is C3.
17. **§5 "Other 2026 systems" and References:** add the separation paper's facts that Jolt Atlas is cited as
    arXiv:2602.17452 (p. 38) and that the paper's Jolt numbers are HyperKZG/BN254, July 2026, Apple Silicon.

### Corrected cost model for one-hot lookups (replacing roadmap §1's one-liner)

Per lookup node with m elements, address width d chunks of log₂ k_chunk bits, table of r bits, structure c
(c = 1 dense, c > 1 prefix-suffix):

    time(node) ≈ κ_commit · d · m                       (Akita, ≈ 90 ns per nonzero on our stack)
               + Σ_instances κ_sumcheck(degree) · m      (Booleanity, Hamming/ra-virtualization, read-check, clamps)
               + κ_table · c · 2^{r/c}                   (per proof, not per lookup; ms at 2^18 dense, 82 ms at 2^64 prefix-suffix)
               + κ_witness · m                           (trace, digit decomposition)

**[paper]** gives the first and third terms (p. 7, p. 18) and shows the third is per-proof (EXP-4d m-sweep, p. 12);
**[paper]** EXP-B3/B4 (p. 21) show the second term dominated the first on the July HyperKZG backend; **[inference]**
the split on Akita is unmeasured and is the first thing the census must produce. Table values enter none of the
per-lookup terms (Prop. 5). The only per-lookup levers are d (encoded width vs reachable range), the number and
degree of sumcheck instances per family, and m itself.

---

## Ranked view: what to adopt, and the one experiment

**From the separation paper**

1. **Width audit as a routine** (adopt now, zero protocol risk): for every one-hot family, compare the encoded
   range-check width with the measured reachable range on the zoo (p. 21–22 method; the PR #258 record is the
   template). Add it to roadmap §2 item 4.
2. **Its measurement method, not its numbers**: the m-sweep (per-proof vs per-lookup), the commitment-count channel
   and the pure-sharing channel (p. 12, p. 21) are exactly the three experiments that tell us whether a roadmap
   item saves time or only digits. Run them on Akita for one einsum node and one softmax node.
3. **Accept Θ(N²_seq) attention and fight the constant** (p. 14–17): C3 is the right item; drop "sub-quadratic" from
   §7 or restate it as approximate/economic selection.
4. **Treat the structured-activation route as a constant-factor substitution**, not a digit saver (p. 19, p. 21):
   low priority for us because of exact-semantics requirements.

**From Celer**

1. **D4 becomes concrete**: a Quark-style grand-product lookup on a *virtual* key (fused-rescale remainders first)
   against a 2¹⁶ range table, with multiplicities committed (n elements, logUp-GKR style) or Pippenger-decomposed
   (Celer) when m ≫ n. Protocol 1 (p. 16) is implementable from the paper. It is the only mechanism we have found
   that removes the 4 remainder digits per einsum output, and it needs no change of commitment scheme.
2. **Decide the soundness budget first**: Celer as published leaves ≈ 84 bits at 2²⁸ × 2¹⁶ over Fp128, LogUp-type
   identities ≈ 100. Either is a policy decision; a Σ e′ᵢ = m check (inference) or 192-bit extension challenges
   (FlashProve's route) closes the gap.
3. **Do not import Celer's benchmark as a speed claim for us**: 0.96 µs/lookup single-threaded BN254 is a different
   field, PCS and workload; the relevant comparison is our own 4-digit one-hot path on Akita.

**The one experiment that settles it** (no cargo in this task; proposed, not run): one micro-benchmark on the
Fp128 + Akita stack, m = 2²⁴ virtual 16-bit keys derived as rem = acc − 2^s·q from a committed q and a sumcheck-
produced acc, table [0, 2¹⁶):

- (a) today's path: 4 one-hot digits committed through Akita + Booleanity + ra-virtualization + read-check;
- (b) a Quark grand product over r − rem_i with n = 2¹⁶ committed multiplicities (logUp-GKR-style side), and
  optionally (c) Celer's Pippenger side with w = 2.

Record prover seconds split into witness / commit / sumcheck, proof bytes, and the soundness bound. This answers,
in one run, the roadmap's open question (commit vs sumcheck dominance per lookup, item 2 above) and D4's viability,
and it either validates or retires the §1 digit-count savings table.

## Sources

- Min Jun Jo, "A Separation Principle for Lookup-Based zkML: Activation-Function Structure Cannot Reduce
  Per-Lookup Proving Cost", ePrint 2026/1390, dated 2026-07-14 — local file
  `/Users/kg/swe/icme/official/jolt-atlas/papers/2026-1390.pdf`.
- W. Qu, Y. Guo, Z. Xuan, X. Liu, J. Zhang, "Celer: A Lookup Argument for Large-Scale Queries", ePrint 2026/1453
  (CRYPTO 2026 per the task brief; the PDF carries no venue line) — local file
  `/Users/kg/swe/icme/official/jolt-atlas/papers/2026-1453.pdf`.
- jolt-atlas PR #258, https://github.com/ICME-Lab/jolt-atlas/pull/258, commit `f5214237e`, merged 2026-07-07
  (read via `gh pr view 258` and `git log`).
- `/Users/kg/swe/icme/official/jolt-atlas/docs/research-directions.md`,
  `/Users/kg/swe/icme/official/jolt-atlas/docs/optimization-roadmap.md` (the statements corrected above).
