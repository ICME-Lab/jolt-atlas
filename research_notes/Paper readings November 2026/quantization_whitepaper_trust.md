# "Bits Under ZK-LLM" and the Jolt Atlas whitepaper, read in full, against our two docs

Scope as revised by the coordinator: agentic_trust.pdf was dropped (question 8 not addressed).
Both remaining PDFs were read cover to cover (Bits: 22 pages incl. appendices A–I; whitepaper:
21 pages), then the Bits paper's setup, tables and cost sections were re-read. The whitepaper is
deleted in the working tree (`git status` shows `D papers/Jolt_Atlas_Whitepaper.pdf`); it was
recovered from `HEAD` into the scratchpad for reading, nothing in the repo was changed.

Sources used throughout:

- BITS = T. Yoon, Y. Zhang, X. Liao (UIUC), "Bits Under ZK-LLM: Evaluating Zero-Knowledge-Friendly
  Quantization for Verifiable Private LLM Inference", arXiv 2609.36437v1, 29 Sep 2026.
  [arXiv](https://arxiv.org/abs/2609.36437); local
  `/Users/kg/swe/icme/official/jolt-atlas/papers/2609.36437v1.pdf`.
- WP = W. Benno, A. Centelles, A. Douchet, K. Gibran (ICME Labs), "Jolt Atlas: Verifiable Inference
  via Lookup Arguments in Zero Knowledge", undated PDF (Feb 2026 per the brief). Local, at git
  `HEAD:papers/Jolt_Atlas_Whitepaper.pdf`. No public URL in the file.
- RD = `/Users/kg/swe/icme/official/jolt-atlas/docs/research-directions.md`;
  OR = `/Users/kg/swe/icme/official/jolt-atlas/docs/optimization-roadmap.md`;
  README = `/Users/kg/swe/icme/official/jolt-atlas/README.md` (Benchmarks section).

"Stated" below means the document says it; "Inference" means my reading of what it implies for
jolt-atlas. Page numbers are the PDF's printed page numbers.

---

## Q1. Bits Under ZK-LLM: proof systems, models, bit widths, quantization schemes, hardware; measurement paper or cost model?

### Takeaway
It is a measurement paper with a two-stage method: a PyTorch quantize-dequantize (QDQ) simulator for
utility across nine models, and a modified DeepProve (sumcheck/GKR, LogUp-style lookups) for
proving cost on GPT-2 Small/Medium/Large at 16 tokens. There is no analytic cost model; the one
"Δ-ZK efficiency" definition (p.4) is never evaluated numerically. Quantization is signed symmetric
per-tensor, static-max calibrated, with float scales by default; power-of-two scales are an
ablation.

### Cited Findings
- Proof system for cost: "we map selected configurations to actual proof-system execution using
  DeepProve (Gailly et al., 2026). We extend DeepProve to independently control weight, activation,
  and nonlinear-LUT precision" — [BITS p.5–6](https://arxiv.org/abs/2609.36437). Floating-point
  baseline: Ernstberger et al. 2025 / Riasi et al. 2025 floating-point SNARK, projections only
  (p.8). The Min-K% appendix uses Plonky3 `p3-batch-stark` over BabyBear with Poseidon2 FRI, and
  SP1 as a zkVM baseline (p.20).
- Models (nine): GPT-2 Small, Medium, XL; TinyLlama-1.1B; Qwen2.5-3B, 7B, 14B; Llama-3.1-8B;
  Qwen3-30B-A3B (MoE) — Table 4 and Fig. 3, p.15–16. Proving-cost runs use GPT-2 Small, Medium and
  Large only (p.8, Table 6 p.16). The paper is explicit: "full-model proof generation is not yet
  feasible at the largest model scale" (p.8).
- Bit widths: W∈{8,10,12,14,16} with A16 fixed, A∈{8,…,16} with W16 fixed (p.6); LUT input
  precision B∈{8,12,16,24} (p.7, p.9); "Auxiliary domains, including embeddings, positional
  representations, and residual paths, are kept at 24 bits" (p.6).
- Quantization scheme: "signed symmetric per-tensor quantization with standard scales and
  static-max activation calibration over eight fixed WikiText-2 training windows" (p.6), context
  512, stride 256 (p.15). Main scale s = 2a/(2^b − 1); "standard-scale ablation" s = a/(2^(b−1) − 1);
  PoT scale s = 2^⌈log2(a/2^(b−1))⌉; c1/c2 integer-ratio scale as in zkGPT (p.15). One scale per
  weight tensor; one static scale per designated activation tensor (p.15). Asymmetric and dynamic
  per-token are ablations (Appendix G, p.18–19).
- Nonlinearities: "GeLU or SiLU activations, normalization inverse square roots, softmax
  exponentials and reciprocals … are replaced by a B-bit LUT over a calibrated input range" (p.7);
  uniform grid of 2^B entries, outputs stored as 24-bit int32, out-of-range inputs clipped
  (Appendix B, p.14). Attention score QKᵀ and softmax(S)V "remain native tensor operations;
  the evaluated policy does not insert a separate QDQ operation after each of these matrix
  products" (p.14). RoPE, masking, reductions, top-k stay in floating point in the utility
  evaluator (p.5).
- Where QDQ is inserted: "projection, normalization, activation-function, and residual-addition
  outputs" (p.5); the LM head "does not apply an additional A-bit requantization to its output
  logits" (p.14).
- Hardware: utility runs on "two Intel Xeon Gold 6526Y CPUs, 251 GiB RAM, and two NVIDIA H100 NVL
  GPUs", PyTorch 2.10.0, CUDA 12.6 (p.15). Proving runs: "context length of 16 tokens, with batch
  size one and 32 CPU threads", time "excludes graph preparation, inference, and verification"
  (p.16). Min-K% proving: one socket of a Xeon Gold 6526Y, 16 cores, 256 GB (p.20).
- QDQ validation (Appendix A, Table 3, p.14): simulation error vs the integer reference is
  2.2–3.2 × 10⁻⁷ in FP32 and 1.6–2.4 × 10⁻³ in BF16; PoT scaling gives exactly 0 simulation error
  at W8A8 and W12A12 in FP32.
- Governance use case (Appendix H): a Min-K% membership-inference statistic over the token
  probability vector, proved in ~3 s for N up to 4096 in Plonky3 against 25.6–33 s (fixed point) and
  25.3–91.2 s (FP32) in SP1 (Table 10, p.21).

### Inferences
- The paper's "activation precision" means the per-tensor integer width after a calibrated scale,
  i.e. relative resolution. jolt-atlas's i32-at-2^14 is not one of its configurations: our step is
  fixed (2^-14) regardless of a tensor's range, and our only "width" decision is the saturating
  clamp. Mapping their results onto us needs the translation in Q5.
- The paper reports no verifier time or proof size for any configuration despite saying it would
  (p.6, "we report prover time, verifier time, and proof size"); only prover time appears.

### Gaps
- Which DeepProve commit, field and lookup argument the modified fork uses is not stated. The field
  discussion on p.9 names BabyBear and Mersenne31 as "commonly used" but does not say which
  DeepProve uses.
- No per-stage profile of DeepProve's prover (commit vs sumcheck vs lookups) other than the
  "joint table proof" column of Table 12.

---

## Q2. Weight vs activation precision: tables, tasks, models, break points

### Takeaway
Weights at 8 bits cost at most a 1.2× perplexity ratio on every model; activations at 8 bits cost
3× to ~500×. With static per-tensor scales, GPT-2-class models are fine at A10–A12, while 7B+
models (Llama-3.1-8B especially) still lose 10–73% perplexity at A12 and need A14–A16. Most of the
A12 loss is a scale-selection problem: dynamic per-token scales recover it in 17 of 18 model–dataset
pairs.

### Cited Findings
- R1 (p.6): "even at the lowest precision of 8 bits, the PPL ratio compared to the BF16 baseline
  remained below 1.2" for weights; for activations "most models exhibited PPL ratios above 10 at
  8 bits. Most models required at least 12 or 14 bits for the ratio to approach 1. This sensitivity
  does not vary monotonically with model size" — [BITS p.6](https://arxiv.org/abs/2609.36437),
  Fig. 1 (p.6) and Fig. 3 (p.15, all nine models, WikiText-2 and C4).
- Fig. 1(b) (log scale, read from the plot): at A8, GPT-2 Small and Llama-3.1-8B are at a few
  hundred, GPT-2 Medium ~30, Qwen2.5-14B ~3, Qwen3-30B-A3B ~1.5; at A10 everything is within
  ~1.0–1.3 except Llama ~1.7; at A14–A16 all ≈1.
- Exact numbers at W12A12 with exact nonlinear functions, static per-tensor scales (Table 9, p.19,
  WikiText-2, W12A12 "Base" vs W16A16 "Base"): GPT-2 Small 27.05 vs 26.35 (ratio 1.027); GPT-2
  Medium 19.77 vs 19.31 (1.024); GPT-2 XL 15.33 vs 15.31 (1.001); TinyLlama 9.05 vs 7.91 (1.144);
  Qwen2.5-3B 7.79 vs 7.57 (1.029); Qwen2.5-7B 8.63 vs 7.41 (1.165); Llama-3.1-8B 10.30 vs 5.97
  (1.725); Qwen2.5-14B 5.52 vs 5.11 (1.080); Qwen3-30B-A3B 8.67 vs 8.39 (1.033).
- Dynamic per-token activation scales at W12A12 (Table 9 "Dyn"): Llama-3.1-8B 5.96 (vs BF16 ≈
  5.97), Qwen2.5-7B 7.40, TinyLlama 7.91, Qwen2.5-14B 5.10 — "improves PPL over the static baseline
  in 17 of the 18 model–dataset pairs" (p.18). "Changing only the scale representation does not
  recover this loss" (p.18).
- Power-of-two scales at W12A12 (Table 9 "PoT"): GPT-2 Small 28.53 (vs Base 27.05, +5.5%), GPT-2
  Medium 19.93, TinyLlama 9.93, Qwen2.5-7B 8.60, Llama-3.1-8B 10.27, Qwen2.5-14B 5.60. At W16A16
  "all evaluated settings remain within approximately 1.2% of the BF16 reference PPL" (p.18).
- Tasks: perplexity on WikiText-2 and C4 (p.6); logit-level fidelity (KL, Top-5 overlap,
  bottom-20% overlap) on 8,192 scored tokens per model and dataset (Table 8, p.18); downstream
  ARC-easy (0-shot, 2,376 questions) and MMLU (5-shot, 14,042 questions) (Fig. 4, p.18).
- Table 8 (p.18), W16A16 full pipeline vs W12A12 full pipeline, WikiText-2 KL / Top-5: Qwen2.5-3B
  0.0037/0.961 vs 0.0407/0.884; Qwen2.5-7B 0.0150/0.936 vs 0.2147/0.776; Llama-3.1-8B
  0.0041/0.963 vs 0.5516/0.653; Qwen2.5-14B 0.0046/0.958 vs 0.0999/0.862; Qwen3-30B-A3B
  0.0055/0.960 vs 0.1053/0.849. "W12A12 with the same LUT settings has lower fidelity across all
  five models. This confirms that perplexity alone is not enough to certify behavioral fidelity"
  (p.17).
- Fig. 4 (p.18): Full16 tracks BF16 on ARC-easy and MMLU for all five models; Full12 drops MMLU on
  Llama-3.1-8B to roughly half (read from the bar chart, ~35% vs ~65%).

### Inferences
- The break point depends on the scale rule more than on the model: the Llama-3.1-8B 1.73× loss at
  A12 vanishes with per-token scales. A static per-tensor scale wastes bits on outlier channels;
  a fixed-step scheme like ours wastes none on range but spends them on the clamp instead (Q5).
- "Activations are more sensitive than weights" is well supported, but the headline reason is
  structural: a weight is a constant with a per-tensor calibrated scale, an activation is a runtime
  value under a static scale calibrated on eight windows.

### Gaps
- The per-model Fig. 1/3 values at A8 and A10 are plotted, not tabulated; only the W12A12 and
  W16A16 columns have exact numbers (Table 9).
- No result isolates per-tensor static vs per-channel; only per-tensor static, dynamic per-token
  and asymmetric are compared.

---

## Q3. Proving cost vs bit width: what scales, what does not, and the bottleneck

### Takeaway
In DeepProve, end-to-end proving time is flat (±5%) across W8–W16 and A8–A16 because the count of
arithmetic operations and lookups is unchanged; the precision-dependent parts (requantization and
range decomposition) are "too small to materially change total prover time". LUT size is flat from
B8 to B16 and jumps 2.6× at B24 because the table (16M entries) then exceeds the query count
(0.823M). The RMSNorm inverse-square-root is a *utility* bottleneck of fixed-grid LUTs on 7B+
RMSNorm models, not a proving-cost bottleneck, and the proposed fix (B24 only there) is cheap in
their system only because that one table is per-row.

### Cited Findings
- R4, Table 6 (p.16), GPT-2 Small mean prover seconds: W16A16 104.474; W8A16 102.718; W10A16
  103.205; W12A16 104.079; W14A16 104.683; W16A8 106.272; W16A10 109.046; W16A12 105.473; W16A14
  104.135. GPT-2 Medium: 222.355 at W16A16, range 215.175–228.912. GPT-2 Large: 525.555 at W16A16,
  range 524.634–546.040. Three runs each — [BITS p.16](https://arxiv.org/abs/2609.36437).
- Stated reason (p.9): "as long as the computations do not lead to overflows in F_p, different
  precisions result in the same number of arithmetic operations and lookups in the ZK protocol.
  … bit precision affects the integer representation and range of values, leading to different
  requantizations and range decompositions. However, in the experiments, we observe that these
  precision-dependent components are too small to materially change total prover time."
- The cliff the paper does name (p.9): "multiplying two integers wider than 16 bits can exceed the
  capacity of a small field, such as the 31-bit BabyBear field p = 15·2^27 + 1 or the Mersenne prime
  p = 2^31 − 1 … Once the overflow occurs, each integer value needs to be represented by multiple
  field elements, which will significantly increase |R_{Q,p}|".
- R5, Table 12 (p.21), GPT-2 Small, LUT precision B for GeLU and softmax-exp: proving time B8
  104.75 s (both B8), B12 103.28, B16 102.65, B24 265.54 (both), 248.44 (GeLU B24 only), 120.49
  (Exp B24 only). LUT payload 0.102 / 1.625 / 26.0 / 6,656 MiB. Direct queries fixed at 0.823 M; all
  queries 17.1–18.2 M. "Joint table proof" 0.483 / 0.592 / 1.325 / 103.761 s. Stated reason (p.9):
  "the proving cost of lookup arguments is proportional to both the size of the LUT and the number
  of lookup queries … reducing table precision changes table size, but not the number of lookups".
- Context length, Table 7 (p.17): GPT-2 Small W16A16 73.1 / 87.2 / 103.7 / 125.9 s at 4 / 8 / 16 /
  32 tokens; "number of intermediate elements increases by approximately 8.4 times from context 4
  to 32" for a 1.72× time increase (p.16–17). "context length has a much larger effect on proving
  time than the change of weight precision" (p.17).
- Floating point vs integer (Table 2, p.8; Table 13, p.22): one Qwen2.5-14B block's seven
  projections take 95.457 s as W16A16 integers in DeepProve against an estimated 121,944 s
  (33.87 h) with a floating-point SNARK; peak RAM 0.20–0.29 GiB vs 14.9–40.8 GiB on the measured
  8-column slices ("73–139 times as much peak RAM").
- RMSNorm (R3, p.7): with B16 LUTs on every nonlinearity, WikiText-2 PPL ratios vs the W16A16
  linear-only baseline are Llama-3.1-8B 1126.8, Qwen2.5-7B 1496.9, Qwen2.5-14B 32.5, Qwen3-30B-A3B
  131.3 (Table 1(a)/Table 4); GPT-2 Small/Medium/XL 1.000–1.002 and Qwen2.5-3B 1.000. Using the B16
  LUT only on RMS rsqrt reproduces the loss (1126.8, 1528.5, 32.6; Qwen3 1.04); making only RMS
  exact restores 0.9993–1.0034; RMS at B24 restores 0.9987–1.0014 (Table 1(b), Table 5, Table 11).
  Root cause: "the calibrated input range extends from 0 to 65,461.31. A B16 LUT contains 65,536
  entries, so the interval between table grid points is about 0.9989 … 66% of the inputs are mapped
  to the grid point at zero" (p.7); B24 gives grid spacing 0.0039.
- Memory of a B24 int32 table: 64 MiB per table vs 256 KiB at B16 (p.8, p.14).

### Inferences
- The flat-cost result is a statement about DeepProve's cost structure (lookup count and matmul
  size dominate; range decomposition is a small share). In jolt-atlas the range decomposition *is*
  the dominant cost (25.2 s of 40.6 s per README), so R4 does not transfer; see Q5.
- The B24 jump is a property of multiplicity-based (LogUp/GKR) lookups, where the table itself is
  committed and proved; in a one-hot Shout lookup the table contents are never committed and the
  cost is the address width in digits (OR §1: "Table contents do not matter"). A 24-bit address is
  6 digits against 4 for 16 bits, a 1.5× per-lookup increase, not 2.6× end to end.
- The RMSNorm result is an artefact of approximating rsqrt by a uniform-grid LUT over a calibrated
  range whose mass sits near zero. jolt-atlas computes rsqrt exactly in fixed point
  (OR B1: ⌊√⌊S³/x⌋⌋, with S = 2^14), so the input resolution is 2^-14 ≈ 6.1 × 10⁻⁵ everywhere, 64×
  finer than their B24 grid (0.0039). The finding does not describe a jolt-atlas bottleneck.
- Sublinear context scaling in DeepProve (1.72× for 8.4× elements) points to a large fixed
  per-layer cost (plausibly weight handling or table setup); jolt-atlas has no per-stage number at
  more than one context length (OR §2 item 5), so no comparison is possible.

### Gaps
- No breakdown of what the "precision-dependent components" cost in absolute terms; the paper
  does not say what share requantization is in its DeepProve build (RD §5 cites "about 27%" from
  the DeepProve paper itself, a different source and possibly a different version).
- The DeepProve GPT-2 Small number here (104 s at 16 tokens, 32 threads) and the "34 s at 64
  positions" that RD §5 attributes to DeepProve cannot both describe the same prover and config;
  neither should be quoted as "DeepProve's GPT-2 time" without the version and machine.

---

## Q4. Recommendations, and whether they are system-specific

### Takeaway
Three stated principles: keep activation precision high, set nonlinear LUT precision per operator,
and reduce precision only where it simplifies the proof relation. The first is general; the second
and third are shaped by DeepProve's cost model (lookups priced by table size and query count,
31-bit-field overflow cliffs) and do not carry over to a one-hot lookup prover on a 128-bit field.

### Cited Findings
- Conclusion (p.10): "preserve sufficient activation precision, allocate nonlinear precision on an
  operator-specific basis, and reduce numerical precision only when it meaningfully simplifies the
  underlying proof relation" — [BITS p.10](https://arxiv.org/abs/2609.36437).
- "for small models that have already been measured in prior work, one could use higher precisions
  such as W16A16 for better utilities, with minimal overhead on the proving cost" (p.9).
- "we only need a B24 LUT for the RMSNorm while keeping other LUTs at B16" (p.9).
- The overflow caveat is tied to 31-bit fields by name (p.9, quoted in Q3).
- Definition 3, static compilability (p.4): "all arithmetic operations, public constants, and
  lookup tables are fixed and independent of the runtime input"; dynamic per-token scales are
  flagged as violating it in spirit ("computing new quantization parameters inside the forward
  pass … can substantially increase the cost", p.4) even though they are the best utility fix (p.18).
- ZK-friendly scales (p.15): "The reason a power-of-two scale is ZK-friendly is that it can be
  compiled as a fixed shift-like integer operation or a multiplication by a public constant."

### Inferences
- Principle 1 (keep activation precision) is the one that constrains us; our scheme already
  satisfies it with margin (Q5), so the paper gives no reason to lower our scale.
- Principle 2 is moot for us on rsqrt (exact) and weak on the activation tables (cost per extra
  address bit is 1/4 digit, not table growth).
- Principle 3 is the interesting one, read against our cost model: for us the "proof relation"
  does get simpler with fewer bits, by exactly one committed digit per 4 bits per element. The paper
  argues against uniform bit-cutting because it did not buy anything in DeepProve; in jolt-atlas it
  would, if the clamp width shrank with it.
- On the field: the paper's reason that W16A16 is the practical ceiling for cheap proving (products
  of >16-bit values overflow a 31-bit field) does not exist on Fp128. We can hold 32-bit values and
  i64 accumulators natively. This is a structural advantage the docs do not currently claim.

### Gaps
- The paper never prices its own principle 2 beyond the single B24 RMSNorm case; there is no
  table of "LUT precision per operator vs cost".

---

## Q5. What the results imply for our fused-rescale design (i32 at 2^14, 4-bit one-hot digits)

### Takeaway
Fidelity: our scheme sits at or above the paper's W16A16-with-exact-nonlinearities column in
resolution, so the paper gives no reason to fear accuracy at our current precision and no reason
to lower it. Cost: the paper's "bit width does not matter" is false for us; each 4 bits of clamp
width is one committed digit per element, so per-operator precision helps us in the one place the
paper says it did not help DeepProve: the output clamp. The direction the paper's data supports is
per-operator (per-site) power-of-two shifts with a per-operator clamp width, not a narrower global
scale and not weight precision.

### Cited Findings
- Our cost model: 12 digits per einsum output (8 clamp + 4 remainder) — [OR §1 table](/Users/kg/swe/icme/official/jolt-atlas/docs/optimization-roadmap.md); 25.2 s of 40.6 s is witness and
  commitment — [README Benchmarks](/Users/kg/swe/icme/official/jolt-atlas/README.md).
- Our constants: `LOG_K_CHUNK = 4`, `XLEN = 32`, `MODEL_SCALE = 14`
  (`/Users/kg/swe/icme/official/jolt-atlas/common/src/consts/general.rs`); tanh table is 2^18
  entries at scale 2^14 (`atlas-trace/src/semantics/tables.rs`, `tanh_table_properties`).
- BITS PoT ablation: no utility cost at W16A16, 0–5.5% at W12A12 (Table 9, p.19).
- BITS: the largest calibrated input the paper reports is the RMSNorm mean-of-squares range
  0–65,461 for Qwen2.5-7B (p.7); W16 keeps all nine models within 1.2% PPL (p.18).
- BITS: weights at 8 bits cost ≤1.2× PPL (p.6), and weight precision has no effect on proving time
  (Table 6, p.16).
- FlashProve (as summarised in RD §3.1): "per-site power-of-two scales; 12-bit clamped activations;
  13-bit weights; a 40-bit residual stream" — [RD §3.1](/Users/kg/swe/icme/official/jolt-atlas/docs/research-directions.md).

### Inferences
- Translation of "activation bits" to our scheme. For a tensor with max magnitude a, our effective
  width is log2(a·2^14)+1 bits with a fixed step 2^-14. A tensor with a = 4 is represented with
  17 bits, a = 1000 with 25 bits. The paper's A16 with static-max scale gives 16 bits of range on
  that tensor with a step of a/2^15. Our resolution is never worse than their A16 unless a < 2,
  and on their one quoted wide-range tensor (max 65,461) we have 2^30 < 2^31, fitting i32 with a
  step 64× finer than their B24 grid. So on the paper's evidence our *current* scheme should match
  their "W16A16, exact nonlinearities" row (within 1.2% of BF16 on all nine models). This is a
  prediction; RD §3.5 correctly says our fidelity is unmeasured.
- Where we are at risk is not resolution but saturation: i32 at 2^14 saturates at |x| ≥ 2^17 =
  131,072. The paper's 24-bit "auxiliary domains" for residual paths and FlashProve's 40-bit residual
  stream both suggest residual-stream magnitudes are the thing to watch; the paper gives no
  magnitude histogram, so this is a gap for our own value census (OR §2 item 4).
- Lowering the global scale (2^12, 2^10) buys nothing on the cost side unless the clamp width also
  drops, and the PoT column of Table 9 shows the utility price of ~12-bit resolution on GPT-2 Small
  (+5.5% PPL) is real. Not recommended.
- Per-operator precision helps us exactly where the paper says it did not help DeepProve: in the
  range decomposition. If an op's output provably fits w bits after a per-op shift, its clamp costs
  ⌈w/4⌉ digits instead of 8. The paper's A12–A16 findings say w = 12–16 (with a *calibrated* scale)
  preserves utility on GPT-2-class models and w = 16 on 7B+ models, so the clamp could drop from 8
  to 3–4 digits per element, giving 7–8 digits per einsum output instead of 12 with no approximate
  range check. This is a SemVersion change (per-site shifts change outputs) and is the exact-
  semantics alternative to OR D1/D2; it reaches a similar digit count as D1's "4 + ~1.4 amortised"
  only at wide rows, but it is exact and width-independent.
- Weights are irrelevant to our cost (public, never committed; they only size the i64
  accumulator). The paper confirms weight precision is irrelevant to DeepProve's cost too (Table 6),
  for a different reason. "Prove the int8 model" as a weight-width statement buys us nothing.
- Nonlinear table precision: going from a 2^18 to a 2^24 address is 5 → 6 digits for us, not 2.6×
  end to end. The rsqrt case is exact in our stack. So the paper's operator-specific LUT advice
  reduces, for us, to "the activation tables are already fine-grained; do not shrink them below the
  range they need". Shrinking an address by τ = 4 (the whitepaper's teleportation) saves half a digit
  per lookup.
- Per-row quantities (softmax normalisation, RMSNorm statistics) are where a wider or two-sided
  range check is cheap for us as well as for them, because the count is per row (OR B1 already says
  this).

### Gaps
- No activation magnitude data in the paper, so the saturation question for our i32 cannot be
  answered from it.
- The paper does not evaluate GPT-2 with any per-site PoT scheme; the FlashProve analogue is on
  Llama 4 Maverick (RD §3.1, 6.81–9.41% top-token divergence, no PPL), so "12-bit per-site works on
  GPT-2" is unmeasured by anyone we have read.

---

## Q6. The Jolt Atlas whitepaper: what we claim publicly, and what no longer holds

### Takeaway
The whitepaper claims a zkVM-shaped seven-stage proof (Spartan R1CS outer sumcheck, PC/bytecode/
memory checking, BlindFold zero knowledge over Pedersen + Nova folding, HyperKZG), a lossy "neural
teleportation" τ = 4 table shrink at a 2^7 fixed-point scale, and a ~38 s GPT-2 (125M) proof on an
M3 with 16 GB plus a ~14 s nanoGPT proof "17×" faster than ezkl. Almost every architectural claim is
no longer true of the code: no R1CS, no zkVM stages, no zero knowledge, no HyperKZG, no τ scaling,
scale 2^14, Akita over Fp128, and a stage profile that has inverted from sumcheck-dominated to
commitment-dominated.

### Cited Findings (what the whitepaper states)
- Title, authors, abstract: "extends the Jolt proving system to model inference … applies it
  directly to the ONNX tensor operations … lookup arguments … neural teleportation to reduce the size
  of lookup tables while preserving model accuracy … streaming … zero-knowledge through the
  BlindFold technique, as introduced in Vega … practical proving times for classification,
  embedding, automated reasoning, and small language models … on-device, without specialised
  hardware … succinctly verifiable" — WP p.1.
- Companion work: "Agentic Trust: Solving the Principal-Agent Problem in Autonomous Commerce Through
  Cryptographic Verification … zkML with Automated Reasoning models for succinctly verifiable
  agentic guardrails with over 99% accuracy" — WP p.5. (Not read, per the revised scope.)
- Contribution 3, commitments: "Dory … is only justified for witnesses over 2^26 elements. We
  therefore replace it with HyperKZG [21], a KZG-based PCS for multilinear polynomials derived from
  the Gemini transformation, trading transparent setup for pairing-based succinct openings that are
  well suited to on-chain verification" — WP p.5. (Reference [21] is MicroNova, not HyperKZG.)
- Architecture, §2: execution trace with "operation type, source tensor addresses and values,
  destination tensor address and value, … advice value" (p.6); Stage 1 "SpartanDag proves the outer
  R1CS constraint" (p.7); Stage 2 inner sumcheck and four product virtualisations
  (LeftInput×RightInput, td_addr×write_flag, ts1_val×select_flag, td_write_val×select_flag) (p.7);
  Stage 3 PCSumcheck (NextPC), ReadRafSumcheck ("Each 64-bit lookup index is decomposed into
  chunks"), Booleanity, HammingWeight, InstructionInput (p.7–8); Stage 4 RASumCheck "D = 8 per-chunk
  ra_i polynomials" and MemoryDag (p.8); Stage 5 memory value evaluation with Inc, Wa, Lt (p.8);
  Stage 6 "Bytecode Verification"; Stage 7 BlindFold (p.8).
- §3 BlindFold: Pedersen commitments to round coefficients, Nova-style folding with a random
  satisfying relaxed-R1CS pair, R1CS encoding of the sumcheck verifier of size O(N·d_max), HyperKZG
  evaluation commitment bound by an extra constraint (p.8–14).
- §4.2 Neural teleportation: "a global, one-sided approximation: every pre-activation input to a
  selected activation is divided by a single factor τ > 1 … y' = σ(x/τ), which is a lossy
  approximation"; "With τ = 4, the effective input range shrinks by 4×, … and the lookup table is
  bounded by 2^16 entries (sufficient for 16-bit fixed-point activations). Empirical evaluation
  shows that τ = 4 introduces output differences of less than 55 units on a 128-scale fixed-point
  representation" — WP p.15–16.
- §5.2 operator list: Add, Sub, Mul, Div; Relu, Sigmoid, Softmax, Tanh, Erf; Gte, Eq; Rsqrt;
  Reshape, Broadcast, Gather, Select; ReduceSum, ReduceMean (→ ReduceSum + Div); Einsum
  (`mk,kn->mn`, `bmk,bkn->bmn`, attention patterns) — WP p.16–17.
- §5.4: "Currently, to verify a tensor operation such as add, mul, or ReLU, we take the ONNX trace,
  decompose it into a CPU trace, and feed that into the Jolt proof system … we instead verify tensor
  relations directly at the polynomial level … Almost all of our checks are performed using virtual
  polynomials, and we do not use or require R1CS" — WP p.17.
- §6 Benchmarks: "MacBook Pro (Apple M3) with 16 GB RAM" (p.17). nanoGPT (~0.25M params, 4 layers):
  vk 0.246 s, pk 0.246 s, proof 14 s, verify 0.517 s; ezkl reported 192 / 212 / 237 / 0.34 s;
  "roughly a 17× speed-up on proof generation alone" (Table 1–2, p.18). GPT-2 (125M), Table 3
  (p.19): pk/vk 0.872 s, witness ~7.5 s, commitment ~3.5 s, sumcheck ~16 s, reduction opening ~7 s,
  HyperKZG prove ~3 s, total ~38 s. No sequence length, proof size, verifier time or peak memory for
  GPT-2; no accuracy numbers anywhere except the τ = 4 sentence.
- §7 Future work: lattice PCS (Hachi, ABBA), on-chain verification ("HyperKZG's G_T operations"),
  extended ONNX support, "Adaptive teleportation for prefix–suffix efficiency" (tune τ jointly with
  the chunk count C) — WP p.18–20.
- Related work claims: ezkl "tables cannot be materialised for large ranges"; Bionetta "doesn't
  support general ONNX"; DeepProve "operations such as Gather require reading the tensor … The most
  recent open-source implementation of DeepProve lacks these lookup arguments" — WP p.5–6.

### Cited Findings (what the code and docs say now)
- Backend: "Akita lattice PCS over the 128-bit Solinas field"; prove 40.6 s (25.2 s witness +
  commitment, 5.28 s IOP, 6.72 s opening reduction + openings), verify 2.21 s; median of 3 without
  tracing 39.9 s / 2.17 s, 2.42 MiB proof, 6.2 GB peak; Dory before removal 26.7 s / 3.42 s /
  2.37 MiB / 6.6 GB — [README](/Users/kg/swe/icme/official/jolt-atlas/README.md);
  [docs/upstream-inventory.md](/Users/kg/swe/icme/official/jolt-atlas/docs/upstream-inventory.md)
  ("BN254 removal (stage 7b): Dory, HyperKZG, the BN254 JoltField impl … are deleted").
- No `blindfold`, `hyperkzg`, `dory` or `pedersen` identifier in any non-vendored `.rs` file
  (grep); the only hits are a README sentence and a Cargo.toml comment about excluded vendored
  crates. RD §7 lists zero knowledge as open: "Upstream's Whiteout stack, when it becomes
  selectable".
- No R1CS, PC, bytecode or memory-checking stage: CLAUDE.md describes per-op `OpLayout`/`VerifyOp`
  relations, one-hot Shout lookups, deferred batches and an opening reduction.
- Teleportation: `atlas-trace/src` has no `teleport` or τ; the `neural_teleport` module in
  `jolt-atlas-verifier`/`jolt-atlas-core` is the Cos/Sin `a mod P` domain reduction and its
  right-shift stage (`division.rs`, `trig_downscale.rs` module docs). Tanh/Erf/Sigmoid tables are
  2^18 entries at `MODEL_SCALE = 14` (`sym_clamp_table`), not 2^16 at a 2^7 scale.
- Current `Op` enum (`atlas-trace/src/op/mod.rs`): Input, Constant, Identity, Add, Sub, Neg, Mul,
  Square, Cube, Div, ScalarConstDiv, Einsum, Sum, MeanOfSquares, SoftmaxLastAxis, Rsqrt, Tanh,
  Sigmoid, Erf, Sin, Cos, ReLU, Clamp, Iff, And, IsNan, GatherSmall, GatherLarge, Slice, Concat,
  Reshape, Broadcast, MoveAxis. Gte and Eq are not variants; Select is Iff; Gather is split.
- DeepProve is treated as a serious comparator in RD §5 (34 s / 176 s on GPT-2 at 64 / 512
  positions), contradicting the whitepaper's dismissal.

### Inferences: claims the next paper must not repeat
1. "Zero knowledge via BlindFold" and the whole of §3: no zk exists in the code today; the zk plan
   is upstream's Whiteout (RD §7) or an in-house lattice zk (memory note "Lattice-only end state").
2. "HyperKZG … well suited to on-chain verification" and the on-chain future-work item: HyperKZG,
   Dory and BN254 are deleted; the stack is Akita over Fp128; RD §3.5 says hash-based verifiers
   (0.8–3.3 ms) beat Akita's 10.5–12.7 ms, so verifier time is "not an advantage".
3. The seven-stage zkVM architecture of §2 (Spartan outer sumcheck, NextPC, bytecode, MemoryDag,
   Inc/Wa/Lt): none of it exists. The paper contradicts itself (§2 vs §5.4 "we do not use or
   require R1CS"); the next paper should describe only the tensor-relation design.
4. "Neural teleportation … τ = 4 … 2^16 entries … 128-scale fixed point": not implemented as
   described; scale is 2^14, activation tables are 2^18, and no lossy τ is applied. The name
   survives for an exact trig domain reduction. RD §3.5 now positions us on "exact, unique-witness
   semantics", which a lossy τ would undercut.
5. "Faster commitments" as a contribution and the Table 3 profile (commitment 3.5 s of 38 s):
   today commitment is 25.2 s of 40.6 s and is the thesis of RD §1. The next paper must present the
   inverted profile honestly.
6. nanoGPT "17× faster than ezkl": no nanoGPT model is in `models/` (measurement plan lists gpt2,
   gpt2-tiny, bge-small, bge-tiny, qwen2-tiny, op_gauntlet); the comparison is unreproduced on the
   current backend. ezkl's numbers are also from a blog post.
7. "Dory is only justified for witnesses over 2^26 elements": the repo later ran Dory on GPT-2 at
   26.7 s, faster than both the whitepaper's HyperKZG 38 s and today's Akita 40 s (README). The
   sentence should not be repeated.
8. The operator list and "ReduceMean decomposed into ReduceSum + Div": stale; use the `Op` enum.
9. The DeepProve "lacks these lookup arguments" remark: stale and now contradicted by RD §5's
   treatment of DeepProve as the de facto GPT-2 comparator.
10. "Streaming" / small-space proving as a demonstrated property (abstract: "We demonstrate that
    Jolt Atlas can prove model inference in memory-constrained environments"): the benchmarks give
    no memory number; README gives 6.2 GB peak on GPT-2, and RD §7 lists memory-bounded proving
    among "requirements more than research, since others ship them".
11. The 38 s GPT-2 number itself: it is on a different PCS, field and probably sequence length
    (unstated); today's like-for-like is 39.9 s median (Akita), with the proof size (2.42 MiB) and
    verify time (2.17 s) the whitepaper never gave.

### Gaps
- The whitepaper gives no date, no sequence length for GPT-2, no proof sizes, no memory, and no
  accuracy methodology for the "<55 units" claim; none of these can be reconstructed from the file.
- Whether any current benchmark used the "automated reasoning" or "classification" workloads the
  abstract names is not determinable from the two docs.

---

## Q7. Which of research-directions.md's directions the whitepaper anticipated, and which contradict it

### Takeaway
The whitepaper anticipated the lattice switch, per-op precision tuning (as τ), and the virtual-
polynomial/one-commitment ethos. It contradicts the docs on what the cost is (sumcheck vs
commitment), on exactness (lossy τ vs unique-witness semantics), on zero knowledge (BlindFold vs
none/Whiteout), on the verifier target (on-chain pairings vs lattice, small proofs by recursion) and
on who the competitors are.

### Cited Findings
- Anticipated: "Lattice-Based Polynomial Commitment Schemes: … replacing the existing pairing-based
  HyperKZG PCS with lattice-based PCS constructions" naming Hachi and ABBA (WP p.18–19) → done with
  Akita instead (README). "Adaptive teleportation … optimize τ jointly with the prefix–suffix
  parameters … (iii) the induced quantization and approximation error" (WP p.19–20) → RD §7
  "Rounding placement as a compiler problem. Given a graph and a bit budget, find where rounding
  must happen and minimise committed digits globally", and Q5's per-site shift direction. "reducing
  commitment cost via virtual polynomials" (WP p.17) → RD §1 "Intermediate tensors are never
  committed" and RD §2 "Commit a value once and derive everything from it".
- Contradicted: WP Table 3 (sumcheck ~16 s, commitment ~3.5 s) vs RD §1 "25 s of the 40 s proof is
  committing one-hot digits"; WP §4.2 "lossy approximation" vs RD §3.5 "Exact, unique-witness
  semantics" and OR §9 "tolerance-based (Newton) rsqrt … makes outputs non-unique"; WP §3 BlindFold
  vs RD §7 "Zero knowledge. Upstream's Whiteout stack, when it becomes selectable"; WP p.5
  "pairing-based succinct openings that are well suited to on-chain verification" vs RD §3.5
  "verifier time … not advantages" and RD §7 "Small proofs by recursion"; WP §1.3 dismissing
  DeepProve vs RD §5 using it as the GPT-2 yardstick; WP abstract's memory-constrained/streaming
  claim vs RD §7 listing memory-bounded proving as unmet.
- Neither anticipated nor contradicted: the approximate row range check (RD §3.3, OR D1), fused
  attention (OR C3), bit-committed hybrids (RD §4 C), fidelity measurement (RD §7) — the whitepaper
  has no accuracy evaluation at all beyond one sentence.

### Inferences
- The whitepaper's τ idea is the ancestor of "per-operator precision" in our stack, but inverted:
  it shrank the table input by a lossy divide; the Bits-paper-supported version is a per-site
  power-of-two shift with exact rounding, which changes outputs deterministically (SemVersion) but
  stays a unique-witness function.

### Gaps
- None material.

---

## Q8. agentic_trust.pdf
Dropped from scope by the coordinator; not read.

---

## What this changes in the docs

Each item quotes the doc, gives the correction and the page.

### research-directions.md

1. §4, "Narrower values do not cut proving cost proportionally. 'Bits Under ZK-LLM' (arXiv
   2609.36437) finds activation precision matters far more than weight precision and the
   inverse-square-root table is the recurring bottleneck."
   **Weakened / reframed.** The paper measured this in DeepProve, where range decomposition is a
   small share of cost and lookups are priced by table size × query count (BITS p.9, Table 6 p.16,
   Table 12 p.21). The flat-cost result is explicitly explained by "the same number of arithmetic
   operations and lookups" — a cost model where bit width is free. In jolt-atlas the range
   decomposition is 25.2 of 40.6 s and costs one committed digit per 4 bits per element, so for us
   bit width is the cost. The sentence should read: "In DeepProve, proving time is flat (±5%) from
   W8 to W16 and A8 to A16 because the lookup and multiplication counts do not change (Table 6);
   that does not hold for a one-hot digit prover, where each 4 bits of clamp width is one committed
   digit per element. The paper's actual utility finding is: weights tolerate 8 bits (≤1.2× PPL on
   nine models); activations under a static per-tensor scale need 10–12 bits on GPT-2 and 14–16 on
   7B+ models (Llama-3.1-8B is 1.73× at A12), with dynamic scales recovering A12 almost entirely
   (Table 9)."
   The rsqrt clause is wrong for us: the bottleneck the paper found is a *utility* loss from a
   fixed-grid B16 LUT whose inputs cluster at zero (66% at one grid point, BITS p.7), on RMSNorm
   models of 7B+; GPT-2 is unaffected (Table 4). jolt-atlas computes rsqrt exactly (OR B1), with
   2^-14 resolution, 64× finer than the B24 fix. Delete the clause or replace with: "The paper's
   recurring RMSNorm bottleneck is an artefact of uniform-grid LUT approximation, which our exact
   fixed-point rsqrt does not have."

2. §4, "FlashProve needs 12-bit activations, DeepProve about 12 bits, zkAgent 20 on GPT-2; small
   models collapse at 8–10 bits."
   **Contradicted in part.** In the paper, GPT-2 Small and Medium are within ~1.0–1.3× at A10 and
   within 3% at A12 (Table 9, Fig. 1b, p.6/19); the models that fail at A10–A12 are the large
   ones (Llama-3.1-8B 1.73×, Qwen2.5-7B 1.17×, TinyLlama 1.14×). All models collapse at A8 (3× to
   ~500×). Replace with: "under static per-tensor scales, every model collapses at 8 bits, GPT-2
   is near baseline from 10–12, and 7B+ models need 14–16 or per-token scales."

3. §4, "The earlier 'prove the int8 model' framing should be read as per-operator precision with
   power-of-two scales, not uniform 8-bit."
   **Strengthened, with a sharper statement.** The paper supports per-operator precision and shows
   PoT scales cost nothing at 16 bits and ≤5.5% at 12 (Table 9 PoT column). For us the payoff is
   specific: a per-site PoT shift plus a per-op clamp width w lets the output clamp cost ⌈w/4⌉
   digits instead of 8, which at w = 12–16 is 3–4 digits, i.e. 7–8 digits per einsum output instead
   of 12, exact, and independent of row width. Add this as the exact-semantics alternative to the
   approximate row check (D1/D2) and as the concrete form of §7 "Rounding placement as a compiler
   problem". Weight precision should be stated as irrelevant to our cost (weights are public and
   uncommitted; the paper finds it irrelevant to DeepProve's cost too, Table 6).

4. §4 lacks the field-size point. **Add:** the paper's ceiling for cheap integer proving (products
   of >16-bit values overflow BabyBear/M31, BITS p.9) does not exist on Fp128; we can carry 32-bit
   values and i64 accumulators natively. This is a structural advantage of our field worth stating
   next to its cost (one-hot digits over 128 bits, RD §1).

5. §5, "DeepProve reports 34 s and 176 s" (GPT-2 at 64 and 512 positions).
   **Weakened.** The Bits paper's DeepProve fork takes 104 s for GPT-2 Small at 16 tokens on 32
   threads (Table 6, p.16; Table 7 context scaling 73–126 s for 4–32 tokens, p.17). The two
   DeepProve numbers differ by ~3× in the wrong direction for context length; cite version, machine
   and config with any DeepProve figure, or cite both with the discrepancy.

6. §3.5, "Integer model diverges from float at 6.81–9.41% of top tokens | Unmeasured for us".
   **Strengthened, with a method.** The paper's protocol (PPL ratio on WikiText-2/C4 with eight
   512-token calibration windows; KL, Top-5 and bottom-20% overlap on 8,192 tokens; ARC-easy and
   MMLU) is a ready-made fidelity protocol (BITS p.6, p.17–18), and its W16A16-exact row (within
   1.2% of BF16 on nine models, p.18) is the bar our scheme should meet on the paper's reasoning
   (Q5). Note the paper's own warning that perplexity alone hides distributional damage (p.17).

7. §1 thesis, "The cost of proving inference is the cost of proving values are small."
   **Qualified.** True of our Akita build (README) and false of our HyperKZG build eight months
   earlier (WP Table 3: commitment ~3.5 s of ~38 s, sumcheck ~16 s). The thesis is a property of the
   commitment scheme and digit encoding, not of inference proving; say so, since the Bits paper
   shows a third profile (DeepProve: lookup count and matmul dominate).

8. Reading-depth table, "Bits Under ZK-LLM" under "Abstract only". Move to "Read in full".

9. §5 "On the use case" paragraph. **Add** the paper's governance framing: its Appendix H proves
   a Min-K% membership statistic over the token-probability vector (3 s in Plonky3, 25–91 s in SP1,
   Table 10 p.21) — an example where what is verified is a statistic *about* the inference, not the
   inference itself, which bears on "who verifies what".

### optimization-roadmap.md

10. §1, "Table contents do not matter: in a one-hot lookup the cost depends on the address width
    only." **Strengthened**, and now contrastable: the Bits paper measures the opposite regime
    (LogUp-style, 2.6× slower at B24 when the table outgrows the 0.823M queries, Table 12 p.21). A
    B24 address costs us 6 digits instead of 4. Worth one sentence so the paper is not misread as
    applying to us.

11. §4 B1 (Rsqrt), "This is exactly today's ⌊√⌊S³/x⌋⌋." **Strengthened**: the paper's finding that a
    fixed-grid LUT rsqrt destroys 7B+ RMSNorm models (Table 1(b), Table 5, p.7/16) is a reason to
    keep the exact relation and not to table it; add to §9 "Not pursuing": "a LUT rsqrt (Bits
    Under ZK-LLM Table 5)".

12. §5 C3, "Whether 12-bit clamping holds accuracy on GPT-2-class models is unmeasured."
    **Partly answered.** With a static per-tensor (not per-site) scale and exact nonlinearities,
    GPT-2 Small is 1.027× PPL at W12A12 and 1.0 at W16 (Table 9, p.19); the paper does not insert
    QDQ after QKᵀ (p.14), so 12-bit *scores* specifically remain unmeasured. Keep the sentence,
    cite the partial answer.

13. §6 D2, "Overflow bounds that make D1 output-preserving … a declared per-bucket width bound".
    **Strengthened and extended**: the paper's data supports a *per-op* width of 12–16 bits after a
    per-site shift as fidelity-safe (Q5); that makes the declared width bound a digit saving in its
    own right (8 → 3–4 clamp digits), not only a companion to D1. Add it as D2′ or as the §5 spec's
    scale model.

14. §9 "Not pursuing", "8-bit one-hot digits". Unchanged; the paper gives no data on digit size.

15. §1 cost model and §2 "Measure first". **Add** to item 4 (value histograms) the fidelity
    measurement of item 6 above, since the same tracer run yields both; and add that no GPT-2
    sequence length is recorded for the 40.6 s figure (the whitepaper omitted it too).

### README.md (one line)

16. The Benchmarks section is current and supersedes WP Table 3; the next paper should take its
    numbers (39.9 s median, 2.17 s verify, 2.42 MiB, 6.2 GB) and state the sequence length.

---

## Ranked view: what to do about quantization precision in jolt-atlas

1. **Measure fidelity of the current scheme before changing anything.** Tracer (no prover build) on
   GPT-2 and the zoo against float ONNX Runtime with the paper's protocol: PPL ratio on WikiText-2
   windows, KL and Top-5 on scored tokens. Prediction from the paper (Q5): within ~1% of float. If
   it is not, the cause is saturation or a table domain, not resolution.
2. **Per-site power-of-two shift + per-op clamp width** as the quantization direction, replacing
   "prove the int8 model". It is the only precision change that cuts committed digits (8 → 3–4 clamp
   digits per element) while staying exact and row-width independent. It is a SemVersion change and
   belongs in the C1–C3 spec's scale model, or as D2′ next to the approximate row check, which it
   competes with on wide rows and beats on narrow ones.
3. **Do not lower the global scale** (2^12 or 2^10). It cuts resolution everywhere for a utility
   price the paper quantifies (+5.5% PPL on GPT-2 Small at ~12-bit PoT) and saves no digits unless
   the clamp width also shrinks.
4. **Ignore weight precision.** Public, uncommitted, only sizes the i64 accumulator; the paper finds
   it cost-neutral in DeepProve too.
5. **Keep the activation tables and the exact rsqrt.** Table precision costs us a quarter digit per
   bit, and the paper's rsqrt-LUT failure mode is a reason never to table rsqrt.

**The one experiment that settles it.** A tracer-only sweep, no prover build: for GPT-2 (then
qwen2-tiny and bge-small), run `atlas-trace` with (a) the current global 2^14 scheme, and (b) a
per-op saturating clamp of width w ∈ {16, 20, 24, 32} applied after each fused rescale and Add,
with per-op PoT shifts chosen from one calibration pass over eight 512-token WikiText-2 windows
(the paper's protocol). Record, per configuration, PPL ratio to float and Top-5 overlap on 8,192
scored tokens, and Σ⌈w/4⌉ + 4 committed digits per einsum output from the value census. The
configuration with the smallest digit count at PPL ratio ≤ 1.01 and Top-5 ≥ 0.96 (the paper's
W16A16 band, Table 8) is the target; if no w < 32 meets it on the residual stream, the answer is
"D1 for wide rows, exact 8 digits elsewhere", and the precision question is closed.
