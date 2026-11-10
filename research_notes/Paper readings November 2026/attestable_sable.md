# Attestable and Sable: two verifiable-inference systems from primary sources

Read on 2026-11-09. Sources actually read in full: Attestable's blog post "Proving LLMs at Scale" (text plus its three chart images), Attestable's company page and policy post, two third-party write-ups of the launch; the Sable repository at commit `0bc6b54` (2026-11-03, "Sable prover and verifier, release 1.0.0"), its README, the 60-page paper `paper/sable.pdf` in full, the early design spec `docs/spec/corrected.tex` in part, and spot-checks of `zk/field.py`, `zk/gpcs.py`, `zk/llm.py`, `zk/recur/`. Nothing from the clone was built or run. Local clone: `/private/tmp/claude-501/-Users-kg-swe-icme-official-jolt-atlas/519d2b41-1999-4a2d-91e9-7c94261caa98/scratchpad/sable-src/Sable/`; extracted paper text: `/private/tmp/claude-501/-Users-kg-swe-icme-official-jolt-atlas/519d2b41-1999-4a2d-91e9-7c94261caa98/scratchpad/sable_paper.txt`.

Access notes: `attestable.com/blog` (index) returned HTTP 404; the two posts and the company page loaded. A raw `curl` with a spoofed browser user agent against attestable.com was blocked by the tool policy; the page text was taken through WebFetch instead and the chart PNGs were downloaded from Attestable's Webflow CDN with a plain `curl` and read as images. Attestable publishes no technical report, paper, docs or code, so there was nothing further to fetch. Sable links only to its own repository and paper.

Jolt-atlas facts used in section 7 are taken from `/Users/kg/swe/icme/official/jolt-atlas/docs/research-directions.md` (GPT-2 proof 40 s with 25 s committing one-hot range digits on an M3 laptop; 2.4 MiB proof; Fp128; Akita; exact unique-witness semantics; arbitrary ONNX); nothing in this file reports on jolt-atlas internals beyond that.

---

## 1. Proof system

### Takeaway
Sable is fully specified: sumcheck + LogUp-GKR over KoalaBear (p = 2^31 − 2^24 + 1) with challenges in F_{p^4}/F_{p^8}, a rate-1/2 Reed–Solomon + BLAKE3 Merkle + mixed-length BaseFold multilinear commitment, transparent, hash-only, 103.4-bit round-by-round soundness including 30-bit grinding, and explicitly *not* zero-knowledge. Attestable names no proof system at all; it states only "hash functions only", "post-quantum", "100 bits of security" and "zero-knowledge" with hidden weights.

### Cited Findings — Sable
- "built on sumcheck and LogUp-GKR lookups over the 31-bit KoalaBear field (p = 2^31 - 2^24 + 1), with a hash-based polynomial commitment (rate-1/2 Reed-Solomon code, BLAKE3 Merkle trees and a BaseFold opening over F_p^8)" — [README](https://github.com/moonmath-ai/Sable/blob/main/README.md); confirmed in code: `zk/field.py` line 15 `P = (1 << 31) - (1 << 24) + 1`; `zk/gpcs.py` lines 49–50 `CAPLOG = 7`, `QUERIES = 153`.
- Extension tower F_{p^2} = F_p[x]/(x²−3), F_{p^4}, F_{p^8}; |F_{p^4}| ≈ 2^124, |F_{p^8}| ≈ 2^248; "Witness values lie in Fp; interactive-proof challenges lie in Fp4 and commitment challenges in Fp8" — paper §3.1 (`paper/sable.pdf`, p. 8).
- Merkle trees send a "cap" of 2^7 nodes at depth 7 instead of a root; Fiat–Shamir uses the BLAKE3 compression function in a sequential (non-standard) Merkle–Damgård mode, argued indifferentiable from a random oracle — paper §3.5, p. 9–10.
- Soundness budget (Table 2, p. 30): sumcheck/GKR round ≤ 19/|F_{p^4}| (119.7 bits); LogUp per chunk with independent tuple coefficients and 11-bit PoW 103.9 bits; proximity gaps (Johnson regime, F_{p^8}, n ≤ 2^24) 163.6 bits; queries "153 at rate 1/2, η = 0.01, 30-bit proof of work" 103.4 bits; "Round-by-round soundness error 103.4" bits. README: "Soundness error is 2^-103 per round in the random-oracle model, with proven proximity bounds and a 30-bit proof of work."
- The one non-off-the-shelf step: "We are not aware of a published theorem that covers exactly this combination of the list-decoding regime, mixed-length batching at later levels and two folds per oracle" — paper §8.2, p. 29.
- Lemma 8.1 (tuple compression): compressing an arity-k tuple with powers of one challenge costs log₂k bits, so Sable "draws one challenge per tuple coordinate" (β₁..β₈) — paper §1.1 and §8.1, p. 5, 28.
- Not zero-knowledge: "A proof reveals evaluations of the trace at random positions, and about 2 x 10^4 proofs together reveal the weights. The system as released is meant for open-weight models and for verifiers entitled to see the prompts in a batch." — README "Limitations"; paper §4 computes ≈1.9 × 10^4 proofs (153 leaves × 4 symbols per proof against a 2^24-symbol codeword of a 2^23-value column).
- What is hidden: nothing. Weights are "a Merkle root over the quantized weights. Anyone can recompute it from the public checkpoint with the deterministic quantization program in scripts/quantize.py" — README. Batches leak prompts to any verifier of the batch — paper §4.
- Transparency: no trusted setup; the only trust anchors are a registry publishing C_W and h_circ (digest of all constants and tables), the client's own tokenization, and the client nonce — paper §4, p. 11.
- Weight range assumption: "The soundness theorem also assumes that CW commits to in-range weights (|w| ≤ 127)"; for closed weights "a one-time range proof per model release would replace the assumption" and "we have not built it" — paper §4 and §8.3, p. 11, 31.
- A ZK mode is designed but not built: salted leaves, hiding commitment with random symbols, Chiesa–Forbes–Spooner masked sumcheck/GKR, per-chunk-only lookup roots; "No overhead is included in the tables" — `docs/spec/corrected.tex` §"Zero knowledge (optional mode)" (line 769); paper §10.3 and §12 "this mode is not yet built".

### Cited Findings — Attestable
- "All of our results are based only on the security of cryptographic hash functions, and thus we are post-quantum secure." "We are not relying on public-key cryptography at all, and uphold 100 bits of security" — [Proving LLMs at Scale](https://attestable.com/blog/proving-llms-scale).
- "The proof binds the execution to a cryptographic commitment to the model weights." "The verifier does not need the model weights and does not rerun the model." — same page.
- Statement form y = F(W, x, r) with committed weights W, input x, random seed r; the proof can also bind "model configuration, decoding parameters, and other policy values" — same page (paraphrase by WebFetch).
- Company page: proofs are "zero-knowledge", "Post-quantum tamper-resistance", "software-only" (no TEE), and hide "model weights, architecture, or data" — [attestable.com](https://attestable.com/).
- No proof system, field, code, hash function, query count, grinding, or soundness analysis is named anywhere on the site — confirmed on all three pages fetched; also the finding of the third-party analysis: "the post does not name the proof system and publishes no field, FRI query count, grinding parameters or soundness analysis" — [noze.it, 12 Aug 2026](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/).
- Founders: Yogev Bar-On (CEO, ex-RAND, co-author of "Securing AI Model Weights"), Shahar Papini (CTO, ex-SSI and ex-StarkWare, "ZKML expertise"), Shahar Samocha (VP R&D, "led development of the proof system S-Two" at StarkWare); advisor Shafi Goldwasser — [attestable.com/company](https://attestable.com/company).
- Funding: $20M seed led by Altimeter and TLV Partners, announced 11–12 August 2026; company founded July 2025 per Globes — [KuCoin news](https://www.kucoin.com/news/flash/ai-security-startup-attestable-completes-20m-seed-round-achieves-zk-breakthrough); [Globes](https://en.globes.co.il/en/article-1001534082); [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/).

### Inferences
- Attestable's stack is very likely a Circle-STARK / S-Two lineage system over Mersenne-31 (M31) with a FRI-family hash commitment and LogUp-GKR lookups: Samocha led S-Two (StarkWare's M31 circle-STARK prover) and Papini is the "Papini" of Papini–Haböck LogUp-GKR (the paper Sable cites as [PH23]). The "hash functions only / post-quantum / 100 bits" phrasing matches STARK conventions. This is my inference from staffing and wording; the site never says so.
- "100 bits of security" without a grinding or query count cannot be compared with Sable's 103.4 bits; we cannot tell whether Attestable's 100 bits is conjectured (up-to-capacity) or proven (Johnson) regime. Sable's is proven-regime at rate 1/2.
- Attestable's claim to hide weights implies a hiding (salted/randomized) commitment and masked sumcheck; the proof sizes of ~5 MiB are consistent with a one-shot hash-based STARK at low-ish rate (FlashProve is 18.8 MB; Sable's non-recursive proof is 100–500 MiB because it opens every column at 153 queries). A 5 MiB proof for a 31B model therefore implies either recursion/aggregation or a far smaller committed trace than Sable's. Unknown which.

### Gaps
- Attestable: field, code rate, hash, query count, grinding, whether zero-knowledge covers inputs as well as weights, whether the "model commitment" is hiding, any soundness analysis. None published. The company page's "formally verified" claim (policy post) has no artifact behind it.
- Sable: no machine-checked proof; the mixed-length BaseFold list-decoding step is argued, not covered by a stated theorem (paper §8.2).

---

## 2. Arithmetization: committed vs virtual, range checks, rounding, nonlinearities, attention, argmax, sampling, quantization, fidelity

### Takeaway
Sable proves every rounding exactly with one lookup per bound into a single 2^26-row tagged table (range check of any width = 1 lookup), commits every lookup *output* (including every causal attention probability) and keeps accumulators, scores, PV partials, softmax denominators and sums of squares virtual via product sumchecks; 8-bit matmul inputs with power-of-two per-row/per-column exponents, 24-bit residual, 10-bit softmax probabilities, exact argmax and exact inverse-CDF sampling; the integer model F_O loses 8.3% perplexity on Qwen2.5-32B and agrees with BF16's greedy token at 86.3% of positions. Attestable states only "non-linear operations in floating point, matrix multiplications dynamically quantized to INT8", which is self-contradictory as written and has no proof-level detail.

### Cited Findings — Sable: what is committed and what is virtual
- Design rule: "every constraint, read as an identity over Z on values that satisfy the gadget's lookups, has both sides below p in absolute value" so "the identity holds in Fp exactly when it holds in Z (Lemma 6.1)"; every gadget "has exactly one satisfying output (Lemma 6.2)" — paper §6, p. 16–17, Table 18 p. 57–58.
- Committed: outputs of every rounding (c, q_i, Q/K/V, P tiles, h'), per-row exponents and regime tuples, selected index bits, gathered key rows, multiplicity columns per chunk, weights (once per model), fixed tables T and RoPE (once). Virtual: linear-layer accumulators (product sumcheck), attention scores acc, block partials N^(b), denominators D, sums of squares Σ in NormScale ("Σ = Σ q_i² + 1 is virtual (a sumcheck over the row)"), the gate product m = a·q_u ("The product m = a · qu is virtual: the next Requant8 reads it as a degree-2 expression"), x_l in MulRound — paper §6.2–6.5, p. 17–21.
- "Every causal score entry costs one committed probability, one lookup and two constraint terms, and the probability must be committed because it is a lookup output." — paper §12, p. 49–50.
- Committed element counts: 1.17 × 10^11 (1×4K), 3.55 × 10^11 (4×4K), 6.38 × 10^11 (16K) for Qwen2.5-32B, "including 3.4 × 10^10 weight elements and 4.7 × 10^8 in the fixed units"; lookups 1.08 × 10^11, 4.31 × 10^11, 7.20 × 10^11 — paper Table 4 and §11.2, p. 38.

### Cited Findings — Sable: range checks and rounding
- "All static lookups go into one table T of 2^26 rows and seven columns (tag, v0, ..., v5) ... A range check u ∈ [0, 2^k) is the single lookup (k, u), so every bound costs one lookup whatever its width" — paper §6.1, Table 1, p. 17. Tags: ranges k ∈ [0,24], ⋆8/⋆9 (8-bit and post-RoPE 9-bit), POW, SR/SRA (shift regimes), NPOW, RSQ, RECIP, SILU, EXP, EXPT.
- ShiftRound(acc, k) = ⌊acc/2^k⌉: two lookups `(t_R, acc + R − s·c) ∈ T` and `(t_B, c + o_B) ∈ T`; the second "bounds |2^k c| ≤ 2^30 ... without it, c + 2^-k p would satisfy the first relation in Fp". Negative shifts need a third lookup SRA bounding |acc| ≤ 2^{23+k}. "Cost per output: 1 committed element and 2 lookups, plus 1 lookup where the shift can be negative." — paper §6.2, p. 17–18.
- Requant8 (per-row dynamic 8-bit): per element `(e+1, 2a_i + s − 2s q_i) ∈ T`, `(⋆8, q_i + 127) ∈ T`; minimality of the exponent proved by a selected witness element and a sign, split into two 15-bit limbs (14-bit for ShiftQ8 whose inputs reach 2^29 — "the undivided form with |4a_j*| up to 2^31 would not be" wrap-free) — paper §6.2, p. 18, A.6.
- Range checks are per value and exact; nothing is approximate or randomly projected. Rounding remainder is never its own column: "a remainder such as acc + 2^{k−1} − 2^k c never needs its own column" because the LogUp leaf layer is merged with the zerocheck and leaf expressions may be arbitrary polynomials in the columns — paper §7.2, p. 24.
- Lookups must be chunked: "Lookups are split into chunks of at most 2^30 lookups, each with its own multiplicity column" because LogUp over a prime field is only sound for N < p and a 16K proof makes "about 7 × 10^11 lookups, more than 300p" — paper §1.1, §7.3, p. 5, 24–25. Chunk counts 103 / 416 / 726 at the three shapes.
- The lookup leaf layer and all constraints of a "domain" are proved in one sumcheck of degree up to 8–10; "Our largest domains have degree 8 to 10" — paper §7.2, p. 24.

### Cited Findings — Sable: nonlinearities, attention, argmax, sampling
- Tables are "correctly rounded ... computed in double precision, and every entry within 10^-4 of a rounding boundary is recomputed in 60-digit arithmetic, so the tables do not depend on the platform's floating point; h_circ pins them" — paper §5.2, p. 13. τ_exp(x) = ⌊2^10 exp(−x·2^{E_attn}/√d_h)⌉ for x < 2^23; τ_SiLU for |c| ≤ 2^14 clipped to 16 bits; τ_rsqrt(δ) = ⌊127·2^15/√δ⌋ and τ_recip(δ) = ⌊2^32/δ⌋ on 16-bit mantissas δ ∈ [2^14, 2^16) or [2^15, 2^16).
- RMSNorm (NormScale): Σ virtual by row sumcheck; signed normalization 2^{E−}Σ = 2^{E+}δ + ρ with a NPOW lookup, range lookups and an input bound `(16, [E<0]Σ)` closing a wrap (example given: 130050·2^14 = p + 32767); then RSQ lookup and per-element rounding lookups. Gain fused into the next weight matrix — paper §5.2, §6.2, p. 13, 18.
- Attention: scores virtual, max via a *gather* (committed key index j*, gathered key row K^g, dynamic LogUp multiset against K̂), M_m = ⟨Q̂_m, K^g_m⟩ by product sumcheck; "the softmax lookup below forces M_m ≥ acc_mj for every valid j. Hence M_m is the maximum (Lemma A.1)" since the EXP table domain x < 2^23 rejects negative inputs — paper §6.3–6.4, p. 19–20. Causal mask is folded into the diagonal-tile lookup input ν(M−acc) + (1−ν)x_min with τ_exp(x_min) = 0. Score matrix tiled 128×128, only tiles on/below the diagonal kept, P committed per tile. PV product blocked over 2^12 keys so partials stay < 2^29 for contexts up to 2^20. Normalisation: D virtual, D = 2^E δ + ρ, RECIP lookup, MulRound. Output o = MulRound(N, recip, ·).
- Argmax (θ = 0): per generated position and every vocabulary row j, `(24, z_{t,y_t} − z_{tj} − [j < y_t]) ∈ T` where the one-hot of y_t and [j<y_t] are public closed-form columns, "so y_t is the first maximum"; vocabulary 152,064 = 297·2^9 rows, unpadded — paper §6.5, p. 20–21.
- Sampling (θ > 0): inverse CDF with p_tj = ⌊2^12 exp(−(z_max − z_tj)/(2^8 T_θ))⌉, seed r = SHA256(seed ∥ n_c ∥ C_W ∥ SHA256(x) ∥ policy), u_t = low 30 bits of BLAKE3(r ∥ t), w_t = ⌊u_t Z_t / 2^30⌋ proved by 10-bit limb products with carries; "Tokens whose probability is below about 2^-13 of the mode's cannot be sampled; this is part of F_O." Cost "21 committed elements in the per-row domain and 18 range lookups, besides the |V| probabilities" per generated token — paper §5.5, §6.5, p. 14, 21. Measured: at temperature 1.0 the gadget "adds 304 lookups and 0.08 MiB" — §11.2.
- Stopping rule is part of the relation: "no y_t with t < G is a stop token, and either y_G is one or G = G_max" — Definition 3.3, p. 10.
- MoE (Qwen3-30B-A3B): router SatShift to 16-bit logits; top-k by a committed threshold index and per-expert selection bits with a tie flag; renormalised softmax over selected experts by table RTEXP + exact division constraints; experts proven as gathered sumchecks over one-hot expert index bits; "proves only the experts each token selects" — paper §5.6, §6.6, §11.3.
- QK-norm (Qwen3): per head row, ShiftQ8 + NormScale + ShiftQ8S with the gain folded, in a second per-head-row domain — §6.6.

### Cited Findings — Sable: quantization and fidelity
- Formats: residual stream A = [−2^23, 2^23) with f_h = 10 fractional bits (8 for Qwen3-32B after the sweep); matmul inputs int8 in [−127,127] with one power-of-two exponent per row (activations) / per output column (weights), "Scales off the reduction axis" so the accumulator is an exact integer, |acc| < 2^29 for ≤ 2^15 terms; static per-head exponents for Q/K/V so one exp table serves all heads/layers — paper §5.1, p. 12.
- Checkpoint → F_O: QuaRot-style fixed random orthogonal rotation fused into weights, RMSNorm gains fused, per-column int8 weights, static exponents calibrated on "1,024 tokens of English technical text (the beginning of a LaTeX document, an early specification of this system)" (= `docs/spec/corrected.tex`), refits so no shift bound binds; "two runs give bit-identical models", under 4 minutes on one GPU — paper §5.7, p. 15–16.
- Accuracy (Table 12, p. 44): Qwen2.5-32B-Instruct BF16 / FP8 / SmoothQuant INT8 / F_O perplexity 5.151 / 5.186 / 5.231 / 5.580 (+8.3% for F_O); greedy agreement with BF16 95.5% / 93.9% / 86.3%; MMLU 83.2 / 83.0 / 83.2 / 81.8; GSM8K 88.6 / 88.4 / 88.2 / 88.0. Qwen3-32B F_O +6.6% ppl, 82.4% agreement, MMLU 80.4 vs 81.8, GSM8K 92.2 vs 93.9; Qwen3-30B-A3B F_O +1.9% ppl, 88.7% agreement, MMLU 78.0 vs 79.6, GSM8K 91.4 vs 92.7 (Table 14, p. 46).
- Attribution: "Only the queries and keys matter: with 12 bits they remove 37% of the gap to BF16"; float softmax, 4 more SiLU bits, 16-bit up projection each < 3% of the gap; on Qwen3-32B the unrounded down-projection input removes 57%; the Q/K exponent split (a calibration choice, zero proof cost) removes 23% / 53% of the Qwen3 gaps — Tables 13, 15, §11.8. 12-bit Q/K would cost ×1.30 (4K) to ×1.68 (16K) proving time by the cost model.
- Partial function: F_O is undefined when SiLU input |c| > 2^14, shifts leave [−16, 24], etc.; the honest prover aborts (`zk.model.WitnessError`). Scan over 3.45M tokens of real text plus 160 adversarial 4,096-token sequences and every vocabulary token at position 0: "No input reached any condition" — Table 16, §11.9, p. 47–48.

### Cited Findings — Attestable
- "Though we prove all non-linear operations in floating points, we quantize all matrix multiplications to integers." and "Our only current limitation is dynamically quantizing the linear operations to 8-bit integers." — [Proving LLMs at Scale](https://attestable.com/blog/proving-llms-scale).
- Gemma 4 31B accuracy chart (image, read directly): GPQA-Diamond Original 0.8485 / INT8 0.8030 / Attestable Quant 0.8283; IFEval 0.9519 / 0.9427 / 0.9353. Page text: "The GPQA-D result demonstrates that our approach can preserve difficult reasoning performance better than [the INT8 baseline]"; "The IFEval result shows where the current quantization still needs improvement." The INT8 baseline is [lokeshe09/gemma-4-31B-it-INT8](https://huggingface.co/lokeshe09/gemma-4-31B-it-INT8) — same page.
- The third-party analysis calls the quantisation wording inconsistent ("all matrix multiplications are quantised to integers", "only limitation is dynamic 8-bit quantisation of linear operations", "almost all inference operations use floating point") — [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/).

### Inferences
- "Prove non-linear operations in floating point" most plausibly means Attestable arithmetizes IEEE/bfloat-style float ops (sign/exponent/mantissa limbs with range-checked rounding) inside the proof, rather than replacing them with integer tables; that would explain why their accuracy loss is smaller than a plain INT8 baseline on GPQA-D (only the matmuls are quantized). It is also consistent with a lookup-heavy M31 design. This is an inference; nothing on the page says how float ops are constrained, how rounding is proven, or whether the float model is bit-exact with respect to any served kernel.
- Sable's exact per-value design means its range-check volume scales with the number of rounded values (≈1 committed element + 2–3 lookups per rounded output) — the same cost structure as jolt-atlas's exact remainder digits, but with lookups of width 24 costing one leaf each rather than 4-bit digits, because LogUp-GKR table size (2^26) is independent of the lookup width.

### Gaps
- Attestable: no information on range checks, rounding, nonlinearity treatment, attention, argmax or sampling at the proof level. Whether the output distribution is bit-exact (a function) or "approximately float" is not stated.
- Sable: no task benchmark on chat-templated prompts; Qwen3 checkpoints were "prompted as base models, without a chat template" (Table 14 caption).

---

## 3. Generation: causal pass vs token-by-token, KV cache, batching, context

### Takeaway
Sable proves the whole reply in one teacher-forced causal pass over prompt + generated tokens (S = P + G − 1 positions) plus the sampling/stopping decisions at the last G positions; batches B equal-length prompts in one proof; contexts up to 16,384 measured (gadgets are bounded for up to 2^20). Attestable proves "the whole benchmarked execution" after decoding, caps context at 16K, and reports batch 1/2/4 throughput; nothing about KV cache or per-token structure.

### Cited Findings — Sable
- "Because F_O is deterministic and causal, all S = P + G − 1 positions can be evaluated in one teacher-forced pass over x followed by the first G − 1 generated tokens ... The prover therefore proves one forward pass over S positions together with the sampling decisions at the last G positions." — paper §3.6, p. 10. Each benchmark sequence is a prompt plus "16 greedily generated tokens"; "the cost depends on the total length and hardly on the split between prompt and generation" — §11.1, p. 37.
- Batched statement: B prompts of equal length, one policy, one nonce; the seed covers all prompts; sequence b samples token t with PRF(r, bG + t) — Definition 3.4, p. 10–11. Measured shapes 1×4K, 4×4K, 1×16K; "a proof of eight 4K sequences runs out of GPU memory in the prove pass, and one of a 32K sequence in its first witness build" (96 GB GPU) — §11.2, p. 39.
- No KV cache in the proof: attention is the full causal tiled score matrix per head; "attention, quadratic in the length, is about half of the work" at 16K — §2, p. 6; "Sliding-window attention in most layers ... would remove most of these entries" — §12.
- The serving engine (not the prover) does normal decoding with a CUDA graph per step and continuous batching, bit-exact with the prover's witness — §12, Table 17.

### Cited Findings — Attestable
- "The prover runs alongside inference. After a sequence is decoded, one proof covers the whole benchmarked execution." (WebFetch paraphrase of the page) — [Proving LLMs at Scale](https://attestable.com/blog/proving-llms-scale).
- "We support up to 16K tokens in the context window." "Proving throughput depends on the batch size and context length." — same page.
- Gemma throughput chart (image, read directly): batch 1 at 1K/2K/4K/8K/16K ≈ 17 / 31.5 / 45 / 50 / 53 tok/s; batch 2 at 1K/2K/4K/8K ≈ 32.5 / 51 / 62.5 / 61 tok/s; batch 4 at 1K/2K/4K ≈ 53.5 / 73 / 77.7 tok/s. Batch 2 stops at 8K and batch 4 at 4K (page text: "Across all successful benchmark configurations", implying the missing points did not run).

### Inferences
- Both systems' throughput rising with context length (Sable: 22.6 ms/token at 4K vs 24.3 at 16K nearly flat; Attestable: 17 → 53 tok/s from 1K to 16K at batch 1) says both are dominated by fixed per-proof costs (weights, tables, openings) at short context and are GPU-saturated only at long context. Attestable's missing batch-4 points above 4K and batch-2 above 8K are consistent with a GPU-memory ceiling like Sable's (my inference; the page says only "successful configurations").
- Attestable's "tokens/s" is almost certainly window tokens (prompt + generated, total sequence length divided by proving time), like Sable's; 53 tok/s on a 16K sequence means ≈309 s per proof. The page never defines the metric.

### Gaps
- Attestable: no KV-cache statement, no per-token vs whole-window definition, no prompt/generation split, no wall-clock proving time (also noted by noze.it).

---

## 4. Numbers

### Takeaway
Sable: Qwen2.5-32B 1×4K in 92.5 s (134.8 MiB, verify 18.8 s one CPU core), 16K in 397.5 s (345.7 MiB, 40.2 s), on one RTX PRO 6000 Blackwell 96 GB; 41–50× the F_O serving engine, ≈90× SGLang FP8; ≈44–67 window tokens/s. Attestable: Gemma 4 31B peak 77.69 tok/s (batch 4, 4K), Llama 8B 195 tok/s, Llama 70B 49 tok/s, "Muse Glimmer 30B" 87 tok/s, proofs 4.94–5.55 MiB, verification 205–251 ms, one H100 — marketing-grade (no wall-clock, memory, overhead, or reproducibility).

### Cited Findings — Sable (reproducible: code + scripts published; measured by the authors, not independently)
- Table 4 (p. 38), Qwen2.5-32B-Instruct, 64 layers, one RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7, sm_120), host Xeon 6776P 24 cores, 214 GB RAM:
  - 1×4K (4,096 tokens): 92.5 ± 0.0 s; overhead 50×; GPU 36.8 GiB; host 101 GiB; proof 134.8 MiB; verify 18.8 s.
  - 4×4K (16,384 tokens): 253.1 s; 45×; 63.5 GiB; 155 GiB; 239.0 MiB; 26.8 s.
  - 16K: 397.5 ± 0.3 s; 41×; 71.8 GiB; 175 GiB; 345.7 MiB; 40.2 s.
  - Mean of three warm runs after a warm-up (103 / 328 / 427 s, which also captures CUDA graphs); first-ever proof on a machine takes ≈1,530 s because of ~24 minutes of NVRTC compilation (cached afterwards).
- Table 7 (p. 40): Qwen3-32B 121.8 / 485.7 ± 25.3 / 668.7 s, proofs 165.1 / 298.0 / 476.7 MiB, verify 26.5 / 36.2 / 55.8 s, overhead 53/68/50×; Qwen3-30B-A3B 84.7 / 181.9 / 265.7 s, 106.2 / 152.6 / 219.5 MiB, verify 20.2 / 23.0 / 29.5 s, overhead 17/17/21× (lower because its F_O engine decodes MoE slowly at 201 ms/token).
- Derived window-token throughput (my arithmetic from Table 4/7): Qwen2.5-32B 44.3 tok/s (1×4K), 64.7 (4×4K), 41.2 (16K); Qwen3-30B-A3B 48.4 / 90.1 / 61.7 tok/s. The paper itself reports ms per token: "22.6 and 24.3 ms" at 1×4K and 16K.
- Overhead baselines: F_O serving engine 1.86 s for the 1×4K work (TTFT 1.30 s, 37.4 ms/token), 5.60 s for 4×4K, 9.60 s for 16K; "Against FP8 in SGLang, which prefills 4,096 tokens in 0.46 s and decodes 15 more in about 0.6 s, the factor is about 90"; cost "6.4 cents" per 1×4K proof at $2.50/GPU-hour — §11.2, p. 39.
- Normalised comparison (Table 8, p. 41): zkLLM 30.2 ps per parameter-token (LLaMa-2-13B, A100); FlashProve 0.56 ps per *active* parameter-token (16.2B active, 24×1,024, H100, 223 s); Sable 0.48–0.75 ps (Qwen2.5-32B), 3.36–6.27 ps per active parameter (Qwen3-30B-A3B).
- Verifier: Python reference 94 / 129 / 145 s; with the compiled C++ core on one Xeon core 18.8 / 26.8 / 40.2 s, of which only 1.3 / 3.5 / 10.2 s is in the core; the authors estimate "a verifier compiled as a whole" at "a few seconds per proof" — §11.7, p. 43–44. Verifier work: 4.22M / 6.81M / 9.19M BLAKE3 compressions, 506K / 761K / 825K challenges, 11.0M / 15.3M / 16.5M F_{p^4} multiplications (Table 11).
- Proof composition at 1×4K: "the leaf data of the query openings (32.6K columns at 4 positions for 153 queries) is 76 MiB" of 134.8 MiB — §11.7.
- Prover phase profile at 1×4K per layer: commit pass 0.26 s, prove pass 1.01 s (88–96% GPU-busy), final passes 0.30 s (Table 6); inside a layer's prove pass: MLP domains 133 ms, softmax tile domains 119 ms, point reductions 100 ms, data prep 268 ms, 512 ms of sumcheck round kernels in total — §11.2, p. 39. At 16K, tile (attention) domains take 1.25 s of the per-layer prove pass.
- Cost model vs measured: prover is 2.45× / 1.73× / 1.56× its own analytic model at the three shapes, with latency of ~3,800 single-block sumcheck rounds per layer (12–18 µs each) blamed at 4K — Table 9, §11.5.
- Memory requirement from README: "Full-model proofs use 18 to 82 GB of GPU memory and 72 to 175 GB of host memory".
- BLAKE3 recursion gadget: 5,344 committed cells per compression, 2.4 µs/compression at 2^16 compressions per block, 1.3 µs at 2^18; "the 4.2 million compressions of a 64-layer 1 × 4K verifier cost about 5.5 s, 6% of the main proof" (F_{p^4} arithmetic of the verifier not yet circuitised) — §10.2, p. 37.

### Cited Findings — Attestable (marketing claims; no artifact, no raw timings)
- Benchmark table (image, "All benchmarked on a single H100, with context size 4K and batch size 4"): Gemma 31B 77.69 tok/s, 5.55 MiB, 251.40 ms; Muse Glimmer 30B 87.25 tok/s, 5.51 MiB, 244.94 ms; Llama 8B 195.03 tok/s, 4.94 MiB, 205.08 ms; Llama 70B 49.21 tok/s, 5.51 MiB, 238.75 ms — [Proving LLMs at Scale](https://attestable.com/blog/proving-llms-scale), image `6a7b36aa...png`.
- Page text: "proof sizes range from 4.35 to 7.92 MiB and verification takes 157 to 648 milliseconds on our reference CPU verifier." "These are alpha results." Gemma 4 31B: "53 tokens per second for a batch-one 16K sequence"; "At batch four, it reaches 77 tokens per second across four 4K sequences." Comparison: ezkl's end-2023 16-minute NanoGPT claim, Attestable's model "roughly 70,000 times larger and about 50,000 times faster" (paraphrase) — same page; ezkl source linked: [blog.ezkl.xyz/post/nanogpt](https://blog.ezkl.xyz/post/nanogpt/).
- Not stated: wall-clock proving time, prover memory, overhead vs plain inference, number of CPU cores, whether tok/s counts window or generated tokens — confirmed absent on the page; also noted by [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/).

### Inferences
- Implied Attestable wall-clock: a 16K Gemma 4 31B sequence at 53 tok/s ≈ 309 s; four 4K sequences at 77.7 tok/s ≈ 211 s for 16,384 window tokens. Those are within ~1.3× of Sable's Qwen2.5-32B numbers (397.5 s and 253.1 s) on comparable-class hardware (RTX PRO 6000 vs H100; Sable's own cost model "prices our GPU at about one H100"), but with a ~25–60× smaller proof and ~100× faster verification, and with claimed zero knowledge. If true, the main difference is not prover speed but proof compression (recursion or a low-rate final layer) and hiding.
- "Muse Glimmer 30B" is not a model I can identify from any public source; treat as an unidentified or internal model name.
- Llama 8B at 195 tok/s vs Llama 70B at 49 tok/s (4×) for ~8.75× the parameters suggests substantial fixed per-proof overhead on the small model, as in Sable.

### Gaps
- Attestable: everything beyond the table and two sentences. No independent reproduction exists; inferenceproofs.ai (per research-directions.md) does not list them as benchmarked.
- Sable: no independent reproduction yet (release 2026-11-03); only one GPU type tested ("other architectures are untested" — README).

---

## 5. Code: openness, licence, language, maturity, reproducibility

### Takeaway
Sable is fully open under MIT (Python + CUDA via NVRTC/CuPy + a small C++ verifier core), 1.0.0 released 2026-11-03 with a README mapping every paper table to a command; reproduction needs a 96 GB Blackwell GPU, CUDA 13, 72–175 GB host RAM and ~25 min of kernel compilation. Attestable has no code, no repository, no API docs, no paper.

### Cited Findings — Sable
- Licence: "MIT License, Copyright (c) 2026 MoonMath.ai" — `LICENSE`. Author Tomer Solberg, MoonMath.ai; contact research@moonmath.ai — README "Citation", paper title page.
- Repository history: two commits, `b76ac59 2026-11-03 Initial commit` and `0bc6b54 2026-11-03 Sable prover and verifier, release 1.0.0` (git log of the clone). Size 3.5 MB. ~16.5K lines of Python under `zk/` plus `zk/cuda/*.cu/.cuh/.cpp`, 37 test files under `tests/`, 17 scripts, two cost-model scripts, `docs/spec/corrected.tex` (1,640 lines), `paper/sable.pdf`.
- Language/runtime: "The prover is written in Python on top of PyTorch, with CUDA kernels compiled at run time by NVRTC" — paper §9; requirements Python ≥ 3.12, PyTorch CUDA build, `cupy-cuda13x`, NumPy, blake3, mpmath, ninja, `g++`; Linux with a CUDA 13 driver — README "Requirements". `requirements.txt` "pins the exact environment of the paper's measurements".
- Reproduction table: `python -m scripts.bench_e2e B S rand:0:3[:verify]`, `scripts.verify_cpu`, `scripts.acc {fo,bf16,fp8,sq8} {wiki,mmlu,gsm8k}`, `scripts.quantize`, `scripts.fo_domain_scan`, `scripts.b3bench`, `costmodel/*.py` — README "Reproducing the paper". Supported `SABLE_ARCH`: Qwen/Qwen2.5-32B-Instruct, Qwen/Qwen3-32B, Qwen/Qwen3-30B-A3B.
- Testing: tiny two-layer Qwen2-shaped model (width 256); every proof message corrupted in turn; wrong-weights proof against C_W; wrong-seed sampling; targeted opening forgeries; no-wrap forgeries (wrapped ShiftQ8 minimality, over-bound shifts, left-shift residues); a NormScale wrapped-left-shift forgery that "verifies once the bound (16, [E < 0] Σ) is removed from prover and verifier"; QK-norm and router wrap forgeries; device vs host transcript byte-identity — paper §11.10, p. 48–49; files `tests/test_llm_tamper.py`, `test_wrap_bounds.py`, `test_norm_bounds.py`, `test_router_wrap.py`, `test_qkn_wrap.py`, `test_moe_clamps.py`, `test_clamps.py`.
- Security hygiene in README: strict byte format `zk.proofio` ("rejects malformed input"), pickles only for local experiments, registry entry (C_W, config, constants) must never come from the prover, package refuses to import under `python -O` because some checks are `assert`s.
- Maturity signals: a research prototype by a single author; protocol logic in Python (verifier 18.8 s of which only 1.3 s in compiled code); 4×4K Qwen3-32B run has unexplained ±25 s variance; graph replay fails to fit memory at large Qwen3 shapes; "other architectures are untested".
- Early spec `docs/spec/corrected.tex` lists defects of a previous draft and their fixes (e.g. "Quantization scale not tied to the data; any scale with q=0 passed", "LogUp inverses committed before their challenge", "Zero knowledge cancelled the mask in the clear") — §"Scope and what changed", lines 54–80. It also specifies gadgets for linear attention (KDA, Kimi-style) and DeepSeek-style sparse attention with an indexer (GLM-5.3-Flash) that are not in the released prover — lines 375, 432.

### Cited Findings — Attestable
- No GitHub, docs, API, technical report or paper on any Attestable page; outbound links are only X, LinkedIn, Hugging Face (the INT8 baseline), ezkl's blog, and policy essays — [attestable.com](https://attestable.com/), [company](https://attestable.com/company), [blog post](https://attestable.com/blog/proving-llms-scale), [policy post](https://attestable.com/blog/pacing-ai-requires-proof). A web search for an Attestable open-source prover found nothing — WebSearch 2026-11-09.
- "no public artefact lets a third party re-run the benchmarks" — [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/).

### Inferences
- Sable's benchmarks are reproducible in principle by anyone with the hardware; the quantized model file is deterministic ("two runs give bit-identical files") so C_W can be independently recomputed, which is the strongest reproducibility story of any system in our survey besides FlashProve.

### Gaps
- Attestable: closed; nothing to read. Whether a commercial API exists is not stated on the site.

---

## 6. Stated limitations and roadmap

### Takeaway
Sable names its limits precisely (no ZK, 100–500 MiB proofs, accuracy loss, one 96 GB GPU, partial-function F_O) and its roadmap (recursive compression via the BLAKE3-in-domain gadget with ZK outer layers, a fully compiled verifier, wider Q/K formats, batching small sumchecks, sliding-window attention). Attestable lists 16K context and INT8 matmuls as "temporary" and says both "will be lifted in the not-too-far future".

### Cited Findings — Sable
- README "Limitations": not zero-knowledge; "Proofs are 100 to 500 MiB, which suits auditors more than every client"; accuracy loss vs BF16; "The prover targets one large GPU ... larger batches or contexts run out of memory."
- Roadmap (paper §10, §12, §13): "A recursive layer that proves the verifier, with zero-knowledge outer layers, is the intended form for every client"; remaining pieces: F_{p^4}/F_{p^8} arithmetic gadgets, a recorder turning a verifier run into a witness, segmentation, outer ZK layers, low-rate final layer — §10.3. Prover: batch the many small sumchecks per layer, move host-side claim arithmetic to GPU, univariate skip (2–4%), sliding-window attention — §12. Accuracy: "finding integer formats that close the gap at a small proving cost is the main open question for F_O" — §12. Also "a machine-checked proof of the protocol" and "proofs of larger mixtures of experts at a lower cost per active parameter" listed as future work motivated by FlashProve — §2, p. 6.
- Spec-only features (KDA linear attention, indexer sparse attention, ZK mode) exist in `docs/spec/corrected.tex` but not in code — lines 375, 432, 769.

### Cited Findings — Attestable
- "We support up to 16K tokens in the context window." "Our only current limitation is dynamically quantizing the linear operations to 8-bit integers." "We believe both limitations will be lifted in the not-too-far future." "These are alpha results." — [Proving LLMs at Scale](https://attestable.com/blog/proving-llms-scale). WebFetch's summary adds: lift the 16K limit, support floating point in linear operations, improve quantization.
- The policy post proposes proofs that bind "an approved model commitment, a defined inference program, and any required input classifier" plus proof-of-useful-work accounting, and calls the system "formally verified" without an artifact — [Pacing AI Requires Proof](https://attestable.com/blog/pacing-ai-requires-proof).

### Gaps
- Attestable: no dated roadmap, no statement on recursion, no statement on proving sampling or the stopping rule.

---

## 7. Against jolt-atlas

### Takeaway
Both systems do things jolt-atlas does not: one causal pass over a whole reply with the stopping rule, committed (and in Attestable's case hidden) weights, 30–70B models on one GPU, 4K–16K contexts, and (Sable) an explicit wrap-free soundness budget with tamper tests for every bound. Jolt-atlas keeps three advantages with evidence: proof size (2.4 MiB on GPT-2 vs Sable's 106–477 MiB; comparable to Attestable's 4.4–7.9 MiB but without the unknowns), generality (arbitrary ONNX graphs vs Sable's one hand-built gadget composition per architecture and Attestable's unknown coverage), and exact unique-witness semantics on a total function (Sable is also exact but partial; Attestable's "float non-linearities" is unexplained).

### Cited Findings (jolt-atlas side from `docs/research-directions.md`; others as above)
- Jolt-atlas: "25 s of the 40 s proof is committing one-hot digits whose only purpose is to range-check rounding and clamping" on GPT-2; "2.4 MiB on GPT-2, with a merged upstream path to sub-100 KB by recursion"; "Exact, unique-witness semantics"; "Arbitrary ONNX graphs" — `/Users/kg/swe/icme/official/jolt-atlas/docs/research-directions.md` §1, §3.5.
- Sable range-check cost structure: one 24-bit-wide range check is one LogUp leaf against a 2^26-row table; rounding costs "1 committed element and 2 lookups" per output — paper §6.1–6.2. Jolt-atlas: "8 exact digits per value" for overflow and "4 exact remainder digits per value" for rounding — research-directions.md §3.2 table.
- Sable's exactness is on a *partial* function with an abort path and a measured zero abort rate on 3.45M real tokens — paper §11.9; jolt-atlas's is total with saturating clamps (research-directions.md §3.3: "out of range means no proof exists, not that the value saturates" describes FlashProve, contrasted with our semantics).
- Sable's proof is "an auditor's proof"; its verifier is 19–56 s on one core — README; jolt-atlas's hash-vs-Akita verifier times are discussed in research-directions.md §3.5 ("not advantages").
- Sable's lookup-soundness lesson: in a 31-bit field LogUp needs N < p chunks and per-coordinate tuple challenges; Sable loses nothing because challenges come from F_{p^4} (124 bits) with 2^30-lookup chunks — paper §7.3, §8.1. (Research-directions.md §4 notes the analogous ~log₂N loss for LogUp/grand-product with 128-bit challenges.)
- Sable's per-row exact maximum via gather + table-domain positivity (Lemma A.1) and its argmax by |V| public-weighted range lookups are exact alternatives to the one-hot argmax designs discussed in research-directions.md §3.4; softmax normalisation is by a virtual denominator sumcheck + RECIP table + MulRound, and the causal mask is folded into the lookup input of diagonal tiles — paper §6.4.
- Sable's "merge the LogUp leaf layer into the zerocheck so remainders never get a column" (§7.2) and "Σ q_i² virtual by row sumcheck" (§6.2) are the same "commit once, derive everything" direction as research-directions.md §2 ("Commit a value once and derive everything from it").

### Inferences
- What Sable does that jolt-atlas does not: whole-reply statements (greedy and seeded sampling, stopping rule, client nonce, policy digest); committed weights opened once per batch (Merkle root recomputable by anyone); streaming two-pass prover with per-layer witness recomputation bounding memory to one layer; MoE routing proofs; per-row dynamic quantization exponents proved minimal; mixed-height Merkle trees and mixed-length BaseFold batching; device-resident transcript and CUDA-graph replay; a published per-term soundness budget. Jolt-atlas does better: proof size by ~50–200× (2.4 MiB vs 106–477 MiB), generality (ONNX import vs three hand-wired Qwen architectures), total semantics (no abort path), and a much smaller trace per value when values are already small (jolt-atlas does not commit every lookup output; Sable commits every attention probability).
- What Attestable does that jolt-atlas does not (as claimed): hidden weights with ZK, 31–70B models at 4K–16K context on one H100, ~5 MiB proofs with sub-second verification, float non-linearities. Jolt-atlas does better on everything verifiable: it has code, a soundness story and a reproducible benchmark; Attestable has none publicly. Proof sizes are in the same class (2.4 vs 4.4–7.9 MiB); Attestable's verifier (157–648 ms) is faster than anything we report, consistent with research-directions.md's note that hash-based verifiers are faster than Akita's.
- Where Sable's cost goes mirrors jolt-atlas's thesis: "Commitment, lookups and constraint terms account for almost all of the prover's work" (§12), and every rounded value costs a committed element plus 2–3 lookups — i.e. the cost of proving values are small, exactly as research-directions.md §1 states. Sable's escape from 4-bit digits is a 2^26-row table with LogUp-GKR, which needs challenges from a ≥124-bit extension and chunking at p; this is option-space our Fp128 already has without chunking, but Sable's table approach commits 1 element + 2–3 leaves per value rather than 8–12 one-hot digits.

### Gaps
- No like-for-like GPT-2 number exists for either system, and neither is on inferenceproofs.ai, so no direct throughput comparison with jolt-atlas is possible from primary sources.
- Attestable's generality (which architectures, whether an importer exists) is unknown.
