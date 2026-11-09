# State of the art in zkML proving for transformers (as of Nov 2026): per-op techniques and zk-friendly reformulations

Source-quality legend used below: **[PR]** peer-reviewed venue (paper text read directly), **[PP]** preprint (arXiv/ePrint), **[ABS]** only the abstract/landing page was readable (ePrint PDFs returned HTTP 403 to my fetcher), **[REPO]** project README, **[BLOG]** vendor blog/marketing, **[SEC]** secondary summary or search-result snippet (not verified against the primary text).

Access caveat: I could read the full text of zkGPT (USENIX Sec '25), zkLLM (arXiv v1 of the CCS '24 paper), ZKTorch (arXiv HTML, first ~80%), Jolt Atlas (arXiv HTML), NanoZK (arXiv HTML, first ~65%), and the DeepProve READMEs. For DeepProve (ePrint 2026/1112), zkAgent (2026/199), OpenLLM (2026/1578) and the "Separation Principle" paper (2026/1390) I only have abstracts plus search snippets; the Lagrange DeepProve-1 blog post returned 403.

## Q1. Per system: proof system, field, PCS, quantization, and how each transformer building block is proven

### Takeaway
Every competitive transformer prover in 2024-2026 is sumcheck/GKR + lookups with a specialised matmul sumcheck; they differ mainly in how they prove rounding. The two ideas jolt-atlas should look at hardest are (a) zkGPT's "advice + one range relation" for division/sqrt with constraint fusion across adjacent roundings, and (b) zkLLM/zkGPT/DeepProve style "one causal forward pass proves the whole sequence".

### Cited Findings

**zkGPT (Qu, Sun et al., USENIX Security 2025) [PR, full text read]**
- Stack: GKR ("ML-friendly GKR" from zkCNN) for arithmetic relations + Lasso for non-arithmetic relations (range and exponentiation); PCS is Hyrax over BN254 via mcl ("around 100 bits of security"), prover O(N), proof/verifier O(sqrt N). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf), [ePrint 2025/1184](https://eprint.iacr.org/2025/1184)
- Quantization: scale/zero-point affine quantization, Q = 16 bits, all quantized values in [0, 2^16]; "Rescaling with appropriate scaling factor is performed at each layer"; rescale constants are a rational C1/C2 with C1, C2 searched in [1, 2^Q]. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Division / rescale: result is prover advice; correctness of q_z = round((C1/C2)·q_x/q_y) is turned into a single two-sided range relation C2·q_y·(2q_z − 1) ≤ 2·C1·q_x ≤ C2·q_y·(2q_z + 1) − 1. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Square root (LayerNorm): q_y = round(sqrt(q_x)) is advice checked by one range relation (2q_y − 1)^2 ≤ 4q_x ≤ (2q_y + 1)^2 − 1; no sqrt/rsqrt lookup table at all. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Range relations: proven by Lasso against a structured range table (decomposable, so not materialised); a 2^32-size naive table is called infeasible for non-Lasso lookups. Lasso cost quoted as committing 3·c·m + c·n^(1/c) field elements for m lookups into a size-n table. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Softmax: q_E = round(scale·q_Q·q_K), q_Emax = row max, q_e = exp(q_Emax − q_E) via one lookup into a shared exponentiation table of (q_t, round(e^{...})) pairs, then q_P = round(S_V·(q_e·q_V)/(S_P·Σ q_e)) i.e. the softmax normalising division is fused with the multiplication by V into one rounding. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- LayerNorm after fusion: one rounding for sigma per row (mean and variance are substituted analytically into the sigma constraint: q_sigma = round(sqrt(Σ_j (d·q_x_ij − Σ_j q_x_ij)^2 / d^2))) and one rounding per element for the output y_ij = round(S_gamma/S_y · gamma_j · (d·q_x_ij − Σ q_x)/(q_sigma·d) + beta term). The division by sigma is a checked division, not a reciprocal lookup. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- GELU: "z-GeLU", GELU(x) ≈ ReLU(x) − D(x) where D is even and approximated by a quadratic a·x^2 + b·|x| + c on |x| below a threshold and 0 above; needs one rounding per element plus one extra lookup for the threshold indicator bit s = 1[|q_x| ≥ q_th]. Reported approximation error vs exact GELU on [−4, 4]: L2 0.0054, L∞ 0.012 (vs x·sigmoid(1.702x): 0.012 / 0.020; i-GELU (I-BERT): 0.0082 / 0.018; h-GELU: 0.031 / 0.068). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Matmul: Thaler-style matmul sumcheck (as in zkCNN) with an optimised bookkeeping-table build: because quantized entries are ≤ 2^Q, precompute eq(y,R)·t for all t and reduce the build to ~n·m field additions instead of ~4·n·m multiplications; skips zeros. Reported: (32×768)·(768×3072) from 0.25 s to 0.037 s single-thread; all GPT-2 matmuls 9.8 s to 1.5 s. Multiple matmuls are batched into one sumcheck by concatenation ("circuit squeeze for matrix multiplication"). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Embedding: not proven; "our LLM circuit takes the input embedding matrix as public input". — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Rounding counts (Table 7; s = sequence length, d = hidden dim, h = heads, D = MLP width): Normalization 3s + 2sd → s + sd (49,224 → 24,608); Attention (3/2)·h·s·(s+1) + sd → (1/2)·h·s·(s+1) + sd (43,584 → 30,912); GELU 2Ds → Ds (196,608 → 98,304). The numbers imply s = 32 tokens (24,608 = 32·769). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)

**zkLLM (Sun, Li, Zhang, CCS 2024) [PR; arXiv v1 read]**
- Stack: sumcheck-based proofs per tensor op + "tlookup" (a parallelised, LogUp-style lookup with committed multiplicities m) + Hyrax/Pedersen commitments on BLS12-381 (|F| ≈ 2^254), all in CUDA. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- Quantization: scaling factor 2^16 for embeddings and parameters. "All rescaling operations were integrated with the subsequent activation functions", each via tlookups of size 2^16 (the number of tlookups equals the number of rescalings the activation input needs). — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- Softmax (zkAttn): shift the row so the result is normalised, then write the (negative) input as a K-digit base-b number and use exp(Σ digits) = Π exp(digit): one tlookup per segment, product of K table outputs. In the experiments the softmax input has cumulative scale 2^64 and uses K = 5 tlookups each of size 2^16, of which L = 3 least-significant segments are treated as constant (output ≈ 1) and the most-significant M segments degenerate to an indicator table; row-sum normalisation is checked to lie within a tolerated error E by another tlookup. Overall L1 output error ≈ 10^-2. Prover overhead O(K·m·n) for an m×n attention input. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- LayerNorm: "The compound non-arithmetic operations of square-root and inverse are managed through two sequential tlookup steps" — a first tlookup that downscales the input (to keep the table small), then one tlookup for the quantized compound 1/sqrt function. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- GELU: uses GELU(z) ≈ z·Sigmoid(1.702z) (SwiGLU handled similarly), via tlookup with rescale fused in. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- zkGPT's critique of zkLLM: proves each transformer block sequentially; uses the unoptimised matmul protocol; "did not propose techniques to reduce the number of non-arithmetic relations to prove from the constraint level". — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)

**DeepProve / DeepProve-1 (Lagrange Labs; ePrint 2026/1112, listed as CCS 2026) [ABS + REPO; PDF not readable]**
- Stack: "sumchecks and logup GKR"; sumcheck/GKR code reused from scroll-tech/ceno. Linear ops (MatMul, Conv, RMSNorm, LayerNorm) get "efficient sum-check protocols"; non-linear ops (Softmax, GeLU, requantization) get LogUp-GKR lookups; per-layer claims chain back-to-front GKR-style. — [deep-prove README](https://github.com/Lagrange-Labs/deep-prove), [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md)
- PCS: the open-source binaries use HyperKZG over BN254 (arkworks; a CUDA HyperKZG exists). — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md). Conflict: the Aug 2025 DeepProve-1 blog reportedly said the PCS was Basefold ("functional but producing large proofs and slow verification") [SEC, blog not directly readable] — [Lagrange blog](https://lagrange.dev/blog/deepprove-1). Read together, they moved from a hash-based PCS to a pairing PCS; I could not confirm the field used in the paper.
- Quantization: post-training static quantization calibrated on a dataset; default 12-bit symmetric range [−2^11, 2^11 − 1], configurable 8/10/12/16 via ZKML_BIT_LEN; requantization after each op is a lookup. — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md)
- LayerNorm: proven as RMSNorm applied to the mean-centred input [SEC snippet of the paper] — [ePrint 2026/1112](https://eprint.iacr.org/2026/1112)
- Embedding: "provable token vocabulary lookup"; also positional encoding (absolute, RoPE), causal mask, EinSum/multi-head attention as batched matmul, argmax/logits. — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md)
- "Certify, don't re-execute": concatenate prompt and claimed output tokens, run one forward pass; the causal mask plus deterministic argmax means position t certifies token t+1, avoiding O(t^2·|C|) work. — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md), [ePrint 2026/1112 abstract](https://eprint.iacr.org/2026/1112)

**ZKTorch (Chen, Tang, Kang; arXiv 2507.07031) [PP, read]**
- Stack: 61 ONNX layers compiled to ~20 "basic blocks", each a KZG-based (BN254) special-purpose argument, aggregated with a parallel generalisation of the Mira accumulation scheme. Lookups use cq (prover time independent of table size). — [arXiv 2507.07031](https://arxiv.org/html/2507.07031v2)
- Quantization: fixed-point scale 2^10 for all models except GPT-J (2^12). — [arXiv 2507.07031](https://arxiv.org/html/2507.07031v2)
- Ops: Softmax subtracts max then exponentiates via lookup; GELU: a compiler rule fuses the ~8 ONNX ops (Pow, Tanh, Mul...) into one cq lookup; Div/Mod follow ZKML (Chen et al., EuroSys '24) and a rescale follows every multiplication; MatMul uses CQLin when one operand is fixed weights, an Aurora-based matmul otherwise; reshape/transpose use a Permute block, Slice etc. a CopyConstraint block. — [arXiv 2507.07031](https://arxiv.org/html/2507.07031v2)

**Jolt Atlas (ICME Labs; arXiv 2602.17452, Feb 2026) [PP, read]**
- As published: Jolt-style sumcheck DAG with lookup arguments applied to ONNX tensor ops; HyperKZG commitments (Dory rejected as "only justified" above 2^26 elements); zero knowledge via BlindFold (Pedersen-committed sumcheck messages + Nova-style folding of the verifier R1CS). — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1)
- Nonlinearities (ReLU, Sigmoid, Tanh, Erf, Softmax) by lookup with prefix-suffix decomposition; "neural teleportation" (tau = 4) bounds activation tables to 2^16 entries, described as lossy but with output differences under 5 units at scale 128; division is non-deterministic advice. The paper does not describe its rescale/range-check or LayerNorm/rsqrt protocols. — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1)
- Note: the paper predates the Fp128 + Akita lattice stack the team now uses, so its numbers are for the BN254/HyperKZG system. — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1)

**zkAgent (SJTU; ePrint 2026/199, listed as CCS 2026) [ABS + SEC]**
- Proves the full pipeline including token-to-embedding lookup, positional encoding, transformer and decoding; "one-shot transcript proving" uses the causal mask to prove a multi-step transcript in one forward pass; a weight-dependent (input-independent) quantization scheme makes the batched proof sound and "unconditionally complete". — [ePrint 2026/199](https://eprint.iacr.org/2026/199)
- Uses "candidate-set extraction and LogUp-based lookups for softmax and LayerNorm" and tracks integer bit-width through each block [SEC snippet]. — [ePrint 2026/199](https://eprint.iacr.org/2026/199)

**OpenLLM (Zhejiang U./Ant Group; ePrint 2026/1578) [ABS + SEC]**
- Decomposes inference into reusable atomic operators with their own succinct non-interactive protocols for GELU, Softmax, normalization; LogUp lookups and matmul reduced to sumcheck [SEC]; claims better numerical fidelity and that it outperforms prior end-to-end approaches. No numbers visible. — [ePrint 2026/1578](https://eprint.iacr.org/2026/1578)

**NanoZK (arXiv 2603.18046) [PP, read partially; several internal inconsistencies noted]**
- Halo2 + IPA over Pallas, per-layer proofs chained by SHA-256 commitments supplied as public inputs. Operation-specific fixed-point formats: hidden states Q1.15, pre-softmax logits Q4.11, RMSNorm sum-of-squares Q6.9. — [arXiv 2603.18046](https://arxiv.org/html/2603.18046)
- Softmax: max-subtract, exp by lookup over [−8, 0] with 1,024 anchors + linear interpolation, reciprocal by lookup with 1,024 anchors, then a multiplication. GELU/SiLU: 2,048 anchors on [−6, 6] + interpolation. RMSNorm: in-circuit sum of squares, rsqrt lookup with 1,024 anchors on [eps, 64]; LayerNorm = RMSNorm + mean subtraction. — [arXiv 2603.18046](https://arxiv.org/html/2603.18046)

**zkPyTorch / Expander (Polyhedra; ePrint 2025/535) [ABS/BLOG]**
- Compiler from PyTorch to the Expander GKR prover. — [IACR news](https://iacr.org/news/item/25360), [Polyhedra blog](https://blog.polyhedra.network/zkpytorch/)

**Bionetta (Rarimo; arXiv 2510.06784) [PP/BLOG]**
- R1CS/Groth16 client-side framework: weights are circuit constants so linear layers cost ~0 constraints; each non-linearity costs 255 constraints (bit decomposition), with a claimed route to ~20. Compilation can be 2000x slower than EZKL. Not aimed at transformers with private weights. — [arXiv 2510.06784](https://arxiv.org/pdf/2510.06784), [Rarimo benchmark post](https://mirror.xyz/0x90699B5A52BccbdFe73d5c9F3d039a33fb2D1AF6/7pA95u4PvLyoNQ6F_X_rkN4OTmfsO_cbmm5KzbamGIo)

**EZKL [SEC]**
- Halo2-KZG circuits from ONNX with a single-region layout and einsum support for attention; nanoGPT was the published transformer target. — [EZKL nanoGPT write-up](https://hackmd.io/mGwARMgvSeq2nGvQWLL2Ww)

### Inferences
- The field has converged on jolt-atlas's own architecture (sumcheck + lookup, specialised matmul sumcheck, advice for division). What differs is the lookup budget per rounding. zkGPT spends one range relation per rounding and no table for sqrt/division; zkLLM spends several 2^16 tlookups per rescale+activation; DeepProve spends a LogUp-GKR lookup per requantization at only 12 bits.
- zkGPT's LayerNorm needs s + s·d range relations per LayerNorm (one sqrt check per row, one checked division per element) and zero rsqrt lookups. If jolt-atlas currently does an rsqrt table lookup plus a fused-rescale multiply per element, the "divide by sigma as a checked division with advice" form is the direct competitor to evaluate; it trades a lookup into a nonlinear table for a range check on a remainder whose bound is data-dependent (C2·q_y·...), which is the awkward part for a fixed-table range check.
- zkGPT's z-GELU and zkLLM's zkAttn digit decomposition are both ways to avoid a single huge table; with jolt-atlas's prefix-suffix Shout tables the table size is not the per-lookup cost driver (see Q5), so their main relevance is accuracy, not cost.
- Nobody in this set reports a special protocol for reshape/concat/transposition beyond ZKTorch's Permute/CopyConstraint blocks; in MLE-sumcheck systems these are index remappings of evaluation points and appear to be treated as free or near-free.

### Gaps
- DeepProve's exact softmax, LayerNorm/RMSNorm (inverse sqrt), range-check and embedding protocols and its field: the paper PDF and the repo's mdBook (docs/src/layers.md, subroutines.md) were not read. These are the most valuable unread sources: https://eprint.iacr.org/2026/1112.pdf and https://github.com/Lagrange-Labs/deep-prove/tree/master/docs.
- zkAgent and OpenLLM per-op protocols and tables: PDFs blocked.
- Modulus Labs Remainder, Giza Orion/LuminAIR, Mina zkML, NovaNet: my searches returned no usable primary material on transformer op proving; I found no evidence any of them publish GPT-2-scale transformer results. Treat as unresearched rather than absent.
- EZKL's current softmax/reciprocal/range-check gadgets were not retrieved.
- Lookups/constraints per element are only stated explicitly for zkGPT (rounding counts) and zkLLM (K = 5 tlookups per softmax element); other systems do not state them in what I could read.

## Q2. Reported benchmarks

### Takeaway
Headline GPT-2 numbers are not comparable: zkGPT's 21.8 s is for a 32-token sequence on 32 threads, zkLLM's 73.9 s is OPT-125M at 2048 tokens on an A100, DeepProve's is 512 tokens in 7.6 min on 24 cores, and Jolt Atlas's ~38 s (M3 laptop) does not state its sequence length. Normalising per token, the public leaders are DeepProve (~1.1 tok/s CPU, repo) and zkAgent (0.30 s/token claimed).

### Cited Findings

| System | Model / seq len | Prover time | Verify | Proof size | Hardware | Status |
|---|---|---|---|---|---|---|
| zkGPT | GPT-2, s = 32 (implied by Table 7) | 21.8 s (all opts); 52.3 s without; 319.6 s single thread without | 0.35 s | 101 KB | Xeon 6126 16-core, 32 threads, 200 GB | [PR] |
| zkLLM (as rerun by zkGPT) | GPT-2 | 15.8 s | 0.54 s | 126 KB | A100 GPU | [PR] |
| zkLLM | OPT-125M, seq 2048 | 73.9 s (commit 11.8 s) | 0.342 s | 141 kB | A100 40 GB + 12 EPYC cores | [PR] |
| zkLLM | LLaMA-2-13B, seq 2048 | 803 s (commit 986 s) | 3.95 s | 188 kB | same; 23.1 GB GPU memory | [PR] |
| ZKML (Chen et al.) | GPT-2 | 4026 s (zkGPT's run) / 3601 s (ZKTorch's run) | 12.1 s / 19 s | 7.8 KB | 32 / 64 threads | [PR] |
| Lu et al. (VOLE, interactive) | GPT-2 | 112.3 s | 31.4 s | 2.24 GB comm. | 32 threads | [PR] |
| Hao et al. (USENIX Sec '24) | GPT-2 | 6096 s | — | 6.85 GB | — | [PR] |
| ZKTorch | GPT-2 (117M) | 599 s | 12 s | not stated | Xeon 8358, 64 threads | [PP] |
| ZKTorch | BERT (110M) / GPT-J (6B) / LLaMA-2-7B | 880 s / 1397.5 s / 2645.5 s | 26.7 s / 62.6 s / 100.1 s | 4.88 / 6.54 / 22.85 MB | same, 4 TB RAM | [PP] |
| DeepProve (repo) | GPT-2, 512 tokens | 7.6 min (1.12 tok/s) | 1.3 s | 10.7 MiB | 24-core, 504 GB CPU | [REPO] |
| DeepProve (repo) | Gemma 3, 512 tokens | 19 min (0.45 tok/s) | 4.3 s | 27 MiB | same | [REPO] |
| DeepProve (paper abstract) | GPT-2 / Gemma 3 | ~174 / ~86 tokens/min single machine; up to 1855 tokens/min distributed | 1 – 3.7 s | — | not on abstract page | [ABS] |
| zkAgent | GPT-2, 512-token transcript | 0.30 s/token amortised; "1047.4x" over zkGPT; coding-assistant run 93.18 s | 1.63 s | — | not on abstract page | [ABS] |
| Jolt Atlas (paper) | GPT-2 (125M), seq len not stated | ~38 s total: witness ~7.5 s, commit ~3.5 s, sumcheck ~16 s, reduction opening ~7 s, HyperKZG prove ~3 s | not stated | not stated | MacBook Pro M3, 16 GB | [PP] |
| Jolt Atlas (paper) | nanoGPT (~0.25M params, 4 layers) | 14 s vs ezkl 237 s (ezkl number taken from ezkl's blog) | 0.517 s vs 0.34 s | — | M3 | [PP] |
| NanoZK | GPT-2-size blocks | d = 768 full block ~68 s "GPU projected"; measured only to d = 128/256 on CPU | ~22 ms per sub-circuit | ~6.9 KB per block | 1 Xeon, 64 GB | [PP], projections |
| zkPyTorch | Llama-3 8B / VGG-16 | ~150 s per token / ~2.2 s per image | — | — | single CPU core (vendor) | [ABS/BLOG] |

- Sources: [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf); [zkLLM arXiv](https://arxiv.org/abs/2404.16109); [ZKTorch arXiv](https://arxiv.org/html/2507.07031v2); [deep-prove README](https://github.com/Lagrange-Labs/deep-prove); [DeepProve ePrint abstract](https://eprint.iacr.org/2026/1112); [zkAgent ePrint abstract](https://eprint.iacr.org/2026/199); [Jolt Atlas arXiv](https://arxiv.org/html/2602.17452v1); [NanoZK arXiv](https://arxiv.org/html/2603.18046); [Polyhedra zkPyTorch](https://polyhedra.network/zkPyTorch).
- DeepProve's own numbers conflict: the repo README gives 67 tokens/min for GPT-2 at 512 tokens on 24 cores and "10–30x faster" than prior work, while the paper abstract gives ~174 tokens/min and "20–60x" — [README](https://github.com/Lagrange-Labs/deep-prove); contradicted by [ePrint 2026/1112](https://eprint.iacr.org/2026/1112). Likely different hardware/GPU or sequence length; unresolved.
- DeepProve characterises zkGPT as "≈ 0.05 tokens/s on similar hardware"; zkAgent cites zkGPT needing over 41 hours and a ~933 MB proof for a 482-token GPT-2 continuation [SEC snippet]. Both reflect that zkGPT proves one next-token inference per proof at short context. — [deep-prove README](https://github.com/Lagrange-Labs/deep-prove), [ePrint 2026/199](https://eprint.iacr.org/2026/199)
- ZKTorch's own blog gives inconsistent GPT-2 figures (~20 min and ~10 min); the paper says 599 s. — [ZKTorch blog](https://ddkang.substack.com/p/zktorch-open-sourcing-the-first-universal), [arXiv 2507.07031](https://arxiv.org/html/2507.07031v2)
- Accuracy: zkGPT perplexity FP32 → quantized: WikiText2 29.3 → 29.5, PTB 41.3 → 41.7, LAMBADA 48.4 → 48.7. zkLLM C4 perplexity: OPT-125M 26.56 → 26.65; LLaMA-2-13B 6.520 → 6.528. DeepProve: ≥ 99.6% cosine similarity to float at 12-bit (GPT-2). ZKTorch: BERT F1 90.104% vs 90.874%; GPT-J ROUGE-1 43.47 vs 42.99. — same sources as above.

### Inferences
- Jolt Atlas's published ~38 s GPT-2 figure cannot be ranked without its sequence length; zkGPT's 21.8 s is for 32 tokens. Any jolt-atlas comparison should state tokens and report s/token; the fair external yardsticks are DeepProve at 512 tokens (7.6 min / 512 ≈ 0.9 s/token on a 24-core CPU, per the README) and zkAgent's claimed 0.30 s/token.
- Proof size is where sumcheck+hash/Hyrax systems diverge: 100 – 190 kB (zkGPT, zkLLM, Hyrax) vs 10.7 MiB (DeepProve) vs MBs (ZKTorch).

### Gaps
- Jolt Atlas GPT-2 sequence length, proof size and verifier time are not in the paper. No published numbers for the Akita/Fp128 version were found.
- Hardware and sequence length behind DeepProve's 174 tokens/min and zkAgent's 0.30 s/token (PDFs blocked).
- No Qwen2 or BERT/bge numbers from any competitor except ZKTorch's BERT.

## Q3. zk-friendly approximations that work with unmodified pre-trained weights

### Takeaway
Published evidence says exact-rounded integer semantics with "verify the inverse" checks (zkGPT) and table lookups at 12 – 16 bits (zkLLM, DeepProve, NanoZK) all hold perplexity within ~0.1 – 0.5 without retraining; low-degree global polynomials do not.

### Cited Findings
- Verify-the-inverse for sqrt and division is standard in zkGPT: sqrt as (2y − 1)^2 ≤ 4x ≤ (2y + 1)^2 − 1 and division as a two-sided product inequality, each a single range relation, with perplexity increase < 0.5 on three datasets at Q = 16. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkGPT z-GELU (ReLU minus a clipped even quadratic) beats I-BERT's i-GELU, h-GELU and x·sigmoid(1.702x) in L2/L∞ error (0.0054 / 0.012) while needing one rounding + one indicator lookup per element. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkLLM uses x·Sigmoid(1.702x) for GELU and a digit-decomposed exp for softmax with tolerated row-normalisation error; perplexity increase < 0.1 on C4, shrinking to < 0.01 at 13B. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- NanoZK: piecewise-linear interpolation tables (1,024 – 2,048 anchors) give mean relative error 0.026% (GELU), 0.003% (rsqrt), softmax abs error 7.2e-6 and claimed ΔPPL < 1e-4; a degree-7 minimax polynomial baseline degrades perplexity by "106–136%". (Preprint with internal inconsistencies.) — [arXiv 2603.18046](https://arxiv.org/html/2603.18046)
- DeepProve treats LayerNorm as RMSNorm on the mean-centred input and uses calibrated 12-bit static quantization with ≥ 99.6% cosine similarity. — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md), [ePrint 2026/1112](https://eprint.iacr.org/2026/1112)
- ePrint 2026/1390 argues the LayerNorm gain 1/sigma is the main data-dependent amplifier of per-layer error in pre-LN transformers; if sigma stays above a floor, one fixed proving precision works at every depth, so cost stays near-linear in layers (measured on EZKL and Jolt Atlas). — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)
- Jolt Atlas's "neural teleportation" (tau = 4) shrinks activation tables to 2^16 with output differences under 5 units at scale 128. — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1)
- zkGPT's error analysis: fusing two roundings into one never increases quantization error (|round(G(F(x))) − G(F(x))| ≤ error of the two-step version). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)

### Inferences
- Since jolt-atlas owns the tracer, it can define rsqrt/division semantics as "the unique integer satisfying the inequality" (exact rounding) rather than "the table value". That makes the op provable by a product + range check with no nonlinear table, and it is at least as accurate as a table.
- Fusion is an accuracy improvement, not a trade-off, which removes the usual objection to changing fixed-point semantics on pre-trained weights.
- Power-of-two-base softmax and Newton-iteration verification were on the question list; I found no 2024-2026 zkML system that reports using either.

### Gaps
- No source measured I-BERT-style integer LayerNorm (iterative integer sqrt) end-to-end in a zk prover; zkGPT only compares i-GELU.
- No task-level accuracy (beyond perplexity/cosine) for BERT-style embedding models under these approximations, except ZKTorch's BERT F1.

## Q4. Reducing truncation / rescale cost

### Takeaway
Nobody eliminates per-element range checks entirely; the published wins are fewer roundings (zkGPT fusion: −50% for LayerNorm and GELU, −30% for attention) and folding the rescale into the following nonlinearity's lookup (zkLLM, ZKTorch's GELU rule, DeepProve's requantization lookup).

### Cited Findings
- zkGPT constraint fusion: classify constraints as Type-I (add/mul), II (division/rescale), III (sqrt), IV (exp lookup); merge adjacent roundings when profitable (II+II, II+III, III+III are profitable; anything with IV is not, because lookup inputs must be integers and a shared exp table must be preserved). Effect on GPT-2: prover 33.2 s → 21.8 s (1.5x), lookup time 22.4 s → 12.1 s (1.8x), advice commitment 1.5 s → 0.8 s (1.9x). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkGPT deliberately does not fuse sigma into the per-element output of LayerNorm because that would raise arithmetic and commitment overhead by a factor of N for a saving of only s roundings. Fusing max into the exp lookup is infeasible (table input would be 2^512). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkGPT cites earlier work (zkCNN line) that fused the post-convolution rescale with the following ReLU. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkLLM: all rescaling is integrated into the subsequent activation's tlookups (ReLU-with-rescale uses two tables to cut O(B·gamma) to O(B + gamma)); softmax input is left at cumulative scale 2^64 (two multiplications, no intermediate rescale) and absorbed by the K = 5 digit decomposition. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- zkGPT circuit squeeze: because all rounded intermediates are committed as advice, sub-circuits are topologically independent and are packed into the same GKR layers; per-layer sumcheck 20.3 s → 5.7 s (1.7x overall) from better parallelism. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkAgent tracks integer bit-width through each block and uses a weight-dependent quantization scheme [SEC]. — [ePrint 2026/199](https://eprint.iacr.org/2026/199)

### Inferences
- Deferred rescaling exists in practice in two forms: zkLLM's "let the scale grow to 2^64 then decompose once", and zkGPT's "substitute the un-rounded expression into the next constraint". Both rely on a field much larger than the accumulated value; a 128-bit field gives jolt-atlas less headroom than BN254/BLS but enough for two or three chained i32 multiplies if the tracer semantics are changed from i64-accumulate-then-clamp.
- Candidate fusions for jolt-atlas mirroring zkGPT: (i) mean, variance, sqrt in LayerNorm into one per-row check; (ii) QK^T rescale fused into the exp-table index computation; (iii) softmax normalising division fused with the product by V (one rounding instead of two); (iv) the two roundings in GELU into one.
- Fusion with saturating clamps is an open question: zkGPT's range relations assume values stay in range via rescale choices, whereas jolt-atlas emits saturating_clamp_i32; a fused constraint has to carry the clamp semantics or prove no saturation.

### Gaps
- I found no published transformer prover using probabilistic truncation or range-check-free rescaling; the VOLE/MPC-style techniques were not surfaced by my searches. This is "not found", not "does not exist".
- DeepProve's requantization lookup width and whether it batches range checks are unread.

## Q5. Lookup-argument trends for a large-field sumcheck prover

### Takeaway
Among the systems surveyed, LogUp variants (tlookup, LogUp-GKR, cq) dominate deployed zkML, Lasso is used by zkGPT for structured range tables, and Shout is used only by Jolt-lineage provers; the one analysis specific to Shout-based zkML says per-lookup cost depends only on the committed address width, not on the table's contents.

### Cited Findings
- Shout (Setty–Thaler–Zhu, ePrint 2025/105): commits only the one-hot address matrix; no grand product/grand sum. With curve commitments, Shout at d = 2 has "at least 8x fewer nonzero committed values and 3.4x fewer field multiplications than Lasso" for structured tables; in Jolt it gave a 3x prover speedup and ~5x smaller proofs. — [ePrint 2025/105](https://eprint.iacr.org/2025/105)
- Shout cost shape: for d = 1, sumcheck is O(K + T) with T nonzero committed ones; splitting the address into d chunks cuts the key/commitment dimension to K^(1/d) but commits d·T nonzeros and raises sumcheck degree from 3 to 2 + d; MLE-structured tables need not be committed because the verifier evaluates them in O(log K). — [Twist and Shout summary](https://georgwiese.github.io/crypto-summaries/Papers/Twist-and-Shout)
- "Separation Principle" (Min Jun Jo, ePrint 2026/1390, single-author preprint): in a Shout-style one-hot argument, per-lookup proving work depends only on the access pattern, not table values; a nonlinearity's structure (low degree, parity, kernel form) only reduces a once-per-proof table term. The only remaining lever is the committed address width, which the author reports yields a bit-exact prover-time reduction "upstreamed" (measured on EZKL/halo2 and Jolt Atlas). — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)
- Lasso as used by zkGPT: 3·c·m + c·n^(1/c) committed field elements for m lookups; lookups are still 55% of its optimised prover time (12.1 s of 21.8 s). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- LogUp-GKR is DeepProve's lookup for softmax, GELU and requantization; cq is ZKTorch's (prover independent of table size, KZG-specific); zkLLM's tlookup commits a multiplicity vector and keeps table size N = O(D), using 2^16 tables. — [zkml README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md), [arXiv 2507.07031](https://arxiv.org/html/2507.07031v2), [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- A "logup*" note positions itself as field- and commitment-agnostic and discusses carrying the one-hot idea to hash-based schemes [SEC snippet only]. — [logup* note](https://levs57.github.io/logup_.pdf)

### Inferences
- For jolt-atlas the 2026/1390 result is directly actionable: effort spent making a nonlinearity "algebraically nicer" does not reduce Shout cost; reducing the bits of address per lookup (narrower input domain via clamping/teleportation, fewer chunks d) and reducing the number of lookups (fusion, Q4) do.
- Range checks on remainders in [0, 2^k) are the smallest-address lookups possible; so a "product + range check" formulation of rsqrt/division (Q3) replaces a wide-address nonlinear lookup with a narrow-address range lookup, which by the separation principle is where real savings are.
- With a lattice PCS where small/one-hot values are cheap, the Shout assumption ("cheap 0/1 commitments") carries over in spirit, but I found no published cost model for Shout over a lattice PCS.

### Gaps
- No source directly compares Shout, Lasso, LogUp-GKR and cq per-lookup prover cost for 2^8 – 2^16 tables or for 2^32 – 2^64 structured tables under a lattice (or any small-value-friendly non-curve) PCS. I found no published lattice-based or folding-based zkML transformer prover at all in 2025-2026 searches; the only folding/accumulation system is ZKTorch (Mira, pairing-based).
- The full Twist-and-Shout paper's concrete multiplication counts were not read (PDF blocked).
- "Lookup-free" algebraic approaches for transformers: nothing found beyond low-degree polynomial activations, which NanoZK reports as destroying perplexity.

## Q6. Published cost breakdowns: what dominates prover time

### Takeaway
The two breakdowns available disagree on the dominant term in a way that tracks the PCS: in zkGPT (Hyrax, Lasso) lookups/range checks are 55% of prover time; in Jolt Atlas's published HyperKZG system sumcheck is ~42% and commitment + opening ~35%.

### Cited Findings
- zkGPT GPT-2 breakdown, all optimisations (21.8 s): commit advice 0.8 s; GKR layer sumcheck 5.7 s ("26%"); GKR combine sumcheck 3.2 s; Lookup (Lasso, range + exp) 12.1 s. Without constraint fusion: lookup 22.4 s of 33.2 s. All GPT-2 matmuls cost 1.5 s after optimisation (9.8 s before). — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- zkGPT: "proving range relations is the bottleneck in our system, as it is more costly" than arithmetic relations; all normalization layers of GPT-2 prove in under 30 s vs > 4000 s in Hao et al. — [zkGPT PDF](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf)
- Jolt Atlas GPT-2 (~38 s): witness generation ~7.5 s, commitment ~3.5 s, sumcheck proving ~16 s, reduction opening proof ~7 s, HyperKZG prove ~3 s. — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1)
- zkLLM: attention-specific (zkAttn) overhead depends strongly on sequence length and only weakly on model size; commit time scales with parameters while proving time scales more slowly. — [arXiv 2404.16109](https://arxiv.org/abs/2404.16109)
- ePrint 2026/1390 (abstract): proof generation cost "concentrates in the nonlinear functions a transformer must evaluate inside the proof system". — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)
- DeepProve blog (secondhand): Basefold PCS was the acknowledged weak point (large proofs, slow verification), consistent with the later switch to HyperKZG and the 10.7 MiB proof. — [Lagrange blog](https://lagrange.dev/blog/deepprove-1), [deep-prove README](https://github.com/Lagrange-Labs/deep-prove)

### Inferences
- Across systems, matmul is a minor cost once a dedicated sumcheck is used (1.5 s of 21.8 s in zkGPT); the per-element nonlinear/rounding work (softmax over s^2·h entries, GELU over s·D, LayerNorm over s·d) dominates. For long sequences the s^2 attention term takes over.
- The lever with the best published evidence is cutting the number of rounding/range relations (1.8x on the dominant term in zkGPT), followed by parallel structure of the sumcheck schedule (1.7x), then matmul micro-optimisations (6.5x on a small term).

### Gaps
- No per-op (softmax vs LayerNorm vs GELU) time breakdown was found in any paper I could read; zkGPT gives only rounding counts per op.
- DeepProve's and zkAgent's breakdowns are in unread PDFs.
- No third-party, same-hardware benchmark across zkGPT / zkLLM / DeepProve / Jolt Atlas exists that I could find; all cross-system numbers are self-reported or rerun by a competitor.
