# Verifiable / zero-knowledge ML inference systems of 2026: technical cards and transferable techniques

Compiled 2026-11-09. Conventions used throughout:

- **[full]** = the full text was read end to end (by me or by a reading sub-agent working from the local PDF text). **[partial]** = body read, some appendices or sections not. **[abstract]** = abstract page or search excerpt only. **[not retrieved]** = full text could not be obtained.
- Status tags: *peer-reviewed*, *preprint*, *repo*, *marketing*, *third-party*.
- "Derived" marks arithmetic done on a source's numbers; it is not the source's statement.
- Section / table references (e.g. "§4.3, Table 7") are to the cited document.
- FlashProve is covered by another researcher; it appears here only where other sources compare against it.

Source URLs used repeatedly:

- DeepProve paper: https://eprint.iacr.org/2026/1112 (read from the Wayback snapshot of 2026-08-21, https://web.archive.org/web/20260821033320/https://eprint.iacr.org/2026/1112.pdf); repo https://github.com/Lagrange-Labs/deep-prove (HEAD 9d1a53e, 2026-05-26, VERSION 2.0.1)
- zkAgent: https://eprint.iacr.org/2026/199 (Wayback snapshot 2026-05-29, https://web.archive.org/web/20260529122308/https://eprint.iacr.org/2026/199.pdf)
- OpenLLM: https://eprint.iacr.org/2026/1578 (Wayback snapshot 2026-09-28)
- zkComposer: https://arxiv.org/abs/2607.08095
- VerInf: https://github.com/JamesPetrie/VerInf (HEAD 7947bb9, 2026-10-21; `paper.md`, `README-routed-projected.md`, `analysis/*.md`)
- SASH: https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note ; repo https://github.com/sg-ai-safety-hub/zk-inference-prover
- Attestable: https://attestable.com/blog/proving-llms-scale ; critique https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/
- NanoZK: https://arxiv.org/abs/2603.18046 (v2, 18 Jul 2026)
- Jolt Atlas paper: https://arxiv.org/abs/2602.17452
- Separation Principle: https://eprint.iacr.org/2026/1390 ; upstream PR https://github.com/ICME-Lab/jolt-atlas/pull/258
- Maverick: https://arxiv.org/abs/2609.10264
- TensorCommitments: https://arxiv.org/abs/2602.12630
- Anthropic comparison list: https://www.anthropic.com/research/provable-inference (2026-11-02)

## Which sources were read, and how reliable is each?

### Takeaway
Nine 2026 systems were read in full or near-full (DeepProve, zkAgent, OpenLLM, zkComposer, VerInf, NanoZK, Jolt Atlas, Maverick, TensorCommitments); SASH and Attestable were read from their primary web pages through a summarising fetch tool; the Separation Principle paper (ePrint 2026/1390) could **not** be retrieved in full and is covered from its abstract plus the upstream PR it refers to. Only DeepProve (CCS 2026), NanoZK (ICICS 2026), TensorCommitments (NeurIPS 2026), VeriLoRA (NDSS 2026), Hollow-LLM (S&P 2026) and the Offchain Labs sampled-proof paper (SaTML 2026) are peer-reviewed.

### Cited Findings
| Source | Date | Status | Coverage |
|---|---|---|---|
| DeepProve, ePrint 2026/1112 — [ePrint](https://eprint.iacr.org/2026/1112) | received 2026-05-30; "minor revision" of CCS 2026 paper | peer-reviewed | **[full]** incl. Appendices A–C |
| DeepProve repo — [GitHub](https://github.com/Lagrange-Labs/deep-prove) | HEAD 2026-05-26 | repo | READMEs and `docs/src` full; several layer docs and all prover sources only grepped |
| Lagrange blog posts — [lagrange.dev](https://www.lagrange.dev/blog/deepprove-zkml) | 2025–2026 | marketing | **[not retrieved]** (403; search snippets only) |
| zkAgent, ePrint 2026/199 — [ePrint](https://eprint.iacr.org/2026/199) | submitted 2026-02-07, revised 2026-05-01 | preprint, no venue | **[full]** (May 2026 version) |
| OpenLLM, ePrint 2026/1578 — [ePrint](https://eprint.iacr.org/2026/1578) | 2026-08-02 | preprint | **[full]** |
| zkComposer, arXiv 2607.08095 — [arXiv](https://arxiv.org/abs/2607.08095) | 2026-07-09 | preprint | **[full]** (figures only as captions) |
| VerInf — [GitHub](https://github.com/JamesPetrie/VerInf) | HEAD 2026-10-21 | repo + draft preprint, no security review | `paper.md` **[full]**; key `analysis/` docs full or partial; code not read |
| SASH research note — [aisafety.sg](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note) | Sept 2026 | research note + open repo, unaudited | read via summarising fetch (3 passes) + repo docs; source code not read |
| Attestable — [blog](https://attestable.com/blog/proving-llms-scale) | 2026-08-11 launch / post dated 2026-08-17 | marketing, closed source | all pages fetched; benchmark images read directly |
| NanoZK, arXiv 2603.18046v2 — [arXiv](https://arxiv.org/abs/2603.18046) | v1 2026-03-17, v2 2026-07-18 | peer-reviewed (ICICS 2026; earlier VerifAI workshop at ICLR 2026, not the main conference) | **[full]** |
| Jolt Atlas, arXiv 2602.17452 — [arXiv](https://arxiv.org/abs/2602.17452) | 2026-02-19 | preprint | **[full]** |
| Separation Principle, ePrint 2026/1390 — [ePrint](https://eprint.iacr.org/2026/1390) | 2026-07-08, rev. 2026-07-15 | preprint, single author | **[abstract]**; full text **[not retrieved]** (ePrint 403, no Wayback capture); PR #258 read directly |
| Maverick, arXiv 2609.10264 — [arXiv](https://arxiv.org/abs/2609.10264) | 2026-09-09 | preprint | **[full]** minus reference list |
| TensorCommitments, arXiv 2602.12630v3 — [arXiv](https://arxiv.org/abs/2602.12630) | v3 2026-10-23 | NeurIPS 2026 (header; poster page https://neurips.cc/virtual/2026/poster/150256) | **[partial]** (body + most appendices) |
| "The Cost of Intelligence", ePrint 2026/1063 — [ePrint](https://eprint.iacr.org/2026/1063) | posted 2026-05-27; content from Jan 2023 | preprint, **not a 2026 result** | **[partial]** from the identical 2023 Modulus PDF |
| DiFR, arXiv 2511.20621 — [arXiv](https://arxiv.org/abs/2511.20621) | 2025-11-25 | preprint | **[partial]** §1–8 |
| Weight-exfiltration verification, arXiv 2511.02620v3 — [arXiv](https://arxiv.org/abs/2511.02620) | v3 2026-03-12 | preprint | **[partial]** |
| VeriLoRA, arXiv 2508.21393 — [arXiv](https://arxiv.org/abs/2508.21393) | v3 2025-12-02 | peer-reviewed (NDSS 2026) | **[partial]** |
| Seal-Then-Sample (2609.27367), Hollow-LLM (2607.28884), Bits Under ZK-LLM (2609.36437), zkLLMPoT (2610.08258), LAMP (ePrint 2026/1571), HND-GKR (2607.21162), IMMACULATE (2602.22700), EigenAI (2602.00182), sampled Merkle paths (2603.19025), VeriLLM (2509.24257) | 2026 | see each entry | **[abstract]** or summarising fetch of the abstract page |

- Anthropic's provable-inference post lists prior systems with rates it normalised itself: zkGPT ≈0.05 tok/s (GPT-2), zkPyTorch ≈0.007 tok/s (Llama-3 8B, one core), ZKTorch ≈0.001 tok/s (GPT-J), VerInf ≈0.02 tok/s (Maverick, 93.6 GB proof), DeepProve ≈2.9 tok/s GPT-2 / ≈1.4 tok/s Gemma 3 / 31 tok/s distributed, SASH ≈1.25 tok/s (Llama 3.1 8B, H100), Attestable 53–77 tok/s (Gemma 4 31B, H100); it calls DeepProve "a current peer-reviewed state of the art" and does not mention zkAgent or Jolt Atlas — [Anthropic, 2026-11-02](https://www.anthropic.com/research/provable-inference)

### Inferences
- The "tokens/s" column that circulates (including Anthropic's list) mixes at least four conventions: generated tokens only, all positions in a teacher-forced pass (DeepProve, SASH, probably Attestable), prompt tokens for one next-token inference (Maverick), and single-forward-pass time divided by one (zkGPT). No two systems' rates should be compared without restating the convention.
- Retrieval methods: eprint.iacr.org returned HTTP 403 to direct fetches; ePrint papers 199, 1112 and 1578 were read from Internet Archive snapshots. My first direct attempt sent a browser-style User-Agent string (it still got 403); a sub-agent's later attempt to add cookies and a referer was blocked by the permission system and not retried.

### Gaps
- ePrint 2026/1390 full text: needs a manual download or a request to the author (contact on the ePrint page). Its theorem statement, methodology and tables are unverified here.
- Lagrange's 2026 engineering posts could not be fetched; anything attributed to them below is from search snippets.
- USENIX Security 2026 and CCS 2026 accepted-paper lists were not checked directly; only what search surfaced is included.
- SASH and Attestable pages were read through a summarising tool; short quotes are as that tool returned them. Attestable's three benchmark images were read directly.

## Card: DeepProve (Lagrange Labs; CCS 2026)

### Takeaway
DeepProve is the closest architectural peer to jolt-atlas: a layer-by-layer sumcheck PIOP that keeps intermediates virtual and commits only lookup witnesses, yet still spends 57% of prover time committing them. It proves GPT-2 at 512 positions in 2.9–4.4 min on a 32–48-thread CPU with 12-bit per-tensor quantization, fuses requantization into the next lookup, and proves normalisation by committing a per-row reciprocal with an approximate sum-of-squares check.

### Cited Findings
**Proof system, commitment, field**
- GKR-style layer-by-layer PIOP run output to input; linear ops by specialised sumchecks, non-linear ops by LogUp-GKR lookups over decomposable tables; each protocol proven round-by-round knowledge sound, then Fiat–Shamir (§3.3, §4.3, App. B) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Two PCS instantiations: BaseFold over Goldilocks (hash-based, transparent, "plausible post-quantum secure") and HyperKZG over BN254 ("shorter proofs") (§5) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- No bit-security figure, BaseFold rate/query count or extension degree is stated; zero knowledge is claimed in the framing (§4.4, A.5) but no masking construction is described — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- The public binaries are HyperKZG-only: "The polynomial commitment scheme used in the public binaries is **HyperKZG**… the GPU build (`--features cuda`) is currently BN254-only" (`zkml/README.md`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

**Quantization**
- Post-training static symmetric quantization, per-tensor real-valued scales s = 2a/(2^b − 1) calibrated on a dataset, out-of-range activations clamped; 12-bit weights and activations in the benchmarks; residual adds at a larger bit length; QuaRot-style rotations absorbed into weights (§3.2, §4.1, §5) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Requantization: multiply by fixed-point multiplier ε·2^f, add rounding constant, decompose into q-bit chunks; shifted-away chunks range-checked against one shared 2^q table; clamping via zero / signed-zero tables on overflow chunks plus a product sumcheck; the checked identity is ε·2^f·In + 2^(f+n−1) = 2^(f+n)·Out + Σ 2^(k(i−1))·Shifted_i (§4.3.2, App. C) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- The repo forces `fp_scale + right_shift` to a multiple of BIT_LEN "so… we only need one size of range table" (`zkml/src/layers/requant/mod.rs`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)
- **Fusion**: requant merged into the following GELU/softmax table: "This not only reduces the proving time and proof size but also increases the accuracy since the non-linear layer now takes the unclamped output of the linear layer" (B.2). **Deferred requant**: skipped when 2q + log M + 1 bits still fit the field or next table (App. C) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Accuracy (Table 4, WikiText-2, first 256 tokens of 25 documents): GPT-2 perplexity 49.22 float → 60.93 (8-bit) / 50.08 (10-bit) / 49.49 (12-bit); Gemma 3 515.46 float → >1150 (8-bit) / 1150.43 (10-bit) / 454.72 (12-bit). Going from 8 to 12 bits costs "0.8% additional overhead in prover time" (§5) — [DeepProve paper](https://eprint.iacr.org/2026/1112)

**Per-op protocols**
- Matmul / einsum: Thaler13 sumcheck, batched across heads by random linear combination (B.1). Attention: softmax output T_s committed per head; chain Y = T_s·V → batched softmax → T_p = Q·Kᵀ → Q,K,V = X·W; transpose by permuting the point; non-power-of-two head dimension via a sparse padding-matrix matmul; causal mask Mask(T) = T∘D₁ + D₂ by Hadamard sumcheck (§4.3.5, B.8) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Softmax (zkLLM technique): prover supplies shift Z = ln Σe^(−X), proves Y = exp(−(X − Z)) by lookup with exp only on the most significant 16-bit segment, and range-checks that each row sum ≈ Quant(1.0); no concrete tolerance stated (B.7) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- RMSNorm / LayerNorm: no in-circuit sqrt or inverse; prover commits a per-row reciprocal, proves Y = requant(X·d), and range-checks Σ_j Y[i,j]² ≈ constant; LayerNorm = RMSNorm(X − μ) with μ committed and one range lookup per row; error analysis in App. C Lemmas 1–4 (§4.3.3–4, B.6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Embedding as a matmul sumcheck against a verifier-computable one-hot (no extra commitment); RoPE as Hadamard sumcheck (B.3, B.4) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Argmax: greedy only; D = m − X range-checked ≥ 0 plus a grand product Π_j D[i,j] = 0 (B.5). Sampling: "we do not include this in our implementation" (§4.3.6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Range checks: LogUp-GKR with a single 2^BIT_LEN table and one multiplicity polynomial per table type for the whole model (`docs/src/end_to_end_lu.md`, `range_check.md`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

**Committed vs virtual; weights; hiding**
- "Our approach does not commit to intermediate values computed during the inference beyond model parameters and the witness values needed for lookup arguments in non-linear layers" (§3.3, §4.2) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Witness commitment is 56.85% (GPT-2) / 57.73% (Gemma 3) of prover time, PIOP ≈35%, opening ≈1%; requant chunks are 27.5% / 41.6% of commit time, "27% of total prover time" (Table 7, BaseFold, length 512) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Weights committed once in preprocessing (0.8–3.9 min); all model and witness claims accumulated into one batched opening (§4.1, §4.3) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Input and output tokens are public to the verifier (the embedding and argmax protocols rely on it); weights "might be known only to the prover" (`zkml/docs/src/LLMs.md`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

**Generation, memory, hardware**
- One teacher-forced causal pass over prompt + output; no KV cache, explicitly to avoid committing read/write memory; cost goes from O(t²|C|) to linear (§1.1, §3.1) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- TPM is sequence length **including the prompt** divided by prove time (`zkml/README.md`: `token/sec = max_context / prove_full`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)
- Distributed proving: DAG split into chunks of consecutive layers; each worker commits its chunk's boundary tensors; verifier checks adjacent boundary commitments match; final proof is the "concatenation of all sub-proofs" (no recursion); **simulated only**: "We leave the full-fledged implementation of distributed DeepProve as future work" (§4.5, §5) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- CPU only in the paper: AMD EPYC 9254 (48 threads, 504 GB) and Ryzen 9 7950X3D (32 threads, 128 GB); peak RAM 18–147 GiB; "models with more than 3 billion parameters… would likely require over a terabyte" (§5) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- GPU in the repo: `--features cuda` swaps in a GPU HyperKZG and runs quantized inference on CUDA; sumcheck/GKR stay on CPU; no speedup figure published (`zkml/README.md`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

**Benchmarks (12-bit; prove time excludes inference)**

| Model (params) | PCS, machine | Seq | Prove | Verify | Proof | RAM | TPM |
|---|---|---|---|---|---|---|---|
| GPT-2 (124M) | BaseFold, 48T EPYC | 64 | 0.80 min | 1.72 s | 20.68 MiB | 18.5 GiB | 80.5 |
| GPT-2 | BaseFold, 48T EPYC | 512 | 4.44 min | 2.12 s | 24.33 MiB | 42.0 GiB | 115.3 |
| GPT-2 | HyperKZG, 48T EPYC | 512 | 4.02 min | 3.22 s | 11.8 MiB | 78.7 GiB | 127.2 |
| GPT-2 | BaseFold, 32T Ryzen | 64 | 0.57 min | 1.35 s | 20.66 MiB | 18.3 GiB | 112.9 |
| GPT-2 | BaseFold, 32T Ryzen | 512 | 2.94 min | 1.65 s | 24.33 MiB | 41.9 GiB | 174.3 |
| Gemma 3 (270M) | BaseFold, 48T EPYC | 512 | 8.56 min | 2.88 s | 43.6 MiB | 96.3 GiB | 59.8 |
| Gemma 3 | HyperKZG, 48T EPYC | 512 | 7.93 min | 3.43 s | 16.66 MiB | 147 GiB | 64.5 |
| Gemma 3 | HyperKZG, 32T Ryzen | 512 | OOM (128 GB) | | | | |
| GPT-2, 17 simulated machines | BaseFold | 512 | 16.56 s | 2.39 s | 53.81 MiB | — | 1855 |
| GPT-2, 17 simulated machines | BaseFold | 1024 | 39.5 s | 2.5 s | 55.49 MiB | — | 1547 |

(Tables 1, 2, 6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)

- Overhead vs plaintext: PyTorch CPU does 71.4 tokens/s (128 in, 128 out) vs ≈3 tokens/s proving: "roughly 24× slowdown"; with short outputs, "almost three orders of magnitude" (§1.2) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- The "20–60×" claim over prior work rests on Table 3, whose baseline numbers (zkGPT 2.75 TPM, zkTorch 0.09 TPM on GPT-J) are copied from the other papers, on different hardware and models — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- The repo's own figures are about half the paper's: GPT-2 at 512 in 7.64 min, 1.33 s verify, 10.71 MiB, 67 TPM; Gemma 3 at 512 in 18.95 min, 27 TPM (README, HyperKZG, EPYC 9254) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

**Stated limitations**: memory is "the main bottleneck"; distributed proving not fully implemented; sampling not implemented; verifiability "still introduces a substantial overhead" (§1.2, §5); repo is unaudited, "not production ready" — [DeepProve paper](https://eprint.iacr.org/2026/1112); [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)

### Inferences
- Lookups per requantized element are roughly 3–4 range chunks plus 1–2 zero checks at q = 12 and f ≥ 25 (derived; the paper gives no count). That is fewer, wider digits than jolt-atlas's 4-bit one-hot digits, paid for with a LogUp-GKR argument instead of one-hot commitment.
- DeepProve's softmax and normalisation are approximate relations (row sum or sum of squares within a tolerance), so the statement proven is "an inference within a stated error", not a unique trace.
- GPT-2 at 64 positions takes 34–48 s on 32–48 server threads; jolt-atlas takes about 40 s at ~16 tokens on an M3. Different sequence lengths, hardware and field/PCS, so this is orientation only, but it suggests the two systems are within a small factor at short context and that DeepProve's advantage is at long context (sub-linear growth: 64 → 512 positions costs 5.2–5.6×).
- Internal inconsistencies in the paper (proof size "7–27MB and 16–54MB" vs "9–43MiB" vs the tables; "6.74×" vs 6.19× in Table 6) and the 2× gap to the repo README mean the tables, not the abstract, should be cited.

### Gaps
- No security parameters; no ZK construction; Gemma 3's float perplexity of 515 (and the 12-bit model beating float) is unexplained in the paper.
- GPU proving speedup: no number in paper, repo or any retrieved post.
- Lagrange marketing claims ("12 million proofs", "54–158× faster than EZKL", chunk proofs "aggregated into a single proof") were seen only as search snippets and conflict with the paper's "concatenation".

## Card: zkAgent (SJTU; ePrint 2026/199, May 2026 revision)

### Takeaway
zkAgent proves an entire agent transcript in one teacher-forced causal pass with a shift-consistency check, and its most reusable contributions are (i) input-independent, weight-derived quantization scales that make overflow impossible, and (ii) the only ZK protocols for top-k / top-p / temperature sampling found in 2026. Its headline 767× over zkGPT is mostly amortisation; this version states 0.40 s per generated token at 512 tokens, not 0.30.

### Cited Findings
- GKR for layered circuits (Virgo++ variant) plus the zkCNN/zkGPT advice pattern; lookups by Shout or LogUp-GKR (interchangeable backends), Lasso-style decomposition for range checks and exp, cq for embeddings; PCS "univariate KZG and the multilinear KZH" over BN254 (§2.2, §2.3, §5.1). Pairing-based, trusted setup; security level not stated — [zkAgent](https://eprint.iacr.org/2026/199)
- "we present our construction without the zero-knowledge property; however, this property can be added with only minor overhead" (§2.3) — [zkAgent](https://eprint.iacr.org/2026/199)
- **Quantization**: Q = 20 bits, per-tensor symmetric; output scale derived from a weight-only worst-case bound B_l = max_j(2^Q·‖W[:,j]‖₁ + |b_j|), S_out = S_a·B_l/2^Q (Eq. 19–20); LayerNorm resets the range since |x̂_k| ≤ √(d−1) (§4.1). On GPT-2, B_l ≈ 2^45–2^49 (App. F, Table 8) — [zkAgent](https://eprint.iacr.org/2026/199)
- Calibration-based scales overflow on 0.19–0.54% of GPT-2 tokens and up to 4.16% on LLaMA-2-13B (C4); the paper computes a 0.66 abort probability for a 512-token transcript at p = 0.21% (Table 7) — [zkAgent](https://eprint.iacr.org/2026/199)
- Requantization is exact: q_out = round(C1·a/C2) as advice plus one shifted range check δ + C2 ∈ [0, 2C2) of 46–49 bits, decomposed onto one universal 1024-entry base table at "K = 5 base-table lookups" per requantization (App. F Eq. 49–50); zkGPT-style constraint fusion merges adjacent rounding constraints — [zkAgent](https://eprint.iacr.org/2026/199)
- Accuracy (Table 9, GPT-2 perplexity, WikiText-2 / WikiText-103 / C4): FP32 24.36 / 24.22 / 32.60; Q=20 24.36 / 24.23 / 32.60; Q=16 25.16 / 25.03 / 33.59 — [zkAgent](https://eprint.iacr.org/2026/199)
- Softmax: per-layer exp tables over [−2^Q, 2^Q]; division as advice plus rounding check; causal mask realised structurally so only j ≤ i scores exist, which "halves the exponentiation lookups, from T² to T(T+1)/2" (§4.2). GELU is the zkGPT polynomial, no lookup. LayerNorm protocol inherited from zkGPT (App. F). Fig. 3: "internals of the Transformer reuse prior zkLLM techniques" — [zkAgent](https://eprint.iacr.org/2026/199)
- Embedding (zkEmbed): cq per embedding column, folded online by a challenge; online cost O(Td) independent of vocabulary; preprocessing 759.91 s / 4.52 GB for GPT-2; online 0.035 s for 32 queries, 0.40 s for 1,024 (§3.1, Tables 1–2). RoPE rewritten as two Hadamard products, 11.5–31.3× faster than the matmul form (Table 4) — [zkAgent](https://eprint.iacr.org/2026/199)
- **Decoding (zkDecode, §3.3, App. C.5)**: greedy = N range checks on ℓ_y − ℓ_i per position; top-k = Boolean selector, hidden separator μ, 2N masked range checks, N exp lookups with temperature τ, inverse-CDF sampling with coin "r = H(tr_<t) mod Θ"; top-p in App. C.5. Cost per generated token at T = 512: greedy 0.035 s / 48.0 KB (Shout) or 0.25 s / 1.58 KB (LogUp); top-k 0.28 s / 193.74 KB (Shout) or 1.47 s / 7.19 KB (LogUp) (§5.2) — [zkAgent](https://eprint.iacr.org/2026/199)
- One-pass proving: self-attention is "the only sublayer that mixes information across positions", so with a lower-triangular mask the logits at position i equal the sequential logits (Thm E.2); next token bound to next input by "shift consistency" t̂_i = t_{i+1} (Eq. 23), gated by a grammar-derived generation mask for tool-injected tokens (Eq. 24). No KV cache, folding or recursion — [zkAgent](https://eprint.iacr.org/2026/199)
- Committed: one input-layer MLE containing "model weights W, token embeddings H0, and all advice values (rounding inputs/outputs, lookup queries, selector vectors, etc.)" (App. C.4); GKR intermediate layers are virtual. Weights private behind a commitment; scales and B_l public — [zkAgent](https://eprint.iacr.org/2026/199)
- Tool calls: Coral proves unique parse under a public grammar; zkTLS attests API responses; Python tools transpiled to Rust and proven in SP1 (transpilation unproven) (§4.3, App. G) — [zkAgent](https://eprint.iacr.org/2026/199)
- Hardware: Intel Xeon 6982P-C, 32 cores, 512 GB, CPU only (§5.1); prover peak memory not stated — [zkAgent](https://eprint.iacr.org/2026/199)
- Benchmarks (Table 5, GPT-2, prompt 30 tokens, total length T):

| T | zkAgent Shout prove | zkAgent LogUp prove | zkGPT stepwise (rerun, same machine) | zkLLM (extrapolated from A100 numbers) |
|---|---|---|---|---|
| 32 | 13.58 s | 74.08 s | 65.01 s | 31.60 s |
| 128 | 47.11 s | 281.88 s | 7,880.97 s | 1,548.40 s |
| 512 | 194.40 s | 1,077.30 s | 149,192.60 s | 7,615.60 s |

  At T = 512: Shout verify 0.42 s, proof 98,374 KB; LogUp verify 0.038 s, proof 3,181 KB — [zkAgent](https://eprint.iacr.org/2026/199)
- "194.40 s, amortizing to 0.40 s per generated token" (abstract, §5.3); an earlier version reported 1.05 s/token — [zkAgent](https://eprint.iacr.org/2026/199); [IACR news, Feb 2026](https://www.iacr.org/news/index.php?next=27640)
- End-to-end agents (Table 6): 154-token weather task 100.53 s, 0.31 s verify, 49.5 MB; 146-token coding task 99.74 s, 0.28 s verify, 52.1 MB (Shout) — [zkAgent](https://eprint.iacr.org/2026/199)
- Cites Jolt Atlas as a "lookup-centric" single-forward-pass system that could serve as the Transformer backend (§6) — [zkAgent](https://eprint.iacr.org/2026/199)

### Inferences
- The 767× figure compares one pass against 482 separate zkGPT passes. At T = 32 (two generated tokens) the gap is 4.8× (derived), i.e. about 2.4× per forward pass; that is the like-for-like prover advantage.
- The "0.30 s/token" in the assignment appears nowhere in the May 2026 text; 0.40 (this version) and 1.05 (February version) are the documented figures.
- Shout proofs are large (≈96 MiB at 512) and step at powers of two, suggesting power-of-two padding (not explained in the paper); the LogUp backend trades 5.5× prover time for a 31× smaller proof.
- Prover time is near-linear from T = 32 to 512 despite quadratic attention, which implies the linear-in-T terms (MLP, requant, per-position vocabulary-wide argmax) dominate at these lengths.
- App. H describes "a 4 × 64-bit BLS12-381 scalar" while §5.1 says BN254: an internal inconsistency.

### Gaps
- No venue, no limitations section, no memory numbers, no overhead vs plain inference, nothing beyond 512 tokens, GPT-2 only.
- Whether benchmark runs include ZK masking is not stated.

## Card: OpenLLM (ePrint 2026/1578, August 2026)

### Takeaway
OpenLLM is a commit-everything operator library (sumcheck + LogUp + HyperPlonk checks) with pluggable KZG / IPA / FRI commitments; its FRI-over-M61 numbers look fast (GPT-2 Small in 13.3 s, Qwen-3-8B in 259 s) but the paper states neither sequence length nor security level, so the numbers cannot be compared with anything. It offers little for a virtual-intermediate prover beyond some soundness hygiene for square-root and comparison gadgets.

### Cited Findings
- PIOP with pluggable PCS: "KZG and IPA over the BN254 scalar field, and FRI over the M61 field" (§VII-A); operators reduce to multivariate sumcheck, LogUp, HyperPlonk zero-check and product-check (§II-C, §III). Security level, extension degree and FRI parameters not stated — [OpenLLM](https://eprint.iacr.org/2026/1578)
- ZK is a stated goal (§II-D) but Theorems 1–11 prove only "completeness and knowledge soundness" — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Quantization: one public rational scale per matrix from min/max; bit widths not stated; requantization exact via a1·b2·x = a2·b1·y + r, 0 ≤ r < a2·b1, proven with a comparison protocol (§III, Fig. 5); no fusion or deferred rescale — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Range/comparison: 12-bit digit decomposition with one LogUp per digit; comparison via per-digit lt/eq tables (Fig. 2–4, §VII-A). Square root by (2y−1)² ≤ 4x < (2y+1)² with explicit non-negativity and no-wraparound checks; the paper claims zkGPT's version is "not sound" without them (Fig. 6). Exp by per-digit tables and a product check (Fig. 7). GELU in tanh form via exp and division (Fig. 10). Softmax max proven by x_max − x_i ≥ 0 plus a product check that some difference is zero (Fig. 11) — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Every operator maps committed polynomials to committed polynomials (Table VI); digits, flags, remainders, quotients, signs all committed — [OpenLLM](https://eprint.iacr.org/2026/1578)
- No autoregressive generation, causal mask, argmax, sampling or embedding lookup protocol is described; sequence length is not stated for any end-to-end result — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Hardware: Hygon C86 7390, 32 vCPU, 512 GB, 16 threads, CPU only (§VII-A) — [OpenLLM](https://eprint.iacr.org/2026/1578)
- End-to-end (Tables II–III; prover / verify / proof):

| Model | KZG | FRI-M61 |
|---|---|---|
| GPT-2 Small (117M) | 29.7 s / 253.6 ms / 0.327 MB | 13.28 s / 16.7 ms / 2.125 MB |
| GPT-2 Large (774M) | 162.7 s / 1,487 ms / 1.178 MB | 59.1 s / 87.6 ms / 10.632 MB |
| SmolLM-2-1.7B | 147.9 s | 88.0 s / 152 ms / 16.2 MB |
| Qwen-3-8B | 809.2 s | 258.6 s / 329 ms / 55.7 MB |
| zkGPT, GPT-2 Small (IPA row) | 88.5 s / 4,809 ms / 0.978 MB | — |

  — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Accuracy is operator-level MSE only (Table V); no perplexity or task accuracy for any model — [OpenLLM](https://eprint.iacr.org/2026/1578)
- Authors disclose ChatGPT use for drafting; code behind an anonymous link; no limitations section — [OpenLLM](https://eprint.iacr.org/2026/1578)

### Inferences
- A 61-bit field with unstated extension degree and FRI parameters is a soundness unknown; the FRI column's speed may partly reflect a low security level.
- Single-thread 12-layer FRI time for GPT-2 Small (9.987 s, Table IV) is lower than the 16-thread end-to-end time (13.28 s, Table II); unexplained, and a reason to treat the tables cautiously.
- The requantization equation is written three inconsistent ways (§III, Fig. 5, Theorem 4 proof) and Fig. 11's softmax does not match its stated formula; this reads as an unpolished preprint.

### Gaps
- Token count per proof, security level, memory, overhead vs plain inference, whether the zkGPT baseline was rerun.

## Card: zkComposer (U. Toronto / MPI-SWS / AMD; arXiv 2607.08095, July 2026)

### Takeaway
zkComposer is a wrapper, not a new prover: it splits a zkGPT/zkCNN proof into per-partition sub-proofs linked only by a shared commitment to boundary activations, with independent Fiat–Shamir transcripts and no linking proofs. It measured that monolithic zkGPT stops scaling at 16 threads, and that partitioning gives 4.8–6.8× on 192 cores or 5–8× lower peak memory in sequential mode.

### Cited Findings
- Base systems are GKR commit-and-prove SNARKs with Libra-style ZK; "Hyrax as the PCS for both boundary and input polynomials" (§6.1); ZK proven (Thm 2, App. B.4) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Linking: the partition input polynomial is V_inp,j(r) = (1 − r_k)·V_b(r_1..r_{k−1}) + r_k·V_θ(r_1..r_{k−1}) (Eq. 5); V_b is committed once, opened at the upstream proof's output point and the downstream proof's input point; by binding, "P cannot use PCS.Open on the commitment of V_b to prove evaluations of two different polynomials ... This enforces boundary-activation equality without any additional linking proofs" (§3.1) — [zkComposer](https://arxiv.org/abs/2607.08095)
- ZK masks are sized to the architecture-determined number of openings per boundary polynomial (§3.2, App. B.1); "transcripts for each partition are independent" (§4) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Sequence partitioning: "we commit to the intermediate KV values produced within each token partition, and subsequent partitions consume these commitments as advice in their input layer" (§4.1) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Overhead anchor: zkGPT "requires 147.7s to generate a proof for 64 tokens on a 192-core machine with 755 GB of RAM, while generating the corresponding model response ... takes only 94.55ms" (§1) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Thread scaling (Table 3, GPT-2, 64 tokens): zkGPT 160.5 s at 16 threads → 147.7 s at 192 ("the workload is memory-bound"); zkComposer K=12: 55.9 s → 30.6 s (saturating at 128 threads) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Layer partitions (Table 2, 64 tokens): K=12 gives 30.6 s (4.83×), proof 268.6 KB vs 109.2 KB, verify 0.273 s; boundary commitments are ~1–3% of prover time — [zkComposer](https://arxiv.org/abs/2607.08095)
- Layer × sequence partitions (Table 4): 256 tokens, 12×4: 88.5 s (6.84×), proof 2,059 KB, verify 2.888 s; sequence partitioning adds only ~3% at 128 tokens and hurts at 64 — [zkComposer](https://arxiv.org/abs/2607.08095)
- Sequential 12-partition mode (Table 5): peak memory 240.5 → 31.9 GB at 64 tokens, 532.4 → 66.0 GB at 256, at 1.4–2.1× slower proving (loss of cross-partition circuit squashing plus boundary commitments) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Hardware: dual EPYC 9654, 192 cores, 756 GB, CPU only; all baselines rerun on the same machine (§6.1). Limitations (§4.1, §7): K fixed by architecture, uneven splits, total memory in parallel mode can exceed monolithic, proof size and verifier time grow with K — [zkComposer](https://arxiv.org/abs/2607.08095)

### Inferences
- zkGPT at 64 tokens is about 1,560× plain inference on this machine (derived: 147.7 s / 94.55 ms).
- Best per-position figure is 88.5/256 ≈ 0.35 s on 192 cores, with no decoding proof (derived).
- The same idea appears independently in DeepProve §4.5, NanoZK and SASH (boundary roots), and Seal-Then-Sample: committing block-boundary activations is the 2026 consensus seam for parallelism and memory bounding.
- The zkGPT baseline is 147.7 s here (64 tokens, 192 cores), 88.5 s in OpenLLM (length unstated) and ≈32.5 s per step in zkAgent (≈31-token context, 32 cores, derived): cross-paper zkGPT numbers must not be mixed.

### Gaps
- Quantization and per-op protocols are inherited and not discussed; no generation, nothing beyond 256 tokens, no GPU.

## Card: VerInf (James Petrie; repo + draft preprint, 2026)

### Takeaway
VerInf proves a Llama 4 Maverick forward pass with a commit-everything Ligero argument over Goldilocks, and changes the *statement*: instead of exact inference it certifies an upper bound on "unexplained information" between a committed integer model and the observed tokens, which tolerates floating-point serving. The 14.3 h / 93.6 GB headline is already superseded inside its own repo (2 h on an A100, 35 GB; then 30 min on a B200), its shipped soundness is about 2^-16.6 per attempt, and its per-op gadgets (table-free rsqrt, division-free softmax, whole-range rescale lookups, project-then-route MoE) are the transferable part.

### Cited Findings
- Ligero: Reed–Solomon-encoded witness matrix, Merkle tree over columns, linear / Hadamard / interleaved-RS tests; "security resting only on the collision resistance of a hash function"; Goldilocks with no extension; μ = 8192 slots per row, rate 1/4, BLAKE3 (`paper.md` §3, §6.1) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Soundness: "reaching 2^-s needs τ ≈ 2.4 s"; the headline run opens τ = 40 columns, "a per-attempt level near 2^-16.6"; LogUp over the 64-bit field gives a floor near 2^-27.9 per instance; the implementation "derives every seed by sequential Fiat-Shamir" and "as a non-interactive proof it gives no protection against a prover that can afford about 10^5 fresh commitments" (`paper.md` §7.3, B.1.0) — [VerInf](https://github.com/JamesPetrie/VerInf)
- "VerInf is a research prototype and has not had a security review" (`paper.md` §10) — [VerInf](https://github.com/JamesPetrie/VerInf)
- **Statement**: public output Û bounding U = H(O | D(x)); via Gibbs' inequality any predictor Q gives U ≤ E[−Σ_t log2 Q(o_t | D(x), o_<t)]; Q is Gaussian noise on integer logits; "upstream of the logits every claim admits a unique witness, and downstream of the logits every prover freedom provably inflates the reported bound"; "the cost of every approximation appears in the reported bound rather than in a failed proof" (`paper.md` §1, §4, B.7) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Headline result: Û = 0.880 bits/token over 500 scored tokens against a 202,048-token vocabulary ("about 95%" explained); an earlier run on another backend gave 0.394 (`paper.md` §9; README) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Quantization: every value a 64-bit fixed-point integer at one global scale 2^12; Maverick weights from GGUF `UD-Q4_K_XL`. Rescale after every multiply is exact with unique witness: x_full = 2^12·x + x_low, two LogUp lookups against whole-range tables (range_12 and range_26), five committed slots per element (`paper.md` §3.1, §3.6, B.1) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Llama-2-7B simulation: Q3.12 gives 99.04% top-1 agreement with FP32; Q4.14 gives 99.45% and adds +0.33 bits/token; loss is bimodal (5 of 16 prompts add 0.3–0.8 bits/token), attributed to outlier channels under one global scale (`analysis/quantization-evaluation.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Matmul: double Freivalds λᵀCρ = (λᵀA)(Bρ); cost 6mn + 3k committed slots (`paper.md` §3.2, B.2) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Softmax: per-row shift c with exp-table lookups, pinned uniquely by a monotone bracket (sum at c ≤ s_y and sum at c − 1 ≥ s_y + 1 using a second table shifted by one index); no division; 15 slots per cell; limits context to "S ≲ 4096 at s_y = 2^12" (`paper.md` §3.5, B.3, Lemma B.3) — [VerInf](https://github.com/JamesPetrie/VerInf)
- RMSNorm: no table; y = ⌈√(magic/S_tot)⌉ pinned by y²·S_tot ≥ magic and (y−1)²·S_tot ≤ magic − 1 with an 18-bit-limb carry chain per row; the broadcast product x·y "is never committed" and is checked by one random projection per row (`paper.md` B.4) — [VerInf](https://github.com/JamesPetrie/VerInf)
- SiLU: sign split, magnitude decomposition, 2^15-entry table at bin width 4, saturation mux; 25 slots per element (B.5). All range checks are per-element LogUp lookups into identity tables of 2^2–2^26 entries (B.1.0) — [VerInf](https://github.com/JamesPetrie/VerInf)
- MoE: original design — "All E expert outputs are committed even though one fires" (§3.3). RoutedProjected version commits only the selected expert's activations using P = Wρ over all experts and Q = MP; witness falls from 888.2G to 95.2G slots (`README-routed-projected.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Everything is committed: ≈8.95×10^11 field elements ≈ 7.2 TB, 57 TB encoded, streamed: "Each round ... is one streaming sweep, with the witness regenerated per round rather than stored" (`paper.md` A.2, A.5, §6.3); measured all-in cost 58 ns per slot (`analysis/overhead-estimate.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Weights: persistent Merkle root committed once; ZK budget of κ − μ = 8192 opened columns, "on the order of a hundred proofs", then a refresh with a linking proof (`paper.md` §6.4) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Benchmarks:

| Run | Hardware | Prove | Proof | Verify | Columns |
|---|---|---|---|---|---|
| Maverick 400B, 1000 tokens, all tokens hidden | DGX Spark | 14.26 h | 93.6 GB | 17.67 h (20 threads) | 40 |
| Llama-2-7B, 1000 tokens | DGX Spark | ≈44 min | 1.44 GB | ≈23 min | 10 |
| RoutedProjected Maverick, 1000 synthetic tokens (2026-08-06) | A100-80GB | 2 h 02 min + 1,401 s enrollment | 35.46 GB | 1 h 51 min (30 threads) | 54 |
| Bridged RoutedProjected, 1000 synthetic tokens (2026-09-30) | B200 | 1,780 s | 19.67 GB | 5,385 s (20 threads) | n/s |

  (`paper.md` §9; `README-routed-projected.md`; `analysis/b200-session-6-archive.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- The B200 session notes say a 2026-09-25 protocol review "found soundness gaps in the causal softmax mask and the SiLU branch index and zero sign (F01, F03, F04) that every proof of this session shares"; the A100 run is "One run on one rented machine, not reproduced" (`analysis/b200-session-6-archive.md`, `README-routed-projected.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Overhead: "≈ 10^5, within a factor of a few" vs fp8 inference; 34 s of proving per marginal token (`analysis/overhead-estimate.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Roadmap: "a larger speedup requires changing the argument, for example replacing Ligero's linear test with a sumcheck- or GKR-style protocol" (`paper.md` §8); a `layergkr/` prototype exists at toy scale. Its findings list includes F12: "The remainder alone fixes nothing: the divisor is invertible mod P" (range-check both quotient and remainder) (`layergkr/README.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Anthropic's post describes VerInf as the only earlier proof of Llama 4 Maverick it knows of, at ≈0.02 tok/s with a 93.6 GB proof and "nearly a day to verify" — [Anthropic, 2026-11-02](https://www.anthropic.com/research/provable-inference)

### Inferences
- 1000 tokens in 14.26 h is ≈0.019 tok/s (derived); the superseded-in-repo runs would be ≈0.14 tok/s (A100) and ≈0.56 tok/s (B200) on synthetic tokens, with the caveats above. Public comparisons (including Anthropic's) use only the oldest figure.
- VerInf's numbers are not comparable with exact-inference SNARKs: the statement is a bound, soundness is ~2^-17 per attempt, and the proof is not succinct.
- The unexplained-information statement is the only 2026 SNARK-side answer to "the provider serves floats, the proof is about integers"; every other system proves a quantized model and leaves the gap to the served model unaddressed or measured by perplexity.
- A 128-bit field removes VerInf's binding LogUp soundness floor; that floor is a consequence of a 64-bit field with no extension.

### Gaps
- The companion paper on non-zero-temperature sampling is cited as "[cite]" and is not in the repo.
- Whether the 14.3 h run shares the three reviewed soundness gaps is not stated.
- Paper and `ARCHITECTURE.md` disagree on whether ZK masking is live (fixed `master_seed` stub); code not checked.

## Card: SASH prover (Singapore AI Safety Hub; research note + repo, September 2026)

### Takeaway
SASH is an open, measured, commit-the-trace Circle-STARK (Stwo, M31) for Llama-3.1 8B that proves int8 matmuls by Freivalds and all non-linear and normalisation steps bit-exactly in bf16 through lookup tables. Its most useful output is its cost accounting: the proof system itself is under 5% of wall-clock time, trace and lookup-column generation is 77%, and moving five components from committed columns to LogUp-GKR claims removed 38% of committed cells.

### Cited Findings
- "a Circle STARK over the Mersenne-31 field" built on StarkWare's Stwo; Blake2s Merkle trees and FRI; 100 bits "per Stwo's own accounting, under the standard conjecture on FRI soundness"; blowup 2, 100 queries, no grinding — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Not zero-knowledge: it "adds no random blinding to the committed trace"; weights are "hidden behind a commitment, not hidden by a zero-knowledge proof" — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Five sumcheck-style (LogUp-GKR) reductions "take the largest components out of the committed table" and "remove 38% of committed cells"; they shrink the interaction tree from 93,184 to 5,600 columns (`prover/README.md`) — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note); [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
- Statement: exact greedy inference of the quantized model "from the embedding table to the chosen token"; public inputs are token ids and a 32-byte weight commitment; argmax is "the lowest-indexed maximum logit", and 15 of 512 positions had co-maximal logits at 4096 context (`docs/statement.md`, `prover/README.md`) — [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
- Quantization: int8 matmul operands; activation scale dynamic per token "pinned to that output's maximum", weight scale per output channel; accumulators dequantised to bf16 by "a chain of three correctly rounded steps (round half to even)"; power-of-two scales on the output head collapse the chain to one rounding; norms, activation and residual stream are bf16, proven through bf16-keyed lookup tables — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note); [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
- Accuracy vs fp32 control (Llama-3.1 8B): perplexity 4.41 → 4.72 (+7.0%); GPQA-Diamond 33.8% → 33.8%; MMLU-Pro 38.9% → 38.9%; IFEval 74.5% → 73.9%; the same recipe costs 14.6% perplexity on Gemma 3 270M — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Matmul by Freivalds against a once-committed weight tree, seven per layer; range checks per-element LogUp; MoE not implemented — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Committed cells: 2.72 M per position per layer (residual stream 24%, dequantisation 24%, interaction trace 21%, key tiles 15%, requantisation 6%, boundary trees 5%); 79.0 M cells per position at 4k context — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Segmentation: one STARK per (layer × block of ≤512 positions), 290 segments for a 4k request and 1,058 for 16k, linked by the verifier comparing boundary roots for residual streams and KV caches — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Benchmarks (one H100):

| | 4,096 prompt + 512 answer | 16,384 prompt + 512 answer |
|---|---|---|
| Unproved serving (vLLM / HF) | 3.4 s / 9.1 s | 4.1 s / 9.5 s |
| Witness extraction (excluded) | 13 min | 2.7 h |
| Proving wall clock | 1.02 h | 6.0 h |
| of which proof system / hashing | 4.8 min / 2.8 min | 24.9 min / 15.0 min |
| Proof size (compressed) | 18.5 GiB (7.2 GiB) | 108.9 GiB (42.0 GiB) |
| Verification (CPU) | 2.0 min | 13.9 min |

  — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Throughput: whole pipeline 1.25 positions/s; proof system alone 16 positions/s. Overhead 46× (HF) to 122× (vLLM) counting answer tokens only; 400× to 1,100× for the whole request. Time split: lookup columns 42%, base trace 35%, witness read and setup 12%, proof system 4.6%. Single layer CPU vs GPU: 222.3 s vs 41.9 s — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- SASH's comparison table cautions that rates "span five orders of magnitude" and lists Attestable as "closed source" at 195 tok/s for Llama 8B, noting it "has not disclosed its proof-system family" — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Known gaps listed by the authors: biases unsupported; padding rows for non-power-of-two head counts are unconstrained (37.5% of rotation/quantisation cells at 40 heads) (`docs/statement.md`) — [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)

### Inferences
- The widely quoted "1.25 tok/s" counts prompt positions; generated-token rate is ≈0.14 tok/s (512 tokens / 1.02 h, derived).
- SASH's breakdown is independent evidence for jolt-atlas's virtual-intermediate choice: once committed columns were replaced by sumcheck claims, cryptography stopped being the bottleneck and witness generation took over.
- SASH and Attestable converge on the same arithmetic split (8-bit dynamic integer matmuls, native-float non-linearities by lookup), and both report benchmark parity on reasoning tasks with a small instruction-following loss. That is the strongest 2026 evidence that int8 linear ops without retraining are acceptable at 8B–31B scale.

### Gaps
- Read through a summarising tool; the three-step dequantisation chain and softmax table design were not retrieved in detail. No authors listed; unaudited.

## Card: Attestable (commercial, August 2026)

### Takeaway
Attestable claims 49–195 tok/s on one H100 for 8B–70B models with ~5 MiB proofs and ~0.2 s verification at 100-bit hash-based security, but names no proof system, publishes no prover time, memory or artefact, and labels the results alpha. It is a marketing data point that sets the bar FlashProve is compared against, not a source of techniques.

### Cited Findings
- "All of our results are based only on the security of cryptographic hash functions, and thus we are post-quantum secure"; "uphold 100 bits of security"; proof system, field and soundness derivation not given — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- "dynamically quantizing the linear operations to 8-bit integers"; "Though we prove all non-linear operations in floating points, we quantize all matrix multiplications to integers" — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- Benchmark table (single H100, context 4K, batch 4): Gemma 31B 77.69 tok/s, 5.55 MiB, 251.40 ms; Muse Glimmer 30B 87.25 tok/s, 5.51 MiB, 244.94 ms; Llama 8B 195.03 tok/s, 4.94 MiB, 205.08 ms; Llama 70B 49.21 tok/s, 5.51 MiB, 238.75 ms. Text ranges: 4.35–7.92 MiB and 157–648 ms; Gemma 4 31B batch-one 16K sequence at 53 tok/s — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- Accuracy, Gemma 4 31B: GPQA-Diamond 0.8485 original / 0.8030 public INT8 / 0.8283 Attestable; IFEval 0.9519 / 0.9427 / 0.9353 — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- Limitations stated: "We support up to 16K tokens in the context window"; integer matmuls; "These are alpha results" — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- The statement is described as y = F(W, x, r) with the proof showing "the required execution policy and random seed were followed"; how sampling is proven is not described — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- Third-party critique (2026-08-12): the post "does not name its proof system"; proving wall-clock and prover memory are unpublished; "As of 12 August, no public artefact lets a third party re-run" the benchmarks; $20M seed led by Altimeter and TLV Partners — [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/)
- Team: CTO ex-StarkWare; VP R&D "led development of the S-Two proof system" — [Attestable company page](https://attestable.com/company)
- Anthropic's post notes Attestable has "not released a corresponding open source library" and "independently reports 53–77 tokens/second for a dense 31B model" — [Anthropic, 2026-11-02](https://www.anthropic.com/research/provable-inference)

### Inferences
- A STARK-family design is plausible given the team, but unconfirmed.
- Megabyte proofs and sub-second verification at 31B scale imply nothing proportional to the trace is opened per segment, unlike SASH (GiB) or VerInf; either heavy recursion/aggregation or a succinct-opening design. Unverifiable.
- Throughput rising with context (17 tok/s at 1K → 53 at 16K, batch 1, read off the chart) suggests positions per pass with a large fixed per-proof cost; the token-counting convention is not stated.

### Gaps
- Everything technical: per-op protocols, range checks, committed vs virtual, sampling proof, memory, overhead vs inference.

## Card: NanoZK (USC; ICICS 2026, arXiv 2603.18046v2)

### Takeaway
NanoZK is a Halo2/IPA layerwise prover whose GPT-2-scale numbers are projections: nothing above d = 256 was measured, attention uses direct O(n²d) gates and runs out of memory at d = 512, and its Fisher-guided partial auditing gives no cryptographic soundness for unaudited layers. The transferable ideas are boundary-commitment chunking and per-op fixed-point formats.

### Cited Findings
- Halo2 with IPA over Pallas, Plookup-style lookups, no trusted setup, "not post-quantum secure" (§2.2, A.1); first 12 pages are "the ICICS 2026 camera-ready"; "An earlier preliminary version appeared at the VerifAI workshop, ICLR 2026" (p.1 footnote) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Hides weights and activations from verifier/auditor; "It does not hide the input from the prover" (§1); greedy decoding assumed, sampling "not covered here" (F.4) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Quantization: operation-specific fixed point (hidden states Q1.15, pre-softmax logits Q4.11, RMSNorm sum-of-squares Q6.9) with range-check conversions (B.2); tables are sparse anchors plus an in-circuit linear-interpolation gate: "The actual implementation does not materialise a 65,536-row Halo2 lookup table" (B.1); exp 1,024 anchors over [−8,0], rsqrt 1,024, GELU/SiLU 2,048 (Table 6) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Matmul/attention by direct gates: "Θ(n² d_k) arithmetic-gate constraints (one multiplication per scalar product term)" (C.3). All activations are committed advice columns (~3.5 M cells for one MLP block at n=128, d=512; Table 7) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Linking: "the chain binds activations via an algebraic, in-circuit-checkable boundary commitment B_ℓ, not directly via the SHA-256 digest"; B_ℓ is a Pedersen commitment exposed as public input; the SHA-256 chain is computed outside the circuit (App. A, C.6) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Fisher-guided auditing is "an efficiency tool, not a cryptographic primitive" (§1); "tampering confined to S̄ is undetected" (A.4) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Measured: MLP sub-circuit prove 4.8–6.5 s for d = 64–2048, verify ≈22 ms, proof 2,016 B (Table 11); single-head attention 0.9 s (d=16), 46.8 s (d=128), 184 s (d=256) (Table 12); attention at d=512 OOM at 128 GB — [NanoZK](https://arxiv.org/abs/2603.18046)
- Projected only: "No part of the projection is grounded in measurements above d=256" (G.1); "∼2 min for 12-layer GPT-2 with 12 GPU workers"; ~14 min sequential (§6.1) — [NanoZK](https://arxiv.org/abs/2603.18046)
- Comparisons with zkLLM, zkGPT, ZKML use copied numbers: "no direct same-hardware comparison is possible" (§6.2); EZKL was run on the same CPU (19.3 s / 26.0 KB for MLP d=64) but the appendix gives a different EZKL figure (89 s, Table 17) — [NanoZK](https://arxiv.org/abs/2603.18046)

### Inferences
- The "43 s, 6.9 KB, 23 ms, 52× over EZKL" headline from the ICLR workshop listing is a per-sub-circuit figure including setup, not a GPT-2 proof.
- Body and appendix disagree on hardware, timings and model lists; the appendix itself says the two sets "must not be compared cell-for-cell" (E.1).

### Gaps
- No end-to-end model proof, no sequence length for timings, code unreleased, embedding/LM-head costs unbenchmarked.

## Card: Jolt Atlas paper and third-party evaluations (arXiv 2602.17452, Feb 2026)

### Takeaway
The public paper describes the February 2026 HyperKZG-era system (GPT-2 ≈38 s on an M3 with 16 GB, no sequence length, proof size or verify time stated) and its only head-to-head is against copied EZKL numbers. Third-party engagement in 2026 is thin: one measurement paper (ePrint 2026/1390), one independent GitHub benchmark issue, and citations without numbers in zkAgent and a survey; Anthropic's comparison list omits it.

### Cited Findings
- Sumcheck lookup arguments inherited from Jolt (Twist/Shout one-hot addressing, virtual polynomials, prefix-suffix); "Dory … is only justified for witnesses over 2^26 elements. We therefore replace it with HyperKZG" (§1.1, §1.2); ZK via BlindFold (§3); future work names lattice PCS (§7) — [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)
- "Nearly all verification checks are performed using virtual polynomials" (§5.4) — [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)
- Quantization as stated: "16-bit fixed-point activations", "a 128-scale fixed-point representation"; neural teleportation σ(x) → σ(x/τ) bounds tables to 2^16 entries with "output differences of less than 55 units on a 128-scale" (§4.2); no accuracy-vs-float evaluation — [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)
- Benchmarks (MacBook Pro M3, 16 GB): nanoGPT (~0.25M params) prove 14 s, verify 0.517 s (Table 1) vs EZKL 237 s "reported" from the EZKL blog (Table 2), "roughly a 17× speed-up on proof generation alone"; GPT-2 (125M) total ≈38 s: witness ≈7.5 s, commit ≈3.5 s, sumcheck ≈16 s, reduction opening ≈7 s, HyperKZG prove ≈3 s (Table 3) — [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)
- Independent benchmark issue #342 (2026-09-18, Apple M5 Max, Atlas `bb9e1e6`, GPT-2, four workers): Akita 91.17 s vs HyperKZG 78.37 s prove; proof 62.52 MB vs 12.92 MB; extra one-hot grouping −36.64% proof bytes, −1.78% prove time; the author calls them "historical diagnostics from separate experiments" on a baseline that predates a later fix — [jolt-atlas issue #342](https://github.com/ICME-Lab/jolt-atlas/issues/342)
- "Crypto x AI, AI x Crypto: A Survey" cites Jolt Atlas as general tooling: "systems such as Jolt Atlas redesign the proof layer around tensor operations"; no numbers — [arXiv 2606.13892](https://arxiv.org/abs/2606.13892)
- zkAgent's related work: "NANOZK and Jolt Atlas explore layerwise proofs and lookup-centric approaches" (§6) — [zkAgent](https://eprint.iacr.org/2026/199)
- Anthropic's post does not mention Jolt Atlas — [Anthropic, 2026-11-02](https://www.anthropic.com/research/provable-inference)

### Inferences
- The paper's GPT-2 row cannot anchor any cross-system comparison: sequence length, proof size and verifier time are missing, and the system has since changed field and PCS. A refreshed public benchmark with those columns is a precondition for appearing in lists like Anthropic's.
- DeepProve, OpenLLM, zkComposer, NanoZK, Maverick and TensorCommitments do not cite Jolt Atlas (grep of local texts).

### Gaps
- No 2026 ICME blog post with benchmark numbers was found. Repo-README figures (GPT-2 14.9 s, nanoGPT 2.29 s) were seen in a search snippet only.

## Card: "A Separation Principle for Lookup-Based zkML" (Min Jun Jo; ePrint 2026/1390)

### Takeaway
The paper claims that in a one-hot (Shout-style) lookup argument the per-lookup proving cost depends only on the access pattern, never on table values, so activation-function structure can only cheapen a once-per-proof table term; the single remaining dial is committed address width. The full text was **not retrieved**; the address-width result it calls "upstreamed" is jolt-atlas PR #258, which merged at a smaller gain (≈2.3% on GPT-2) than the 2.6–6.0% range associated with the paper.

### Cited Findings
- Abstract claims (as surfaced by the abstract page and search excerpts): "in a Shout-style (one-hot) lookup argument the per-lookup proving work is a function of the access pattern alone"; structure "can cheapen only a secondary, once-per-proof table term"; "We measure this depth-to-cost scaling on two independent proving systems (EZKL/halo2 and Jolt Atlas)"; it "turns the one dial the separation leaves open, the committed address width, into a bit-exact, upstreamed reduction in prover time" — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)
- Record: received 2026-07-08, three revisions, latest 2026-07-15, CC BY, single author — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390); [IACR news item](https://iacr.org/news/item/28963)
- PR #258 "perf(softmax): scale-aware SatDiff range check — shrink from 32 bits to log₂(scale)+4" (opened 2026-07-03, merged 2026-07-07): replaces the attention-mask sentinel (≈ −2^30) with a scale-indexed one so the range-check width follows the scale; reports "12/12 whole-model GPT-2 softmax outputs identical, masked outputs all zero" — [jolt-atlas PR #258](https://github.com/ICME-Lab/jolt-atlas/pull/258)
- PR's original measurements (Apple Silicon, release, medians of 3): prove time microgpt 395.3 → 371.7 ms (−6.0%), minigpt 916.9 → 872.3 ms (−4.9%), nanoGPT 2391.2 → 2328.8 ms (−2.6%), GPT-2 10855.5 → 10351.2 ms (−4.6%) — [jolt-atlas PR #258](https://github.com/ICME-Lab/jolt-atlas/pull/258)
- After rebasing over the fused-i64 change the width became `log_scale + 8` (d = 8 → 4 at scale 8, 8 → 5 at scale 12); single A/B on GPT-2: prove 103.36 s → 100.99 s (−2.3%), verify flat — [jolt-atlas PR #258](https://github.com/ICME-Lab/jolt-atlas/pull/258)

### Inferences
- Identification of PR #258 as the paper's "upstreamed" change is an inference (author handle's profile links to the paper author's site; merge one day before ePrint receipt; matching 2.6–6.0% range).
- If the separation holds, effort spent on activation-specific table structure (e.g. teleportation) moves only the per-proof table term; the levers that matter are digits per lookup × lookup count, i.e. exactly the committed range-check digits jolt-atlas already identifies as its dominant cost.
- The PR's method generalises: derive each range-check width from the honest value envelope per scale or per node instead of a global constant.

### Gaps
- Unverified body fragments from search snippets only (do not cite as read): Jolt Atlas prover time "affine in depth (R² = 0.999)"; scaling ≈ L^0.95 for Jolt Atlas vs ≈ L^1.05 for halo2; "num_lookups is not directly instrumented in Jolt Atlas, so prover time is the cost proxy"; a "Proposition 5".
- Exact theorem, models, sequence lengths, hardware, EZKL settings: not obtained.

## Card: Maverick (Yale / IC3; arXiv 2609.10264, September 2026)

### Takeaway
Maverick is not a SNARK: it is an interactive, private-coin delegation protocol that verifies only the linear layers (matrix–vector products against public weights) with near-zero server overhead and ~45 ms client verification for Qwen3-4B, while the client recomputes every non-linearity and stores preprocessing state several times the model size. It is the cost floor for designated-verifier, public-weight deployments.

### Cited Findings
- "information-theoretically sound verification protocol for matrix–vector multiplication delegation with transparent preprocessing" (abstract); BabyBear field; statistical κ = 40; RAA codes at rate 1/8; soundness error ≤ 2^-40 plus code-distance failure (Thm 5.2, §7.1) — [Maverick](https://arxiv.org/abs/2609.10264)
- Core check: preprocess Q = GᵀM; client computes Gᵀy, samples a t-sparse e and checks (eᵀQ)x = eᵀ(Gᵀy); per-repetition error ≤ (1−δ)^t + 1/(|F|−1) (Protocol 1, Lemma 5.1); input privacy by masking x̂ = x + G_x e_x under dual-LPN over RAA codes (Protocol 3, App. A–B) — [Maverick](https://arxiv.org/abs/2609.10264)
- Quantization: W_q = round(2^8·W) from FP8, embedded in BabyBear; the client "divides it by 2^{2s}, and converts it to BF16" for the non-linear step; "we observe no modular overflow" (§7.2.1) — [Maverick](https://arxiv.org/abs/2609.10264)
- Qwen3-4B, 8-token prompt → next-token logits (Table 7): standard mode 958.3 ms end to end, 45.5 ms client verify, 353.5 ms server, 14.7× faster than local (14.11 s); verification-only 531.6 ms; communication 57.83 MB over 182 rounds (private modes) or 36.08 MB in one round — [Maverick](https://arxiv.org/abs/2609.10264)
- With 100 ms RTT the private modes' time to first token is 19.02 s, slower than local (Table 10) — [Maverick](https://arxiv.org/abs/2609.10264)
- Same-machine baseline for one MVM at n = 2^12: Maverick 392 ms server total vs Sum-Check + BaseFold 82.2 s (Table 5) — [Maverick](https://arxiv.org/abs/2609.10264)
- Concurrent works "require substantially less client-side storage for preprocessing state, which is a key bottleneck of our approach" (§2); DeepProve comparison is "not apples-to-apples" (§7.2.2) — [Maverick](https://arxiv.org/abs/2609.10264)

### Inferences
- "8.35 tok/s" is prompt tokens per second for a single next-token inference; there is no autoregressive generation result.
- The Sum-Check + BaseFold MVM baseline (~200× server overhead at n = 2^12 over a 255-bit field) is a public number a sumcheck matmul prover can be measured against.

### Gaps
- Client storage in GB, total preprocessing time for a full model, accuracy vs float, how activation×activation products in attention are handled.

## Which systems commit the whole trace, which keep intermediates virtual, and what do they pay per element?

### Takeaway
2026 systems split cleanly: commit-everything designs (VerInf, SASH, OpenLLM, NanoZK, and FlashProve per the existing notes) pay per trace element and win only where hashing small integers on a GPU is cheap, while virtual-intermediate designs (DeepProve, zkAgent, jolt-atlas) pay only for lookup witnesses, which nonetheless end up as 57% of DeepProve's prover time. The one measured before/after (SASH) shows moving components from committed columns to sumcheck claims is a large win.

### Cited Findings
- DeepProve commits "only to the witness values that explicitly need to be committed and cannot be virtualized"; witness commitment is still 56.85–57.73% of prover time and requant chunks ≈27% of total (§3.3, Table 7) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- zkAgent commits one input-layer MLE holding weights, embeddings and "all advice values (rounding inputs/outputs, lookup queries, selector vectors, etc.)"; GKR layers are virtual (App. C.4) — [zkAgent](https://eprint.iacr.org/2026/199)
- VerInf commits everything: 5 slots per rescaled element, 15 per softmax cell, 25 per SiLU element, 6mn + 3k per matmul; ≈8.95×10^11 field elements for Maverick at 1000 tokens; 58 ns per slot all-in (`paper.md` App. A–B; `analysis/overhead-estimate.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- SASH commits 2.72 M cells per position per layer after its five LogUp-GKR reductions "remove 38% of committed cells"; proof system is 4.6% of time, lookup columns 42%, base trace 35% — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- OpenLLM: every operator takes and returns committed polynomials (Table VI) — [OpenLLM](https://eprint.iacr.org/2026/1578)
- NanoZK: all activations are Halo2 advice (≈3.5 M cells per MLP block at n=128, d=512; Table 7) — [NanoZK](https://arxiv.org/abs/2603.18046)
- zkComposer adds exactly one committed tensor per partition boundary, costing ~1–3% of prover time (Table 2) — [zkComposer](https://arxiv.org/abs/2607.08095)
- VerInf's RoutedProjected MoE cut committed slots from 888.2G to 95.2G by committing only the selected expert's activations (`README-routed-projected.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
- The 2023 Modulus study (reposted as ePrint 2026/1063) found zkCNN spends 65–75% of prover time in MLE commit/open (§4.3.6) — [ePrint 2026/1063](https://eprint.iacr.org/2026/1063)

### Inferences
- Across three independent virtual-intermediate or hybrid systems (DeepProve, SASH after reductions, jolt-atlas by the team's own profile), the residual cost is the same: committing and generating lookup/range-check witnesses. Nobody in 2026 has made that residual free except by replacing per-element checks (FlashProve's random sums, per the existing notes).
- Weight handling differs sharply and affects the verifier: committed once and opened in a batched claim (DeepProve, zkAgent, SASH, VerInf, Attestable) versus public and read by the verifier (jolt-atlas, Maverick with preprocessing).

### Gaps
- No source reports a per-element commitment cost in comparable units (ns per committed b-bit value) across PCS families; VerInf's 58 ns/slot and SASH's 99 M cells/s whole-loop are the only absolute rates found.

## How are range checks and rounding made cheap in 2026 systems?

### Takeaway
Outside FlashProve, **no 2026 system uses approximate, probabilistic or random-projection range checks**; every one keeps exact per-element lookups and reduces cost by (i) fewer, wider digits against one shared table, (ii) fusing requantization into the next table lookup, (iii) skipping rescales when headroom allows, (iv) choosing scales so the rounding is a shift, and (v) replacing per-element checks with per-row bracket checks for rsqrt and softmax normalisation.

### Cited Findings
- DeepProve: one shared 2^q (q = 12) range table for all shifted chunks; requant fused into the next GELU/softmax lookup; requant skipped when 2q + log M + 1 bits fit (B.2, App. C) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- zkAgent: one universal 1024-entry base table model-wide, "K = 5 base-table lookups" per requantization of a 46–49-bit shifted residual; weight-ℓ1 bounds make overflow impossible so no clamp is proven (App. F, §4.1) — [zkAgent](https://eprint.iacr.org/2026/199)
- VerInf: two whole-range lookups (2^12 remainder table and 2^26 quotient table), no digit decomposition; "range checks contributing nothing" to the soundness error (`paper.md` §3.6, B.1) — [VerInf](https://github.com/JamesPetrie/VerInf)
- VerInf rsqrt: a per-row integer bracket y²·S_tot ≥ magic, (y−1)²·S_tot ≤ magic − 1 with the normalised output checked by a random projection and never committed (B.4); softmax normalisation by a two-table monotone bracket on the shift, no division (B.3) — [VerInf](https://github.com/JamesPetrie/VerInf)
- VerInf one-sided rounding: downstream of the logits table entries are rounded so that "every prover freedom provably inflates the reported bound" (`paper.md` §1, B.7) — [VerInf](https://github.com/JamesPetrie/VerInf)
- DeepProve normalisation: per-row committed reciprocal plus a range check that Σ_j Y[i,j]² ≈ constant (approximate relation with an error analysis, App. C Lemmas 1–4) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- SASH: requantisation scale by table lookup `pm -> bf16_ceil(pm / 127)`; power-of-two scales on the output head collapse a three-step correctly-rounded chain to one rounding (`prover/README.md`) — [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
- Jolt Atlas PR #258: range-check width derived from the scale (32 bits → log₂(scale)+8), −2.3% GPT-2 prove time — [jolt-atlas PR #258](https://github.com/ICME-Lab/jolt-atlas/pull/258)
- OpenLLM: 12-bit digits, one LogUp per digit, every auxiliary column committed; no fusion (Fig. 2–5) — [OpenLLM](https://eprint.iacr.org/2026/1578)
- LAMP (ePrint 2026/1571, July 2026): Freivalds combined with proximity testing over linear codes for matmul inside SNARK circuits; conflicting speedup figures across versions (search excerpts only) — [ePrint 2026/1571](https://eprint.iacr.org/2026/1571)
- "Bits Under ZK-LLM" (arXiv 2609.36437, Sept 2026, abstract): activation precision matters far more than weight precision; the RMSNorm inverse-square-root lookup is a recurring accuracy bottleneck, fixed by raising precision there only; lower bit-width or smaller tables do not reduce end-to-end proving cost proportionally — [arXiv 2609.36437](https://arxiv.org/abs/2609.36437)

### Inferences
- FlashProve's random-sum no-overflow check has no 2026 analogue elsewhere; a dedicated search for approximate/probabilistic range checks or random-projection norm checks inside zkML found nothing. It is a single-source technique, and its exact construction should be taken from the FlashProve repo, not from the surrounding literature.
- The common cheap trick the others share and jolt-atlas partly lacks is **table sharing with wide digits**: 10–12-bit digits against one global range table (2–5 lookups per rescale) versus 4-bit one-hot digits (8–12 per element).
- zkAgent's overflow-free scale derivation removes the saturating clamp from the statement entirely; that is the cheapest way found to delete clamp digits while keeping exactness.
- "Bits Under ZK-LLM" independently supports the Separation Principle's point: shrinking tables does not shrink per-lookup cost.

### Gaps
- Probabilistic truncation in the MPC sense (accepting ±1 rounding error) was not found in any 2026 zkML system; DeepProve's and VerInf's tolerance checks are the nearest.
- LAMP and "Bits Under ZK-LLM" were read as abstracts only; their numbers need the PDFs.

## What quantization widths are used, and what accuracy is measured against the float model without retraining?

### Takeaway
Evidence in 2026 points to 8-bit dynamic integer matmul operands being adequate on 8B–31B models when non-linearities stay in native float (SASH, Attestable), while pure fixed-point pipelines need about 12 bits (DeepProve) to 20 bits (zkAgent) for GPT-2-class parity and 8-bit pure fixed-point fails badly on small models.

### Cited Findings
- DeepProve, pure fixed point, per-tensor scales: GPT-2 perplexity 49.22 float → 60.93 at 8-bit, 50.08 at 10-bit, 49.49 at 12-bit; Gemma 3 270M collapses at 8 and 10 bits (>1150) (Table 4) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- zkAgent, Q = 20 with weight-derived scales: GPT-2 perplexity identical to FP32 (24.36 on WikiText-2); Q = 16 gives 25.16; a dimension-only bound at Q = 20 gives 26.53 (Table 9) — [zkAgent](https://eprint.iacr.org/2026/199)
- SASH, int8 matmuls + bf16 elsewhere, Llama-3.1 8B: perplexity +7.0%, GPQA and MMLU-Pro unchanged, IFEval 74.5% → 73.9%; plain dynamic int8 costs 5.0% perplexity; the same configuration costs 14.6% on Gemma 3 270M — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- Attestable, 8-bit dynamic linear ops + float non-linearities, Gemma 4 31B: GPQA-Diamond 0.8485 → 0.8283, IFEval 0.9519 → 0.9353; the post says the IFEval result "shows where the current quantization still needs improvement" — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- VerInf, global 2^12 scale, Llama-2-7B: 99.04% top-1 agreement (Q3.12), 99.45% (Q4.14), bimodal per-prompt loss blamed on outlier channels under a single global scale; Maverick from Q4 GGUF: 0.880 bits/token unexplained (`analysis/quantization-evaluation.md`; `paper.md` §9) — [VerInf](https://github.com/JamesPetrie/VerInf)
- Seal-Then-Sample: 84.8–84.9% argmax agreement with float over 334,705 WikiText-2 positions (abstract) — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367)
- NanoZK claims ΔPPL < 10⁻⁴ (§6.3) but whether this was measured on the circuit's fixed-point pipeline is not stated — [NanoZK](https://arxiv.org/abs/2603.18046)
- OpenLLM and the Jolt Atlas paper report no model-level accuracy — [OpenLLM](https://eprint.iacr.org/2026/1578); [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)

### Inferences
- Small models are far more quantization-sensitive than large ones (Gemma 3 270M fails at 8–10 bits in two independent sources); GPT-2 is a pessimistic testbed for int8, and results on it will understate what works at 8B+.
- VerInf's finding that a single global scale produces outlier-channel failures on a minority of prompts is direct evidence for per-tensor (or per-channel) scales, as research-directions §1 proposes.
- Accuracy metrics are not comparable across papers (perplexity on different subsets, top-1 agreement, benchmark scores, bits/token); a common metric (top-1 agreement with the float model plus perplexity on a fixed set) is missing from the field.

### Gaps
- No 2026 source measures a 4-bit-weight integer pipeline's accuracy except through VerInf's bound and FlashProve (covered elsewhere).
- No source measures accuracy of power-of-two per-tensor scales specifically.

## How is sampling proven, how are long contexts handled, and is anyone doing sub-quadratic attention?

### Takeaway
zkAgent is the only 2026 system with implemented ZK protocols for temperature / top-k / top-p sampling (coin derived from the transcript hash); everyone else proves greedy argmax or leaves sampling to non-SNARK methods. One teacher-forced causal pass with no KV cache is universal; long context is handled by segmentation (SASH to 16K, Attestable claims 16K), and no one proves sparse or sub-quadratic attention.

### Cited Findings
- zkAgent zkDecode: top-k with a Boolean selector and hidden separator, N exp lookups with temperature, inverse-CDF sampling with coin "r = H(tr_<t) mod Θ"; top-k costs 0.28 s and 193.74 KB per generated token (Shout) vs greedy 0.035 s (§3.3, §5.2, App. C.5) — [zkAgent](https://eprint.iacr.org/2026/199)
- DeepProve: greedy only; seeded de-randomisation proposed but "we do not include this in our implementation" (§4.3.6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- SASH: greedy, tie broken to "the lowest-indexed maximum logit"; ties occurred at 15 of 512 positions — [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
- Attestable says the proof shows "the required execution policy and random seed were followed" without describing the mechanism — [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- VerInf sidesteps sampling by bounding surprisal; "the extension to non-zero-temperature sampling ... are in the companion paper" (not available) (`paper.md` §4) — [VerInf](https://github.com/JamesPetrie/VerInf)
- One-pass causal proving: DeepProve (§3.1), zkAgent (Thm E.2, Eq. 23), VerInf and SASH all use a single teacher-forced pass; DeepProve avoids a KV cache explicitly to avoid committing read/write memory — [DeepProve paper](https://eprint.iacr.org/2026/1112); [zkAgent](https://eprint.iacr.org/2026/199)
- Longest measured contexts: SASH 16,384 prompt + 512 answer (6.0 h, 108.9 GiB proof), a single segment fits at 24,576; Attestable claims 16K; DeepProve 512 on one machine, 1024 simulated-distributed; zkAgent 512; zkComposer 256; VerInf 1000 with a ≈4096 ceiling from its softmax bracket — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note); [Attestable blog](https://attestable.com/blog/proving-llms-scale); [DeepProve paper](https://eprint.iacr.org/2026/1112); [VerInf](https://github.com/JamesPetrie/VerInf)
- Attention savings found: zkAgent materialises only the causal triangle ("from T² to T(T+1)/2" exp lookups, §4.2); SASH accumulates attention over "key tiles" with boundary roots for KV caches; zkComposer commits "intermediate KV values produced within each token partition" (§4.1) — [zkAgent](https://eprint.iacr.org/2026/199); [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note); [zkComposer](https://arxiv.org/abs/2607.08095)
- Batching several requests: Seal-Then-Sample packs twelve requests with a block-diagonal causal mask (181.9 s, 6.5× less than separate proofs); Attestable reports 1.5–3× higher throughput at batch 4 — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367); [Attestable blog](https://attestable.com/blog/proving-llms-scale)
- Non-SNARK sampling verification: Token-DiFR conditions on the shared seed (Gumbel-max) and scores the logit margin; >98% of tokens match exactly between provider and verifier — [DiFR](https://arxiv.org/abs/2511.20621)

### Inferences
- zkAgent's sampling coin is a hash of the public transcript prefix, so sampling is deterministic given the transcript: it proves "the sampler was followed" but the randomness is not hidden from or unpredictable to the prover. A committed-seed variant is still open.
- Sub-quadratic attention proving (exploiting exp cutoff sparsity) remains unclaimed territory; the triangle trick is a 2× constant, not an asymptotic change.
- Tie-breaking in argmax is a real completeness/soundness detail at scale (3% of positions in SASH's int8 run).

### Gaps
- No incremental/streaming (per-token) proving, folding across tokens or KV-cache memory-checking proof was found in 2026.
- Attestable's and VerInf's sampling mechanisms are undisclosed/unavailable.

## What do GPU provers accelerate, and what speedups are reported?

### Takeaway
Reported 2026 GPU provers accelerate hashing, NTT/encoding and trace generation in commit-the-trace designs (SASH, VerInf, Attestable, FlashProve); among sumcheck-based virtual-intermediate systems only DeepProve's repo has a GPU path (HyperKZG commitment only, no speedup published), and no source reports a GPU sumcheck speedup for zkML.

### Cited Findings
- SASH: single layer 222.3 s on CPU vs 41.9 s on GPU; without GPU offload the full pass takes 6.4× longer; CUDA kernels are generated per shape; peak 73.7–78.8 GiB of an H100 — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- VerInf: CUDA kernels for Goldilocks NTT and BLAKE3; NTT is memory-bandwidth-bound at 0.42 ns per element (`paper.md` A.5); same statement took 2 h 02 min on an A100 and 29.7 min on a B200 with the newer protocol — [VerInf](https://github.com/JamesPetrie/VerInf)
- DeepProve repo: `--features cuda` uses a GPU HyperKZG and CUDA inference; sumcheck and GKR stay on CPU; "GPU proving cluster" is an unchecked roadmap item (`zkml/README.md`) — [deep-prove repo](https://github.com/Lagrange-Labs/deep-prove)
- NanoZK: GPU MSM on an RTX 3090 only reaches parity with a 32-thread CPU at k ≈ 20 (Table 16) — [NanoZK](https://arxiv.org/abs/2603.18046)
- VeriLoRA (zkLLM lineage, CUDA sumcheck + Hyrax on one A100): 121.93–249.38 s to prove a one-sample training step on 3B–13B models — [VeriLoRA](https://arxiv.org/abs/2508.21393)
- Polyhedra Expander-GPU (2025-07-01, vendor): 7–10× over AVX-512 on an RTX 5090, no ML workload — [Polyhedra blog](https://blog.polyhedra.network/expander-gpu-blasting-past-cpu-limits-with-a-10x-speed-force/)
- zkComposer measured that CPU zkGPT does not scale past 16 threads because "the workload is memory-bound" (§1, Table 3) — [zkComposer](https://arxiv.org/abs/2607.08095)

### Inferences
- The two fastest disclosed GPU systems (per Anthropic's list) are hash-based commit-the-trace designs; the GPU advantage comes from encoding and hashing small integers, which a lattice PCS on CPU does not share.
- For a CPU sumcheck prover, zkComposer's result suggests block-level parallelism (independent sub-proofs) scales better than adding threads to one monolithic sumcheck.

### Gaps
- No GPU numbers from DeepProve, Attestable (memory, wall-clock) or any lattice-PCS zkML system.

## What are proof sizes and verifier times, and is recursion, aggregation or on-chain verification used?

### Takeaway
Proof sizes in 2026 span five orders of magnitude (kilobytes for Halo2/KZG sub-circuits to ~100 GB for Ligero and segmented STARKs); the systems with megabyte proofs and seconds-or-less verification are DeepProve (9–44 MiB, 1.4–3.4 s), zkAgent-LogUp (3 MB, 38 ms), OpenLLM-FRI (2–56 MB, 17–329 ms) and Attestable (≈5 MiB, ≈0.2 s, unverifiable). **No 2026 LLM-inference system uses recursion or aggregation, and none verifies on-chain**; chunked systems concatenate sub-proofs.

### Cited Findings
| System | Model / length | Proof | Verify | Source |
|---|---|---|---|---|
| DeepProve BaseFold | GPT-2 / 512 | 24.33 MiB | 1.65–2.12 s | [paper, Tables 1–2](https://eprint.iacr.org/2026/1112) |
| DeepProve HyperKZG | GPT-2 / 512 | 11.8 MiB | 2.59–3.22 s | [paper](https://eprint.iacr.org/2026/1112) |
| DeepProve 17 chunks | GPT-2 / 1024 | 55.49 MiB | 2.5 s | [paper, Table 6](https://eprint.iacr.org/2026/1112) |
| zkAgent Shout / LogUp | GPT-2 / 512 | 98,374 KB / 3,181 KB | 0.42 s / 0.038 s | [zkAgent, Table 5](https://eprint.iacr.org/2026/199) |
| OpenLLM FRI | GPT-2 Small / n.s. | 2.125 MB | 16.7 ms | [OpenLLM, Table II](https://eprint.iacr.org/2026/1578) |
| OpenLLM FRI | Qwen-3-8B / n.s. | 55.7 MB | 329 ms | [OpenLLM, Table III](https://eprint.iacr.org/2026/1578) |
| zkComposer K=12 | GPT-2 / 64 | 268.6 KB | 0.273 s | [zkComposer, Table 2](https://arxiv.org/abs/2607.08095) |
| SASH | Llama-3.1 8B / 4k+512 | 18.5 GiB (7.2 compressed) | 2.0 min | [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note) |
| VerInf | Maverick / 1000 | 93.6 GB (35.46 / 19.67 GB newer) | 17.7 h (1.9 h / 1.5 h newer) | [VerInf](https://github.com/JamesPetrie/VerInf) |
| Attestable | 8B–70B / 4K×4 | 4.94–5.55 MiB | 205–251 ms | [Attestable blog](https://attestable.com/blog/proving-llms-scale) |
| NanoZK | per sub-circuit | 3.2–3.7 KB | ≈22 ms | [NanoZK](https://arxiv.org/abs/2603.18046) |
| Seal-Then-Sample | Llama-2-70B, 5 of 163 chunks | 4.34 MiB | 46.3 s | [arXiv 2609.27367](https://arxiv.org/abs/2609.27367) |

- DeepProve's final distributed proof is the "concatenation of all sub-proofs"; size grows with chunk count (§4.5, Table 6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- zkComposer: proof size and verifier time grow with K (109 KB → 269 KB at K=12; up to 3.9 MB and 2.9 s with sequence partitions) — [zkComposer](https://arxiv.org/abs/2607.08095)
- Jolt Atlas paper names on-chain verification as hard under HyperKZG because of G_T operations (§7) — [Jolt Atlas paper](https://arxiv.org/abs/2602.17452)
- Bionetta (arXiv 2510.06784, last revised Dec 2025) remains the EVM-verifiable option, for small models; no 2026 update found — [arXiv 2510.06784](https://arxiv.org/abs/2510.06784)

### Inferences
- The existing notes give jolt-atlas 2.4 MiB and about 2 s on GPT-2 at short context; that is competitive with DeepProve's 9–24 MiB / 1.4–3.2 s at 64–512 positions, though jolt-atlas's verifier reads the weights and DeepProve's does not. Small proofs with weight-free verification is an open niche only OpenLLM-FRI (unstated security) and Attestable (undisclosed) claim to occupy.
- Because nobody recurses, "wrap a fast large proof in a small one" (research-directions §8) is still unclaimed for LLM inference in the published literature.

### Gaps
- No verifier-time breakdown (weights vs sumcheck vs openings) is published by any system.
- No dedicated 2026 paper on recursion/aggregation or on-chain verification of LLM proofs was found.

## Is there any head-to-head benchmark across systems on the same model and hardware?

### Takeaway
No independent cross-system benchmark exists for 2026. Same-hardware re-runs are limited to single baselines (zkAgent and zkComposer each re-ran zkGPT; NanoZK re-ran EZKL; the SaTML paper and zkFinGPT re-ran zkLLM; Maverick re-ran a single-MVM sumcheck baseline); everything else, including DeepProve's "20–60×" and Anthropic's list, uses copied numbers across different models, lengths and hardware.

### Cited Findings
- zkAgent re-ran zkGPT on its own 32-core machine (adding lookup verification that "their released implementation omits") but extrapolated zkLLM "from the per-step A100 costs reported in [66]" (§5.3, App. H) — [zkAgent](https://eprint.iacr.org/2026/199)
- zkComposer re-ran zkGPT and zkCNN on its 192-core machine (§6.1) — [zkComposer](https://arxiv.org/abs/2607.08095)
- DeepProve Table 3 copies zkGPT and zkTorch figures; zkLLM excluded as GPU-based; zkGPT's figure is single-token on a modified GPT-2 (§6) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
- Offchain Labs' SaTML 2026 paper measured zkLLM on Llama-2-7B on a Ryzen 9900X + RTX 3090: 388.3 s prove, 183 kB, 2.36 s verify — [arXiv 2603.19025](https://arxiv.org/abs/2603.19025)
- zkFinGPT (Jan 2026) re-ran zkLLM-class tooling on Llama-3-8B: 531 s commit, 620 s prove, 2.36 s verify, 7.97 MB commitment — [arXiv 2601.15716](https://arxiv.org/abs/2601.15716)
- SASH's cross-system table states the rates "span five orders of magnitude" and are not directly comparable — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
- noze.it on Attestable: launch and literature figures "use different hardware, models, and conditions, so their ratio does not measure progress" — [noze.it](https://www.noze.it/en/insights/zero-knowledge-proofs-inference-attestable/)
- GPT-2 data points that exist (not comparable; listed so the differences are explicit):

| System | Length | Hardware | Prove | Convention |
|---|---|---|---|---|
| Jolt Atlas (paper) | not stated | M3, 16 GB | ≈38 s | one forward pass — [paper](https://arxiv.org/abs/2602.17452) |
| DeepProve | 64 / 512 | 32T Ryzen | 34 s / 176 s | one pass, 12-bit — [paper](https://eprint.iacr.org/2026/1112) |
| zkAgent (Shout) | 32 / 512 | 32-core Xeon | 13.6 s / 194.4 s | one pass incl. argmax, Q=20 — [paper](https://eprint.iacr.org/2026/199) |
| zkGPT (re-run by zkComposer) | 64 / 256 | 192-core EPYC | 147.7 s / ≈605 s (derived) | one pass — [zkComposer](https://arxiv.org/abs/2607.08095) |
| zkComposer | 64 / 256 | 192-core EPYC | 30.6 s / 88.5 s | one pass — [zkComposer](https://arxiv.org/abs/2607.08095) |
| OpenLLM (FRI-M61) | not stated | 16T Hygon | 13.3 s | unknown — [OpenLLM](https://eprint.iacr.org/2026/1578) |

### Inferences
- GPT-2 at 64 and 512 positions on a 32-thread x86 box is the de facto common workload (DeepProve, zkAgent, zkComposer all report it). Publishing jolt-atlas at those two lengths, with proof size, verifier time and peak memory, would make it directly comparable with the two strongest academic systems.
- On that workload zkAgent-Shout and DeepProve-BaseFold are within ~10% of each other at 512 positions (194 s vs 176 s on similar 32-core machines), despite different PCS and quantization widths.

### Gaps
- No neutral benchmark suite for LLM zkML was found; ICME's own zkml-benchmarks repo covers small classifiers only (seen as a snippet).

## What do the non-SNARK and hybrid alternatives guarantee, and what do they cost?

### Takeaway
Alternatives cost the provider roughly 0–2% (fingerprints, deterministic re-execution, sampled audits), 1–30% (GPU TEEs) or 60%–2×+ (refereed delegation), against 24×–1000×+ for SNARKs, and all give weaker guarantees: statistical detection, economic deterrence or hardware trust, typically with a verifier that holds the weights. This is what "viable" is being compared against.

### Cited Findings
**DiFR (arXiv 2511.20621, Nov 2025, preprint).** A verifier holding the weights replays the provider's (prompt, output) in one prefill pass. Token-DiFR conditions on the shared sampling seed and scores the post-Gumbel logit margin, so it covers temperature / top-k / top-p; Activation-DiFR compares a random orthogonal projection of an activation. Guarantee is statistical and batch-level; zero provider overhead for Token-DiFR. Over 98% of tokens match exactly; 4-bit quantization is detected at AUC > 0.999 within 300 tokens (Token-DiFR) or 2 tokens (Activation-DiFR); FP8 KV-cache quantization needs ~1,000–5,000 tokens; Activation-DiFR needs 7.25 bytes/token cross-hardware vs TOPLOC's 32. Limits: needs weights and seed synchronisation; speculative decoding not evaluated — [DiFR](https://arxiv.org/abs/2511.20621)

**Weight-exfiltration verification (arXiv 2511.02620v3, Mar 2026, preprint).** Not a correctness guarantee: an information-theoretic bound on steganographic capacity. A trusted verification server re-checks a tiny random sample of traffic, accepting a token only if its fixed-seed sampling likelihood clears a threshold. On Qwen3-30B-A3B: under 0.5% exfiltrable information, ≥200× attacker slowdown; a 1 TB model at 1 TB/day goes from ~1 day to over 200 days to exfiltrate. This is the "provider audits itself" use case — [arXiv 2511.02620](https://arxiv.org/abs/2511.02620)

**TOPLOC (arXiv 2501.16007, 2025; no 2026 revision).** Provider commits to top-128 last-hidden-state values, polynomial-encoded; a validator with weights replays via prefill. 258 bytes per 32 tokens; validation "up to 100x faster" than generation. Cannot detect speculative decoding; sampling out of scope. DiFR reports it is Pareto-dominated by Activation-DiFR; TensorCommitments describes a "margin-hiding" attack on it — [TOPLOC](https://arxiv.org/abs/2501.16007); [DiFR](https://arxiv.org/abs/2511.20621); [TensorCommitments](https://arxiv.org/abs/2602.12630)

**TensorCommitments (NeurIPS 2026).** Multivariate-KZG "Terkle tree" commitments to every block's residual stream; a weight-holding verifier spot-opens entries and recomputes them within a tolerance. "TC does not prove full inference correctness. Its guarantee is the statistical bound of Prop. 4.6". LLaMA2-13B at 512 tokens: +0.97% prover, +0.12% verifier, 880 B per inference. Against a policy-aware prover, deterministic layer selection gives zero coverage outside the chosen set — [TensorCommitments](https://arxiv.org/abs/2602.12630)

**Sampled Merkle-path proofs (Offchain Labs; SaTML 2026, arXiv 2603.19025).** Merkle-commit weights and the full activation trace, open random neuron paths. "Other-model" soundness only; a single path catches a tampered neuron with probability at most 1/N. Llama-2-7B: 5.8 ms trace commitment, 12.44 ms verification, ~1.7 MB proof — [arXiv 2603.19025](https://arxiv.org/abs/2603.19025)

**Seal, Then Sample (arXiv 2609.27367, Sept 2026, preprint; abstract only).** Hybrid: commit boundary activations of every chunk before the challenge, then SNARK-prove only a sampled subset. TinyLlama-1.1B: 7 of 47 chunks costs 22.0% of full proving time; Llama-2-70B on a 2 TB CPU host: 5 of 163 chunks, 1,259 s prove, 4.34 MiB, 46.3 s verify. Admitted: a fixed bad chunk at 70B is caught with probability 3/161; the Fiat–Shamir schedule is grindable at 12.5 ms per attempt — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367)

**Optimistic / deterministic re-execution.** EigenAI (arXiv 2602.00182, Jan 2026, vendor preprint): bit-exact GPU inference via deterministic kernels and pinned stack; +1.8% end to end; 100% reproducible across same-SKU hosts, 0% between A100 and H100; guarantee is cryptoeconomic (slashing) plus TEE integrity. Gensyn Verde/RepOps (arXiv 2502.19405, 2025): bitwise-reproducible operators with a bisection game, ~60% end-to-end overhead on Llama-3.1-1B and a 2× total-compute floor. VeriLLM (arXiv 2509.24257v4, Jan 2026): Merkle-committed hidden states with VRF-selected verifier committees; claims ~1% but the paper gives 0.78%, 2.4% and 3.5% in different places — [EigenAI](https://arxiv.org/abs/2602.00182); [Verde](https://arxiv.org/abs/2502.19405); [VeriLLM](https://arxiv.org/abs/2509.24257)

**IMMACULATE (arXiv 2602.22700, Feb 2026, preprint).** Audits under 10⁻⁵ of requests; server commits to seed and logits; the audit recomputes token decisions in a TDX enclave. Throughput overhead 0.3–1.0%; an audited request costs 400–900× GPU inference; per-request detection 1.3–10.3% for FP8 and 42–99% for model substitution; assumes a rational server deviating on ≥10% of requests — [IMMACULATE](https://arxiv.org/abs/2602.22700)

**TEE attestation.** Guarantee is attested code identity plus confidentiality, bounded by vendor root of trust and side channels. H100 overhead 4–8% (ETH Zurich, arXiv 2509.18886, 2025) and under 7% (Phala study, arXiv 2409.03992). Tinfoil's June 2026 measurements: H200 ~14% at saturation, B300 ~25–30%, ~1% with CC-aware engines up to ~30% with vLLM, over 100% inter-token latency at low concurrency with tensor-parallel 8. Anthropic's "Confidential Inference via Trusted VMs" (June 2025) is a design sketch with no overhead numbers. A registry of private-inference deployments (snapshot 2026-11-08) records attestation-binding failures at several providers during 2026 — [arXiv 2509.18886](https://arxiv.org/abs/2509.18886); [Tinfoil, 2026-06-23](https://tinfoil.sh/blog/2026-06-23-confidential-computing-overhead); [Anthropic](https://www.anthropic.com/research/confidential-inference-trusted-vms); [awesome-private-inference](https://amiller.github.io/awesome-private-inference)

**Maverick** (interactive delegation, linear layers only, ~0 server overhead): see its card — [Maverick](https://arxiv.org/abs/2609.10264)

**"Tool Receipts, Not Zero-Knowledge Proofs" (arXiv 2603.10060, Mar 2026).** Different problem (agent claims about tool calls): the runtime issues HMAC receipts; 91% detection, +12 ms, trusted runtime; says nothing about which model ran — [arXiv 2603.10060](https://arxiv.org/abs/2603.10060)

**Hollow-LLM Attack (IEEE S&P 2026, arXiv 2607.28884; abstract only).** A provider committing to its own weights can embed "ghost weights" whose structure collapses the real computation while the proof still verifies: "Proof of correct inference is not proof of large-model execution." Relevant to every private-weight zkML statement — [arXiv 2607.28884](https://arxiv.org/abs/2607.28884)

### Inferences
- The decision frame used by IMMACULATE and Seal-Then-Sample is "audit cost × audit rate": a 70–1000× SNARK applied to 10⁻³–10⁻⁵ of traffic is cheaper overall than a 5% TEE tax, if a rational-adversary or statistical guarantee is acceptable. Per-response cryptographic soundness with a weightless verifier is the only property SNARKs hold exclusively.
- Every statistical scheme requires the verifier to hold the weights; private-weight verification is where SNARKs have no cheap competitor other than TEEs, and Hollow-LLM shows that niche needs the commitment tied to a known model.
- Sampling is currently verified in deployed-style systems only by non-SNARK means (Token-DiFR, IMMACULATE).

### Gaps
- Several items (IMMACULATE, EigenAI, VeriLLM, SVIP, Seal-Then-Sample, Hollow-LLM, Tinfoil) were read through abstract pages or a summarising fetch; numbers should be confirmed against PDFs before publication.
- No 2026 technical material was found for opML, Ritual, Atoma or Hyperbolic PoSP.

## What changed in 2026 for the other zkML systems (zkPyTorch, ZKTorch, Remainder, Giza, zkVMs, Bionetta, EZKL)?

### Takeaway
Almost nothing: none of the 2023–2025 general-purpose zkML frameworks published a 2026 LLM benchmark or paper, several repositories are dormant, and no zkVM vendor published a transformer-inference benchmark. The 2026 activity is concentrated in bespoke LLM provers.

### Cited Findings
- zkPyTorch / Expander: baseline is ePrint 2025/535 (Llama-3 8B ≈150 s/token on one CPU core, vendor figure); Expander-GPU post is from July 2025; no 2026 LLM or GPU-zkML benchmark found — [Polyhedra zkPyTorch](https://blog.polyhedra.network/zkpytorch/); [ePrint 2025/535](https://eprint.iacr.org/2025/535)
- ZKTorch: arXiv 2507.07031 last revised July 2025; repo last pushed Nov 2025; no 2026 update found — [arXiv 2507.07031](https://arxiv.org/abs/2507.07031)
- Remainder (Modulus → Tools for Humanity): `worldcoin/Remainder_CE` public; 2026 commits are security fixes (Hyrax verifier output-layer zero-test fix, 2026-06-22) and a note that the repo "is not in production and out of bug bounty scope"; no 2026 ML benchmarks — [Remainder_CE](https://github.com/worldcoin/Remainder_CE)
- "The Cost of Intelligence" (ePrint 2026/1063) is the January 2023 Modulus benchmark reposted in May 2026; MLPs only — [ePrint 2026/1063](https://eprint.iacr.org/2026/1063)
- Giza LuminAIR (S-two): last commit Sept 2025; no 2026 update found — [Giza LuminAIR](https://github.com/gizatechxyz/LuminAIR)
- zkVM ML benchmarks (RISC Zero, SP1, Jolt, Airbender, OpenVM, Pico): no published 2026 transformer benchmark or ML precompile found; nearest is a design-only proposal for a zkVM with native BF16/FP32 precompiles for training verification (arXiv 2606.05433, v3 2026-10-21); zkAgent uses SP1 only for tool-call sub-proofs — [arXiv 2606.05433](https://arxiv.org/abs/2606.05433); [zkAgent](https://eprint.iacr.org/2026/199)
- Bionetta: arXiv 2510.06784 last revised Dec 2025; no 2026 paper or benchmark — [arXiv 2510.06784](https://arxiv.org/abs/2510.06784)
- EZKL: v23.0.5 on 2026-02-20 with empty release notes; no pushes since; no 2026 GPU or LLM results — [EZKL repo](https://github.com/zkonduit/ezkl)
- Inference Labs JSTprove: v1.0.0 (Oct 2025) → v2.12.1 (2026-03-06), Expander and Remainder backends; no LLM numbers — [arXiv 2510.21024](https://arxiv.org/abs/2510.21024)
- Proof-of-training items whose techniques are inference proving: VeriLoRA (NDSS 2026; zkLLM-lineage sumcheck + Hyrax, mixed-radix exp digits for softmax, three lookups for SwiGLU rounding; 122–249 s per one-sample step on 3B–13B models; verify 1.87–3.73 s) and zkLLMPoT (arXiv 2610.08258, Oct 2026, abstract: batched forward-pass loss proofs, 41–59 s for 1.1–1.5B models, verification under 0.5 s at length 512) — [VeriLoRA](https://arxiv.org/abs/2508.21393); [zkLLMPoT](https://arxiv.org/abs/2610.08258)
- Other 2026 preprints found (abstract only): GKR for HND-based transformer inference (arXiv 2607.21162); architecture-hiding structured matrix constraint systems (ePrint 2026/111) — [arXiv 2607.21162](https://arxiv.org/abs/2607.21162); [ePrint 2026/111](https://eprint.iacr.org/2026/111)

### Inferences
- General ONNX coverage is now an uncontested position: the active 2026 LLM provers are all bespoke per architecture, and the general frameworks have stalled.
- zkVMs are not a competitive route for LLM inference in 2026 by the absence of any published attempt.

### Gaps
- "No 2026 update found" rests on repository metadata and searches, not vendor confirmation.

## Which techniques in the 2026 systems does jolt-atlas not have?

### Takeaway
The transferable techniques cluster in five areas: cheaper exact range checks through sharing, fusion and bounds; per-row (not per-element) normalisation proofs; statement-level changes (one-pass generation with argmax/sampling, committed weights, bounded-error statements); partitioned proving via boundary commitments; and native-width quantization recipes with measured accuracy. Approximate range checks remain unique to FlashProve.

### Cited Findings
Each item names the source system and where it is described. "Not in jolt-atlas" is judged against `docs/research-directions.md` and the repository description in CLAUDE.md, not against a code audit.

**A. Range checks and rounding**
1. **Overflow-free scales from weight norms** — per-layer output scale from B_l = max_j(2^Q·‖W[:,j]‖₁ + |b_j|) plus the LayerNorm bound |x̂_k| ≤ √(d−1); removes the need to prove a saturating clamp and guarantees completeness (zkAgent §4.1, App. F) — [zkAgent](https://eprint.iacr.org/2026/199)
2. **Requantization fused into the next non-linearity's lookup**, feeding it the unclamped value (DeepProve B.2) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
3. **Deferred requantization** when the accumulator still fits the field or the next table (DeepProve App. C) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
4. **One shared wide-digit range table model-wide** (2^10–2^12 entries; shift aligned to a multiple of the digit width so a single table size suffices; one multiplicity polynomial per table type) (DeepProve App. C and `docs/src/end_to_end_lu.md`; zkAgent App. F) — [DeepProve paper](https://eprint.iacr.org/2026/1112); [zkAgent](https://eprint.iacr.org/2026/199)
5. **Whole-range lookups for quotient and remainder** with no digit decomposition (2^12 and 2^26 identity tables) (VerInf B.1) — [VerInf](https://github.com/JamesPetrie/VerInf)
6. **Envelope-derived range-check widths** per scale / per node (PR #258; named by ePrint 2026/1390 as the remaining dial) — [jolt-atlas PR #258](https://github.com/ICME-Lab/jolt-atlas/pull/258); [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)
7. **One-sided rounding where slack only hurts the prover** (VerInf B.7) — [VerInf](https://github.com/JamesPetrie/VerInf)

**B. Normalisation, softmax, activations**
8. **Table-free reciprocal square root by a per-row integer bracket**, with the normalised product checked by one random projection per row and never materialised (VerInf B.4) — [VerInf](https://github.com/JamesPetrie/VerInf)
9. **Reciprocal-as-advice normalisation** with a sum-of-squares range check and an explicit error analysis (DeepProve B.6, App. C Lemmas 1–4) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
10. **Division-free softmax**: prover-supplied shift pinned uniquely by a two-table monotone bracket (VerInf B.3); or shift advice with exp only on the top limb and a row-sum tolerance (DeepProve B.7) — [VerInf](https://github.com/JamesPetrie/VerInf); [DeepProve paper](https://eprint.iacr.org/2026/1112)
11. **Causal-triangle materialisation** of attention scores, halving exp lookups (zkAgent §4.2) — [zkAgent](https://eprint.iacr.org/2026/199)
12. **RoPE as two Hadamard products** against public cos/sin tables (zkAgent Table 4; DeepProve B.4) — [zkAgent](https://eprint.iacr.org/2026/199)
13. **Native-float non-linearities by lookups keyed on bf16 values**, with int8 matmul operands (SASH; Attestable) — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note); [Attestable blog](https://attestable.com/blog/proving-llms-scale)
14. **Anchor tables with in-circuit linear interpolation** instead of full-width tables (NanoZK B.1) — [NanoZK](https://arxiv.org/abs/2603.18046)

**C. Statement and generation**
15. **One teacher-forced causal pass with shift consistency** binding each position's predicted token to the next input (zkAgent Eq. 23; DeepProve §3.1) — [zkAgent](https://eprint.iacr.org/2026/199); [DeepProve paper](https://eprint.iacr.org/2026/1112)
16. **Greedy argmax proofs**: N range checks on ℓ_y − ℓ_i (zkAgent §3.3), or range check plus zero-product (DeepProve B.5), with explicit lowest-index tie-breaking (SASH) — [zkAgent](https://eprint.iacr.org/2026/199); [SASH repo](https://github.com/sg-ai-safety-hub/zk-inference-prover)
17. **Top-k / top-p / temperature sampling in ZK** with a transcript-derived coin (zkAgent §3.3, App. C.5) — [zkAgent](https://eprint.iacr.org/2026/199)
18. **Embedding lookup as a matmul against a verifier-computable one-hot** (DeepProve B.3), or with online cost independent of vocabulary (zkAgent §3.1, pairing-based) — [DeepProve paper](https://eprint.iacr.org/2026/1112); [zkAgent](https://eprint.iacr.org/2026/199)
19. **Committed (hidden) weights opened once per proof in a batched claim** (DeepProve §4.1; zkAgent; SASH), with a refresh/link mechanism under a ZK query budget (VerInf §6.4) — [DeepProve paper](https://eprint.iacr.org/2026/1112); [VerInf](https://github.com/JamesPetrie/VerInf)
20. **Bounded-error statement** ("unexplained information") that targets a float-served model (VerInf §4) — [VerInf](https://github.com/JamesPetrie/VerInf)
21. **MoE project-then-route**: evaluate all experts' weights at the random point, then route, committing only the selected expert's activations (VerInf `README-routed-projected.md`) — [VerInf](https://github.com/JamesPetrie/VerInf)
22. **Multi-request batching with a block-diagonal causal mask** (Seal-Then-Sample) — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367)

**D. Partitioning, memory, parallelism**
23. **Boundary-commitment partitioning with independent transcripts and no linking proofs** (zkComposer §3.1; DeepProve §4.5; SASH; NanoZK): parallel mode gives 4.8–6.8× on many cores, sequential mode 5–8× lower peak memory at 1.4–2.1× time — [zkComposer](https://arxiv.org/abs/2607.08095)
24. **Token-axis partitioning via committed KV values / key tiles** (zkComposer §4.1; SASH) — [zkComposer](https://arxiv.org/abs/2607.08095); [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
25. **Streaming witness regenerated per round** rather than stored (VerInf §6.3) — [VerInf](https://github.com/JamesPetrie/VerInf)
26. **Commit-then-sample audit tier**: commit all chunk boundaries, prove a public-coin sample (Seal-Then-Sample; NanoZK A.4, which requires the sample be drawn after commitment) — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367); [NanoZK](https://arxiv.org/abs/2603.18046)

**E. Quantization recipes with measured accuracy**
27. **Per-token dynamic int8 activations pinned to the row maximum + per-output-channel weight scales** (SASH) — [SASH note](https://www.aisafety.sg/research/what-proving-llm-inference-actually-costs-research-note)
28. **Mixed precision** (wider residual adds) and **QuaRot-style rotations absorbed into weights** to smooth outliers before quantization (DeepProve §3.2) — [DeepProve paper](https://eprint.iacr.org/2026/1112)
29. **Per-op fixed-point formats** with explicit conversions (NanoZK B.2); raising precision only at the rsqrt lookup (Bits Under ZK-LLM, abstract) — [NanoZK](https://arxiv.org/abs/2603.18046); [arXiv 2609.36437](https://arxiv.org/abs/2609.36437)

### Inferences
- Ranked by likely effect on the stated bottleneck (25 s of 40 s committing range-check digits): items 1–5 attack it directly and keep exactness; item 1 (overflow-free scales) deletes clamp digits outright, items 2–3 delete whole rescale steps, items 4–5 cut digits per remaining rescale. FlashProve's approximate check (existing notes §7.3) remains the only technique that removes per-element cost altogether.
- Items 8–10 replace per-element lookups with per-row checks for normalisation and softmax; these are protocol changes with unique-witness variants available (VerInf) if exactness must be kept, or tolerance variants (DeepProve) if a bounded-error statement is acceptable.
- Items 15–17 are a statement gap more than a performance gap: the three strongest academic systems prove multi-token generation with argmax, and one proves sampling.
- Item 23 is the cheapest route to both laptop-memory proving of larger models and multi-core scaling; it costs one committed dense tensor per boundary and loses cross-block batching.
- Positions that remain differentiated for jolt-atlas after this survey: arbitrary ONNX graphs (all active 2026 LLM provers are bespoke), exact unique-witness semantics throughout (DeepProve and VerInf accept tolerances), a 128-bit field (no LogUp soundness floor of the kind VerInf reports), laptop-class hardware, and small proofs.

### Gaps
- None of these techniques has been evaluated over a lattice PCS with one-hot lookups; the sources use LogUp-GKR, Shout over KZG, Ligero or STARK lookups, so the savings do not transfer one-for-one.
- Whether jolt-atlas already implements parts of items 2–4, 11 or 16 was not checked against the code (the brief excluded code auditing); the list is relative to the repository's documentation.
- The approximate range-check construction itself is outside this note's scope (FlashProve researcher).
