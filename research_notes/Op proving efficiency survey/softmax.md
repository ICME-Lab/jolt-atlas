# Softmax in jolt-atlas: current protocol, literature, and redesign options

Conventions used in these notes:
- **[CODE]** = verified by reading the repository on branch `dev` (paths relative to `/Users/kg/swe/icme/official/jolt-atlas`). No cargo command was run; nothing here was tested.
- **[LIT]** = claim from a paper or blog, with URL.
- **[EST]** = my own estimate or design reasoning, not verified by code or measurement.
- Notation: a softmax node has `F` rows (product of leading dims) and `N` lanes (last axis); `T = F·N` elements; `S = 2^14` (MODEL_SCALE = 14). "Chunk" = one committed one-hot `ra` polynomial with `K_CHUNK = 16` addresses per element (LOG_K_CHUNK = 4).

## 1. What exactly does the current protocol define, commit to and look up, and where is the cost?

### Takeaway
Per element the protocol commits **22 one-hot chunk polynomials** (22 nonzeros, 352 padded cells per element at scale 14) and runs **19 sumcheck relation instances** over four stages; per row it sends **3 field elements in the clear** (`exp_sum_q`, `max_k`, `argmax_k`). 14 of the 22 chunks serve only `z = max − x` (an 8-chunk 32-bit clamp lookup plus two 3-chunk exp-digit lookups) and 8 serve two 14-bit remainders. At today's benchmark sizes softmax is a rounding error (0.43 s of a 50.3 s prove in the only trace on disk); it only matters at long context because it scales as seq².

### Cited Findings

**Fixed-point definition (atlas-trace) [CODE]**
- Per row over live lanes `0..active_len`: `max_k`, `argmax_k` (first position of the max) — `atlas-trace/src/op/softmax.rs:87-94`.
- Per lane: `z = max_k − x` (i64, error if > i32::MAX), saturated `zu = min(z, z_bound−1)`, split `z_hi = zu >> log2_base`, `z_lo = zu & (base−1)`, two sub-table reads `hi_val = lut_hi[z_hi]`, `lo_val = lut_lo[z_lo]`, `exp_q = ⌊hi_val·lo_val / S⌋`, `r_exp = hi_val·lo_val − exp_q·S` — `atlas-trace/src/op/softmax.rs:97-127`.
- Per row: `exp_sum_q = Σ exp_q` (must fit i32), `inv_sum = ⌊S²/exp_sum_q⌋` — `softmax.rs:128-140`.
- Per lane: `softmax_q = ⌊exp_q·inv_sum / S⌋`, `R = exp_q·inv_sum − softmax_q·S` — `softmax.rs:142-149`. Floor everywhere (no round-to-nearest).
- Masked lanes (`j ≥ active_len`) are zero in output and every witness column — `softmax.rs:12-16,150`.
- Witness columns captured per element: `x, exp_q, r, z_hi, z_lo, exp_hi, exp_lo, r_exp`; per row: `max_k, argmax_k, exp_sum_q, inv_sum` — `softmax.rs:68-84`.
- Table generation: `needed = ⌈S·(s+1)·ln2⌉ + 2` (cutoff beyond which `S·e^(−z/S)` rounds to 0), `base = 2^b` with smallest `b` s.t. `4^b ≥ needed`, `hi_size = needed/base + 2`, `lut_hi[h] = round(S·e^(−h·B/S))`, `lut_lo[l] = round(S·e^(−l/S))` — `atlas-trace/src/semantics/tables.rs:246-295`. At scale 14 this gives [EST, arithmetic on the code's formula] `needed ≈ 170,350`, `base = 512`, `hi_size = 334` (padded to 512), so both sub-tables are 2^9 entries and `z_bound = 2^18`; `SOFTMAX_CLAMP_BOUND` is derived the same way and unit-tested as 11 at scale 8 and 16 at scale 12 — `common/src/consts/softmax.rs:7-54`.
- Importer lowers ONNX `Softmax` to `SoftmaxLastAxis { active_len: None }` and requires the softmax axis to be last — `atlas-onnx-import/src/lower.rs:344-356`.

**Committed polynomials [CODE]** — `jolt-atlas-verifier/src/onnx_proof/ops/softmax_last_axis/mod.rs:78-101`, chunk count `d = ⌈log_K / 4⌉` from `atlas-sumcheck/src/config.rs:43-58` and `common/src/consts/general.rs` (`XLEN = 32`, `LOG_K_CHUNK = 4`, `MODEL_SCALE = 14`):

| Family | log K | chunks d | purpose |
|---|---|---|---|
| `SoftmaxRemainderRaD` | 14 | 4 | range check `R ∈ [0,S)` |
| `SoftmaxExpRemainderRaD` | 14 | 4 | range check `r_exp ∈ [0,S)` |
| `SoftmaxClampRaD` | 32 (XLEN) | 8 | saturating clamp lookup on `z` |
| `SoftmaxZHiRaD` | 9 | 3 | Shout address for `lut_hi` |
| `SoftmaxZLoRaD` | 9 | 3 | Shout address for `lut_lo` |
| **total** | | **22** | |

No dense witness column is committed; `exp_q, R, r_exp, exp_hi, exp_lo, z_hi, z_lo, z` are virtual polynomials whose claims are prover advice bound by the relations below.

**Per-row auxiliary vectors [CODE]**: `SoftmaxSumOutput(idx,k)`, `SoftmaxMaxOutput(idx,k)`, `SoftmaxMaxIndex(idx,k)` appended to the transcript for every row `k` — verifier `mod.rs:260-264`, prover `jolt-atlas-core/.../softmax_last_axis/mod.rs:266-287`. The module doc states they "add O(F) field elements to the proof size" and carries `TODO(#218)` — verifier `mod.rs:4-8`. The verifier computes `inv_sum[k] = ⌊S²/exp_sum_q[k]⌋` itself (`mod.rs:310-334`) and evaluates the argmax indicator MLE in `O(F·log N)` (`max.rs:95-107`).

**Stages and relations [CODE]** (degree = `fn degree`, rounds = `fn rounds`):

| Stage | Relation | degree | rounds | file |
|---|---|---|---|---|
| 1 (inline) | RecipProduct: `S·out(r)+R(r) = Σ eq·exp_q·inv_sum(k)` | 3 | log T | verifier `recip_mult.rs:22,123-154` |
| 1 | ExpSum: `exp_sum(r_k) = Σ eq(r_k,k)·exp_q` | 2 | log T | `exp_sum.rs:15,79-109` |
| 1 | IdentityRangeCheck on `R` (14-bit) | 2 | 14 + log T | `atlas-sumcheck/src/identity_range_check/mod.rs:22,160-178`; built at verifier `mod.rs:403-405` |
| 2 (inline) | ExpProduct: `S·exp_q(r)+r_exp(r) = Σ eq·exp_hi·exp_lo` | 3 | log T | `exponentiation/mult.rs:14,69-100` |
| 2 | Max indicator: `max(r_k) = Σ eq(r_k,k)·e·x`, `e` public from `argmax_k` | 3 | log T | `max.rs:18,119-148` |
| 2 | IdentityRangeCheck on `r_exp` (14-bit) | 2 | 14 + log T | verifier `mod.rs:455-457` |
| 2 | RaVirtualization for `R` (d = 4) | d+1 = 5 | log T | `atlas-sumcheck/src/ra_virtual/mod.rs:77-83`; verifier `mod.rs:459-461` |
| 2 | Booleanity (+Hamming linear term) for `R` | 3 | 4 + log T | `atlas-sumcheck/src/booleanity/mod.rs:20,141-147` |
| 3 (deferred batch `SoftmaxStage3`) | Shout ReadRaf `lut_hi` (`rv + γ·raf = Σ_a ra(a,r)(val(a)+γ·id(a))`) | 2 | 9 (address only) | `atlas-sumcheck/src/shout/mod.rs:33,96-134`; verifier `mod.rs:495-501` |
| 3 | Shout ReadRaf `lut_lo` | 2 | 9 | verifier `mod.rs:503-509` |
| 3 | Prefix-suffix ReadRaf, `SoftmaxClampTable<32>`: `z_hi·B + z_lo = Clamp[z]` | 2 | 32 + log T | `atlas-sumcheck/src/ps_shout/mod.rs:30,314-320`; `significance_clamp.rs:9-16,56-67`; verifier `mod.rs:511-514` |
| 3 | RaVirtualization + Booleanity for `r_exp` (d = 4) | 5 / 3 | log T / 4 + log T | verifier `mod.rs:516-519` |
| 4 (deferred one-hots) | RaVirtualization + Booleanity for `z_hi` (d = 3) | 4 / 3 | | verifier `mod.rs:209-213,571-577` |
| 4 | RaVirtualization + Booleanity for `z_lo` (d = 3) | 4 / 3 | | same |
| 4 | RaVirtualization + Booleanity for clamp `ra` (d = 8) | **9** / 3 | | verifier `mod.rs:579-583` |

That is 3 + 5 + 5 + 6 = **19 relation instances** per softmax node. Stages 3–4 are built inline (challenges drawn, claims appended) and proven after the node loop in deferred batches — prover `mod.rs:194-215`.

- Operand link (no sumcheck): `X(r2) = max_k(r2_lead) − z(r2)` using the clamp lookup's witness claim — verifier `mod.rs:536-560`. `z ≥ 0` (hence `max_k ≥ x` everywhere) follows from the clamp lookup's 32-bit address / raf; attainment follows from the max-indicator sumcheck.
- The clamp table clamps to `[0, 2^SOFTMAX_CLAMP_BOUND − 1]`, floor-at-0, and is const-generic on `MODEL_SCALE`, so softmax supports only that one scale — `atlas-lookup-tables/src/clamp.rs:260-274`; verifier `mod.rs:10-14`.
- **Masked softmax is not provable today.** Prover asserts `active_len ∈ {None, last_dim}` ("use a power-of-two softmax axis"), verifier returns an error — prover `mod.rs:85-94`, verifier `mod.rs:115-125`; `op_surface.rs:595-596` pins it as `should_panic(expected = "masked softmax")`. The tracer sets `active_len` whenever the last axis is not a power of two (`softmax.rs:179-187`), so a non-power-of-two attention width cannot currently be proven.
- The design doc linked from the module (HackMD) describes an **older variant**: `S = 4096`, tables of ~146 and 256 entries, a `sat_diff` identity range check instead of the clamp table, separate HammingWeight instances, "24 sumcheck instances" — [HackMD](https://hackmd.io/@R5TO3fi7TlWQKnY6Ejr2eg/Byjucoqjbx). The code differs (clamp is a 32-bit prefix-suffix lookup; Hamming is folded into Booleanity).

**Timing data found [CODE/measured from an existing file]**
- No softmax timing exists in `docs/` or `specs/` (grep for "softmax" finds only migration notes: `specs/adoption-program.md:240-243`, `docs/upstream-inventory.md:125-126`, `specs/verifier-audit-simplification.md:99,188,341`). `reports/` is empty.
- The only data is `trace-1792493663983876.json` at the repo root (397 MB, mtime Oct 20). It is **stale**: it contains `DoryScheme::prove_rlc`, `BN254::multi_pair` and `TractModelLoader` spans, i.e. it predates the Akita/Fp128 and tract-free importer cut-over. Aggregated by me with a throwaway script: `ONNXProof::prove` 50.3 s, 2172 nodes, 24 softmax nodes; `SoftmaxLastAxis::prove` (stages 1–2 inline) **0.426 s total** (stage 1 0.103 s, stage 2 0.237 s); `SoftmaxLastAxis::verify` 0.145 s of a 9.37 s verify; softmax stages 3–4 are inside `deferred_lookups::prove_all` (9.7 s for all ops) and are not separately attributable; commitment is 17.1 s for all ops.

### Inferences
- [EST] Cost per element is dominated by one-hot chunk count (commit, Booleanity, RaVirtualization, opening reduction all scale with it). Breakdown of the 22: clamp 8 (36%), exp digits 6 (27%), `R` 4 (18%), `r_exp` 4 (18%). The clamp also brings the two most expensive sumchecks: a 32 + log T round prefix-suffix read-raf and a degree-9 RaVirtualization.
- [EST] 2172 nodes / 24 softmax matches Qwen2-0.5B at a short prompt (the memory notes say a 9-token prompt padded to 16). At that size each node has about 16·16·16 = 4096 elements, ~10^5 softmax elements in total — under 1% of the proof. The team's sense that softmax "can definitely be improved" is true per element, but the payoff appears only at long context: GPT-2 at 1024 tokens has 12 layers × 12 heads (16 padded) × 2^20 ≈ 1.5–2·10^8 softmax elements, i.e. 3–4·10^9 one-hot nonzeros at 22 chunks, roughly an order of magnitude more than the entire short-prompt Qwen proof described in the memory notes (2^28.9 nonzeros).
- [EST] The attention-score einsum feeding softmax is also seq²-sized and carries its own clamp chunks, so any long-context plan has to treat `QKᵀ → softmax → ·V` as one unit (see §7, proposal D).
- **Suspected soundness gap, from code reading only (not tested).** The verifier parses `exp_sum_q[k]` twice with different widths. `SoftmaxLastAxisVerifier::new` truncates the advice to i32 (`to_u64() … as i32`, verifier `mod.rs:285-288`) and that truncated vector is what `cache_exp_sum` binds to the ExpSum relation (`mod.rs:352-359`). `inv_sum_evals` re-reads the same advice at full 64-bit width (`mod.rs:319-331`). A prover that sends `exp_sum_q[k] + 2^32` therefore passes the ExpSum check with the honest sum, while the verifier computes `inv_sum[k] = ⌊2^28 / (2^32 + s)⌋ = 0`; RecipProduct is then satisfied by `out[k,·] = 0, R[k,·] = 0`. If this reading is right, a malicious prover can zero any softmax row. It needs a targeted test (e.g. a `mutate::Site` adding 2^32 to that claim) before being treated as confirmed.
- [CODE-derived] The same function panics rather than erroring on malformed advice (`.expect("auxiliary scalar should fit within 64 bits")`, `mod.rs:278`; `.expect("exp_sum_q[k] should fit in u64")`, `mod.rs:323-324`), which is a verifier DoS surface and conflicts with the "no panics" goal in `specs/verifier-audit-simplification.md`.
- [EST] The clear aux vectors leak `max`, `argmax` and `Σexp` per attention row, so the current softmax cannot be made zero-knowledge without #218-style changes.
- [EST] Accuracy note on the current semantics: `inv_sum = ⌊S²/sum⌋ ∈ [S/N, S]`, so for diffuse rows at large `N` the reciprocal carries a relative truncation error up to `N/S` (6% at `N = 1024, S = 2^14`), biased downward, on top of the 1/S output quantization. This is invisible at 16 tokens and worth measuring at long context.

### Gaps
- No post-Akita timing exists in the repo, and I was not allowed to run the prover; the share of commit / deferred-lookup time attributable to softmax at any context length is unmeasured.
- I did not read `deferred_lookups.rs` in full; whether stage-4 one-hot checks are packed into shared buckets (as the memory notes describe for clamp/remainder families) is unverified. The notes say softmax/activation one-hots (840 polys) were "not packed" as of August 2026.
- `gh issue view 218` returned no output in this environment; the #218 text was not retrieved. Its content is inferred from the module TODO and the user's memory file.

## 2. zkLLM's zkAttn: how many lookups per element, how does it map onto Shout, can it remove max/argmax and division?

### Takeaway
zkAttn removes the max, the argmax and the division entirely: the prover commits a per-row normalizer `ẑ ≈ log-sum-exp`, the exponential of `ẑ − x` is the probability itself, and the only normalization check is that each row sums to 1 within a tolerance. Its digit count (K = 5 tables of 2^16) is a consequence of its 2^64-scale inputs and does not transfer; what transfers to jolt-atlas is the structure, which would cut the per-element chunk count from 22 to roughly 8–14, at the price of a prover-chosen shift (bounded malleability, not bit-exactness).

### Cited Findings
- Shift invariance: `ẑ := γ√d·ln(Σ_j exp(z_j/(γ√d)))`, `s(z) = s(z − ẑ)`; the prover uses `⌊ẑ⌉` and `Z' = Z − ⌊ẑ⌉·1ᵀ`; the paper states that computing ẑ "is highly intricate and not directly verifiable", so correctness is enforced through the row sums instead — [zkLLM, arXiv 2404.16109](https://arxiv.org/html/2404.16109).
- Committed objects in the zkAttn protocol: `Z`, `⌊ẑ⌉`, the digit tensors `X^(k)`, `Y`, `Y^(k)`, and row sums `ŷ`; no max or argmax step exists — [zkLLM](https://arxiv.org/html/2404.16109).
- Digit decomposition: `x = Σ_k x^(k)·B^(k)`, `θ·exp(−x/(γ√d)) = Π_k θ^(k)·exp(−B^(k)x^(k)/(γ√d))`; each segment has a table of `b^(k)` entries. The most significant `M` segments are indicator tables (`y = 1{x^(k) = 0}`), the least significant `L` segments are only range-checked (their factor is treated as 1), the middle ones are real exp tables — [zkLLM](https://arxiv.org/html/2404.16109).
- Normalization: division is avoided ("to circumvent the verification of real division operations"); `ŷ = Y.sum(axis=1)` must lie in `T_R = [θ−E, θ+E]`, checked by one tlookup per row — [zkLLM](https://arxiv.org/html/2404.16109).
- Error bound: `ε_attn = O((K−M−L)·(n/θ)^{1/(K−M−L+1)})`; experiments use `K = 5` tlookups of size 2^16 each with `L = 3`, input scale accumulating to 2^64, and report total L1 error about 10^-2 — [zkLLM](https://arxiv.org/html/2404.16109).
- Cost statement: zkAttn prover overhead `O(K·m·n)`; the overhead depends mostly on sequence length rather than model size — [zkLLM](https://arxiv.org/html/2404.16109).
- System: BLS12-381 scalar field, Hyrax commitments, CUDA prover on an A100; sequence length 2048; OPT-125M 73.9 s prove / 141 kB, LLaMA-2-13B 803 s / 188 kB; perplexity change under 0.1 (e.g. OPT-125M 26.56 → 26.65) — [zkLLM](https://arxiv.org/html/2404.16109).
- A later paper (VeriLoRA) is described as reusing the shift-invariance + product decomposition as "zkSoftmax" — [search summary of arXiv 2508.21393](https://arxiv.org/pdf/2508.21393) (PDF body could not be parsed; secondary description only).

### Inferences
- [EST] Lookups per element in zkLLM: K = 5 digit range lookups plus K − L = 2 value lookups, plus one tolerance lookup per row. In a one-hot/Shout system a 2^16-entry digit is 4 chunks, so a literal port would cost 20 chunks per element — no better than today's 22. zkLLM needs 5 digits because it does not rescale before softmax; jolt-atlas's `z` has only 18 significant bits.
- [EST] Mapping onto jolt-atlas: per row a normalizer `c_k` (clear or a small committed dense column); `z = c_k − x`; digit lookups on `z`; output `= EXP[z]` directly at scale `S`; row-sum relation (the existing ExpSum shape) producing a per-row sum, then a per-row range check `sum ∈ [S−E, S+E]`. Removed relative to today: `max_k`, `argmax_k`, `exp_sum_q` advice, `inv_sum`, the Max relation, RecipProduct, the `R` remainder (4 chunks + identity RC + RaVirtualization + Booleanity). `z ≥ 0` still holds for an honest prover because log-sum-exp ≥ max.
- [EST] The tolerance is the catch. With per-element rounding at scale `S`, an honest row sum can deviate by up to about `N/2` units, so completeness forces `E ≈ N/2`; the prover may then pick any shift whose sum lands in the window, i.e. scale a whole row of probabilities by up to `±E/S` (0.05% at N = 16, about 3% at N = 1024). That is zkLLM's approximate soundness and it conflicts with this repo's invariant that "the verifier never trusts the prover for anything the trace semantics define" (CLAUDE.md) and with golden-hash determinism.
- [EST] The malleability can be shrunk to about one quantization step without extra commitments: in Shout, table *values* are free, so the same address `ra` can read a second, high-precision table (say 2^40-scale entries) used only for the row-sum check. Then successive shifts change the sum by a factor `e^(1/S)` while rounding noise is negligible, and a window of ratio slightly above `e^(1/S)` admits one shift (rarely two adjacent ones). Exact uniqueness would require also proving `sum(c−1) > threshold`, i.e. a second full set of lookups.

### Gaps
- zkLLM does not state numeric values for `M`, `b^(k)`, `γ`, `θ` or `E` in the text I could retrieve, and gives no softmax-only timing (only a figure).
- Whether "accept any shift within one step" is acceptable for jolt-atlas is a policy question for the team, not something the literature settles.

## 3. How do the other zkML systems prove softmax, and at what cost?

### Takeaway
Everyone except zkLLM uses the same recipe as jolt-atlas — subtract the max, look up `exp`, divide with a range-checked remainder — and none publishes a softmax-only cost. The one transferable trick beyond zkLLM is zkGPT's **constraint fusion**: softmax's division is merged into the following `·V` matmul so the per-(i,j) work is only the exp lookup.

### Cited Findings
- **zkGPT** (USENIX Security 2025): attention constraints are `qEmax,i = max_j qE_ij`, `qe_ij = exp(qEmax,i − qE_ij)`, `qs_ij = round(qe_ij / (S_s·Σ_j qe_ij))`, then `qP = round(S_s·S_V/S_P · qs·qV)`; exp is a Lasso lookup of pairs `(q_t, round(e^(S_t·q_t)/S_z))`; division and sqrt are taken as advice and reduced to "a single range relation" — [zkGPT paper](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf).
- zkGPT fusion: after merging, attention becomes `qe_ij = exp(qEmax,i − qE_ij)` and `qP_ij = round(S_V·(qe·qV)_ij / (S_P·Σ_k qe_ik))`; the number of rounding constraints for attention drops from `(3/2)·h·s(s+1) + s·d` to `(1/2)·h·s(s+1) + s·d` (43,584 → 30,912 in their GPT-2 instance). Merging the exp lookup with anything else is called unprofitable because the lookup key must be an integer and the table must stay shared across layers — [zkGPT paper](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf).
- zkGPT setup and numbers: GKR + Lasso, Hyrax commitments, 16-bit quantization, 16-core Xeon with 32 threads; GPT-2 proved in 21.8 s (101 kB proof, 0.35 s verify) versus 15.8 s for zkLLM on an A100, 4026 s for ZKML; the attention row vector length used in their analysis is 32; perplexity 29.3 → 29.5 on WikiText2 — [zkGPT paper](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf); headline also on the [USENIX page](https://www.usenix.org/conference/usenixsecurity25/presentation/qu-zkgpt).
- **ZKTorch**: softmax is exponentiation, sum and division with a max subtraction so that `e^(z_i − m)` stays inside the lookup range; nonlinearities and scale corrections use the CQ lookup, division follows ZKML (Chen et al.) with a non-negative-remainder check; BN254 + KZG; GPT-2 599 s vs ZKML 3601 s on a 64-thread Xeon; 10-bit quantization for most models — [ZKTorch, arXiv 2507.07031](https://arxiv.org/html/2507.07031v2).
- **DeepProve / DeepProve-1** (Lagrange): sum-check for linear layers, LogUp-GKR lookups over decomposable tables for nonlinear ones; softmax "gets its own protocol with range checks", later consolidated into a generic parameterizable lookup layer that folds requantization into the lookup — [Lagrange Q4 2025 update](https://lagrange.dev/engineering-updates/lagrange-engineering-update-q4-2025), [DeepProve-1 announcement](https://lagrange.dev/blog/deepprove-1). The paper abstract reports about 174 tokens/minute for GPT-2, verification 1–3.7 s — [ePrint 2026/1112](https://eprint.iacr.org/2026/1112).
- **Jolt Atlas paper**: softmax is only listed among "activations (via lookup tables)"; no fixed-point definition, protocol or softmax-specific numbers. End-to-end: nanoGPT 14 s vs EZKL 237 s; GPT-2 about 38 s (HyperKZG era, M3 16 GB) — [arXiv 2602.17452](https://arxiv.org/html/2602.17452v1).
- **EZKL**: nonlinear functions such as ReLU or `e^x` are fixed-column lookups (logUp variant of the halo2 argument), with fixed-point rescaling by lookup — [search results on EZKL](https://arxiv.org/pdf/2402.02675) (general description; I found no source for its softmax specifically).
- **ePrint 2026/1390** ("A Separation Principle for Lookup-Based zkML", measured on EZKL and Jolt Atlas): in a Shout-style argument "the per-lookup proving work is a function of the access pattern alone, never of the table values"; the remaining lever is the committed address width — [ePrint 2026/1390](https://eprint.iacr.org/2026/1390).
- **NANOZK** (arXiv 2603.18046, March 2026): layer-wise proofs with lookup approximations for softmax/GELU/LayerNorm, claimed no measurable accuracy loss; authors' claims only — [arXiv 2603.18046](https://arxiv.org/html/2603.18046v1).

### Inferences
- [EST] zkGPT's fusion is directly relevant: jolt-atlas already has a fused-rescale einsum (`acc = q·D + r`). Making the divisor the per-row exp sum removes the per-(i,j) `R` remainder and moves one division to each `(i, c)` output of `softmax·V`. It wins when `seq > d_head` (GPT-2: 64) and loses below that.
- [EST] ePrint 2026/1390's principle matches the cost model in §1: the levers are address width (chunk count) and the number of separately range-checked remainders, not the function being tabulated.

### Gaps
- zkGPT's paper text does not say how the `max` itself is constrained (attainment and dominance); I could not find it in the extracted pages.
- No source gives DeepProve's softmax definition or table sizes (the "Inside DeepProve" post returned HTTP 403).
- Mystique, zkPyTorch and Kang's ZKML were not individually researched within the call budget; ZKML appears only as a baseline (4026 s / 3601 s for GPT-2) in the zkGPT and ZKTorch papers cited above.

## 4. Alternative normalizers: base-2 softmax, integer softmax, polynomial replacements

### Takeaway
None of these helps here. Base-2 and I-BERT-style polynomial exponentials optimize multiplier/shift hardware cost, which is not the cost model of a one-hot lookup prover, and the ones that change the function (Softermax, 2Quad) need fine-tuning or distillation, which is excluded.

### Cited Findings
- I-BERT integer softmax: subtract the max, reduce the exponent to `p ∈ (−ln2, 0]`, approximate `exp(p)` by `0.3585·(p+1.353)² + 0.344` (max gap 1.9·10^-3), then bit-shift; evaluated with 8-bit quantization-aware fine-tuning on RoBERTa — [I-BERT, arXiv 2101.01321](https://arxiv.org/pdf/2101.01321).
- Softermax: replaces `e^x` by `2^x`, splits integer and fractional parts, 4-segment piecewise-linear table for the fraction, integer running max so renormalization is a shift, piecewise-linear reciprocal; 8-bit I/O. It is **not** drop-in: models are fine-tuned with Softermax in the loop; BERT-Base/Large accuracy within 0.5% of an 8-bit baseline after that fine-tuning; no numbers without fine-tuning — [Softermax, arXiv 2103.09301](https://ar5iv.labs.arxiv.org/html/2103.09301).
- MPCFormer's `2Quad` replaces softmax with a quadratic and relies on knowledge distillation to recover accuracy; without distillation some GLUE tasks collapse (near-zero CoLA on BERT-large in SecFormer's comparison). PUMA keeps exact softmax/GeLU and matches plaintext accuracy at higher MPC cost — [MPCFormer, arXiv 2211.01452](https://arxiv.org/pdf/2211.01452), [SecFormer, arXiv 2401.00793](https://arxiv.org/pdf/2401.00793).

### Inferences
- [EST] Faithful base-2 (`e^(−z/S) = 2^(−z·log2e/S)`) needs a constant multiplication and rescale of `z` first, which costs a range-checked remainder (about 4 chunks) — as much as it could save. The "shift" for the integer part is not free in a lookup prover either: `2^(−n)` times a fractional table value is the same digit-product structure jolt-atlas already has (`lut_hi × lut_lo`). True base-2 softmax (dropping the `log2e`) rescales the temperature by `ln 2` and is a different model.
- [EST] I-BERT's polynomial needs two fixed-point multiplications with remainders per element, which is strictly worse than one small-table read in Shout.
- [EST] Useful ideas that do survive: Softermax's "integer max" (shift by a coarse per-row value so no exact max is needed) is the same observation as zkLLM's free shift; and "probabilities need few bits" suggests the output could be narrower than 14 bits, shrinking the `R` range check, if the consumer einsum tolerates it (untested).

### Gaps
- No source reports accuracy of base-2 or polynomial softmax applied to unmodified pre-trained GPT-2/Qwen/BERT without any fine-tuning; BOLT was not retrieved.

## 5. Can the division be one multiplication plus one range check per element, and can the per-row max be eliminated?

### Takeaway
The division already is one multiplication relation plus one fixed-width range check per element (the reciprocal trick), and that is the cheap form — a direct `q·sum + r = e·S, r < sum` would be more expensive because the bound is per-row and wider. The max can be eliminated only by accepting a prover-chosen shift (zkLLM) or kept exactly at O(F) cost (#218 style); it is not a per-element cost today.

### Cited Findings
- [CODE] Current division: `S·out + R = exp_q·inv_sum[k]`, `R ∈ [0, S)` via one 14-bit identity range check (4 chunks), `inv_sum` computed by the verifier from the clear row sum — verifier `recip_mult.rs:79-87,143-153`, `mod.rs:310-334,403-405`. The HackMD rationale: reciprocal-multiply gives the remainder a constant bound `S`, moving the variable-bound check to `O(F)` — [HackMD](https://hackmd.io/@R5TO3fi7TlWQKnY6Ejr2eg/Byjucoqjbx).
- [CODE] The max costs per element only the degree-3 Max relation, whose prover exploits the one-hot sparsity of the indicator; no committed column is involved — `max.rs:95-148` (verifier), memory note `softmax-phase1-eq-hoist.md`.
- [LIT] zkGPT proves division as one range relation on advice (`C2·q_y·(2q_z−1) ≤ 2C1·q_x ≤ C2·q_y·(2q_z+1) − 1`), with a variable divisor inside the relation — [zkGPT paper](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf).
- [LIT] zkLLM has no max and no division; the shift is constrained only by the row-sum tolerance — [zkLLM](https://arxiv.org/html/2404.16109).

### Inferences
- [EST] Direct division with remainder `r < sum_k` needs two range checks of about `14 + log2 N` bits (`r ≥ 0` and `sum − 1 − r ≥ 0`): 12 chunks at N = 1024 against 4 today. The reciprocal form should stay; the `O(F)` check `inv·sum + r_div = S², 0 ≤ r_div < sum` is where the variable bound belongs.
- [EST] The per-element remainder `R` disappears entirely only if (a) the output is the exp value itself (zkLLM shift, §2) or (b) the division is fused into the next einsum (zkGPT, §3).
- [EST] Without the attainment check the shift is free; with division still present correctness would be preserved up to quantization (softmax is shift-invariant), but outputs would no longer be a deterministic function of the input, so some replacement constraint (sum window, or `exp_sum ≥ S`) plus a policy decision is needed.

### Gaps
- None of these variants was prototyped; remainder widths are computed by hand.

## 6. The #218 plan ("division sumcheck, max_k = z(r)+x(r), one-hot ⟨e,z⟩=0 for argmax") versus the alternatives

### Takeaway
The #218 plan is sound in outline, fixes proof size, verifier work and the ZK leak, and would also remove the code path behind the suspected `exp_sum` width bug — but it leaves the per-element prover cost (22 chunks, 19 instances) untouched, so it is an O(F) clean-up, not the efficiency win. It should be combined with a per-element redesign rather than done in isolation.

### Cited Findings
- [CODE] The only in-repo statements are the module TODO ("Remove auxiliary vectors and derive them inside the protocol", verifier `mod.rs:8`) and scheduling notes: softmax moved "as code" because "the #218 aux-vector rework had not merged" (`docs/upstream-inventory.md:125-126`); #218 is an explicit non-goal of the verifier-audit spec (`specs/verifier-audit-simplification.md:99,341`). No spec for #218 exists in `specs/`; `git log` shows no commit implementing it.
- [Memory file, user-authored plan, Oct 2026] (1) keep ExpSum, add a `log F`-round division sumcheck `Σ_k eq(r,k)·(inv_sum[k]·exp_sum[k] + r_div[k]) = S²` with range checks `r_div ≥ 0`, `exp_sum − r_div − 1 ≥ 0`; (2) `max_k(r_lead) = z(r) + x(r)` from existing openings; (3) committed one-hot `e` with the standard one-hot checks plus `Σ eq·e·z = 0`, replacing MaxIndicator; result `O(F) → O(log F)` field elements per node, arithmetization and accuracy unchanged — `/Users/kg/.claude/projects/-Users-kg-swe-icme-official-jolt-atlas/memory/softmax-aux-vector-removal.md`.

### Inferences
- [EST] Step (2) needs care: `z(r) + x(r)` at a single random point does not by itself show that `z + x` is independent of `j`. Either open at a second independent `r_j'`, or define things the other way round — let the committed one-hot `e` (Booleanity + Hamming weight 1 per row) define `M(r_k) = Σ eq(r_k,k)·e·x` by the existing Max relation and treat `z := M⊗1 − x` as a virtual polynomial; then `⟨e,z⟩ = 0` holds automatically and no separate check or second opening is required.
- [EST] Prover cost of #218: one committed one-hot over `F` rows with `log N` address bits, two range checks over `F` values, one `log F`-round sumcheck — negligible next to the `F·N` work. Proof size drops by `3F` field elements per node (at N = F = 1024 per head that is thousands of 16-byte elements per node) and the verifier loses its `O(F)` divisions and `O(F log N)` indicator evaluation.
- [EST] Positioning: #218 keeps bit-exact, deterministic semantics, which zkLLM's approach gives up. That is a legitimate differentiator ("exact quantized softmax with polylog communication"), but the paper-worthy efficiency claim needs the per-element reductions in §7 as well.

### Gaps
- The actual issue text and any discussion on it were not retrievable (`gh` returned nothing).

## 7. Ranked redesign proposals

### Takeaway
Ranked by value per unit of risk: (A) fix/guard the aux-vector parsing now; (B) collapse the clamp + two digit lookups + product rescale into one lookup on `z` with bit-identical outputs (22 → about 12 chunks, 19 → about 9 instances); (C) do #218 for the per-row part and add a public mask selector; (D) for long context, fuse the division into the `·V` einsum (zkGPT) or adopt a zkLLM-style prover-chosen shift (down to about 8 chunks, at the cost of exactness). Base-2/polynomial softmax should not be pursued.

### Cited Findings
- Cost model inputs are the code facts in §1 (22 chunks; instance list; clamp = 8 chunks + 32-round prefix-suffix + degree-9 RaVirtualization) and the literature in §2–§4 ([zkLLM](https://arxiv.org/html/2404.16109), [zkGPT](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf), [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)).
- [CODE] `lut_hi` is zero-padded past `hi_size` and the clamp saturates at `z_bound − 1`, so every `z ≥ 2^18` yields `exp_q = 0` — `atlas-trace/src/op/softmax.rs:60-65,112-118`. This is what makes proposal B output-preserving.
- [Memory notes, user-authored] prior measurements in this codebase: LogUp-GKR range checks were a net regression against one-hot Booleanity on a 256-bit field; static clamp widths and bucketed global lookup instances were large wins; prefix-suffix phases must be even-width (a `phase_schedule` was added) — `memory/qwen-sub-minute-brainstorm.md`.

### Inferences (all [EST]; chunk counts at scale 14)

| # | Proposal | Semantics | Chunks/elt | Instances | Touches |
|---|---|---|---|---|---|
| — | Today | — | 22 | 19 | — |
| A | Harden aux parsing (single width, error not panic, add a mutation site) | none | 22 | 19 | verifier |
| B | One structured lookup on `z`: `exp_q = EXP32[z]`, `EXP32[z] = [z < 2^18]·⌊hi[z≫9]·lo[z&511]/S⌋` | **bit-identical output** | 12 (8 for `z` + 4 for `R`) | about 9 | lookup tables, prover, verifier; trace witness columns only |
| C | #218 per-row rework + public mask selector | bit-identical | same as base | +3 per-row | prover, verifier |
| D1 | Fuse division into `softmax·V` einsum (zkGPT) | changes outputs (more accurate) | about 8 per (i,j) + wider remainder per (i,c) | — | importer pass, tracer op, einsum prover |
| D2 | zkLLM shift: output `= EXP[c_k − x]`, row sum in window | changes outputs; prover-chosen shift | about 8–9 | about 5 | tracer, prover, verifier, policy |
| ✗ | Base-2 / I-BERT / 2Quad | changes model or adds remainders | ≥ 22 | — | — |

**A. Guard the aux vectors (do first, tiny).** Parse `exp_sum_q`, `max_k`, `argmax_k` once, reject values outside `[0, 2^31)` (and `argmax ≥ N`) with `ProofVerifyError`, and use the same parsed integers for both the MLE and `inv_sum`. Add a soundness-sweep site that adds `2^32` to a `SoftmaxSumOutput` claim. This is independent of everything else and becomes moot once C lands.

**B. Single lookup on `z` (largest exact win).** Today `z` is looked up three times: a 32-bit clamp (8 chunks), then `z_hi` and `z_lo` (3 + 3), and the two table values are multiplied and rescaled with a range-checked remainder (4 more chunks, ExpProduct, an identity RC, a RaVirtualization and a Booleanity). The clamp's one-hot already addresses the full 32-bit `z`. Define the table over that address directly so that it returns `exp_q`; the tracer's composite `⌊hi·lo/S⌋` on the low 18 bits times the indicator "top 14 bits are zero" reproduces today's outputs exactly, so golden hashes do not move. Removed: `z_hi`/`z_lo` chunks and their two read-rafs and one-hot checks, `r_exp` and its RC and one-hot checks, ExpProduct. Kept: the 8-chunk `z` one-hot (still degree-9 RaVirtualization), `R`, RecipProduct, ExpSum, Max. Required work: a prefix-suffix (or sparse-dense) read-raf whose suffix is a dense 2^18 table — the existing even-width phase schedule would need an uneven 14 | 18 split, or two 16-bit phases with the dense table split 2 + 16; the table stays owned by `atlas_trace::semantics`. Caveats: a 2^18 dense table per instance is a fixed cost that exceeds `T` for small nodes (4096 elements at 16 tokens), so it should be one model-global instance across softmax nodes or gated on `T ≥ 2^18`; verifier evaluates the 2^18-entry table MLE once per instance. Further narrowing: if the importer can bound the logit span statically (e.g. causal masks expressed as a public selector instead of a large negative addend), `z` fits in 20 bits and the address drops to 5 chunks (total 9). Estimated saving: about 45% of softmax commit/one-hot work and the removal of 10 of 19 instances.

**C. #218 plus masking (per-row, exactness-preserving).** As in §6, with `z` defined from the committed argmax one-hot. Add a public selector `m[k,j]` to ExpSum/RecipProduct/Max so that `active_len < N` and causal masks are provable; today non-power-of-two widths are rejected (§1), and padded lanes/heads are pure waste (12 heads padded to 16 is 33% overhead on every softmax and attention einsum). Prover cost is negligible; the gains are proof size, verifier time, a ZK-compatible protocol and a functional gap closed.

**D1. Fuse the division into the value einsum (long context, exact and deterministic).** Follow zkGPT: keep `exp` per (i,j), never materialize normalized probabilities, and let the `softmax·V` einsum divide by the row sum: `P[i,c] = ⌊Σ_j e_ij·V_jc / sum_i⌋` (scales adjusted). Per (i,j) cost becomes only the `z` lookup (8 chunks with B); the remainder moves to `s·d_head` outputs per head with a row-dependent bound (two range checks of roughly 30–40 bits, about 16–20 chunks per output), which amortizes to about 1 chunk per (i,j) at `s = 1024, d_head = 64` and is a loss when `s ≤ d_head`. Accuracy improves (one rounding instead of two). Cost: an importer fusion pass, a new fused op in atlas-trace with a variable divisor, and an extension of the shared fused-rescale einsum machinery; output hashes change.

**D2. zkLLM-style shift (cheapest prover, weakest guarantee).** As mapped in §2: 8–9 chunks per element, no `R`, no division, no max, with a second high-precision table on the same `ra` to keep the accepted shift within about one step. Soundness caveat: outputs are no longer a function the verifier pins exactly; a prover can choose among adjacent shifts (relative change about `2^-14` per step with the fine-table variant, up to `±N/(2S)` with a plain tolerance). Requires an explicit relaxation of the "verifier never trusts the prover for trace semantics" invariant and a new kind of soundness test (accept-set bound rather than mutation rejection). Accuracy: zkLLM reports perplexity change under 0.1 with an L1 error near 10^-2; jolt-atlas's version would be tighter but is unmeasured.

Ordering rationale: A is a correctness fix. B gives the biggest reduction without touching semantics or the importer. C is orthogonal and cheap once softmax is being rewritten anyway (doing B and C in one spec avoids migrating the stages twice). D1/D2 only pay off beyond roughly 64–128 tokens and should be decided by a long-context measurement; D1 keeps exactness and is the better fit for this codebase's invariants.

### Gaps
- All savings are chunk-count and instance-count estimates. No measurement exists for the Akita/Fp128 build, where the relative cost of commitment versus sumcheck may differ from the Dory-era notes.
- Feasibility of an uneven-phase prefix-suffix read-raf with a dense 2^18 suffix in the current `ps_shout` implementation was not checked in code.
- Whether real GPT-2/Qwen/bge logit spans (including mask addends after import) admit a static bound below 2^20 is unknown; it needs a trace histogram.
- The accuracy effect of the current floor-reciprocal semantics at long context (the `N/S` truncation in `inv_sum`) has not been measured, so the accuracy benefit claimed for D1 is qualitative.
