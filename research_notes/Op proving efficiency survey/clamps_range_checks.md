# Saturating clamps and range checks in jolt-atlas: current mechanisms, literature, proposals

Conventions. **[CODE]** = verified by reading the source on branch `dev` (paths relative to
`/Users/kg/swe/icme/official/jolt-atlas`, cited `file:line`). **[LIT]** = literature claim with URL.
**[MEMO]** = historic measurement from the developer's session memory file
`/Users/kg/.claude/projects/-Users-kg-swe-icme-official-jolt-atlas/memory/qwen-sub-minute-brainstorm.md`
(Aug 2026, Dory/BN254 era, M3 Pro; not in the repo, not re-measured under Akita). **[EST]** = my own
estimate or inference. No cargo command was run; nothing below was confirmed by execution.

Global constants [CODE]: `XLEN = 32`, `LOG_K_CHUNK = 4` (every one-hot chunk is K = 16),
`LOG_K = 64` (`common/src/consts/general.rs:1-7`), `MODEL_SCALE = 14` (`general.rs:13`), default
compile scale 14 (`atlas-trace/src/model.rs:241`). Chunk count for a `log_K`-bit address is
`ceil(log_K / 4)` (`atlas-sumcheck/src/config.rs:44-59`). `OneHotConfig::validate` accepts chunk
widths 4 or 8 only (`config.rs:131-140`).

Generic sumcheck degrees [CODE]: prefix-suffix read-raf 2 (`atlas-sumcheck/src/ps_shout/mod.rs:30`),
identity range check 2 (`identity_range_check/mod.rs:22`), Booleanity 3 (`booleanity/mod.rs:20`),
RaVirtual `d + 1` where d = chunk count (`ra_virtual/mod.rs:81-83`), Hamming weight 1 (now folded into
Booleanity as a `LinearTerm`, `booleanity/mod.rs:45-63`). The generic one-hot check bundle is
`[RaVirtual, Booleanity(+Hamming)]` (`shout/mod.rs:337-370`).

---

## Q1. What does each clamp / range-check mechanism cost today (relations, degrees, chunks, committed columns)?

### Takeaway
The hot path (fused rescale ops and Add/Sub/Sum) is already heavily optimized: per element it commits
8 four-bit one-hot chunks for the clamp output plus 4 for the remainder (7 for Cube), in model-wide
packed buckets, with no RaVirtual and no read-raf. Everything else (MeanOfSquares / Div / Rsqrt / Cos /
Sin range checks, softmax, activation clamps, ONNX Clamp, ReLU) still uses per-node committed chunks
with a 32- or 64-address-round read-raf and a degree-9 or degree-17 RaVirtual.

### Cited Findings

**A. Fused rebase clamp ("clamp split"), also used by Add/Sub/Sum** [CODE]
- Statement per fused node: `acc(r) = q(r)·2^S + R(r)`, `out = SatClamp(q)`, `R ∈ [0, 2^S)`. `q`
  (`VirtualPoly::ClampAcc`) and `R` (`VirtualPoly::RescaleRemainder`) are *virtual* advice claims opened
  at the node's output point, not committed — `jolt-atlas-verifier/src/onnx_proof/fused_rebase.rs:9-19, 103-132`.
- Rebase widths: Einsum / Mul / Square `S = scale = 14`; Cube `2·scale = 28`; MeanOfSquares divides by
  `count·2^S` and takes a separate path — `fused_rebase.rs:62-71`.
- No per-node committed polys: clamp and remainder one-hots live in packed buckets —
  `fused_rebase.rs:137-142`, `clamp_lookups/mod.rs:266-272`.
- Buckets: nodes grouped by width, first-fit-decreasing into a cycle space whose capacity is the largest
  clamped node; one set of committed chunks and checks per bucket — `global_clamp.rs:1-19, 113-160`.
- Clamp bucket columns: 8 output chunks (`OUT = out + 2^31`, 32 bits) + `w/4` slack chunks, with the
  static width `w ∈ {40, 48, 56, 64}` → 10/12/14/16 slack chunks — `clamp_split.rs:11-18, 87-117`,
  `atlas-trace/src/model.rs:248-286`. Slack chunks are `None` on interior cycles (`clamp_split.rs:329-336`).
- Static widths come from `OpSpec::clamp_bound`: Add/Sub `2·2^31` (→ W40), Sum `n·2^31`, Mul / Square /
  MeanOfSquares `2^62 >> scale` (→ W56 at scale 14), Cube `u128::MAX` (→ W64), Einsum
  `2^31·max_row Σ|w|` when one operand is constant else `2^62·contracted size`, then `>> scale` —
  `atlas-trace/src/op/linear.rs:55-57, 186-195`, `op/fused.rs:68-70, 104-106, 145-148, 211-213`,
  `op/einsum.rs:278-303`. Every bound assumes operands span the full i32 range (no value-range propagation).
- Split sumcheck (one per bucket, degree 4, `log_t(bucket)` rounds):
  `Σ_t W(t)·[OUT + β₁/M·IND·S·(2·OUT − M) + β₂·IND·OUT·(OUT − M) + β₃·IND·(1 − IND)]`, `M = 2^32 − 1` —
  `clamp_split.rs:20-32, 492-512`. `IND` is virtual (Hamming weight of the slack chunks).
- Chunk check: one Booleanity sumcheck (degree 3, `4 + log_t` rounds) over all the bucket's chunks, with
  a linear term proving Hamming weight 1 for output chunks, Hamming weight `IND` for slack chunks, and
  the chunk-value recompositions of `OUT` and `S` — `clamp_split.rs:34-55, 541-591`.
- Exact mode: a prover-declared 0/1 flag per bucket (`GlobalClampExact`); if 1 the verifier checks
  `ClampAcc(r_n) = out_n(r_n)` per node, and the bucket runs only a degree-2 value sumcheck plus
  Booleanity on the 8 output chunks; slack chunks stay empty and unopened —
  `deferred_lookups.rs:228-292`, `clamp_split.rs:749-819, 340-360`.
- Public-output shortcut: an output node whose public output has no `±2^31` element proves `acc = out`
  directly and is a gap in its bucket — `clamp_split.rs:160-178, 842-853`, `clamp_lookups/mod.rs:242-259`.
- Scalar nodes (`log_T = 0`): `q` and `R` opened in the clear and checked by the verifier —
  `fused_rebase.rs:161-175, 197-214`, `clamp_lookups/mod.rs:44-54, 277-294`.

**B. Remainder range check for Einsum / Mul / Square / Cube** [CODE]
- Membership: `global_clamp.rs:85-111`. Bucket width = rebase bits; chunks = `ceil(width/4)` → 4 chunks at
  S = 14, 7 for Cube (`global_clamp.rs:163-166`).
- Value sumcheck (degree 2): `Σ_t W(t)·V(t) = Σ_n γ_n·R_n(r_n)` — `clamp_split.rs:594-688`.
- Booleanity with linear term (degree 3): Hamming weight 1 per chunk and
  `V(r') = Σ_d 2^{4(d−1−i)}·Σ_k k·ra_d(k, r')` — `clamp_split.rs:690-747`.
- No identity read-raf and no RaVirtual on this path (the legacy `RescaleRemainderRCProvider` /
  `RescaleRemainderRaEncoding` types still exist in `fused_rebase.rs:216-302` but the deferred path uses
  `ValueParams`, `deferred_lookups.rs:295-336`).

**C. Deferred-batch machinery** [CODE]
- Clamp and remainder jobs are registered during the node loop and proven after it:
  `DeferredClampSplit` → `DeferredClampChunkChecks` → `DeferredRemainderRC` → `DeferredRemainderRaChecks`,
  each one batched sumcheck across all buckets — `deferred_lookups.rs:199-336`. The verifier checks that
  registrations equal the model-derived bucket membership (`deferred_lookups.rs:211-223, 295-306`).
- Four further batches, each followed by one batched one-hot-check sumcheck: `ActivationExec`,
  `ActivationClamp`, `RangeCheck` (MeanOfSquares remainder, Rsqrt div/sqrt), `SoftmaxStage3` —
  `deferred_lookups.rs:42-107, 338-363`. These batch the *sumchecks* only; their one-hot chunks remain
  per-node committed polynomials (e.g. `mean_of_squares.rs:84-94`, `rsqrt.rs:61-69`,
  `activation_clamped/mod.rs:238-247`).

**D. Operand range checks (`range_checking/`)** [CODE]
- Mechanism: prefix-suffix binary read-raf into the UnsignedLT table over the interleaved
  (remainder, bound) pair, `rv_claim = 1` — `range_checking/mod.rs:41-45, 178-192`; indices interleave two
  i32 operands (`range_check_operands.rs:472-478`). Address width `LOG_K = 64` → 16 chunks per checked
  value (`range_checking/mod.rs:247-253`); hence 64 + log_T read-raf rounds at degree 2 and RaVirtual of
  degree 17.
- Users: Div (`R < divisor`, inline per node, `ops/div.rs:43-51, 295-315`); Rsqrt (two checks → 32 chunks
  per element, deferred, `ops/rsqrt.rs:61-69, 250-273`); MeanOfSquares (`R < D = N·2^S`, constant bound,
  deferred, `ops/mean_of_squares.rs:84-94, 140-149`, `range_check_operands.rs:263-339`); Cos / Sin
  (`R < τ`, constant, `ops/cos.rs:32-43, 82`, `range_check_operands.rs:351-409`).
- ScalarConstDiv never range-checks its committed remainder (documented soundness gap) —
  `docs/upstream-inventory.md:129-132`.

**E. Identity range check** [CODE]
- Degree 2 (`atlas-sumcheck/src/identity_range_check/mod.rs:22`); today used only by softmax, for two
  remainders with `log_K = scale = 14` → 4 chunks each — `ops/softmax_last_axis/rc.rs:37-56`,
  `softmax_last_axis/mod.rs:404-405, 456-457`. Softmax also commits a 32-bit significance clamp (8 chunks)
  and exp hi/lo digit tables — `softmax_last_axis/mod.rs:77-102`.

**F. ONNX `Clamp` op** [CODE]
- Prefix-suffix read-raf into `ClampTable<32>` (symmetric clamp to `[-2^9, 2^9 − 1]`, `CLAMP_BOUND = 9`,
  `atlas-lookup-tables/src/clamp.rs:196-210`), `LOG_K = XLEN = 32` → 8 committed `SymmetricClampRaD`
  chunks; inline per node; then `[RaVirtual (deg 9), Booleanity]` — `ops/clamp.rs:29-131`.

**G. `activation_clamped` (Erf / Sigmoid / Tanh)** [CODE]
- Stage 1: small-table lookup `out = Table[clamped]`, degree 2, table of
  `ACTIVATION_TABLE_VARS = MODEL_SCALE + 4 = 18` variables → 5 chunks (`common/src/consts/activation.rs:3-8`,
  `activation_clamped/mod.rs:49-50, 126-136, 189-195`).
- Stage 2: `clamped = ActivationClampTable<32>[input]`, 32-bit prefix-suffix read-raf → 8 chunks
  (`activation_clamped/mod.rs:72-119, 213-224`).
- 13 per-node committed chunk polys per element; both stages deferred; one-hot checks (RaVirtual deg 6
  and deg 9, Booleanity) after their batches — `activation_clamped/mod.rs:198-247`.

**H. ReLU** [CODE]
- Prefix-suffix read-raf into `ReluTable<32>` (degree 2, 32 + log_T rounds), 8 committed `NodeOutputRaD`
  chunks, then `[RaVirtual (deg 9), Booleanity]`, inline per node — `ops/relu.rs:17-68`. No sign-bit
  decomposition.

**Cost table per checked element** [CODE for counts; the tabulation is mine]

| Mechanism | Committed one-hot chunks (K=16) | Sumchecks |
|---|---|---|
| Fused clamp, bucket in exact mode | 8 (slack chunks empty) | value deg 2; Booleanity deg 3 |
| Fused clamp, saturating bucket | 8 + w/4 (18–24; slack sparse) | split deg 4; Booleanity deg 3 |
| Remainder (Einsum/Mul/Square) | 4 | value deg 2; Booleanity deg 3 |
| Remainder (Cube) | 7 | same |
| MeanOfSquares / Div / Cos / Sin range check | 16, per node | read-raf deg 2 (64 addr rounds); RaVirtual deg 17; Booleanity deg 3 |
| Rsqrt | 32, per node | two of the above |
| Softmax | 4 + 4 + 8 + exp digits, per node | identity RC ×2, clamp read-raf, digit lookups |
| ONNX Clamp / ReLU | 8, per node | read-raf deg 2 (32 addr rounds); RaVirtual deg 9; Booleanity |
| Clamped activation | 8 + 5, per node | read-raf; small-table deg 2; RaVirtual deg 9 and 6; Booleanity |

So one Einsum/Mul/Square element costs 12 committed one-hot nonzeros when its bucket is exact (the
common case), and one MeanOfSquares element costs 8 + 16 = 24.

**Zoo incidence**
- Raw ONNX op counts (python `onnx` over `models/*/network.onnx`, before import rewrites; my count):
  gpt2 — Mul 74, Add 64, Gemm 48, MatMul 25, LayerNormalization 25, Sqrt 36, Div 12, Softmax 12, Tanh 12,
  Pow 12; bge-small — Add 172, MatMul 96, Mul 74, Sqrt 61, ReduceMean 50, Div 49, Sub 26, Pow 25,
  Softmax 12, Erf 12; qwen2-tiny — Mul 39, Add 22, MatMul 20, Div 9, ReduceMean 5, Sqrt 5, Softmax 2,
  Sigmoid 2, Cos 1, Sin 1; gpt2-tiny — Add 14, Mul 14, Gemm 8, MatMul 5, LayerNormalization 5, Softmax 2,
  Tanh 2; bge-tiny — Add 32, MatMul 16, Mul 14, ReduceMean 10, Div 9, Softmax 2, Erf 2.
  **No model in `models/` contains a `Relu` or `Clip` node.**
- Full Qwen: 2,172 nodes, about 750 clamped / fused nodes — `jolt-atlas-verifier/src/onnx_proof/deferred_lookups.rs:3-8` [CODE comment] and [MEMO].
- [MEMO] Qwen clamp widths: Add and weight-einsum 40; Mul, MeanOfSquares, attention-einsum 56. Qwen had
  zero saturated elements in every bucket, so every bucket ran in exact mode.

**Timing data**
- `docs/upstream-inventory.md:156-158` [CODE/doc]: GPT-2 with Akita, release `prove`, M3: prove 39.9 s
  (vs 26.7 s Dory), verify 2.17 s, proof 2.42 MiB; "the commitment phase dominates (24.4 s), and two of
  the six 2^24-variable one-hot groups take about 10 s each of it."
- [MEMO] Qwen (9-token prompt, Dory): 222.8 s baseline → 46.3 s. Relevant steps: deferred batching
  → 160.5 s (deferred lookups then 46 s: clamp read-raf 14.8, clamp one-hot checks 19.3, remainder
  2.7 + 4.0); static clamp widths → 127.3 s; bucketed global clamp → 114.3 s (committed polys 13k → 5.4k);
  remainder buckets → 3,412 polys; clamp split 75.4 → 65.2 s; remainder value sumcheck (identity RC and
  RaVirtual dropped) → 66.5 s after an unrelated fix; per-bucket exact mode → 49.7 s. At that point the
  un-bucketed families were "MoS/Sqrt RC 2,352 polys via binary read-raf, softmax/activation 840".
- [MEMO] LogUp-GKR range checks were built and measured on BN254 for the remainder: net regression
  46.3 → 55.4 s ("GKR fractional sum costs ~30-60 BN254 mults/element vs one-hot's ~8 batched affine adds").

### Inferences
- [EST] Under Akita the one-hot *commitment* is the dominant cost (24.4 of 39.9 s on GPT-2), so the right
  metric for clamp/range proposals is committed one-hot nonzeros per element, not sumcheck degree.
- [EST] On transformer models the per-element chunk budget is dominated by fused ops (12 per element) and
  by the 16/32-chunk per-node range checks on LayerNorm-shaped nodes (MeanOfSquares, Rsqrt, Div), which
  are small tensors (one value per row) so their absolute weight is lower than their chunk count suggests.

### Gaps
- Lowered-IR node counts per op for the zoo models (needs `dump_lowered_ops`, a cargo run, which was out
  of bounds). The raw ONNX counts above overstate/understate after constant folding and fusion.
- No per-phase timing under Akita for the deferred clamp / remainder batches exists in `docs/` or
  `specs/`; there are no bench targets in the repo. The [MEMO] numbers are Dory-era.

---

## Q2. How many committed columns and lookups does one remainder range check / one saturating clamp cost? Is the quotient's i64 range decomposed?

### Takeaway
Remainder: 4 one-hot chunks of 4 bits (7 for Cube), one degree-2 value sumcheck and a share of one
degree-3 Booleanity per bucket. Clamp: 8 chunks for the 32-bit output, plus `w/4` (10–16) slack chunks
that are only populated on saturated elements; the quotient's full i64 range is no longer decomposed
densely.

### Cited Findings
- All counts and citations are in Q1 A–B. The previous protocol looked the accumulator up in a `w`-bit
  saturation table with `w/4` committed chunks and a degree-`w/4 + 1` RaVirtual —
  `clamp_split.rs:3-9` [CODE doc comment].
- **Suspected soundness gap (code reading only, not executed).** The remainder bucket for S = 14 commits
  `ceil(14/4) = 4` chunks with weights `2^12, 2^8, 2^4, 2^0` and Hamming weight 1 over K = 16 per chunk,
  with no restriction on the top chunk — `clamp_split.rs:693-732, 822-833`, `global_clamp.rs:163-166`,
  `config.rs:44-59`. That proves `R ∈ [0, 2^16)`, not `[0, 2^14)`. A prover could then use
  `q' = q − j`, `R' = R + j·2^14` for `j ≤ 3`, which satisfies `acc = q'·2^S + R'`. The module comment
  claims "`V ∈ [0, 2^width)`" (`clamp_split.rs:605-609`). The witness-tamper sweep only adds 1 to a single
  quotient or remainder element (`jolt-atlas-core/src/onnx_proof/testing/soundness.rs:342-360`), so a
  coordinated `(q − 1, R + 2^14)` tamper is not exercised. Cube (28 bits, 7 chunks) is exact; the clamp
  slack widths are multiples of 8 and exact.

### Inferences
- [EST] If confirmed, the gap lets a malicious prover bias every Einsum/Mul/Square output by up to −3 LSB.
  Cheap fixes: (i) make the linear claim `β·4·R(r')` so the chunks decompose `4R ∈ [0, 2^16)` (sound
  because `R = acc − q·2^S` is an integer once `q` is pinned by the output chunks); (ii) generalize
  `LinearTerm` weights from `γ' + w·k` to an arbitrary per-chunk table `f_d(k)` and add
  `Σ_k [k ≥ 4]·ra_top(k, r') = 0`; (iii) move to a scale that is a multiple of 4.
- [EST] The softmax identity range checks at `log_K = 14` go through a different path
  (`compute_r_address_chunks` front-pads the address point with zeros, `config.rs:81-95`), which likely
  pins the unused high bits; I did not trace it end to end.

### Gaps
- The suspected gap needs a `known_gap_*` test (coordinated quotient/remainder tamper through
  `prove_from_trace`) to confirm or refute.

---

## Q3. Can the clamp be eliminated statically (interval analysis) or by a no-overflow semantic?

### Takeaway
The dynamic "exact bucket" mode already delivers most of what a no-overflow semantic would: when nothing
saturates, a clamp costs one 32-bit range check (8 chunks). What remains to win is (a) shrinking that
range check with importer-side value bounds and (b) code simplification; the range check on the output
cannot vanish entirely because it is what forces the quotient to be an integer.

### Cited Findings
- Exact mode and its verifier check: Q1 A (`deferred_lookups.rs:228-292`, `clamp_split.rs:799-819`) [CODE].
- Slack chunks of an exact bucket are committed empty and "a class whose members are all identically
  zero commits nothing" — `clamp_split.rs:352-359`, `atlas-pcs/src/akita.rs:31-32` [CODE].
- Node outputs, `q` and `R` are virtual; only one-hot chunks tie them to integers
  (`fused_rebase.rs:15-19`, `clamp_split.rs:44-51`) [CODE].
- Current static bounds assume full-range i32 operands (Q1 A citations) [CODE].
- [MEMO] Qwen had zero saturated elements in every clamp bucket.

### Inferences
- [EST] **Why the output range check cannot be dropped outright.** With only `R ∈ [0, 2^S)` proven,
  `q = (acc − R)·2^{−S}` in the field; a prover choosing `R ≠ acc mod 2^S` gets a non-integer (huge) `q`
  that still satisfies the linear identity. A range check on `q` of any width `b` with `2^{b+S} ≪ p`
  forces divisibility. So one range check on the output is the floor for this architecture unless the
  output is bound by a commitment that itself enforces small coefficients (see Q6).
- [EST] **Static width for the output check.** The check needs `b ≥` the true bit-width of `q`, not 32.
  If the importer propagated value intervals (LayerNorm outputs bounded by `sqrt(n)·2^scale`, softmax in
  `[0, 2^scale]`, activations bounded by their tables, constant weights with known row sums), many nodes
  would get `b ≤ 20–24` → 5–6 chunks instead of 8 (25–37% fewer clamp nonzeros on those nodes). Nodes on
  the residual stream need worst-case additive bounds and may stay at 32.
- [EST] **No-overflow semantic** (trace fails if `q ∉ i32`; prover proves only `OUT ∈ [0, 2^32)` and
  `acc = out`): prover time equal to today's exact mode; deletes the split sumcheck, slack chunks, `IND`,
  the exact flag and the public-output special case. Trade-off: completeness is lost for inputs that
  saturate (an honest execution that overflows cannot be proven), and model accuracy is unaffected on
  inputs that do not saturate. Since exact mode is a prover-declared per-bucket flag, the current design
  already gets the speed without the completeness loss; the semantic change buys simplicity, not time.
- [EST] Combining static interval bounds with a no-overflow rule makes the bound a *verifier-known model
  property*: the importer proves `|q| < 2^b` for all inputs within declared input ranges, so no dynamic
  flag is needed and the chunk count is fixed at compile time. Inputs must then be range-constrained
  (public inputs are known to the verifier, so this is free).

### Gaps
- How tight interval analysis is on GPT-2 / Qwen / BGE is unmeasured; needs an importer experiment
  reporting per-node bit-widths. Worst-case interval growth through attention (product of two
  activations) may give useless bounds without LayerNorm-aware reasoning.
- Saturation frequency on GPT-2 and BGE under Akita is not recorded anywhere I found.

---

## Q4. Can remainder check + clamp be merged into one joint decomposition? 2×16-bit vs 4×8 vs 8×4 chunks under a one-hot-friendly lattice PCS?

### Takeaway
A joint decomposition of `acc + 2^{31+S} = OUT·2^S + R` is possible and removes one value sumcheck, one
advice claim and one bucket family, but needs the scale to align with the chunk width (S = 14 does not
align with 4-bit chunks). Wider chunks cut committed nonzeros proportionally (8-bit: 12 → 6 per fused
element); the vendored Akita adapter has a K = 256 family, but atlas hardcodes K = 16.

### Cited Findings
- Upstream `jolt-akita` defines one-hot families K16 and K256 — `third_party/jolt-akita/src/one_hot_family.rs:8-17` [CODE].
- atlas-pcs uses only `AKITA_ONE_HOT_K16`, `LOG_ONE_HOT_K = 4`, groups up to 64 one-hot columns per class,
  and packs one-hot polys narrower than 16 variables densely — `atlas-pcs/src/akita.rs:54, 82-90` [CODE].
- `OneHotConfig` accepts `log_k_chunk ∈ {4, 8}`; `lookup_index_chunk` returns `u8` — `config.rs:76-78, 131-140` [CODE].
- Akita: lattice PCS commitment time is proportional to the number of nonzero entries, not polynomial
  length; reports a 2.2×–4.2× Jolt prover speedup over Dory — [Akita, ePrint 2026/1983](https://eprint.iacr.org/2026/1983) [LIT, abstract].
- Twist and Shout: "Shout at d=2 has at least 8x fewer nonzero committed values and 3.4x fewer field
  multiplications than Lasso for structured tables" (2^64 tables) — [ePrint 2025/105](https://eprint.iacr.org/2025/105) [LIT, abstract].
- [MEMO] Under Dory, "chunk-width changes don't help (2-bit halves tier-2 but doubles tier-1)". That
  analysis is Dory-specific.

### Inferences
- [EST] Joint value at S = 14 is 46 bits → 12 four-bit chunks, the same count as 8 + 4 today; the chunk
  containing bit 14 straddles `R` and `q`, so `out` is not a linear function of the chunks. With S = 16
  (or 12) the boundary aligns: `R` = low 4 (or 3) chunks, `OUT` = next 8, one Booleanity instance, and
  the Q2 width gap disappears. Changing the global scale is a tracer/importer semantic change with
  accuracy consequences (S = 16 leaves 15 integer bits in i32; S = 12 loses 2 fractional bits).
- [EST] 8-bit chunks (K = 256): OUT 4 nonzeros, remainder 2 (S = 16) → 6 per fused element vs 12, i.e.
  about 2× fewer one-hot nonzeros for the clamp/remainder families. Costs: Booleanity address rounds
  4 → 8 and per-chunk `G` tables 16 → 256 entries (negligible for big buckets); the one-hot polynomial
  arity grows by 4 variables, which matters for Akita's exact-shape setups (`MAX_ONE_HOT_NUM_VARS = 35`,
  `akita.rs:86-87`) and opening cost. 16-bit chunks (K = 65,536): 2 + 1 = 3 nonzeros, but no Akita family
  exists for it, arity grows by 12, and small nodes/buckets (T < 2^16) pay table-size costs.
- [EST] Because the clamp path has no RaVirtual, widening chunks does not raise any sumcheck degree
  there; for the legacy per-node lookups (ReLU, Clamp, range checks) it would *lower* RaVirtual degree
  (17 → 9, 9 → 5).

### Gaps
- Akita's K = 256 commit/open cost relative to K = 16 per nonzero is not documented in the repo and was
  not measured. Whether the packaged schedule catalogs cover K256 at the needed arities is unknown
  (`docs/upstream-inventory.md:150-153` only mentions the K16 `w1r0` one-hot profile).

---

## Q5. Global batching: what is model-wide now, what remains per node?

### Takeaway
Clamp and power-of-two remainder checks are already model-wide (bucketed by width, one batched sumcheck
per stage). The remaining per-node committed families are the UnsignedLT range checks (MeanOfSquares,
Rsqrt, Div, Cos/Sin), softmax, clamped activations, ONNX Clamp and ReLU; Div, Clamp and ReLU are not even
sumcheck-batched.

### Cited Findings
- Bucketing and batching: Q1 A–C citations [CODE].
- Buckets are capped at the largest clamped node's size, so there are several buckets per width rather
  than one global vector — `global_clamp.rs:16-19, 113-119` [CODE].
- Per-node remaining: `mean_of_squares.rs:84-94`, `rsqrt.rs:61-69`, `div.rs:43-51, 295`,
  `cos.rs:32-43`, `softmax_last_axis/mod.rs:77-102`, `activation_clamped/mod.rs:238-247`,
  `clamp.rs:77-85`, `relu.rs:17-25` [CODE].
- Opening reduction: Akita has no additive homomorphism; committed polys are grouped into classes
  (dense by `(encoding, num_vars)`, one-hot up to 64 columns of one arity) and each class is opened
  natively at its point — `atlas-pcs/src/akita.rs:4-28` [CODE].
- LogUp-GKR: for M looked-up columns "the prover has to commit only to a single extra column, i.e. the
  multiplicities of the table entries" — [ePrint 2023/1284](https://eprint.iacr.org/2023/1284) [LIT, abstract].
- zkGPT proves division and square root with advice plus a single range relation, using Lasso for range
  and exponentiation checks and GKR for arithmetic — [zkGPT, USENIX Security 2025](https://home.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf) [LIT, via search summary].
- zkLLM's tlookup is a sumcheck-based tensor lookup argument avoiding bit decomposition, claimed to have
  no asymptotic overhead — [zkLLM, arXiv 2404.16109](https://arxiv.org/html/2404.16109v1) [LIT, via search summary].

### Inferences
- [EST] The constant-bound range checks (MeanOfSquares `R < N·2^S`, Cos/Sin `R < τ`) can move into the
  bucketed value-check machinery as two plain range checks, `R ∈ [0, 2^k)` and `B − 1 − R ∈ [0, 2^k)`
  with `k = ceil(log2 B)`: about `2·ceil(k/4)` chunks (10 for a 20-bit `D`) instead of 16, no 64-round
  read-raf, no degree-17 RaVirtual, and shared buckets. Variable-bound checks (Div, Rsqrt) become
  `R` and `b − 1 − R` both in `[0, 2^32)`: the same 16 chunks, but bucketed and without read-raf /
  RaVirtual.
- [EST] LogUp-GKR is structurally attractive here because `R` and `q` are *virtual*: a GKR fractional
  sum needs only evaluation claims on the looked-up vector, so a remainder check would commit nothing
  per element (only a 2^14–2^16 multiplicity vector per bucket). The [MEMO] regression was measured when
  one-hot commitment was cheap (Dory batched affine adds) and field mults were 256-bit; under Akita, with
  commitment at 60% of prover time and a 128-bit field, the balance may flip. A 32-bit output check would
  still need committed 16-bit limbs (dense), so the gain is clearest for remainders.

### Gaps
- No measurement of LogUp-GKR vs one-hot Booleanity on Fp128 + Akita exists. The [MEMO] says the BN254
  primitives (`fractional_sum.rs`) lived in the since-removed `joltworks` crate; I did not check whether
  they survive on any branch.
- I could not obtain zkGPT's or zkLLM's per-element range-check costs from primary text (search
  summaries only), nor anything on DeepProve or Binius-style range checks; treat those as not surveyed.

---

## Q6. Can the lattice PCS (Akita) give range proofs for free or cheaply?

### Takeaway
Possibly, and it is the highest-upside open question: the Akita adapter already commits dense polys under
a "dense-bounded" flavor that admits only coefficients in `[0, 2^64)` via 8-bit digits, and the Akita
abstract advertises an "optimized digit range check". If binding/extraction guarantees the digit bound
exactly, committing a remainder (or the node output) as a narrow dense-bounded polynomial would be its
own range proof. I could not confirm that guarantee.

### Cited Findings
- Dense classes are "committed under the dense-bounded flavor. The flavor admits coefficients in
  `[0, 2^64)`"; signed classes are shifted by `2^31` / `2^63` — `atlas-pcs/src/akita.rs:19-25, 101-135` [CODE].
- The packaged dense profile is `AkitaChunkProfile::new(8, 2)` (catalog `w8r2`), one-hot is `w1r0` —
  `akita.rs:93`, `docs/upstream-inventory.md:150-152`; config name `jolt-fp128-dense-bounded` —
  `third_party/jolt-akita/src/configs.rs:150-154` [CODE].
- Akita abstract: mentions an "optimized digit range check" and "exact Euclidean norm checks" giving
  tighter Module-SIS parameters; commitments compress to 128 bytes; builds on Hachi —
  [ePrint 2026/1983](https://eprint.iacr.org/2026/1983) [LIT, abstract].
- Hachi: lattice multilinear PCS over extension fields combining Greyhound with ring switching —
  [ePrint 2026/156](https://eprint.iacr.org/2026/156) [LIT, search summary].
- LatticeFold+: "a new purely algebraic range proof which is much more efficient than the one in
  LatticeFold", which required bit-decomposition range proofs on all witnesses; prover 5–10× faster —
  [ePrint 2025/247](https://eprint.iacr.org/2025/247) [LIT, abstract].
- Grand Danois adapts Hachi's relation to Johnson–Lindenstrauss projections for norm bounds —
  [ePrint 2026/1196](https://eprint.iacr.org/2026/1196) [LIT, search summary].

### Inferences
- [EST] Three caveats before relying on this. (1) Lattice knowledge extraction usually yields a witness
  with *relaxed* norm (slack from challenge differences); an ℓ2 "exact Euclidean norm check" bounds the
  whole vector, not each coefficient, so it does not by itself give `R_i < 2^14`. A per-digit range check
  would. (2) The guarantee must hold for the *extracted* polynomial the opening binds to, as an Fp128
  multilinear with integer coefficients below the bound, and must be part of Akita's stated security
  theorem. (3) atlas would need a narrower dense profile (e.g. one or two 8-bit digits) and catalog;
  today everything dense is 64-bit bounded.
- [EST] If it holds, the fused path could commit `R` as a 14/16-bit dense column (bound enforced by the
  PCS) and `out + 2^31` as a 32-bit dense column, replacing 12 one-hot nonzeros per element by two small
  dense coefficients and removing both Booleanity sumchecks. A dense 2-digit coefficient vs four one-hot
  nonzeros is not obviously cheaper in Akita; needs measurement.
- [EST] LatticeFold+'s algebraic range proof targets folding-scheme witness norms (monomial-commitment
  based); it is evidence that purely algebraic range proofs over lattice commitments exist, not a
  drop-in for a PCS-only setting.

### Gaps
- I did not read the Akita PDF body or the upstream `akita-*` crates (git dependencies, not vendored), so
  whether the digit range check yields exact per-coefficient bounds for the extracted witness is
  unverified. This should be asked of the Akita authors or checked in the paper's security theorem.
- No source found for Greyhound / LaBRADOR giving exact (slack-free) coefficient range guarantees usable
  as an application-level range proof.

---

## Q7. ReLU: how is it proven, and what is the cheapest alternative?

### Takeaway
ReLU is a 32-bit prefix-suffix Shout lookup with 8 committed chunks and a degree-9 RaVirtual, per node
and inline. A split-style protocol (8 chunks, no read-raf, no RaVirtual) is available by reusing the
clamp-split pattern, but no zoo model contains ReLU, so the priority is low.

### Cited Findings
- Current protocol: `jolt-atlas-verifier/src/onnx_proof/ops/relu.rs:17-68` [CODE]; no `Relu` node in any
  `models/*/network.onnx` (my ONNX count, Q1).
- The Jolt Atlas paper positions lookups as the tool for non-linear and piecewise primitives (comparisons,
  range checks) — [arXiv 2602.17452](https://arxiv.org/abs/2602.17452) [LIT, search summary].

### Inferences
- [EST] Split-style ReLU: commit the 8 chunks of `X = x + 2^31`; with a generalized Booleanity linear term
  (arbitrary per-chunk weight table) obtain virtual `X(r')` and the sign `s(r') = Σ_{k<8} ra_top(k, r')`
  (top chunk below 8 ⇔ negative `x`); prove `out = (1 − s)·(X − 2^31)` with one degree-3 eq-weighted
  sumcheck, bucketed like clamps. Same 8 nonzeros, but no 32-round read-raf and no degree-9 RaVirtual.
- [EST] Alternative without sign extraction: `out ∈ [0, 2^31)`, `out − x ∈ [0, 2^31)`,
  `out·(out − x) = 0` — 16 chunks, strictly worse on commitment.
- [EST] The same pattern replaces the activation input clamp and the ONNX Clamp (both 32-bit read-rafs
  with degree-9 RaVirtual): range-check the input's 8 chunks once and express the clamp with the
  interior/saturated split against a narrower bound.

### Gaps
- No measurement; ReLU-heavy models (CNNs/MLPs) are not in the zoo, so there is no workload to rank this
  against.

---

## Q8. Ranked proposals

### Takeaway
Ranked by expected value under Akita, where one-hot commitment dominates: (1) confirm and fix the 14-bit
remainder gap; (2) 8-bit chunks for clamp/remainder buckets; (3) bucket the remaining range checks as
plain value checks; (4) static output widths from importer interval analysis; (5) resolve whether Akita's
bounded commitment is a range proof; (6) LogUp-GKR for virtual remainders; (7) scale alignment + joint
decomposition; (8) split-style ReLU / Clamp / activation clamp.

### Cited Findings
- All supporting facts are cited in Q1–Q7; commitment dominance is `docs/upstream-inventory.md:156-158`.

### Inferences (all [EST]; savings are per affected element unless stated)

| # | Proposal | Estimated saving | Required change | Soundness / completeness caveat |
|---|---|---|---|---|
| 1 | Fix remainder width (prove `R < 2^14`, not `2^16`) | none (correctness) | Scale linear claim by `2^(4d − width)`, or per-chunk weight tables, or scale multiple of 4; add coordinated-tamper `known_gap` test first | Gap is unconfirmed by execution; the `4R` fix relies on `q` being integer-pinned by the output chunks |
| 2 | K = 256 chunks for clamp + remainder buckets | ~2× fewer one-hot nonzeros (12 → 6–7 per fused element; exact clamp 8 → 4) | atlas-pcs K256 one-hot class + catalog; `LOG_K_CHUNK` per family; Booleanity with 8 address rounds | None new; S = 14 remainder becomes 2 chunks = 16 bits, so needs fix #1 |
| 3 | Bucket MeanOfSquares / Cos / Sin / Div / Rsqrt range checks as value checks (`R` and `B−1−R`) | MoS: 16 → ~10 chunks; all: drop 64-round read-raf and deg-17 RaVirtual; ~2.3k fewer per-node polys on Qwen [MEMO count] | New bucket kind; bound claim linear in existing openings | `B − 1 − R` must be a claim the verifier derives (constant or opened operand); Rsqrt bound `2v + 1` already linear |
| 4 | Importer interval analysis → static output-check width `b < 32` | 8 → 5–6 chunks on bounded nodes; none on residual-stream nodes | Value-range propagation in atlas-onnx-import / `clamp_bound`; declared input ranges; per-node width in `NodeMeta` | Bounds must be sound for *all* admissible inputs; pair with a no-overflow rule (trace error) or keep saturating fallback |
| 5 | Akita bounded-coefficient commitment as the range proof | Up to all 12 one-hot nonzeros and both Booleanity sumchecks per fused element | Narrow dense profile/catalog; commit `R`, `OUT` dense | Only valid if Akita's theorem gives exact per-coefficient bounds for the extracted witness (unverified); ℓ2-only or slack-relaxed bounds are insufficient |
| 6 | LogUp-GKR for remainders (virtual vector, commit only multiplicities) | Remainder commits 4 → ~0 per element; adds GKR fractional-sum work | Port fractional-sum GKR to relations; multiplicity vector per bucket | Regressed on BN254/Dory [MEMO]; must be re-measured on Fp128/Akita before adoption |
| 7 | Scale 16 (or 12) + joint decomposition of `OUT·2^S + R` | One fewer value sumcheck, advice claim and bucket family; subsumes #1; with #2 gives 6 nonzeros | Global scale change (tracer, importer, golden hashes, tables keyed on `MODEL_SCALE`) | Accuracy / headroom change: S = 16 leaves 15 integer bits, S = 12 loses 2 fractional bits |
| 8 | No-overflow semantic replacing saturating clamp | ~0 prover time (exact mode already achieves it); large code deletion | atlas-trace overflow → error; drop split sumcheck, slack, `IND`, exact flag | Completeness loss on saturating inputs; none on soundness |
| 9 | Split-style ReLU / ONNX Clamp / activation clamp | Same chunks; drops 32-round read-raf and deg-9 RaVirtual; enables bucketing | Generalized linear-term weights; new relation | Low value for the transformer zoo (no ReLU / Clip nodes) |

### Gaps
- None of the savings are measured. The first experiments I would run: (a) the coordinated-tamper test
  for #1; (b) a per-phase Akita profile of GPT-2 splitting one-hot commit time by family (clamp OUT,
  remainder, per-node range checks) to size #2–#4; (c) a K16-vs-K256 Akita commit/open microbench at
  equal information content.
