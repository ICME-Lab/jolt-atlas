# Proving optimization roadmap

This document lists the prover optimizations worth pursuing, in the order we should do them. It
comes out of the November 2026 survey of our per-op protocols against the zkML literature
(`reports/Op proving efficiency survey.md`, with per-topic notes under `research_notes/`).

Every saving quoted here is an estimate from counting committed digits and sumcheck instances.
Nothing has been measured on the Fp128 + Akita backend. Section 2 says what to measure before
committing to the larger items; `docs/measurement-plan.md` says what to build and run, in what
order.

Revised after the second research round (`reports/Recent proof system advances 2026.md`), which
read FlashProve's repository, the Akita paper and upstream a16z/jolt's pull requests since our
pin. Groups D and E changed most. Items marked "measured upstream" cite numbers from upstream
pull-request descriptions, not our own runs.

## 1. Cost model

On GPT-2 (M3, Akita over Fp128) a proof takes 40.6 s: 25.2 s witness and commitment, 5.3 s IOP,
6.7 s opening reduction and openings. Intermediate tensors are never committed. What we commit is
one-hot digit polynomials for lookups and range checks, at 4 bits per digit (`LOG_K_CHUNK = 4`).

The working assumption is that the number to drive down is **committed one-hot digits per tensor
element**, with sumcheck degree and instance count second. Measured on Akita on 2026-11-10
(measurement plan, Phase 1 results): the commit is 23.6 s of the 40.2 s GPT-2 prover and witness
generation 0.8 s; the three largest one-hot classes (192 polynomials of 2^20 K=16 digits) are
20.9 s on their own, about 104 ns per committed digit, and the share rises to 72% at 128 tokens.
The separation paper's ablations on the July 2026 HyperKZG backend found sumcheck work dominant;
that does not transfer to Akita. The digit table below is therefore the first-order model, with
one correction: the packed clamp buckets are 40–64 bits wide (10–16 digits per element), not the
32 bits (8 digits) assumed below, so per-op clamp widths (D2) have more to take. Table contents
do not matter: the separation
principle says per-lookup cost is independent of table values, with a per-proof table term that is
nearly free (a 2^64 prefix-suffix table costs 82 ms).

Digits committed per element today, and the target after this roadmap:

| Pattern | Today | Target | Item |
|---|---|---|---|
| Einsum output (rescale + clamp) | 12 | 7–8 with per-op clamp widths (D2), or 4 exact remainder digits plus 0.6–1.4 amortised at row width 1,024 (D1) | D1, D2 |
| Linear layer with bias (Einsum, Add) | 20 | 12 | C1 |
| Einsum followed by tanh / erf / sigmoid | 25 | 12 | C2 |
| Standalone tanh / erf / sigmoid | 13 | 5 | B2 |
| Attention pair: score einsum + softmax | 34 | 10–12 | C3 |
| Softmax alone | 22 (the separation paper counted 16 in July and 11–12 after PR #258; the census must reconcile) | 12 | B3 |
| Rsqrt | 33 polynomials per node | shared buckets, about 22 digits per row | B1 |
| Div, MeanOfSquares, Sin, Cos range check | 16 per node, 64-bit read-raf | shared buckets, no read-raf | B1 |
| Reshape, Slice, Concat | 0, but a full sumcheck | 0, no sumcheck | B4 |

## 2. Measure first

No per-op timing exists on the current backend. The ranking below assumes commitment volume
dominates, which the stage split supports but does not prove per op. Before starting group C or D:

1. One `--trace-terminal` run each of GPT-2 and Qwen. The spans on `Einsum::prove`,
   `EinsumDotProver::initialize` and the deferred batches already exist.
2. A census of committed one-hot digits by producing op (not by arity class), so each item below
   can be sized as a share of the 25 s.
3. A lowered-op histogram per zoo model from `dump_lowered_ops`.
4. Value histograms from the tracer: einsum output magnitudes, and the spread of `max − x` in
   attention rows. Items D2 and C3 depend on these.
5. A long-context run (GPT-2 at 256 or 512 tokens). Softmax and the score einsum grow with the
   square of sequence length and are invisible at 16 tokens.

## 3. Group A: prover-only work

No transcript, verifier or tracer change. These can start immediately and need no spec.

Upstream landed deferred-reduction and product kernels after our pin (a16z/jolt #2255, #2211,
#2191). Bump the pin (section 8) and vendor what is shared before reimplementing A1 and A2.

### A1. Split-eq dot engine

`jolt-atlas-core/src/onnx_proof/ops/einsum/dot.rs` binds `HighToLow` over dense tables with fully
reduced arithmetic. It is the only large prover that does not use `GruenSplitEqPolynomial`.

- Bind `LowToHigh` and have the batched layouts (`amk,ank->amn`, `abmk,abnk->abmn`) pass the eq
  point instead of a dense eq table.
- During the eq rounds, compute the quadratic inner polynomial and multiply by the linear eq factor:
  2 field multiplications per pair instead of 6.
- After eq is fully bound, compute a quadratic and scale it once. Keep the degree-3 wire encoding
  so messages stay bit-identical.
- Accumulate products unreduced and reduce once per thread per round.
- Use `sumcheck_evals_array::<2>` in the degree-2 arm (the current call allocates a `Vec` per
  index) and delete the unused `EqSchedule::Low`.

Estimated 2–3x on batched einsum rounds. `relation_prover_equiv_contraction` already checks the
messages against the reference prover.

### A2. Fold kernel

The fold `Σ_i eq(r, i) · A[i, k]` is where a weight matmul spends its time (2.4M weights against a
1024-entry sumcheck table for GPT-2's MLP projection).

- Accumulate `field × i64` products unreduced (`mul_u64_unreduced`, two accumulators for sign) and
  reduce once per output coefficient.
- Walk memory row-major (`acc[j] += eq[h] · W[h][j]` with per-thread accumulators). The current
  loops stride by `k` for the left operand, the `[n, k]` weight case and the batched layout.
- Iterate the unpadded extents. 768 → 1024 and 3072 → 4096 padding makes about 44% of a padded
  GPT-2 MLP weight matrix zeros.

Estimated 2–4x on fold time. The same kernel speeds up the verifier's `Constant` MLE evaluation.

### A3. Evaluation reduction

`atlas-sumcheck/src/evaluation_reduction/mod.rs` builds its univariate sequentially with one
heap-allocated `UniPoly` per coefficient. Make it parallel and allocation-free, deduplicate
identical points, and skip it for public tensors. In the one available trace (pre-Akita) it cost
four times as much as all structural-op sumchecks combined.

### A4. Housekeeping

- Remove the `padded_pow2()` copies and the `LookupBits` → `u64` → clone chain around the fused
  rebase.
- Confirm the tracer's i64 einsum kernel is blocked and vectorized; an earlier profile had it
  costing as much as proving the contraction.
- Resolve the two contradictory comments about whether witness generation re-executes the fused
  accumulation (`prover.rs:299-311` against `fused_rebase.rs:143`).

## 4. Group B: protocol changes with identical outputs

The tracer's outputs and the golden hashes do not move. Each item changes relations, so it needs
the zoo tests, the soundness sweeps over the new witness columns, and one real-model run.

### B1. One range-check mechanism for everything

Fused-rescale remainders already use model-wide packed buckets: a value sumcheck plus Booleanity,
with no read-raf and no ra-virtualization. Rsqrt, Div, MeanOfSquares, Sin and Cos instead commit
per node and run 64-bit `UnsignedLessThan` read-rafs with degree-17 ra-virtualization. Move them
all onto the buckets.

- **Rsqrt** becomes one relation, `out² · x ≤ S³ < (out + 1)² · x`, with the two slacks
  `S³ − out²·x` and `(out + 1)²·x − S³ − 1` range-checked in buckets. This is exactly today's
  `⌊√⌊S³/x⌋⌋`. The dense quotient column disappears; the relation degree rises from 3 to 4. The
  main gain is the removed 64-bit read-rafs, which the July ablations showed dominate rsqrt's cost.
  Never replace rsqrt with a fixed-grid table: "Bits Under ZK-LLM" shows that destroys 7B+ RMSNorm
  models.
- **Div** and the **MeanOfSquares** and **trig** remainders become `R` and `B − 1 − R` bucket checks.
- The new relations must range-check `out` explicitly and keep `x ≥ 1` enforced. A bound whose
  width is not a multiple of the digit size has to be enforced exactly. FlashProve does this two
  ways: multiply the lookup key by a power of two so one table enforces the range, or check both
  `R` and `B − 1 − R`. Norm quantities are per row, so these checks are per row too.

### B2. Fuse the activation clamp into the table lookup

Tanh, Erf and Sigmoid pay a 32-bit clamp lookup (8 digits) and then a 2^18 table lookup
(5 digits). The table's 5 address digits already prove the clamped value is in range, so the clamp
can use the interior/saturated split that the fused rebase uses: 5 dense digits per element, sparse
slack digits only on saturated elements, one read-raf and the degree-9 ra-virtualization removed.

### B3. Softmax: one lookup on `z`

Today `z = max − x` is looked up three times (clamp, high exp digit, low exp digit) and the two
table values are multiplied and rescaled with a range-checked remainder. Define one table over the
clamp's existing 32-bit address that returns `exp_q` directly:
`EXP[z] = [z < 2^18] · ⌊hi[z ≫ 9] · lo[z & 511] / S⌋`. This drops 10 of 22 digits and 10 of 19
relation instances.

It needs a prefix-suffix read-raf whose suffix is a dense 2^18 table, run as one model-wide
instance because the table is larger than a short-context softmax node. Whether `ps_shout` can host
that phase split has not been checked. C3 supersedes this item; do B3 only if C3 is deferred.

### B4. Structural ops as ties

- **Reshape**: `check_pad_coherent_reshape` already guarantees the padded input and output are the
  same flat array, so `out(r) = in(r)`. Make it a zero-round tie like MoveAxis.
- **Slice and Concat**: when offsets are aligned to the padded extent they reduce to fixing
  selector bits. Make those ties. For misaligned cases, either evaluate the selector in closed form
  (it is a shifted eq) or have the compiler reject them, depending on whether any survive lowering
  in the zoo.
- **IsNaN**: fixed-point values are never NaN. Rewrite it to constant zero in the importer and fold
  the dead `Iff`, removing a degree-3 sumcheck per attention layer.
- **Importer fusions**: compose MoveAxis chains, push permutations into einsum index strings, elide
  Broadcast where the consumer can index directly.

The prover saving is small. The gain is verifier time (it currently materializes a 2^n selector
table per node) and less code.

### B5. Softmax per-row rework (#218) and masking

Replace the three per-row vectors sent in the clear with a committed argmax one-hot, a `log F`-round
division sumcheck and two per-row range checks. Define `z` from the argmax one-hot
(`M(r_k) = Σ eq · e · x`, `z = M − x`) rather than checking `max = z + x` at one point, which does
not show the sum is constant along the row.

Add a public mask selector to the row relations so `active_len < N` and causal masks are provable.
Today a non-power-of-two softmax axis is rejected.

## 5. Group C: fusions that change output bits

Each needs a new `SemVersion`, re-pinned golden hashes, an accuracy run on GPT-2, Qwen and bge, and
a spec (`/new-spec`). C1–C3 share machinery and belong in one spec.

### C1. Fuse bias and residual adds into the einsum rebase

`Gemm` lowers to `Einsum` then `Add`, and `Add` carries its own 8-digit saturating clamp. Put the
addend in the accumulator: `acc' = Σ L·R + b · 2^S`. Because `b · 2^S` is a multiple of the divisor,
the quotient is `q + b` and the remainder is unchanged. One clamped node disappears per linear
layer (20 → 12 digits per element). The same works for a residual `Add` when the einsum has a
single consumer, with the residual as an extra opening.

Outputs differ from today only where the un-biased quotient would have saturated. `clamp_bound`
must add `|b| · 2^S`.

### C2. Rebase digits as lookup addresses

The fused rebase already commits a one-hot decomposition of the quotient for its range check. A
one-hot decomposition is a Shout address. When an activation consumes an einsum output, read the
activation table through those digits instead of committing a second decomposition: einsum followed
by GELU or tanh costs 12 digits per element instead of 25, and the einsum output stays virtual.

This generalizes to any lookup whose input is a fused-rebase output. It needs the activation table
defined over the rebase's address space (clamp folded into the table, as in B2).

### C3. Fused attention block

Prove `QKᵀ → mask → softmax → ·V` as one unit instead of three ops.

1. **Score einsum emits `z` directly, unrescaled.** The per-row max enters the accumulator like a
   bias: `z_acc = max_acc − Σ Q·K`. No rescale, so no score remainder.
2. **Exp is one lookup on the digits of `z_acc`.** The table ignores the low bits and returns 0
   above the cutoff. The digits serve as both the range check on `z_acc` and the exp address.
3. **Normalization moves into the `·V` einsum** as a per-row divisor:
   `P[i, c] = ⌊Σ_j e_ij · V_jc / sum_i⌋`. The per-pair output remainder disappears; a remainder
   with a row-dependent bound is checked per output instead. This rounds once where today rounds
   twice.
4. **Per-row work** follows B5: committed argmax one-hot, row-sum sumcheck, public causal mask.

Per attention pair this commits only the digits of `z_acc`: about 10–12, against 34 today, and
nothing softmax-specific. Unlike zkLLM's design it has no prover-chosen shift, so outputs stay a
deterministic function of the input.

Open questions to settle in the spec:

- How wide `z_acc` is on real models. FlashProve, which has this design working, keeps the address
  near 16 bits: operands are clamped to 12 bits, the score is shifted by 8, and the cutoff and
  causal mask are rows of the table. It commits five byte cells per pair and stores the argmax as
  index bits. Whether 12-bit clamping holds accuracy on GPT-2-class models is unmeasured.
- Whether the prefix-suffix read-raf can host a table that is constant in its low digits, dense in
  the middle and an indicator in the high digits.
- The step-3 remainder costs more per output than today's fixed-width one. It wins when sequence
  length exceeds head dimension (64 for GPT-2) and loses below that, where absolute cost is small
  anyway.

### C4. GELU as one table op

Collapse the importer's Mul/Add/Cube/Tanh (or Erf) GELU subgraph into a single table lookup. This
removes the surrounding nodes with their clamp and remainder checks. Combined with C2, an MLP block
`fc → GELU` commits 12 digits per element in total.

## 6. Group D: commitment-level experiments

Each is a measurement first and a change only if the measurement is positive.

- **D1. Row-wise approximate no-overflow check.** For each row of a wide tensor, the verifier
  draws n random combinations of the columns with coefficients 0 (probability 1/2) and ±1 (1/4
  each), shared across rows and drawn after every commitment the tensor depends on. The prover
  commits the digits of each combination as a second batch and range-checks those through the
  existing one-hot check, in place of 8 output-range digits per element. One degree-2 eq-weighted
  sumcheck over rows and columns links the combinations to the tensor's existing evaluation claim.
  Remainder digits stay exact. The soundness lemma is Lyubashevsky–Nguyen–Seiler 2020/1448
  Lemma 2.5: if every combination is below T, every entry is below 2T except with probability
  2^-n. Slack 2, exact, ℓ∞, no field-size condition.
  - **Cost and saving.** Fixed per row: n projections times 5 digits (honest values below 2^7),
    7 (below 2^16) or 11 (full i32). At n = 128 against 8 digits per element: break-even row width
    80 / 112 / 176, savings at width 1,024 of 12.8x / 9.1x / 5.8x, at 4,096 of 51x / 37x / 23x.
    Sharing one matrix across M rows costs a union bound, n = 128 + ⌈log₂ M⌉.
  - **Completeness can be perfect.** A deterministic ℓ1 threshold T = k·2^w cannot be exceeded by
    an honest row, at the cost of at most one extra digit per projection. FlashProve's 2^-40 is a
    packing choice.
  - **It cannot reproduce a saturating clamp**, so it needs D2 or partial-function semantics.
  - **Verifier cost** is about n·k additions per row width, once. The tensor-train succinct variant
    loses at our widths.
  - **The prototype must include the negative test**: drawing the matrix before the remainder
    commitment lets the prover steer the tensor into its kernel.
- **D2. Per-op clamp widths and overflow bounds.** Two roles.
  - **As a digit saving in its own right:** per-site power-of-two shifts with a per-op clamp width
    of 12–16 bits cut the clamp digits from 8 to 3–4 per element, exactly, independent of row
    width. "Bits Under ZK-LLM" (Table 9) puts GPT-2 Small at 1.027x perplexity at 12-bit
    activations with exact nonlinearities, and 7B+ models at 14–16 bits under static scales. A
    tracer-only sweep settles it (measurement plan, Phase 2). If GPT-2-class models land at 12–16
    bits, D1's remaining territory is the 7B+ residual stream.
  - **As D1's companion:** compile-time bounds from weight ℓ1 norms showing a clamp never engages
    (zkAgent); a declared per-bucket width bound into the transcript, like the existing exact flag;
    or partial-function semantics, a `SemVersion` change. A scale-derived width landed in
    jolt-atlas #258 (commit `f5214237e`, July 2026: −4 to −5 polynomials per softmax node, −2.6%
    to −6.0% prover); the sentinel survives in the importer, the range check it narrowed was
    replaced in the rewrite, so the census must re-verify the saving.
- **D3. Dense tiny-digit columns for exact range digits.** Akita gives no range guarantee on
  committed values, so the range check must be ours: a dense digit column with values in [0, b), a
  zero-check term Π(z − a) batched into the sumcheck, and homomorphic recomposition. SuperNeo
  (ePrint 2026/242) is the published, security-proved form. Two constraints from it: the
  zero-check has degree 2b−1, so b = 2 (ternary) is the natural choice and a 4-bit digit would
  need degree 16–31; and the commitment itself is only relaxed-binding, the range comes from the
  check. Upstream commits increments as one bounded-dense object (a16z/jolt #2145). No source
  measures dense tiny cells against one-hot nonzeros; the microbenchmark sweeps b ∈ {2, 4, 16}
  with the zero-check's cost included, and adds the BitZ arm below as a third comparator.
- **D4. Lookups on virtual keys.** FlashProve never commits the low 16 remainder bits: one lookup
  on a virtual key covers them. It needs a multiplicity-based lookup. Celer (ePrint 2026/1453,
  CRYPTO 2026) is a Quark grand product that takes its query as an oracle, so a virtual key works
  and it can sit on Akita at about 10 field multiplications per lookup; its soundness error is
  O(m·n/|F|), about 84 bits at 2^28 lookups into a 2^16 table over Fp128, against about 100 for a
  LogUp-type argument, so the budget (a fix to its multiplicity sum, or 192-bit challenges) must be
  set first. Promoted from research to microbenchmark candidate: a Quark grand product on virtual
  16-bit remainders against today's four one-hot digits, recording the witness, commit and
  sumcheck split. It is the only mechanism found that removes the remainder digits without
  changing the commitment.
- **D5. BitZ bit commitment for the range digits.** BitZ (ePrint 2026/2141) commits bits under a
  hash tree, proves evaluation claims over a fixed 128-bit prime with one extra round, and gives
  an exact per-coefficient range for free; measured 0.52 ns per bit on a laptop against our
  ~90 ns per one-hot nonzero. Hash-based, not zero-knowledge, proof-of-concept prover. See
  `docs/research-directions.md` section 4. Gated on running its proof-of-concept harness on our
  M3 against an isolated Akita commit, and on an explicit decision about the lattice-only end
  state.

Dropped from this group:

- **8-bit digits.** Measured upstream on Akita at 2^29 (a16z/jolt #1994): commit 153.9 s at K = 256
  against 80.7 s at K = 16, and a slower prover overall. Upstream standardised on K = 16.
- **Range bounds from the Akita commitment.** The Akita paper (Def. 10.11, Remarks 4.1 and 10.30)
  rules it out: no exact, relaxed or aggregate bound on the committed values.

## 7. Group E: sumcheck engine

These apply to the generic relation engine and the lookup subprotocols. The small-value prover,
univariate skip and the high-degree kernel do not apply to the einsum contraction, whose tables are
full field elements after the fold and whose degree is 2; delayed reduction and the one-reduction
small-by-big fold do, and they are A1 and A2 (ePrint 2026/587 measures 1.31–1.41x on degree-2
rounds and 2.2–4.5x on the fold, on BN254). Every number in this group is from BN254; none exists
at 128 bits.

- **Four-factor virtual selectors.** Keep 4-bit committed chunks but make each virtual selector a
  product of four of them. Measured upstream (a16z/jolt #2188): 38–44% off ra-virtualization and
  4.5–8.9% off the whole prover. This changes relation geometry, so it is a proof-format change.
- **Balanced product trees** for the high-degree ra-virtualization checks. ePrint 2026/587
  splits the factor list recursively and extrapolates with integer binomial coefficients, dropping
  the earlier Toom-Cook mappings; it measures 1.70–2.18x at degrees 5–33 on BN254, trading
  multiplications for additions, so expect 1.2–1.6x on Fp128 (our estimate). Its implementation
  is on a fork branch (`quangvdao/jolt`, `zkproof-submission`); check what reached mainline.
  Groups B and C delete most degree-9 and degree-17 instances, so apply this to what remains.
- **Pattern-based Booleanity kernels** (upstream #1923, #2257). Prover-only.
- **Folding the Hamming-weight check into read-raf** with a random constant, as Magnetar does for
  read-only lookups. Small.
- **Small-value sumcheck** for `eq · small · small` relations and Booleanity's first rounds. The
  paper's cost model gives κ ≈ 10 for a two-limb field, and the speedup scales as a root of κ, so
  its measured 4.1–6.0x on BN254 scales to roughly 2–3x on Fp128, unmeasured. Earlier versions of
  this document said "under 2x".
- **Univariate skip over large prime fields.** ePrint 2026/587 §7 adapts it to small values in a
  large prime field, implements it in Jolt's Spartan outer sumcheck, reports 1.11–1.24x over round
  batching and calls it the current default in Jolt. The cost is a higher-degree verifier
  interpolation. Applies where the small-value prover applies. Earlier versions of this document
  parked it on a misreading.

## 8. Order

1. Measurements, in the phases of `docs/measurement-plan.md`.
2. Bump the Jolt and Akita pins on a separate branch and re-run the baseline. Upstream's commit
   and field kernels since the pin are the cheapest speedup available, and an upstream change of
   the 128-bit modulus (a16z/jolt #2379, open) will reach the vendored field.
3. Group A (after the pin bump, for what upstream did not already supply), in parallel with B1
   and B2.
4. D2's tracer sweep of per-op clamp widths (no prover build), and the BitZ and D3 microbenchmarks
   in parallel with everything above.
5. D1 if the census shows wide rows hold most of the remaining clamp digits after D2.
6. B4 and B5.
7. One spec for C1, C2 and C3; C4 follows.
8. D4's grand-product microbenchmark; D5 and D3 per their gates and the end-state decision.
9. Group E on whatever relations remain large.

## 9. Not pursuing

- 8-bit one-hot digits, and range bounds from the Akita commitment (section 6).
- Small-value sumcheck, univariate skip or Toom-Cook inside the einsum contraction.
- Committing weights, a Freivalds-style extra projection, or cross-node batching of contractions.
- Piecewise-polynomial or holonomic activations, symmetric half-tables, a normalized-mantissa
  rsqrt table, or tolerance-based (Newton) rsqrt. The separation paper measured a holonomic erf
  at 0.27–0.56x with 19 → 11 commitments, so the reason is exactness and uniqueness of outputs,
  not digit count.
- A fixed-grid lookup table for rsqrt (destroys 7B+ RMSNorm models), and lowering the global
  scale (+5.5% perplexity at 12 bits for no digit saving; per-op clamp widths are the right lever).
- Base-2 softmax, I-BERT polynomial softmax or 2Quad: they need fine-tuning or add range-checked
  remainders.
- zkLLM's prover-chosen softmax shift: C3 reaches the same commitment count with exact outputs.
