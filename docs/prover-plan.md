# Prover plan

One document for where the proving time goes, what we will do about it, in what order, and
which strategic questions stay open. It condenses `docs/optimization-roadmap.md` and
`docs/research-directions.md` (both removed; their full text is in git history and their sources
are `reports/` and `research_notes/`) after the measurements of `docs/measurement-plan.md`
Phases 1–3 replaced the estimates they were built on. Numbers marked *measured* are from those
phases on an M3 (11 cores, 18 GB), commit `25c7218f1`; everything else is an estimate and says so.

## 1. Where the time goes (measured)

GPT-2, 16 tokens, 40.2 s to prove (±1.2% over four runs), 2.24 s to verify, 2.38 MiB proof:

| stage | s | share |
|---|---|---|
| trace execution (unspanned) | ~3 | 7% |
| witness generation | 0.8 | 2% |
| Akita commit | 23.6 | 59% |
| IOP: op sumchecks, of which deferred lookup sumchecks | 5.7 (3.1) | 14% |
| opening reduction sumcheck (1,833 claims) | 1.8 | 4% |
| per-class Akita openings | 5.0 | 12% |

Everything scales linearly in context (74 / 145 / 286 s at 32 / 64 / 128 tokens); the commit's
share rises to 72% at 128 tokens, and trace execution to 9%. bge-small (16.9 s) is the other
regime: 27 small one-hot classes, 4,902 commitments, and the opening stage (7.4 s, 44%) exceeds
the commit.

**What is committed.** Only one-hot digit polynomials (K = 16, 4 bits per digit): nothing else.
On GPT-2 at 64 tokens, by committed digit rows (one row = one cycle of one one-hot polynomial;
empty columns are free, see below):

| family | share of rows | what it is |
|---|---|---|
| clamp output chunks (`OUT = out + 2^31`, 8 per element, `clamp_split.rs`) | ~71% | the clamped output of every einsum, Add, Sub, Sum, Mul, Cube, MeanOfSquares |
| rescale remainders (14 bits = 4 chunks; 28 for Cube) | ~22% | `acc mod 2^S` |
| Tanh (GELU) lookups | ~3% | the activation's clamp and table address |
| softmax | ~1.5% | clamp, exp digits, two remainders |
| rsqrt, MeanOfSquares, gather | < 0.3% | — |

The clamp buckets also carry `w/4` slack chunks per static width `w ∈ {40, …, 64}`, nonzero only
on saturated rows (2 elements in all of GPT-2); they are committed as empty columns and cost
nothing.

**Three facts that set the plan.**

1. **Akita charges ~144 ns per committed one-hot row and ~12 ns per dense value.** A K = 16
   one-hot column costs 143 ns per digit; a dense column of the same digits in [0, 16) 12 ns,
   two base-4 columns 25–37 ns, four Boolean columns 57–69 ns; an empty one-hot column costs
   nothing (`atlas-pcs/examples/commit_bench.rs`). The one-hot encoding pays about 12x for
   spreading each digit over 16 cells, and row sparsity cannot help because every row holds a
   digit.
2. **Every clamped element commits 8 output chunks plus 4 remainder chunks, whatever its
   width.** The traced quotients on real text need 13–29 bits (residual Adds 17–23, einsum
   outputs 16–24, the LayerNorm chain 27–29); one node in GPT-2 saturates (a Cube, 2 elements).
   A declared per-op width can take the 8 output chunks to `⌈w'/4⌉`: 1.47x at per-node observed
   widths, 1.29x at a uniform 24 bits (112 nodes excepted). The static 40–64-bit widths size only
   the free slack chunks.
3. **92% of the clamp output chunks sit on rows at least 1,024 wide, 63% on rows of 4,096**
   (GPT-2); 48% on rows of 1,024 for bge-small. A per-row check amortises over them.

The July 2026 HyperKZG profile, which found sumcheck work dominant per lookup, does not transfer
to Akita. Digits are the cost.

## 2. Thesis and strategy

**On our stack, proving inference is proving values are small, and we prove it in the most
expensive encoding available.** The arithmetic (Thaler's matmul sumcheck, virtual intermediates)
commits nothing; 93% of what we commit exists to range-check rounding and clamping (the clamp
output chunks and the remainders), and it is committed in the one-hot encoding Shout needs for
lookup addresses, at 12x the cost of a dense digit, when a range check needs no lookup at all
(the clamp buckets already run none: a split sumcheck and a Booleanity chunk check).

The strategy that follows, in order of leverage:

- **Range checks become dense digit decompositions, not lookups** (section 3, items 1–2). A
  value `v` is committed as base-4 or Boolean digit columns with a zero-check (degree 4 or 2 in
  the digit) and
  the linear recomposition `v = Σ 4^i d_i`; no read-raf, no Hamming weight, no ra-virtualization.
  SuperNeo (ePrint 2026/242) is the published, security-proved form (the commitment is relaxed
  binding; the range comes from the zero-check, which is what Akita needs, since it certifies no
  range itself).
- **Wide rows get the row check** on top (item 3): 128 random ternary combinations per row, shared
  across rows, each range-checked instead of every element. Exact lemma (LNS 2020/1448 2.5):
  slack 2, error 2^-n, no field condition; completeness can be perfect at one extra digit.
- **Widths are declared per op** (item 4): 1.3–1.5x on the output chunks, and what the row check
  needs (it cannot reproduce a saturating clamp).
- **One-hot columns remain only where a table is read** (Tanh, softmax, gather: 7% of rows), or
  are virtualised from the digits.
- **The commitment scheme stays Akita** (decided 2026-11-10): the encoding change above is a
  4x on the commit with no scheme change, and keeps zero knowledge (upstream's Whiteout stack),
  recursion (the Akita verifier guest) and small proofs. BitZ is kept in view as the future
  option for the range digits (section 5); the FlashProve-style hash architecture is declined
  (section 4).

The estimated ceiling, with the measured per-row costs and today's chunk counts: items 1–4 take
the 23.6 s commit to roughly 5–7 s on GPT-2 at 16 tokens (output chunks ÷ 3.8–5.7 by the
encoding, ÷ 1.3–1.5 by declared widths, and ÷ ~10 on wide rows by the row check; remainders ÷
3.8–5.7), i.e. the prover from 40 s to the low 20s before anything is done to the IOP or the
openings, which then become half of what is left. These are estimates; the first prototype
measures them.

## 3. The plan, in order

Each item names its measured target, the mechanism, and what it needs. Items 1–4 change the
range-check protocol only; outputs and golden hashes do not move unless the item says so.

### 1. Dense-digit encoding for the clamp output chunks (roadmap D3) — 71% of rows, 3.8–5.7x on them

**Implemented on `dev`** (spec `specs/clamp-dense-digit-range-check.md`): `OUT` is 16 base-4
`U8Scalars` digit columns under `atlas_sumcheck::digit_range`, and the Booleanity check keeps the
slack chunks only. The base is the one constant `clamp_split::DIGIT_BASE`. Not yet measured: the
Phase 1 gate run and decision 3 below are pending. A first GPT-2 run (16 tokens) proved in
34.5 s against the 40.2 s baseline, verified in 2.20 s, with a 2.24 MiB proof; the stage split
is not yet measured.

Today each bucket commits `OUT = out + 2^31` as 8 one-hot chunks per row plus empty slack chunks,
checked by a split sumcheck and the Booleanity chunk check with a linear term
(`clamp_split.rs`); there is no lookup on the buckets already. Replace the one-hot chunks by
base-4 digit columns (`U8Scalars` in [0, 4), sixteen per element) with a batched zero-check
`d(d−1)(d−2)(d−3) = 0` (degree 4 in the digit, 5 with the eq factor) in place of Booleanity and the same linear recomposition of
`OUT` and `S`; the split sumcheck keeps its shape. Measured commit cost 25–37 ns per 4-bit digit
against 143. Decide base 4 against Booleans (57–69 ns; `d(d−1) = 0`, degree 3 with eq) in the prototype by
total time, not commit time alone. Needs: new witness columns in
`global_clamp.rs`/`clamp_split.rs`, a model-level bucket relation (no `type Spec`; those are
node-level), the soundness sweeps over the new columns, zoo
tests, one real-model run. Prototype on `gpt2-tiny` in the test profile first.

### 2. The same for the rescale remainders — 22% of rows, 3.8–5.7x on them

**Implemented on `dev`**: each remainder bucket commits `⌈w/2⌉` base-4 `U8Scalars` digit columns
under the same `digit_range` relation, which also tightens the bound to `R < 2^w` (the 4-bit
one-hot chunks proved `R < 2^{4⌈w/4⌉}`). Not yet measured.

A 14-bit remainder is seven base-4 digits with the same zero-check; the 28-bit Cube remainders
fourteen. Same machinery as item 1, same PR or the next. (Celer-style grand products on virtual
keys, the former D4, are superseded: they only beat one-hot digits, not dense ones.)

### 3. Row check on wide rows (roadmap D1) — ~10x on 63–92% of the clamp output chunks (estimate)

Per row width, after every commitment the tensor depends on, draw n = 128 + ⌈log₂ rows⌉
combinations with coefficients 0 (p = 1/2) and ±1 (p = 1/4 each). The prover commits the digits
of each combination (5 at honest values < 2^7, 7 at < 2^16, 11 at full i32; one more for perfect
completeness) through item 1's range check, and one degree-2 eq-weighted sumcheck over rows and
columns links the combinations to the tensor's evaluation claim. Element digits disappear on
those rows; remainders stay exact (item 2). Break-even row width 80–176; at width 1,024 the
saving is 6–13x, at 4,096 23–51x. The prototype is a standalone relation test on one 3,072-wide
GPT-2 einsum output with the **negative test** that draws the matrix before the remainder
commitment (the prover can then steer the tensor into its kernel). Needs item 4 or
partial-function semantics, since the check bounds but does not saturate.

### 4. Per-op declared clamp widths (roadmap D2) — 1.3–1.5x on the clamp output chunks, exact

Replace the static 40/48/56/64 ladder by a per-op width declared at compile time (observed +
margin, or a uniform 32), absorbed into the transcript like the existing exact flag. On the
three GPT-2 texts and bge-small every quotient but the saturating Cube stays inside its observed
width, so this is output-identical on those inputs; what is traded is saturation semantics for a
margin, which is a `SemVersion` decision (saturating at the declared width, or partial-function
semantics as FlashProve and Sable use, with the tracer asserting). Compile-time overflow bounds
from weight ℓ1 norms (zkAgent) are the stronger form where they apply. At scale 2^14 the honest
values need 17–29 bits, so "12–16-bit activations" is a change of scale, not of width, and is
not pursued (section 6).

### 5. Fusions that remove clamps (roadmap C1, new LayerNorm item, B4) — up to ~15% of rows

- **Bias into the einsum rebase (C1):** 72 einsum → constant-Add pairs on GPT-2, 12% of the
  clamp output chunks; `acc' = Σ L·R + b·2^S` leaves the remainder unchanged and the quotient `q + b`,
  so one clamped node disappears per linear layer. Outputs differ only where the un-biased
  quotient would have saturated. Nothing on the encoder models.
- **Fused LayerNorm:** each norm is nine lowered ops, eight of them clamped, holding the widest
  quotients in the model (27–29 bits). One op with per-row range checks (VerInf, DeepProve) and
  the rsqrt relation of item 7 removes most of them. Needs an importer pass and a tracer op.
- **Structural ops as ties (B4):** Reshape, aligned Slice and Concat, MoveAxis chains as
  zero-round ties; IsNaN rewritten to constant zero and the dead Iff folded. Every Slice in the
  zoo is aligned. Small prover gain; verifier time and code.

### 6. Activations and softmax (roadmap B2, C2–C4, B3, B5) — 7% of rows, more at long context

Softmax and the attention-score einsum grow with the square of the sequence length and are small
at 16 tokens; measure at 256–512 tokens before sizing this group.

- **Activation clamp fused into the table lookup (B2):** the table's address digits already
  prove the clamped value is in range; drop the separate clamp (8 digits → 5).
- **Rebase digits as lookup addresses (C2) and GELU as one table op (C4):** when an activation
  consumes an einsum output, read the table through the output's digits (after item 1 these are
  base-4 digits: virtualise the one-hot address from them at degree 4, or keep a one-hot copy
  for lookup ops only). On GPT-2 no activation reads an einsum directly today because the bias
  Add sits between; C1 first.
- **Fused attention (C3):** scores emitted as `z = max − score` unrescaled, one table for the
  range check, exp and causal mask on the digits of `z`, normalisation in the `·V` einsum's
  per-row divisor, committed argmax one-hot and a public mask selector (B5). FlashProve has this
  working at ~16-bit addresses with 12-bit operands and a coarse shift; our `z` width on real
  models is measured by the census (half of GPT-2's exp lookups already sit at the saturated end,
  `max − x ≥ 2^17`). Supersedes B3 (one lookup on `z`). Needs a spec.

### 7. Per-node range checks onto the shared mechanism (roadmap B1) — few rows, removes 64-bit read-rafs

Rsqrt, Div, MeanOfSquares, Sin and Cos commit per node and run 64-bit `UnsignedLessThan`
read-rafs with degree-17 ra-virtualization. Rsqrt becomes one relation,
`out²·x ≤ S³ < (out+1)²·x`, with its two slacks range-checked by item 1's mechanism; the others
become `R` and `B − 1 − R` checks. Never table rsqrt: fixed-grid tables destroy 7B+ RMSNorm
models (Bits Under ZK-LLM).

### 8. Openings (17%, 44% on bge-small): one commitment, one opening

Per-class Akita openings scale with the class count. After items 1–2 the classes change shape
anyway; lay every committed polynomial on an aligned subcube of as few classes as the catalog
allows (upstream's one-native-group layout, a16z/jolt #2012; `akita.rs` already uses aligned
subcubes within a class). JaggedPCS is a different construction and would cost 0.4–0.6 s under
our padding. The reduction sumcheck (1.8 s, 1,833 claims) shrinks with the IOP item below.

### 9. The IOP: the digits are the trace (own analysis)

After item 1 every op output that is clamped is committed as its digit columns, so a consumer
relation can define its input as the recomposition `Σ 4^i d_i` of the producer's digits instead
of taking a claim from the producer's sumcheck. Then every op relation is independent: all
relations of all layers go into one batched sumcheck (upstream's per-round degree schedule,
a16z/jolt #2336), same-shape instances stack across layers (twelve copies of one matmul become
one instance with a layer axis; the weights one MLE the verifier evaluates once per shape), the
evaluation-reduction stage disappears, and the lookup tables join one tagged instance. No
committed volume changes; the targets are the IOP, the reduction, proof size, verifier time and
memory: 7.5 s of 40 s today, about half of what is left after items 1–4. Softmax and table
outputs are not digit-decomposed and keep a short local chain or gain a decomposition. No
shipped transformer prover does this (FlashProve runs ~2,000 sumchecks per request). Needs a
relation compiler from the ONNX graph; a Phase 4 candidate after the range-check items.

### 10. Engine (roadmap A, E) and the pin bump

- Bump the Jolt and Akita pins on a branch and re-run the baseline first; upstream's commit and
  field kernels since the pin (#2255, #2211, #2191, #2145, #2188) are the cheapest speedup
  available, and the open modulus change (#2379) will reach the vendored field.
- Einsum dot engine on `GruenSplitEqPolynomial`, `LowToHigh`, unreduced accumulation (A1), the
  row-major unreduced fold over unpadded extents (A2), a parallel allocation-free evaluation
  reduction (A3). Estimated 2–4x on their kernels, which are inside the 5.7 s IOP.
- Four-factor virtual selectors (upstream #2188, 38–44% off ra-virtualization), balanced product
  trees (1.2–1.6x on Fp128, our estimate from 2026/587's BN254 numbers), small-value sumcheck
  (~2–3x on Fp128, estimate) and univariate skip (1.11–1.24x, Jolt's default) for what remains
  high-degree after items 1–7. Microbenchmark on Fp128 before writing any into the prover;
  every published number is BN254.

### 11. The tracer

Trace execution is 3 s at 16 tokens and 12.8 s at 64 (7–9% of the prover, linear in context),
inside `prove` and unspanned. Profile its i64 einsum and LayerNorm kernels; the prover's own
witness generation over the same data takes 0.8 s.

### Order and gates

1. Item 1 prototype on `gpt2-tiny` (test profile), then release and the Phase 1 run table
   against `bench/bin/prove-25c7218f1`. This is the first protocol change and sets the encoding
   everything else uses. **Gate:** total time per clamp digit (commit + sumchecks) below
   today's.
2. Item 2 in the same encoding. Item 4's declared widths as the `SemVersion` decision.
3. Item 3's standalone relation test with the negative test; then on the MLP rows.
4. The pin bump, in parallel on its own branch.
5. Items 5 and 7; one spec for item 6 (C1–C4 share machinery); item 8.
6. Item 9's relation compiler as its own spec; item 10's microbenchmarks; item 11 any time.

## 4. Positions (research, condensed)

- **Magnetar** (Setty, ePrint 2026/2423): binary field, packed bits, constraints instead of
  lookups, one commitment and one opening. Its field cannot count, so its arithmetisation does
  not transfer; its layout discipline (commit once, derive everything; one opening) does, and is
  items 8–9. Folding the Hamming-weight check into read-raf transfers for read-only lookups.
- **FlashProve** (Anthropic/EPFL, Nov 2026): Goldilocks with a cubic extension, Reed–Solomon at
  rate 0.104, committed byte-cell trace, per-row approximate range check (item 3's source),
  exact rounding by one 2^16 lookup on a virtual key, fused attention (item 6), partial-function
  semantics, 12-bit activations, 6.8–9.4% top-token divergence, 18.78 MB proofs, 21 s verifier,
  Llama 4 Maverick at 111 tok/s on an H100 at batch 24. Hardware explains 10–100x of the ~10^4
  gap in parameters × tokens/s; the rest is encoding, which items 1–4 address.
- **Sable** (MoonMath, 3 Nov 2026): KoalaBear + LogUp-GKR, BaseFold/BLAKE3, every rounding exact
  at one committed element plus 2–3 lookups, Qwen2.5-32B at 4K in 92.5 s with 135 MiB proofs,
  partial semantics, not ZK. **Attestable:** no paper or code; ZK, ~5 MiB proofs, within ~1.3x of
  Sable's wall clock. Positions still unclaimed: small proofs with an open design, ONNX
  generality, total exact semantics, fidelity to the served model.
- **Bit-level commitments.** BitZ (ePrint 2026/2141) commits bits under a hash tree, gives an exact
  per-coefficient range for free, and proves claims over a fixed 128-bit prime with one extra
  round: 0.52 ns per bit measured on an M5 (2^30 bits in 561 ms), i.e. ~2 ns per 4-bit digit
  against Akita's 37 (base-4 dense) or 143 (one-hot). Hash-based, not ZK, proof-of-concept.
  Limber (no range check, 3–4x at 128 bits), LaBinius (characteristic 2 only) and Mikros (no
  packing at 128 bits) are out. 8-bit one-hot digits are slower on Akita (upstream #1994).
- **The separation principle** (ePrint 2026/1390): per-lookup cost is independent of table
  values; attention needs Ω(N²) comparisons, so only the per-pair constant is in play.
- **Quantization** (Bits Under ZK-LLM): activations are the sensitive side (GPT-2 fine at 10–12
  bits, 7B+ need 14–16 under static scales; power-of-two scales free at 16 bits); weights
  irrelevant to us (public). For us bit width is the cost, unlike DeepProve.

## 5. Decisions and open points

1. **Decided: Akita now, BitZ in view.** Akita after item 1 is ~37 ns per digit; BitZ's paper
   number is ~2 ns, from a proof-of-concept prover that is hash-based and not zero-knowledge.
   Staying on the lattice keeps zero knowledge, recursion and small proofs and takes the 4x that
   needs no scheme change. Revisit BitZ when item 1 has landed and the commit is measured again,
   or when BitZ ships a maintained prover with a zero-knowledge variant. The proof-of-concept is
   checked out at `/Users/kg/swe/icme/scratch/BitZ-PoC` with the run commands in the measurement
   plan, Phase 3, for whenever that comparison is wanted; it is not on the critical path.
2. **Semantics under declared widths:** saturate at the declared width, or partial-function
   semantics with the tracer asserting. Item 4; a `SemVersion` change either way.
3. **Base 4 or Booleans** for item 1, by measured total time. Open: item 1 ships base 4 on
   `dev`; switching to Booleans is the one constant `clamp_split::DIGIT_BASE = 2`. Decided by
   the Phase 1 gate run, not yet made.

## 6. Not pursuing

- Lowering the global scale or 12–16-bit activations (the honest values need 17–29 bits at
  2^14; the saving is in the encoding, not the scale); 8-bit one-hot digits; range bounds from
  the Akita commitment (it gives none).
- A FlashProve-like committed-trace hash architecture for general ONNX: it means leaving Fp128,
  megabyte proofs and exact semantics, in a space several teams occupied within weeks.
- Grand products on virtual keys for the remainders (D4): dominated by dense digits.
- Small-value sumcheck, univariate skip or Toom-Cook inside the einsum contraction; committing
  weights; Freivalds-style projections; cross-node batching of contractions (item 9 does this
  structurally instead).
- Piecewise-polynomial or holonomic activations, half-tables, mantissa or Newton rsqrt, tabled
  rsqrt, base-2 or polynomial softmax, prover-chosen softmax shifts: exactness and uniqueness of
  outputs.
- Sub-quadratic attention (ruled out by theorem).

## 7. Reads still owed

Upstream a16z/jolt #2040 (binary-field track; whether the LaBinius decoupling reaches an odd
prime); Cyclo (ePrint 2026/359, the only benchmark of a dense tiny-digit range check); the
Whiteout paper when public. The whitepaper's claims not to repeat (R1CS stages, BlindFold,
HyperKZG/Dory/BN254, on-chain verification, 2^7-scale teleportation, the old stage profile,
unreproduced comparisons) are listed in the research report.

## References

FlashProve technical report and repository (v0.16.0); Magnetar 2026/2423; BinarySpartan
2026/1656; Akita 2026/1983; LNS 2020/1448 and GHL 2021/1397 (approximate range); BitZ 2026/2141;
Limber 2026/1635; LaBinius 2026/2103; Mikros 2026/2646; Celer 2026/1453; Separation Principle
2026/1390; tensor-train projections 2026/2146; PikkuFold 2026/1809; Dao et al. 2026/587;
JaggedPCS 2025/917; Neo/SuperNeo 2026/242; Bits Under ZK-LLM arXiv 2609.36437; Sable
(github.com/moonmath-ai/Sable); DeepProve 2026/1112; zkAgent 2026/199; a16z/jolt #1911, #1994,
#2012, #2040, #2145, #2188, #2336, #2379. Full notes: `reports/` and `research_notes/`.
