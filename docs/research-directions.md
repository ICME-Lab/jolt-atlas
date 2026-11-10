# Research directions

`docs/optimization-roadmap.md` lists the optimizations to our per-op protocols. This document
covers the larger questions: where proving cost comes from, what the newest systems do about it,
and which direction jolt-atlas should take. `docs/measurement-plan.md` says what to build and run
to settle the open points.

Revised twice: after the second research round (`reports/Recent proof system advances 2026.md`,
abstracts and repositories) and after full readings of the papers it rested on
(`reports/Paper readings November 2026.md`, with per-paper notes under
`research_notes/Paper readings November 2026/`). Nothing here has been measured or prototyped by
us.

Reading depth, since conclusions depend on it:

| Read in full or from source | Abstract only |
|---|---|
| Magnetar; FlashProve's technical report and repository; the Akita paper's commitment, extraction and benchmark sections and the pinned Akita code; BitZ, Limber, LaBinius, Neo/SuperNeo, Mikros, Celer, the Separation Principle paper, Dao et al. 2026/587, BinarySpartan, JaggedPCS, PikkuFold, tensor-train projections, Lyubashevsky–Nguyen–Seiler 2020/1448, Gentry–Halevi–Lyubashevsky 2021/1397, "Bits Under ZK-LLM", the Jolt Atlas whitepaper, Sable's paper; DeepProve, zkAgent, OpenLLM, VerInf; upstream a16z/jolt pull-request descriptions | Cyclo, Maltese, zkLLMPoT, LAMP, IMMACULATE and other minor sweep items |

## 1. Thesis

**On our stack, the cost of proving inference is the cost of proving values are small.** On GPT-2,
25 s of the 40 s proof is the stage that generates witnesses and commits one-hot digits whose
only purpose is to range-check rounding and clamping. The arithmetic itself (matmul by sumcheck,
virtual intermediates) commits nothing.

Three qualifications from the full readings:

- **It is a property of our commitment and digit encoding, not of inference proving.** The
  HyperKZG build had commitment at 3.5 s of 38 s; DeepProve's profile is lookup count and matmul;
  and on the one backend anyone has profiled per lookup (our own, July 2026, HyperKZG), the
  separation paper's ablations found sumcheck folding, not commitment, dominated: halving the
  committed families on rsqrt moved per-lookup time to 0.90x. Whether digits or sumcheck instances
  dominate on Akita is unmeasured, so every digit-count saving in the roadmap is an upper bound on
  time saved until the 25 s stage is split into witness generation, commit and lookup sumchecks.
- **The fastest systems still prove every rounding exactly.** What they stopped doing is proving,
  value by value, a bound that exists only to keep field arithmetic from wrapping (section 3).
- **Akita certifies nothing about ranges, but that is not a law of commitments.** BitZ's
  commitment relation includes an exact per-coefficient range (section 4).

## 2. Magnetar

Magnetar (Setty, ePrint 2026/2423, October 2026) is a RISC-V zkVM built on Spartan and Twist/Shout
over binary fields with a hash-based commitment (Ligerito). It reports 7.2–8.7 million cycles per
second on an M4 Max, 6–20x the proving throughput of Jolt-Akita. That comparison changes the
arithmetisation as well as the commitment, so it does not isolate either.

Its design choices:

- **Bit-valued witness, packed.** 128 bits pack into one element of F₂¹²⁸, so commitment cost is
  per bit and a committed bit needs no Booleanity check.
- **Constraints instead of lookups.** Instruction semantics are R1CS constraints and bit routings
  over committed bits, where Jolt uses lookups with one-hot addresses.
- **One commitment, one opening.** MosaicPCS places every committed polynomial on an aligned
  power-of-two subcube of one polynomial, so a claim lifts by appending fixed selector bits.

**We cannot adopt its field.** Characteristic 2 cannot count: a sum in F₂ is an XOR. Our matmul
sumcheck relies on integers embedding in a large-characteristic field, which is what lets a matmul
cost one small multiplication per weight with nothing committed. Magnetar proves RV64I only, with
no multiply instruction.

**What transfers:**

- **Commit a value once and derive everything from it.** For a matmul output: the rescale, the
  range check, the clamp, the output value and the next activation's lookup address all read one
  committed object. Roadmap item C2 is the first step.
- **One commitment, one opening.** We already pad to powers of two and pack clamp buckets into
  aligned segments, and `atlas-pcs/src/akita.rs` already uses aligned subcubes within a
  commitment class. Upstream Jolt has a one-native-group layout (a16z/jolt #2012). JaggedPCS
  (ePrint 2025/917) is a different construction: it handles arbitrary column heights with an
  adapter sumcheck that would add about 0.4–0.6 s on GPT-2, and it pays only if padding is
  dropped. The saving here is bounded by the per-class opening share of the 6.7 s opening stage,
  which is unmeasured; the reduction sumcheck in that stage does not shrink with relayout.
- **Folding the Hamming-weight check into the read check** with a random constant, for read-only
  lookups. Its other one-hot refinement uses the Frobenius map and is binary-only.
- **Chunking with recursion** for memory. Upstream's Akita verifier guest merged on 2 November
  2026 (a16z/jolt #1911).

## 3. FlashProve

### 3.1 What it is

FlashProve (Anthropic and EPFL, November 2026; Apache 2.0) is a zero-knowledge succinct argument
that a committed model produces the claimed tokens under greedy decoding.

| | |
|---|---|
| Model | Llama 4 Maverick: 400B-parameter mixture-of-experts, about 17B parameters active per token |
| Hardware | One H100 with 24 CPU cores and about 270 GB of host memory. Ports to a 16 GB consumer GPU and Apple silicon appeared within a week |
| Throughput | 17, 76, 111 and 123 tokens/s at batch sizes 1, 8, 24 and 64, counting every window token; 39.1 tokens/s counting only checked reply tokens |
| Overhead | 71x to 93x plain prefill by their measure; 61x on the neutral benchmark at inferenceproofs.ai |
| Proof and verifier | 18.78 MB per 1,024-token request; about 21 s single-threaded |
| Field and code | Goldilocks with a cubic extension (192-bit challenges); Reed–Solomon at rate 0.104; BLAKE2b; just inside the proven (Johnson) regime. 128 bits per round counts 20–32 bits of grinding |
| Quantization | Per-site power-of-two scales; 12-bit clamped activations; 13-bit weights (the 4-bit codes are decoded and re-quantized); a 40-bit residual stream |
| Fidelity | The integer model's top token differs from float32 at 6.81% of positions, 9.41% on natural text. No task benchmark |
| Formalization | Protocol and its completeness, soundness and zero-knowledge proved in Lean at the IOP layer; the C++ prover and verifier are generated and not proved equivalent |

A crude normalisation by active parameters times tokens per second puts FlashProve near
1.9 × 10¹² on an H100 and jolt-atlas near 5 × 10⁷ on an M3 CPU. Hardware might explain 10–100x of
the gap. This indicates the order of magnitude only.

### 3.2 Architecture, against ours

| | jolt-atlas | FlashProve |
|---|---|---|
| Intermediate values | Virtual | Rounded outputs are committed as byte cells. Accumulators, attention scores, logits, lookup keys and the low 16 remainder bits are virtual |
| What is committed | One-hot digits | Signed bytes, three packed per field element |
| Commitment | Lattice (Akita) over Fp128 | Reed–Solomon and Merkle tree, WHIR-style folding |
| Overflow checks | 8 exact digits per value | 128 random combinations per row (section 3.3) |
| Rounding | 4 exact remainder digits per value | Exact: one 2^16 lookup on a virtual key plus one small lookup |
| Nonlinearities | One-hot Shout lookups | Grand-product lookup through GKR product trees; keys never committed, multiplicities committed as bit-planes; tables of 2^8, 2^10 or 2^16 rows |
| Matmul | Thaler's sumcheck, virtual accumulators | The same |
| Endgame | Opening reduction, then openings | All claims become linear claims, batched into one proximity test |
| Weights | Public; the verifier reads all of them | Committed once, hidden, opened once per batch |

No per-stage profile of FlashProve's prover is published.

### 3.3 The approximate range check

For each row of a trace matrix, the verifier draws 128 random combinations of the columns with
coefficients 0 (probability 1/2) and ±1 (1/4 each). The same combinations are shared by every row.
The prover commits two digits (10 and 8 bits) of each combination, and a small lookup checks them.

The soundness lemma is Lyubashevsky–Nguyen–Seiler (ePrint 2020/1448, Lemma 2.5), restated with
128 columns by Gentry–Halevi–Lyubashevsky (ePrint 2021/1397, Lemma 3.4): for any fixed vector and
n random ternary combinations with this distribution, if every combination is below T then every
entry of the vector is below 2T, except with probability 2^-n. It is an ℓ∞ statement, exact, with
no heuristic and no condition on the field size. It is not an ℓ2 Johnson–Lindenstrauss projection.

- **Soundness slack is exactly 2.** FlashProve's gap from 2^7 (honest) to 2^18 (proven) is almost
  all completeness margin: the honest combination has standard deviation about √(k/2) times the
  value bound, and FlashProve's threshold sits far above it (GHL's heuristic constant, made
  rigorous by PikkuFold Theorem 2).
- **Perfect completeness is available.** A deterministic ℓ1 threshold T = k·2^w cannot be exceeded
  by an honest row, at slack 2k, which at 4-bit digit granularity costs at most one extra digit per
  projection. FlashProve's imperfect completeness (2^-40) is a packing choice, not a property of
  the method.
- **It is per row, and pays only on wide rows.** Per projection the prover commits 5 digits when
  honest values are below 2^7, 7 below 2^16, 11 for full i32. With 128 projections that gives,
  against 8 clamp digits per element: break-even row width 80 / 112 / 176, savings at width 1,024
  of 12.8x / 9.1x / 5.8x, and at 4,096 of 51x / 37x / 23x. The earlier figures in this document
  assumed full i32 honest values.
- **Sharing one matrix across rows costs a union bound.** Error is M_rows·2^-n, so strict 2^-128
  over 2^20 rows needs 148 projections.
- **The verifier is not succinct in the random matrix.** It does about n·k additions per row width,
  once, because the matrix is shared. The tensor-train succinct variant (ePrint 2026/2146) targets
  2^20–2^24-wide vectors, worsens slack to 2^μ and completeness to 1/2, and costs the verifier more
  than the dense matrix at our widths (k ≤ 4,096). Not worth adopting unless the verifier becomes a
  circuit.
- **It does not make rounding exact.** Every remainder and every table input still gets an exact
  lookup, about 2–3 per matmul output in FlashProve.
- **Linearly consumed values get no exact check at all.** A quotient that feeds only linear ops
  (the residual stream, attention output) is covered by the approximate check alone. That is a
  valid statement only because the model is defined as a partial function: out of range means no
  proof exists, not that the value saturates.

**It works on virtual tensors.** The papers use the commitment only to fix the vector before the
random matrix is drawn; nothing requires the checked vector itself to be committed. The protocol
for us (written out in the approximate-range reading note): draw one matrix per row width after
every commitment the tensor depends on; the prover commits the projection digits as a second
batch through the existing one-hot range check; one degree-2 eq-weighted sumcheck over rows and
columns links the projections to the tensor's existing evaluation claim, batched so a single
terminal evaluation survives. The one hard precondition is the ordering: if the matrix is drawn
before the remainder commitment, the prover can steer the tensor into its kernel. The negative
test for that attack belongs in the prototype.

For us, the output clamp check (8 of 12 digits per einsum output, and all 8 on Add, Sub and Sum)
could become this check, with the 4 remainder digits staying exact. It cannot reproduce a
saturating clamp, so it needs a companion: a compile-time no-overflow bound, a declared width, or
partial-function semantics (roadmap D1 and D2). Section 3.6 gives the exact alternative that
competes with it.

### 3.6 The exact alternative: per-op clamp widths

"Bits Under ZK-LLM" (arXiv 2609.36437) measured accuracy on nine models and proving cost on a
modified DeepProve. Its cost result (prover time flat within ±5% from 8 to 16 bits) is specific to
DeepProve, whose op and lookup counts do not change with width. For us bit width is the cost: one
committed digit per 4 bits per element. Its accuracy results therefore matter more to us than to
its authors:

- weights at 8 bits cost at most 1.2x perplexity on every model, so weight precision is irrelevant
  to us (weights are public and uncommitted);
- activations are the sensitive side: 8 bits costs 3x to 500x; at 12 bits with exact
  nonlinearities GPT-2 Small is at 1.027x, Qwen2.5-7B at 1.165x, Llama-3.1-8B at 1.725x; dynamic
  per-token scales recover 12 bits in 17 of 18 cases;
- power-of-two scales cost nothing at 16 bits and at most 5.5% at 12;
- GPT-2-class models are fine at 10–12 bits; 7B+ models need 14–16 under static scales;
- the inverse-square-root finding is an accuracy failure of fixed-grid tables on 7B+ RMSNorm models
  (66% of inputs land on the zero grid point). We compute rsqrt exactly at 2^-14 resolution, so it
  does not describe us, and it is a reason never to table rsqrt.

The design this supports: per-site power-of-two shifts with a per-op clamp width of 12–16 bits.
That cuts the clamp digits from 8 to 3–4 per element, exactly, output-preserving by construction,
independent of row width, with 7–8 digits per einsum output instead of 12. It can be evaluated
with the tracer alone (sweep the clamp width, score perplexity and top-5 agreement against float).
If GPT-2-class models land at 12–16 bits, the row check's remaining territory is the 7B+ residual
stream. The risk to check is i32 saturation on residual streams.

### 3.4 Techniques it has that we had only proposed

- **Fused attention.** Scores are virtual, the max is a committed argmax index in bits, and one
  table does the range check on `max − score`, the exponential and the causal mask. Normalisation
  is folded into the `·V` division. Five byte cells and two lookups per pair. The address stays
  near 16 bits through 12-bit operands and a coarse shift. This is roadmap C3.
- **Two-sided and scaled-key range checks** for the reciprocal square root in RMSNorm, per row.
  This is roadmap B1.
- **Fused range, clamp and function tables.** Roadmap B2, C2 and C4.
- **Segments with carried boundaries**, for memory.

### 3.5 Where it leaves room

| FlashProve | Our position |
|---|---|
| One architecture, a bespoke protocol | Arbitrary ONNX graphs |
| 18.78 MB proofs | 2.4 MiB on GPT-2, with a merged upstream path to sub-100 KB by recursion. Other committed-trace provers are larger still (Sable 106–477 MiB) |
| Integer model diverges from float at 6.81–9.41% of top tokens | Unmeasured for us; a prover that tracks the served model more closely has a real claim |
| Partial-function semantics, imperfect completeness | Exact, unique-witness semantics |
| 1,024-token context | Unmeasured beyond short prompts. SASH has measured 16K |
| Greedy decoding | Not addressed by us. zkAgent has top-k, top-p and temperature protocols |

Not advantages, despite earlier drafts of this document: verifier time (hash-based verifiers run
0.8–3.3 ms against Akita's 10.5–12.7 ms on Akita's own table), and laptop-class hardware (the
ports erased that).

## 4. Commitment options for small integers

**Akita gives no range guarantee on committed values.** Its commitment relation does not require
the canonical short decomposition (Def. 10.11), extraction yields only a weak opening
(Remark 10.30), and its internal digit check covers the next recursive witness, not the source.
`DenseBounded` in the code is a prover-side sizing contract.

| Option | What it is | Status after the full readings |
|---|---|---|
| A. One-hot digits, fewer of them | Today's mechanism, with per-op clamp widths or the row check removing overflow digits and fusions removing duplicate decompositions | Available now |
| B. Dense tiny-digit columns with our own zero-check | A dense column of digits in [0, b) plus a zero-check term Π(z − a) batched into the sumcheck, with homomorphic recomposition | SuperNeo (ePrint 2026/242) is the published, security-proved form: exact range from the zero-check, commitment only relaxed-binding. The zero-check has degree 2b−1, which is why SuperNeo uses b = 2 (ternary digits); a dense 4-bit digit would need degree 16–31. Unimplemented, no benchmarks, no 128-bit parameters. Upstream commits increments as one bounded-dense object (a16z/jolt #2145). The microbenchmark must sweep b ∈ {2, 4, 16} against one K = 16 one-hot nonzero |
| C. Bit commitment under a hash tree: BitZ | Commits the witness's bits, 128 per element of F₂¹²⁸, Reed–Solomon plus Ligerito/Merkle, and proves evaluation claims over any finitely generated ring, a fixed 128-bit prime included | Read in full (85 pages). Range check exact per coefficient to [0, 2^B) for a declared B (Theorem 4.15), so a remainder in exactly [0, 2^s) is B = s. For p ≈ 2^128 an extra round lifts the claim to the integers and projects through a random ~110-bit prime, costing O(ℓ) integer additions and 23 bits of grinding (26 ms). Measured (Table 10, MacBook Air M5, 10 threads): 2^30 bits plus one evaluation claim in 561 ms, 373 KB proof, 11 ms verifier: 0.52 ns per bit, about 2.1 ns per 4-bit digit against our ~90 ns per one-hot nonzero. Transparent, "plausibly post-quantum", **not zero-knowledge**; proof-of-concept prover, measured without the extra round; about 16 GB at 2^32 bits; lookups not addressed |
| D. Hash-based commitment of dense small integers in a small field | FlashProve's and Sable's design | Requires leaving Fp128. BaseFold commits 128-bit elements at 38.4 ns each on one core (LaBinius Table 1); BitZ shows a hash commitment can keep the PIOP over Fp128, so this option is dominated by C |

Dropped from option C after the readings:

- **Limber** (ePrint 2026/1635) is a compiler for non-native integer arithmetic over a base PCS. It
  provides no range check (it consumes an external one and names range proofs its major cost), its
  "o(1) overhead" holds at a 256-bit field and is 3–4x at 128 bits by its own parameter script, and
  BitZ's head-to-head puts it 20–25x slower.
- **LaBinius** (ePrint 2026/2103) commits integer coefficients in a ring with a small composite
  modulus and produces claims over a *binary* field for Binius-style statements. Its "no bitness
  check" is definitional: the statement is defined on the extracted witness mod 2. For b-bit
  integers or a 128-bit prime it yields nothing. Its "within 1.2x" is against Binius's own prover
  on hash circuits, single core, 100-bit security; proofs are 0.3–5.4 MiB, or 73–100 KiB with a
  LaBRADOR compressor at a 77–713 ms verifier. Revisit only if upstream's binary-field track
  (#2040) shows the modulus trick applied to an odd prime. It does show a lattice commitment
  committing bits at 0.30 ns per bit single-core, which is the lattice side's existence proof.
- **Mikros** (ePrint 2026/2646) packs small-field vectors into extension elements by a factor
  λ/log₂q, which is 1 at 128 bits. It does not apply.

Two findings that bound expectations:

- **8-bit one-hot digits are slower on Akita.** Measured upstream at 2^29 (a16z/jolt #1994): commit
  153.9 s at K = 256 against 80.7 s at K = 16.
- **The lattice-only end state now has a price.** Every condition this document set for option C
  (exact range over a fixed prime, an order of magnitude cheaper per bit) is met by BitZ on the
  paper's evidence, and no lattice construction read here matches it: Akita certifies nothing,
  LaBinius works only in characteristic 2, SuperNeo exists on paper at base 2. Staying lattice-only
  is a decision against 0.52 ns per bit and for zero knowledge (Whiteout), recursion (the Akita
  verifier guest) and small proofs. It should be made explicitly, after the BitZ proof-of-concept
  has been run on our machine against an isolated Akita commit.

On lookup arguments: the soundness loss depends on the family. A LogUp-type argument loses about
log₂(m+n) bits; Celer (ePrint 2026/1453) loses log₂(m·n), about 84 bits at 2^28 lookups into a
2^16 table over our field; BitZ's GKR is analysed round by round and reaches 128 bits with two
grinding bits per round. FlashProve uses a grand product with 192-bit challenges. Celer takes its
query as an oracle, so a virtual key works and it can sit on Akita; that makes "never commit the
remainder's low bits" (roadmap D4) a microbenchmark candidate rather than research.

## 5. Other 2026 systems

Techniques from systems read in full that we do not have:

- **Overflow-free scales from weight ℓ1 norms**, removing the clamp (zkAgent).
- **Requantization fused into the next lookup**, and skipped when headroom allows (DeepProve,
  where requantization chunks are about 27% of prover time).
- **One shared digit table model-wide** (DeepProve, zkAgent).
- **Per-row bracket checks** for reciprocal square root and softmax normalisation (VerInf,
  DeepProve).
- **Zero-knowledge sampling protocols** (zkAgent).
- **Boundary-commitment partitioning** for memory and cores (zkComposer: 5–8x lower peak memory at
  1.4–2.1x time).

The de facto common workload is GPT-2 at 64 and 512 positions on a roughly 32-thread x86 machine:
DeepProve reports 34 s and 176 s, zkAgent 194 s at 512, while the Bits paper's DeepProve fork takes
104 s for GPT-2 Small at 16 tokens, so every DeepProve figure needs its version and configuration.
No head-to-head is independent. A neutral benchmark, inferenceproofs.ai, launched on 3 November
2026 and lists jolt-atlas as tracked but not benchmarked. Our own July 2026 HyperKZG GPT-2 figure
(about 10 s, PR #258) and today's 40 s are almost certainly different workloads; the baseline run
must pin one.

**Sable** (MoonMath, released 3 November 2026; 60-page paper and code, benchmarks reproducible)
is the open, exact committed-trace design: sumcheck plus LogUp-GKR over the 31-bit KoalaBear field
with extension challenges, rate-1/2 Reed–Solomon with BLAKE3 and BaseFold, proven-regime soundness
2^-103, not zero-knowledge. Every rounding is exact at one committed element plus 2–3 lookups,
against our 8–12 one-hot digits; one 2^26-row tagged table serves every range width; the LogUp leaf
layer is merged into the zerocheck so remainders never get a column. Matmul accumulators, attention
scores and softmax denominators are virtual; every lookup output, including every attention
probability, is committed. Int8 matmul inputs with power-of-two per-row and per-column exponents,
a 24-bit residual stream, 10-bit probabilities, partial-function semantics with zero aborts on
3.45M real tokens. It implements exact inverse-CDF sampling from a client nonce, the stopping rule
in the relation, and MoE top-k. Qwen2.5-32B at 4K tokens proves in 92.5 s with a 135 MiB proof
verified in 18.8 s on one RTX PRO 6000; 16K tokens in 398 s with 346 MiB; +8.3% perplexity and
86.3% greedy agreement with BF16. Its own statement of where the time goes: "Commitment, lookups
and constraint terms account for almost all of the prover's work."

**Attestable** has no paper, documentation or code. Its blog claims hash-only, post-quantum,
zero-knowledge with hidden weights, "100 bits of security", floating-point nonlinearities with
dynamically int8-quantized matmuls, a 16K context cap, and alpha results: Gemma 4 31B at 77.7
tokens/s on one H100 at 4K with 5.5 MiB proofs verified in 251 ms; Llama 70B at 49 tokens/s. Its
implied wall-clock is within about 1.3x of Sable's on comparable hardware, so its differentiator is
proof compression and hiding, not prover speed.

After these two, the positions still unclaimed are: proof size in Attestable's class with an open
design, ONNX generality (Sable is three hand-wired Qwen variants), and total rather than partial
exact semantics.

On the use case: Anthropic renamed its goal from "provable inference" to "inference verification"
on 19 October 2026, calling cryptographic proof narrower than needed. Non-SNARK schemes cost the
provider 0–2% but give statistical, economic or hardware-trust guarantees. "Hollow-LLM"
(arXiv 2607.28884) argues that committed-weight proofs do not show a large model was run.

## 6. Direction

Three strategies were considered.

- **A. Adopt a FlashProve-like committed-trace hash architecture for general ONNX.** It is the
  only design shown within two orders of magnitude of serving cost. It means leaving Fp128, giving
  up megabyte-scale proofs and exact semantics, and entering a space several teams occupied within
  weeks, with generality as the only differentiator.
- **B. Keep the lattice stack; adopt the row check, the fusions and upstream's improvements.** It
  attacks the largest cost with no new commitment scheme and keeps arbitrary ONNX, exact
  semantics, small proofs and a recursion path. Its ceiling is perhaps 1.5–3x on GPT-2 (our
  estimate), not a change of class.
- **C. Bit commitment of the range digits under BitZ, with the Fp128 sumcheck layer unchanged.**
  The only route found to "commit once, derive everything, range checks included". The paper
  meets every condition this document set for it (section 4). It is hash-based and not
  zero-knowledge, which conflicts with the lattice-only end state, and its numbers are from a
  proof-of-concept prover.

**Recommendation: B now, C promoted from "next question" to a funded experiment in parallel, A
declined.** The B ceiling stays at perhaps 1.5–3x on GPT-2 (our estimate), and its mechanism must
be sumcheck instances removed (B1, B2, C2) as much as digits, since the July ablations found
sumcheck folding dominant per lookup. What decides between B and C:

- the BitZ proof-of-concept run on our M3 at 2^30 bits with a 128-bit prime target and the extra
  round on, against an Akita commit of 2^28 one-hot digits on the same machine with witness
  generation separated out (threshold proposed by the reader: under 3 s single-threaded and more
  than 5x below Akita means write the spec);
- the dense digit-base sweep for option B (section 4);
- an explicit decision on the lattice-only end state against those two numbers.

## 7. The digits are the trace: one batched relation

This is our own analysis, not from a paper. No shipped transformer prover does it: FlashProve runs
about 2,000 sumchecks per request.

### 7.1 Why the proof is a chain today

Op N's sumcheck produces the evaluation claims that op N−1 consumes, so every node runs its own
sumcheck in sequence, and a 12-layer model runs twelve copies of the same ~150 sumchecks. The
chain exists because intermediate tensors are virtual: op N's sumcheck is the only thing that
pins down op N−1's output.

### 7.2 The observation

For most ops the output is already committed. The 8 clamp digits per einsum, Add, Sub, Sum and
Mul output are a one-hot digit decomposition of the output value, which is a commitment to the
value. We derive the output claim from the consumer's sumcheck and then check the digits against
it; we could instead let each relation define its input as the recomposition of the producer's
committed digits, `Σ 2^(4i) · value(ra_i)`, which Shout's read-value machinery already provides.

Then:

- every op relation is independent of every other, so all relations for all layers can be one
  batched sumcheck under a random linear combination (upstream's per-round degree schedule for
  batched sumchecks, a16z/jolt #2336, handles heterogeneous degrees);
- same-shape instances stack: twelve layers of the same matmul become one instance of the batched
  einsum layout with a layer axis, and the weights become one MLE over (layer, k, j) that the
  verifier evaluates once per shape;
- the evaluation-reduction stage disappears, since each consumer opens the digits at its own
  point;
- claim routing, the claim lifecycle and per-op prover scheduling are replaced by one relation
  compiler.

The lookups collapse the same way. Range checks are already model-wide buckets; the activation,
softmax and range tables can join them in one instance with a public tag polynomial selecting the
table per site (`Σ_tag eq(tag, ·) · T_tag`). The tag costs no committed digits because the site's
table is public. FlashProve does this with a random shift; Jolt's decomposable tables are the
same idea.

### 7.3 What it buys and what it does not

It does not touch committed volume: the same digits, laid out differently. The separation
principle says per-lookup cost is independent of instance count.

It attacks everything else: the IOP (a few large parallel sumchecks in place of ~2,000 small
sequential ones), the opening reduction (one stage, one layout), proof size, verifier time,
prover memory and parallelism across layers, and code complexity.

### 7.4 What does not fit directly

Softmax output and activation-table outputs are not digit-decomposed today (softmax commits only
its remainder digits). They would gain a decomposition, at a commit cost, or keep a short local
chain. The soundness argument is the standard Shout read-value argument that the committed digits
bind the recomposed value; it must be written for the batched setting. A relation compiler that
emits the batched relation from the ONNX graph is real work.

### 7.5 Where it sits

An IOP-side redesign, orthogonal to the commit-volume work in Phases 1–3 of
`docs/measurement-plan.md`, and a candidate for Phase 4. It pairs with the one-commitment,
one-opening item. The IOP and opening stages it targets are about 12 s of the 40 s GPT-2 proof, so
the ceiling is known before anything is built.

## 8. Open directions

- **Rounding placement as a compiler problem.** Given a graph and a bit budget, find where rounding
  must happen and minimise committed digits globally.
- **Attention's per-pair constant.** Sub-quadratic attention is ruled out: the separation paper's
  Theorem 3 shows content selection needs Ω(N²) certified comparisons. Only the constant per pair
  is in play, which is the fused-attention item.
- **Fidelity to the served model.** Quantify and reduce the divergence between the proved integer
  model and the float model (FlashProve 6.8–9.4% of top tokens, Sable 86.3% greedy agreement).
- **Small proofs by recursion.** Wrap through upstream's Akita verifier guest.
- **Zero knowledge.** Upstream's Whiteout stack, when it becomes selectable.

Now requirements more than research, since others ship them: one causal pass over all positions,
memory-bounded proving by segment, committed weights, a published like-for-like benchmark.

## 9. Reads done and still owed

Done: BitZ, SuperNeo, the tensor-train paper, the two range-proof originals, Celer, the separation
paper, the sumcheck paper, Sable. The virtual-tensor soundness argument is written in the
approximate-range reading note; FlashProve's Lean theorem is no longer needed for it, since the
lemma is LNS 2.5.

Still owed:

1. Upstream a16z/jolt #2040 and the binary-field track: whether the LaBinius decoupling is applied
   to an odd prime.
2. Cyclo (ePrint 2026/359): the only benchmark of a dense tiny-digit range-check design.
3. The Whiteout paper, when it is public.

## 10. The whitepaper's claims not to repeat

From the audit of `papers/Jolt_Atlas_Whitepaper.pdf` (February 2026) against the current code:
the seven-stage zkVM-style proof with R1CS and Spartan (the code has none, and the paper
contradicts itself on this between sections 2 and 5.4); BlindFold zero knowledge; HyperKZG, Dory
and BN254 (Akita over Fp128 now); on-chain verification; lossy neural teleportation at a 2^7 scale
with 2^16 tables (exact at 2^14 with 2^18 tables now); the stage profile (commit 3.5 of 38 s then,
25 of 40 s now); the operator list; the DeepProve dismissal; the unreproduced nanoGPT comparison;
"Dory only justified above 2^26"; and any GPT-2 benchmark without sequence length, proof size and
verifier time.

## References

- R. Rinberg et al. FlashProve: an open-source library for provable inference. November 2026.
  https://www.anthropic.com/document/flashprove-technical-report ;
  https://github.com/anthropics/FlashProve (read at commit 9e2bdd8, v0.16.0)
- S. Setty. Magnetar: High-speed post-quantum RV64 zkVM from standard hashes. ePrint 2026/2423.
- S. Setty. BinarySpartan: Spartan over binary fields. ePrint 2026/1656.
- Q. Dao et al. Akita: A high-performance lattice-based polynomial commitment scheme.
  ePrint 2026/1983.
- V. Lyubashevsky, N. K. Nguyen, G. Seiler. ePrint 2020/1448, Lemma 2.5 (the approximate range
  proof). C. Gentry, S. Halevi, V. Lyubashevsky. ePrint 2021/1397, Lemma 3.4.
- BitZ, ePrint 2026/2141. Limber, ePrint 2026/1635. LaBinius, ePrint 2026/2103. Mikros,
  ePrint 2026/2646. Celer, ePrint 2026/1453. A Separation Principle for Lookup-Based zkML, ePrint
  2026/1390. Tensor-train random projections, ePrint 2026/2146. PikkuFold, ePrint 2026/1809.
  Q. Dao et al., ePrint 2026/587. JaggedPCS, ePrint 2025/917.
- Bits Under ZK-LLM, arXiv 2609.36437. Sable, https://github.com/moonmath-ai/Sable.
- DeepProve, ePrint 2026/1112. zkAgent, ePrint 2026/199. OpenLLM, ePrint 2026/1578.
- E. Ben-Sasson et al. On proximity gaps for Reed–Solomon codes. ePrint 2025/2055.
- W. Nguyen, S. Setty. Neo and SuperNeo. ePrint 2026/242.
- G. Arnon, A. Chiesa, G. Fenzi, E. Yogev. WHIR. EUROCRYPT 2025.
- J. Thaler. Time-optimal interactive proofs for circuit evaluation. ePrint 2013/351.
- a16z/jolt pull requests #1911, #1994, #2012, #2040, #2145, #2188, #2379.
