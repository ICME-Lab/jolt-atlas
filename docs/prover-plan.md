# Prover plan

One document for where the proving time goes, what has been done about it, what is left and in
what order, and which strategic questions stay open. Numbers marked *measured* are from the
benchmark machine (`scripts/remote.sh`) on 2026-11-18 unless a line says otherwise; everything
else is an estimate and says so. Timings are comparable only within one machine and session:
the machine was a 16-core Graviton with 30 GB until midday on 2026-11-18 and a 32-core one
with 61 GB after, and each table says which.

## 1. Where the time goes (measured)

**At 16 tokens** (32 cores, `bench/gate.sh`, medians of three, commit `bb50ed8c3`):

| | GPT-2 | Qwen |
|---|---|---|
| prove | 1.75 s | 4.43 s |
| trace execution | about 0.27 s | about 0.55 s |
| commit | 0.14 s | 0.51 s |
| sumchecks (`iop`) | 0.68 s | 1.77 s |
| openings | 0.65 s | 1.40 s |
| verify | 0.14 s | 0.33 s |
| proof | 1.33 MiB | 1.77 MiB |
| peak memory | 2.7 GB | 7.5 GB |

Huge pages for the allocator take a further 16% off Qwen on this machine (5.82 s to 4.90 s
on an earlier commit, kernel time halved) and are not in these numbers (`CLAUDE.md`, "Remote
runs").

**By sequence length** (32 cores, single runs, commit `5f30bbf2a`, about 10% before the
table above):

| Model | Tokens | Prove | Verify | Proof | Peak memory |
|---|---|---|---|---|---|
| GPT-2 | 32 | 3.14 s | 0.25 s | 1.62 MiB | 4.6 GB |
| GPT-2 | 64 | 5.41 s | 0.24 s | 1.66 MiB | 8.0 GB |
| GPT-2 | 128 | 13.34 s | 0.26 s | 2.38 MiB | 17.6 GB |
| Qwen | 32 | 8.34 s | 0.54 s | 2.40 MiB | 13.8 GB |
| Qwen | 64 | 15.62 s | 0.53 s | 2.49 MiB | 25.1 GB |
| Qwen | 128 | 32.29 s | 0.62 s | 3.55 MiB | 46.5 GB |

Proving is close to linear in context: about 100 ms and 140 MB per token on GPT-2 (10 tokens
a second), about 250 ms and 360 MB per token on Qwen (4 tokens a second). The fixed cost is
small, so a longer context amortises almost nothing. Verify time and proof size are nearly
flat. Memory is the limit on context.

**The cores are not the limit; the serial part is.** Qwen at 16 tokens, by thread count
(32-core machine, before the changes of the afternoon of 2026-11-18): 10.9 s at 8 threads,
7.51 s at 16, 6.57 s at 32. Doubling the cores from 16 to 32 bought 14%, and the proof used
about 18 of the 32 on average, with 14% of its CPU time in the kernel (page faults and
memory mapping). Comparing every span at 4 and at 32 threads gives the part of it that does
not shrink (`serial ≈ (8·t32 − t4)/7`); at commit `8b9fb9a79` that estimate was 3.9 s of a
5.4 s proof: sumchecks 1.5 s, openings 1.4 s (the grand product 0.54 s, the reduction
sumcheck 0.29 s, Ligerito 0.21 s), trace 0.63 s, commit 0.40 s. This comparison is how the
next serial piece is found.

**Where the growth is** (16 cores, commit `1dedd8911`; Qwen, traced single runs, 16 against
64 tokens):

| Stage | 16 tokens | 64 tokens | Growth |
|---|---|---|---|
| sumchecks | 3.05 s | 7.96 s | 2.6x |
| openings | 2.54 s | 10.20 s | 4.0x |
| trace | 1.48 s | 4.19 s | 2.8x |
| commit | 0.82 s | 2.92 s | 3.6x |

At 64 tokens the commit and the openings are half of the time, and they scale with the
committed bits. Inside the openings at 64 tokens: the grand product 4.4 s, Ligerito 1.5 s,
building the opening's inputs 1.25 s, the projection 1.0 s, the reduction sumcheck 0.8 s.
Inside the sumchecks at 16 tokens: the op batches 1.3 s, the serial steps between waves
1.15 s (0.66 s of it lookup witness setup), the bucket-value batch 0.67 s, the one-hot
checks 0.28 s.

**What is committed** (`prove --census` at 64 tokens, before the public reshaped output):

| | Qwen | GPT-2 |
|---|---|---|
| range-check digit columns | 91.1% of rows | 92.4% |
| of which 32-bit outputs | 62% | 76% |
| activation lookups | 7.7% | 6.0% |
| softmax, gathers, the rest | 1.2% | 1.6% |

A rescaled product commits 46 bits: a 14-bit remainder and a 32-bit output. The values are
far narrower than their columns. On Qwen 36% of einsum outputs are exactly zero (padding
included; the trace stores tensors padded, so the two are not separated), 83% fit in 16 bits
and 99.8% in 20; no value in either model needs more than 29 bits. Add, Sub and Sum outputs
are an eighth of Qwen's checked cells and about 40% of GPT-2's.

**The projection's proof of work is a lottery per proof.** The grind before the prime is
drawn is 23 to 26 bits, and how long the search runs is fixed by the transcript: a protocol
change that alters the transcript moves it by up to 100 ms either way on GPT-2 with no change
in work (70 ms to 165 ms across the public reshaped output, while every other part of the
opening got faster). Read an openings change of that size against the sub-stage spans before
calling it a regression.

## 2. How we got here

The plan's first measurements were 40.2 s for GPT-2 at 16 tokens on an M3 under Akita, with
59% of it the commit of one-hot digit columns. The thesis then was that proving inference on
this stack is proving values are small, in the most expensive encoding available. What
followed, by spec:

| Spec | What it did |
|---|---|
| `clamp-dense-digit-range-check`, `one-range-check`, `no-scalar-special-cases` | range checks as dense base-4 digits in packed buckets; every remainder bound on one mechanism |
| `one-clamp` | a result outside `i32` is an execution error; nothing saturates |
| `post-loop-batches`, `structural-ops-as-ties`, `log-partition-softmax`, `fused-gelu` | lookups in two batches; Reshape, Slice and Concat as zero-round ties; softmax and GELU as single table ops |
| `stream-proof`, `verification-schedule`, `relations-as-values` | the proof is the transcript; the verifier walks one compiled schedule |
| `bitz-commitment` | BitZ replaces Akita: one commitment of bits, one opening. GPT-2 24.4 s to 9.1 s on the benchmark machine |
| `digits-as-trace` stage 1 | a digit-backed output's claims are checked against its digits, at a point of the node's own |
| `wave-schedule` | the schedule runs in waves, one batched sumcheck per wave (9 for GPT-2 and for Qwen at 16 tokens) |

Without a spec, from 9.1 s to 2.9 s on GPT-2 and 13.1 s to 7.9 s on Qwen: the tracer's einsum
kernels and shared constants; dense groups in the opening reduction; the grand product
(NEON slice kernels, byte tables, in-place folds, a stored layer at a sixteenth of the bits,
skipped empty columns, per-layer histograms); the split-eq unreduced fold for `mk,kn->mn`;
cached activation tables; the output of a node a model output is a reshape of taken as public
(Qwen 6.7% at 64 tokens); the BitZ size cap raised from 2^31 to 2^36 bits.

On the 32-core machine, from 7.06 s to 5.55 s on Qwen and 2.63 s to 2.22 s on GPT-2, with
verify from 0.88 s to 0.54 s and 0.35 s to 0.24 s: the tracer's matrix product tiled over
columns and vectorised, and its rescale, narrowing and elementwise passes in parallel
(`specs/einsum-kernels.md`, step 2); witnesses of set-up relations and table reads built with
their batch; a small table's one-hot scatter sized to its cells; bucket weights and the
committed witness polynomials built side by side; and one unreduced split-eq kernel for
evaluating a tensor of at most 32-bit values at a point, which the matrix product's fold, the
prover's constant ties and the verifier share (steps 3 and 5).

Then to 4.89 s and 1.84 s, with verify at 0.33 s and 0.14 s: a wave's points drawn before
their claims, so its output and remainder claims are computed together (a protocol change);
a minimum task size of 128 for the shared parallel loops, which had split to single items
(verify 37 to 39% faster from this alone); Ligerito's folds in reused buffers and its
queried rows added in place; the grand product's small layers; the tracer kernel reshaped,
its rescale passes fused, and its tables generated once and beside the execution.

Then to 4.43 s and 1.75 s with `specs/deferred-rescale.md`: a chain of products keeps its
exact accumulation as a wide value and is rescaled once. The range buckets hold 41% fewer
bits on Qwen and 33% fewer on GPT-2, the openings stage is 27% and 12% faster, and peak
memory is 12% and 11% lower.

## 3. What is left

**Targets (set 2026-11-17).** GPT-2 at 16 tokens in about 1 s and Qwen in about 3 s on the
benchmark machine. Today 1.75 s and 4.4 s on 32 cores.

Every step is gated by `bench/gate.sh` on GPT-2 and Qwen (`--model models/qwen`) against the
commit it lands on: stage times within 3%, peak memory within 5%, unless a spec states
tighter bounds. Gains below are estimates until a gate replaces them.

### A. Engineering on the current protocol (no spec)

Found by the 4-against-32-thread comparison of section 1; largest first.

1. **The allocator.** Every change that moved work onto worker threads slowed the openings
   stage by 3 to 4% with an identical transcript, and huge pages are worth 16%. Fewer and
   reused large buffers, or an allocator that keeps and reuses them, is the fix; nothing is
   in code yet.
2. **The opening's parts that do not scale** (about 0.8 s on Qwen): the grand product's
   large layers scale 5.5x from 4 to 32 threads, which is memory bandwidth; the reduction
   sumcheck pays about 7 ms a round over some 900 small instances.
3. **The per-node steps still serial** (about 0.35 s on Qwen): claim reductions, mostly of
   axis permutations, table-read setup, and the structural ties that chain one node's claim
   to the next.
4. **The projection's proof of work** is a search whose length is random per proof, 80 to
   240 ms on Qwen, as large as most single optimisations now. A cheaper hash for it, or a
   smaller difficulty from a larger prime interval, would cut both its mean and its spread;
   it changes the protocol.
5. **The grand product's remainder:** write the stored layer already folded, keep the
   accumulators in registers, a prepared fold multiplier.

### B. Committed volume (each needs a spec; this is where long contexts are won)

Commit and openings are half the time at 64 tokens and follow the committed bits; so does
memory. One measured point for sizing: removing 14% of Qwen's bits took 6.7% off the prove
time and 7% off peak memory at 64 tokens.

1. **Per-node output widths** (`specs/per-node-output-widths.md`, proposed, parked). A width
   below 32 bits stated per node at compile time, enforced by the tracer. 15 to 18% fewer
   committed bits on Qwen with classes and margin, 23% at exact widths: an estimated 7 to 9%
   of prove time at 64 tokens. It changes the contract (a value above its node's width is an
   execution error) and needs a calibration tool; parked because the gain does not justify
   that on time alone.
2. **The row check on wide rows.** After every commitment a tensor depends on, draw
   128 + ⌈log₂ rows⌉ combinations of each row with coefficients 0 and ±1, commit the digits
   of the combinations and range-check those in place of every element (LNS 2020/1448,
   lemma 2.5: slack 2, error 2^-n). Element digits disappear on wide rows; remainders stay
   exact. It is the one known change that removes most of the 32-bit columns and not a
   fraction of them. Prototype first as a standalone relation test on one wide einsum
   output, with the negative test that draws the combinations before the remainder
   commitment.
3. **Fewer rescales: landed** (`specs/deferred-rescale.md`) for products read by products
   or sums. Left: deferred `Square` and `Cube`, wide axis sums, a fused LayerNorm with one
   Rsqrt relation, and why the commit stage did not follow a 41% fall in bucket bits.
4. **Dropping the output check of Add, Sub and Sum.** Their outputs are exact functions of
   checked operands. Worth it on GPT-2 (about 40% of checked cells), not on Qwen (an
   eighth); needs a statement of what bound consumers then rely on.
5. **Activations and softmax.** The activation clamp fused into the table lookup, rebase
   digits as lookup addresses, fused attention. Lookups are 6 to 8% of rows at 64 tokens;
   softmax grows with the square of the sequence length, so measure at 256 tokens before
   sizing this.

### C. One sumcheck for the model (`digits-as-trace` stages 2 and 3)

Digit-backed relations stack across identical layers (twelve in GPT-2, twenty-four in Qwen):
one instance with a layer axis per shape, the weights one polynomial the verifier evaluates
once. Targets the op batches and the bucket-value batch (2 s of Qwen's 3.1 s), proof size and
verifier time. Needs a memory plan and a relation compiler.

### D. Beyond the current machine

- **Memory per token.** 370 MB per token on Qwen caps the context before time does.
  Nothing is planned yet; the committed-volume items help in proportion.
- **A GPU backend.** Estimated 5 to 10x end to end on one H100 against sixteen cores, when
  Qwen proved 2.7 tokens a second on them: the trace and the hashing gain the most, the binary-field arithmetic of the opening the least (estimated assuming no
  carry-less multiply on a GPU; NVIDIA GPUs since Ampere have one, `clmad`, from CUDA 13.3), the many small nodes and the transcript little. Nothing measured. The first step is
  one kernel, the grand product's inner loop, on a small NVIDIA instance behind a feature
  flag; CUDA and not Metal, since a Mac's numbers say little about a deployment.
- **Sumcheck research** is not expected to give 2x by itself: the stage is a third of the
  time, and half of it is not round arithmetic. A cheaper rescale argument (fewer than 46
  committed bits per product) is the research question that would.

### Order

A in its listed order, as far as each needs no decision. Then the prototype of B2, since it decides whether B1
is worth its contract change at all. C after B. D's GPU kernel whenever an instance exists.

## 4. Positions (research, condensed)

- **Magnetar** (Setty, ePrint 2026/2423): binary field, packed bits, constraints instead of
  lookups, one commitment and one opening. Its field cannot count, so its arithmetisation does
  not transfer; its layout discipline (commit once, derive everything; one opening) does, and is
  what `bitz-commitment` and `digits-as-trace` took. Folding the Hamming-weight check into read-raf transfers for read-only lookups.
- **FlashProve** (Anthropic/EPFL, Nov 2026): Goldilocks with a cubic extension, Reed–Solomon at
  rate 0.104, committed byte-cell trace, per-row approximate range check (B2's source),
  exact rounding by one 2^16 lookup on a virtual key, fused attention (B5), partial-function
  semantics, 12-bit activations, 6.8–9.4% top-token divergence, 18.78 MB proofs, 21 s verifier,
  Llama 4 Maverick at 111 tok/s on an H100 at batch 24. Hardware explains 10–100x of the ~10^4
  gap in parameters × tokens/s; the rest is encoding and precision.
- **Sable** (MoonMath, 3 Nov 2026): KoalaBear + LogUp-GKR, BaseFold/BLAKE3, every rounding exact
  at one committed element plus 2–3 lookups, Qwen2.5-32B at 4K in 92.5 s with 135 MiB proofs,
  partial semantics, not ZK. **Attestable:** no paper or code; ZK, ~5 MiB proofs, within ~1.3x of
  Sable's wall clock. Positions still unclaimed: small proofs with an open design, ONNX
  generality, fidelity to the served model.
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

1. **Decided 2026-11-16: BitZ, owned, in place of Akita** (`specs/bitz-commitment.md`). Every
   committed column is bit planes of one vector, with one commitment and one opening. Zero
   knowledge is set aside; it returns, if it does, as an in-house construction. Recursion is
   not weighed.
2. **Decided: a result outside `i32` is an execution error** (`SemVersion::V3`,
   `specs/one-clamp.md`). A narrower declared width (B1) keeps this form.
3. **Base 4 digits.** Under BitZ a digit column is committed at two bits a coefficient, so
   the base no longer sets the commit cost; it sets the number of columns in the value
   sumcheck. Not reopened.
4. **Open: whether an operator-stated width is an acceptable contract** (B1), and whether the
   row check's bound without saturation is enough for every op that checks its output (B2).

## 6. Not pursuing

- Lowering the global scale or 12–16-bit activations (the honest values need 17–29 bits at
  2^14; the saving is in the encoding, not the scale); 8-bit one-hot digits.
- A FlashProve-like committed-trace hash architecture for general ONNX: it means leaving Fp128,
  megabyte proofs and exact semantics, in a space several teams occupied within weeks.
- Grand products on virtual keys for the remainders (D4): dominated by dense digits.
- Small-value sumcheck, univariate skip or Toom-Cook inside the einsum contraction; committing
  weights; Freivalds-style projections; cross-node batching of contractions (C does this
  structurally instead).
- Piecewise-polynomial or holonomic activations, half-tables, mantissa or Newton rsqrt, tabled
  rsqrt, base-2 or polynomial softmax, prover-chosen softmax shifts: exactness and uniqueness of
  outputs.
- Sub-quadratic attention (ruled out by theorem).
- One range check for all the remainders, or none. The bound `rem ∈ [0, 2^S)` is what makes the
  quotient of `acc = out·2^S + rem` unique: without it any `out` has a balancing `rem`. Each
  remainder is a separate free value and needs its own bound; the claims are already proved
  together in the post-loop buckets, and what is paid per node is the committed digits.
- No rescaling. Values at twice the scale cannot be multiplied again inside the field, and
  every table read needs its index at the model's scale. Rescaling less often is B3.

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
