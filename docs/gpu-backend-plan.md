# GPU backend plan

The over-arching plan for a second prover: GPU-first, transformer-specific, with no dependency
on the code in this repository today. It fixes the goal, the design rules we take from the CPU
prover, the decisions still open and what settles each, and the order of work. It is not a
spec: each area below becomes one or more specs (`/new-spec`) when its phase starts. Written
2026-11-25. Sizing numbers are arithmetic from the model's published configuration; nothing in
this document is a measurement unless it says so.

## 1. The decision

- The code in this repository today becomes the **CPU ONNX backend**. It is kept working and
  frozen in scope: bug and soundness fixes, no new features. It stays the reference for exact
  fixed-point semantics and for the numbers we compare against (Qwen2-0.5B at 16 tokens in
  2.7 s on 32 cores, `README.md`).
- A new **GPU backend** is built in `gpu/`, as its own Cargo workspace with no dependency on
  any crate here (section 4.1). We write and own all of it. It has no ONNX frontend: each
  supported architecture is a model definition in Rust over a small set of shared layer
  proofs, the approach PR #409 illustrates on CPU and DeepProve takes.
- The first model is Qwen2-0.5B, which the CPU backend already proves; the target is a
  7-billion-parameter Llama-family decoder built from the same block.

## 2. Goal

Prove, on one data-center GPU, that a 7B Llama-family model produced a given continuation of a
given prompt, at a rate stated in **tokens per second at 1,000 to 4,000 tokens of context**,
with a proof a CPU verifies in well under a second per thousand tokens.

The target rate is set at the end of phase 0, from a cost model built on measured kernel
rates. It is not set here.

**Not goals for the first target:** ONNX or any general graph; models other than decoder-only
transformers; zero knowledge; more than one GPU; non-NVIDIA GPUs; training.

## 3. What hindsight says

Each rule is something the CPU prover taught us at a cost.

| Rule for the new backend | What it cost us here |
|---|---|
| Fix the numeric contract and the field together, first | Fp128 was chosen for Akita; Akita left, the field stayed, and it is the worst field for a GPU |
| Give every tensor its own declared bit width from day one | Everything is `i32`; 93% of committed rows are range-check digits, and narrowing widths afterwards (`specs/per-node-output-widths.md`) was parked as too invasive |
| The model's structure is code, not data | 34 general ops, a runtime claim accumulator, shape inference and an 84,000-line codebase; PR #409 proves the same models in about 6,500 lines |
| Public structure is never proven | RoPE's sines and cosines and the embedding read of public tokens were general `Sin`, `Cos` and `Gather` proofs with committed addresses |
| The verifier and a slow reference prover exist before any fast prover | The verifier was carved out of the prover after the fact (`specs/verifier-audit-simplification.md`) |
| Padding is part of each layer's definition, with a test that padded and unpadded execution agree | Broadcast filled padded lanes and every LayerNorm on a non-power-of-two width proved the wrong variance, unnoticed until 2026-11-19 |
| Generic soundness tests from the first relation: mutate every proof value, tamper the witness, forge every claim | These sweeps found real bugs here, and were added after the ops they test |
| Inference and proving are one program | Trace execution is 23% of Qwen's prove time, run as a separate CPU pass |
| Measure before choosing; keep a gate that compares commits on one machine | The per-instance sumcheck overhead we planned a rewrite around was the profiler |
| Every sumcheck is a declared relation and the verifier is derived from it | Hand-written verifiers had to be kept in step with their provers by hand until the relation layer replaced them |

## 4. Shape of the new backend

```
model definition (Rust)        Llama block: RMSNorm, attention + RoPE, gated MLP, residual
        │
integer inference engine       runs the model on the GPU in exact integers, keeps the witness
        │
prover (GPU)                   commit the witness → one batched sumcheck → one opening (4.2)
        │  proof bytes
verifier (CPU, own crate)      compiled from the model definition; reads the proof once
```

Three properties hold the design together:

1. **One protocol, two provers.** A reference prover on the CPU, written for clarity, and the
   GPU prover implement the same protocol and must produce the same proof bytes for the same
   input. That is the GPU prover's main correctness test, and it lets protocol work proceed on
   a laptop.
2. **Every GPU kernel has a CPU twin behind one backend trait** and is tested against it.
3. **The verifier is its own crate** with no prover and no GPU dependency, written and
   documented for audit before the GPU prover exists.

PR #409 shows what ONNX-free means: a small layer set (attention, MLP, LayerNorm and RMSNorm,
softmax, GELU and SiLU, residual), claims passed as Rust values, a public RoPE table, direct
embedding reads, weights from safetensors. The new backend takes that shape and none of that
code: everything in `gpu/` is written here and owned end to end.

### 4.1 Layout and build isolation

`gpu/` must never make a build slower, or be made slower, by the CPU backend:

- `gpu/` has its own `Cargo.toml` with its own `[workspace]`, its own `Cargo.lock` and its own
  `target/`. The root workspace does not list it, and no path dependency crosses the boundary
  in either direction. A build inside `gpu/` compiles nothing of the CPU backend.
- CI jobs are filtered by path: a change under `gpu/` does not run the CPU backend's jobs, and
  a change outside it does not run the GPU backend's.
- The CUDA kernels sit behind a feature, so the reference prover, the verifier and the tests
  build and run on a laptop with no CUDA toolkit.
- The crates inside `gpu/` are named and split by the phase 2 specs. The verifier is one of
  them from the start.

### 4.2 The skeleton of the proof

The whole proof is two commitments, one batched sumcheck and one opening. Everything else in
this plan fills in one of these steps.

Once per model:

0. **Commit the weights.** The verifier keeps the commitment, not the weights.

For each proof:

1. **Run the integer model** and keep the witness: every tensor the verifier cannot
   recompute. That is each op's output, each rescale's remainder and each lookup's address.
2. **Commit the witness.**
3. **One batched sumcheck.** The verifier draws one random point, and the sumcheck proves
   every op's relation at that point at once:
   - a matrix product, `Σ_k W[i,k]·a[k] = 2^s·q[i] + r[i]`, with `q` the rescaled output and
     `r` its remainder;
   - a lookup, `b[i] = T[a[i]]`;
   - an addition or a residual, which is linear and costs almost nothing.

   It ends in a handful of claims: the weights' polynomial has this value at this point, the
   witness's polynomial has that value at that point.
4. **One opening**: a joint proof that those claims agree with the two commitments.

The verifier checks the sumcheck's rounds, evaluates the public parts itself (the prompt, the
output, the tables) and checks the opening.

**Where the prover's time goes.** For a matrix product the sumcheck is about 14 rounds over
the layer's width, however many weights or tokens there are. The work is the two weighted
passes before it, one over the weights and one over the activations: the weight pass and the
witness pass of the floor (5.7). A prover built on this skeleton is the floor plus what its
commitment costs.

**What the count depends on.**

- **One sumcheck needs every tensor between ops committed.** Then every relation is local
  and independent of the others, and they batch into one. With less committed, the layers
  are proven in sequence, each sumcheck feeding the next, as the CPU backend does; its serial
  steps between sumchecks are 1.15 s of the 3.05 s it spends in sumchecks on Qwen at 16
  tokens. This is open decision 9.
- **One opening needs both commitments under one scheme.** If the weights and the witness
  are committed differently, as 5.6 hypothesises, there are two openings.
- **The claims land at different points.** A short second sumcheck brings them to one point
  per commitment, unless the opening takes several points.
- **Range checks** are free where the witness is committed as bits, and are further terms of
  the batched sumcheck where it is not.

## 5. Decisions

### 5.1 Made (2026-11-25)

| Decision | Outcome |
|---|---|
| Weights | **Committed** once per model and opened in every proof. The verifier holds the commitment, not the weights |
| First model | **Qwen2-0.5B**: the same block as the 7B targets, and the CPU backend's 2.7 s to compare against. Which 7B follows is still open (5.2) |
| Where it lives | **`gpu/` in this repository**, its own workspace (4.1) |
| Ownership | **All code written here.** No dependency on this repository's crates, on upstream Jolt, or on outside proof-system crates |
| Not the commodity stack, and not a copy of anyone's | **We do not build on a small prime field with a FRI-family commitment** (Goldilocks or a 31-bit field with WHIR, BaseFold or FRI); Sable, FlashProve and SP1 already do. Nor do we adopt Magnetar or any other system as it stands. The core of the proof system is chosen by a search for the best fit to our workload (5.5), with the commodity stack as the bar to beat |

### 5.2 Open

Each has a recommendation and the measurement or check that settles it. None should be decided
by argument alone.

| # | Decision | Options | Recommendation | Settled by |
|---|---|---|---|---|
| 1 | Numeric representation | (a) 32-bit fixed point everywhere, as today; (b) per-tensor widths, e.g. 8-bit weights and 16-bit activations; (c) floating point | (b), with the widths chosen by experiment. (c) has no affordable proof. See 5.3 | Accuracy experiment, phase 0 |
| 2 | Where the arithmetic runs | A prime field (Fp128 as today, a 127-bit Mersenne prime, a smaller prime with an extension); the integers mod `2^k` (read and dropped, 5.5); a binary field | Not decided; the search of 5.5 decides. It must be a ring that counts: a sumcheck for a matrix product touches each weight once only where integers add and multiply. In a binary field each multiply-add is a few hundred bit constraints (an estimate), and Qwen2-0.5B at 16 tokens is several billion multiply-adds against under half a billion weights. The incumbent is a prime field larger than our values, with a reduction that is cheap in machine words | The search of 5.5; GPU sumcheck-round rates per candidate; the widths of decision 1 (an 8-bit by 16-bit product over 11,008 terms needs about 38 bits) |
| 3 | How small values are committed | Packed bits in a binary field with ring switching and Ligerito (BitZ, Magnetar); integers under a code over the rationals (Zinc, Zinc+; read and dropped, 5.5); a pay-per-bit lattice scheme (SuperNeo, LaBinius; read and dropped, 5.5); a linear or quasi-abelian code over the arithmetic's own ring (Rotor, dropped; Quasar, unread) | Not decided, but packed bits stand after the reading of 5.5: nothing read commits a small value as cheaply at 120 bits with a small proof. They price a value by its bits, give range checks for free and beat Akita here by 14 to 15 times (PR #409); NVIDIA GPUs since Ampere have a carry-less multiply, so the field is no obstacle there. Their cost is the bridge to a ring that counts (the grand product, our largest opening cost today), and the open work is that bridge (5.5) | GPU rates for binary-field multiplication with `clmad`, for the encoding and the hash tree, and for the bridge; the Mersenne pair of 5.5 written out and checked |
| 4 | The 7B model | Llama-2-7B; Mistral-7B; Qwen2.5-7B; Llama-3.1-8B | One config-driven block serves all four, so this is a licence question: Llama 2 is under Meta's community licence and gated; Mistral-7B and Qwen2.5-7B are Apache 2.0 | Licence review, before phase 4 |
| 5 | GPU kernels | CUDA through a Rust backend trait, with which toolchain (C++ kernels over FFI, or Rust-side code generation) | CUDA C++ kernels behind the trait; benchmark an existing proving library only as the bar to beat | Phase 0 |
| 6 | Statement | What is public (prompt, output, sampling seed); greedy or sampled decoding | Public prompt and output, greedy decoding first; one proof covers a whole sequence, since a causal model's forward pass over N tokens checks every next-token choice | Product owner |
| 7 | Security level | 128 bits as today; 100 bits as the Gemma result; proven or conjectured soundness for the commitment | Decide explicitly and state it next to every benchmark | Phase 0 spec |
| 8 | Padding | Pad every dimension to a power of two; lay tensors out without it | Do not assume power-of-two padding: Llama's MLP width 11,008 pads to 16,384, half as many weight cells again in every MLP matrix | Phase 0 cost model |
| 9 | What the witness holds | (a) Every tensor between ops, so one batched sumcheck proves every relation (4.2); (b) only what nothing else establishes, with the layers proven in sequence, as the CPU backend does and as 5.4 took from Magnetar | Leaning to (a), not decided. It commits roughly 16 more bits per activation (an estimate) and removes all depth from the proof; on a GPU more bits run in parallel and depth does not | The committed bits per activation under each, from the floor calculator; GPU rates for the commit pass against the serial steps |

### 5.3 On quantized weights

A proof system works over a finite field, so the model must be integers in some form. The
choice is not whether to quantize but how narrow, and who defines the result.

- **Today's contract (32-bit fixed point)** is already quantization, to a format nobody serves.
  It is exact with respect to our own semantics and close to the float model.
- **Narrower widths** are the largest single cost lever we know: they shrink the committed
  weights four-fold at 8 bits, shrink every range check, and let values fit a GPU-friendly
  field. The cost is accuracy, and it is measurable before any prover exists.
- **The statement stays honest either way** if our integer inference engine is what serves the
  tokens: we prove the model that ran, not an approximation of another one.

So the plan does not commit to a width. It commits to making width a declared property of each
tensor, and to choosing the defaults from an experiment: perplexity of the integer model
against the half-precision original at several weight and activation widths (32/32, 16/16,
8/16, 8/8), on the target model.

### 5.4 What Magnetar shows, and what it does not

Magnetar (Setty, ePrint 2026/2423) is a zkVM for RV64I built entirely over `F_2` and
`F_{2^128}`: BinarySpartan for the constraints, Twist and Shout over binary fields for memory,
and MosaicPCS over Ligerito for the one commitment. On 12 cores of an M4 Max it proves 7.2 to
8.7 million cycles a second at 2^22 cycles, with 600 KiB proofs verified in 11 to 13 ms.

**What we take:**

- **One polynomial for everything committed together.** Every committed vector sits on an
  aligned subcube of one polynomial, so a claim on a part lifts to a claim on the whole by
  appending fixed coordinates, with no adapter sumcheck. `specs/bitz-commitment.md` already
  took this. For us that is one polynomial for the weights and one for the witness (4.2).
- **Commit only what nothing else establishes.** A segment another argument proves correct is
  read through an evaluation claim and never committed. For us this would mean no
  intermediate tensor between layers is committed; whether we take it is open decision 9,
  since it is what puts the layers in sequence.
- **No argument whose soundness error grows with the instance.** A permutation or grand-sum
  argument loses about the instance's size over the field's; a sumcheck loses its logarithm.
  That is how Magnetar keeps 120 bits in a 128-bit field while SP1 Hypercube and RISC Zero,
  with such arguments over roughly 124-bit extensions, target 96 to 100. Our lookup argument
  follows Shout, not LogUp, for this reason, and every argument we add, BitZ's bridge
  included, states its error bound against it.
- **Provable security from standard hashes**: no algebraic hash, and no conjectured
  proximity gap for the code. This is the bar for open decision 7.
- **Two relations over one witness**: bit constraints where bits are cheap and packed
  field constraints where words are, tied by the single commitment.
- **Recursion as the answer to memory**: prove chunks and compose the proofs, cheap when the
  verifier is hashing and binary-field arithmetic. Magnetar proposes this and has not built
  it; it is an option for the 7B model's memory bound, not a plan.

**What does not transfer:**

- **Its arithmetisation.** RV64I has no multiply instruction, and Magnetar's adder is bit
  constraints. Our workload is integer multiply-add, by the billion (open decision 2).
- **Its speed, to a GPU, unmeasured.** Every figure is from a CPU, and the paper measured SP1
  on that CPU too, not on the GPUs SP1 is built for. The field itself is no obstacle: NVIDIA
  GPUs since Ampere have a carry-less multiply (`clmad`, CUDA 13.3, July 2026), with which
  NVIDIA reports a sumcheck over `F_{2^128}` 3 to 4 times faster than bit-slicing on an RTX
  5090 and up to 13 times on a B200.

### 5.5 The search for something better

Open decisions 2 and 3 are one question: what is the best core for proving integer matrix
products and lookups over tiny values on a GPU? Nobody's published system is built for that.
The search runs in phase 0 and is judged against these, in order:

1. **It counts.** A matrix product costs one ring operation per weight, not per multiply-add.
2. **Small values are cheap to commit**: priced by their bits, or so few are committed that it
   does not matter.
3. **Its arithmetic is the GPU's own**: a few 32- or 64-bit multiplies per operation with a
   cheap reduction, or carry-less multiplies.
4. **120 bits or more, provable**, with losses that grow with the logarithm of the instance.
5. **A fast verifier and a small proof.** This is where the commodity stack is weakest and
   where we mean to differ.
6. **Two commitments and one opening** (4.2): the weights once per model, the witness once
   per proof, and one joint opening for both. Two openings are accepted only if committing
   the two differently is measured to be faster overall (5.6).

**The bar** (the commodity stack on one GPU): Sable proves 4,096 tokens of Qwen2.5-32B in
92.5 s on an RTX PRO 6000, at a soundness error of 2^-103, not zero-knowledge, with a verifier
that takes 19 to 56 s on one core. FlashProve reports about 110 tokens a second for a 400B
mixture-of-experts model on an H100, with 19 MB proofs and a 21 s verifier. Both are their
authors' figures.

**Candidates.** Every paper below has been read in full except Quasar, which is known from
its abstract.

| Candidate | The idea | What is known | What we must learn |
|---|---|---|---|
| Integers mod `2^k` (Rotor, ePrint 2026/2175; Nenya, 2026/2672) | Prove directly over machine words | **Read in full and dropped as a core** (see below) | Nothing further; two ideas are kept |
| Bits in `F_{2^128}`, a prime field for the arithmetic (BitZ; our CPU backend) | Pay per bit to commit; bridge to a field that counts by a grand product | Qwen2-0.5B in 2.7 s on 32 CPU cores. The bridge is 0.6 s of a 0.85 s opening | The bridge's and the prime field's cost on a GPU; whether a cheaper bridge exists |
| Bits in `F_{2^128}` throughout (Magnetar) | Never leave the binary field | 7 to 13 million RISC-V cycles a second on 12 CPU cores at 120 provable bits | Whether any argument makes integer products cheap there; without one it fails criterion 1 |
| Packing by the field trace (Frigo and shelat, ePrint 2026/2770) | A claim on bits packed `m` to an element of `GF(2^m)` becomes one linear relation on the elements, by the trace's linearity alone; error 2^-m, no sumcheck, any `m`, any coefficients | **Read.** It replaces ring switching, which is not where our time goes, and its gain is zero knowledge: no quadratic constraint in the commitment. It needs a commitment that answers a general linear query | Nothing for speed. Kept for when zero knowledge returns, and for packing in a field whose degree is not a power of two (the Mersenne pair below) |
| Integers committed directly (Zinc, ePrint 2025/316; Zinc+, 2026/855) | Encode integers with a code over the rationals; check the constraints modulo a random prime | **Read in full and dropped** (see below) | Nothing further |
| A code over the arithmetic's own field (Quasar, ePrint 2026/1839) | Field-agnostic commitment from quasi-abelian codes; implemented over the 127-bit Mersenne prime | GPU encoding 33 to 148 times a 32-thread CPU; 100-bit security | Cost per committed small value, since it pays per element; whether 120 provable bits is reachable |
| Pay-per-bit lattice commitments (SuperNeo, ePrint 2026/242; LaBinius, 2026/2103) | Commit small values cheaply under Module-SIS | **Read in full and dropped as a core** (see below). Akita, not pay-per-bit, was 14 to 15 times slower than BitZ here | Nothing further for SuperNeo; LaBinius is worth a second look if its verifier is fixed |

A candidate is dropped when it fails a criterion on paper, and prototyped at kernel level on
the GPU when it does not. The search ends with one core written into the protocol spec.

**Dropped: the integers mod `2^k`** (read 2026-11-25).

- **The arithmetic is not word-sized where it matters.** Rotor gets soundness from a gap of
  bits between the statement's ring and the ring the protocol runs in. Made non-interactive,
  that gap is about λ plus a term in the number of rounds for the first and largest round,
  and about λ per round in all: by its Lemma 4, some 300 to 400 bits in the first round at
  120 bits of security over 2^30 values, and thousands by the last. Nenya runs in a Galois
  ring of degree 128 with 32 values packed to an element, and measures one such
  multiplication at 1.2 times the speed of the 32 multiplications in a 31-bit field's
  extension that it replaces (0.39 times at 64-bit words). A 127-bit prime field is cheaper
  than either.
- **Their proofs are large and their verifiers slow.** Both commit in the style of
  Brakedown, with proofs that grow as the square root of the witness. Nenya reports 52 to
  237 MB and a 1.8 to 7.1 s verifier at 2^25 constraints, at 100 bits, on 192 virtual CPUs.
  Rotor has no figures: its evaluation was withdrawn when its parameters changed.
- **What they are for, we do not need.** Their gain is arithmetic that wraps at `2^k` for
  free. Our contract has no wrapping: a value outside its width is an error. A ring only
  has to be larger than our values.

Two things are kept. Rotor's first analysis broke under Fiat–Shamir, because a prover can
retry a round until its challenges favour it and the gains add up across rounds; any argument
of ours whose error is not a per-round field fraction is checked against that attack. And
its compatible commitment, one committed vector answering both integer queries and
binary-field queries on its bits, is the cleanest statement of what a bridge must do, though
its own construction costs a word of codeword for every committed bit.

So the counting ring that fits a GPU is a prime with a reduction that is cheap in words (a
Mersenne prime), not a power of two, and the search turned to the other half: committing bits
cheaply and bridging them to that prime.

**Dropped: committing integers directly** (Zinc and Zinc+, read 2026-11-25).

- **Zinc is theory only.** It has no implementation, no analysis of the non-interactive
  setting, and an opening in the style of Brakedown. Its code multiplies every committed
  value by integers of about 4λ bits (2λ conjectured), so a codeword symbol is hundreds of
  bits however small the value: the opposite of paying per bit.
- **Zinc+ is built for small traces.** It replaces that code with a Reed–Solomon transform
  over a 16-bit prime run without reduction, and it is implemented, at a 100-bit target. Its
  figures stop at 2^18 elements. Encoding 2^18 polynomials of 32 bits takes 177 ms on a
  MacBook Air M4 with all cores, about 47 million bits a second, where packed-bit commitments
  run at about 4 billion bits a second on one core (Ligerito and BaseFold at rate 1/2 in
  LaBinius's Table 1; both rates are our arithmetic from the papers' tables). A bit costs
  about a third of a 32-bit integer there, not a thirty-second, and that each one is a bit is
  a separate sumcheck. The opening is in the style of Ligero, one per claim, and the authors
  call it large.
- **Neither proves over a fixed prime.** The sumcheck runs modulo a random prime of more
  than λ bits (about 192 in Zinc+'s implementation). The authors' own open question 7.5 is
  our case, a large prime with witness values far below it.

Nothing is taken from them that we lack: the projection onto a random prime is the step BitZ
already has.

**Dropped: the lattice commitments** (SuperNeo and LaBinius, read 2026-11-25).

- **SuperNeo is a folding scheme without an opening.** It has no implementation, no
  measurement, no proof size and no final proof: succinctness is left to a second proof
  system that the paper does not construct. Committing a bit costs 960 field additions (our
  count from its parameters, 15 ring elements of degree 64).
- **LaBinius is fast and does not bridge.** On one core it commits 2.7×10^9 bits in 636 ms
  and opens them in 167 ms: per bit, level with BaseFold and Ligerito at rate 1/2 to commit,
  and five times faster than Ligerito to open. But its claims are in a binary field only, so
  it would replace our encoding, hash tree and Ligerito and leave the bridge where it is. It
  targets 100 bits. Its proof is 2.9 MiB at that size and grows as the square root, or 83 KiB with a
  713 ms verifier.

**What the reading leaves standing.** Nothing read commits small values more cheaply than
packed bits at 120 bits with a small proof, and nothing read has a cheaper bridge than
BitZ's. Among published schemes taken as they are, the incumbent of decision 3 stands. That
is a comparison with what we have, and it does not show the incumbent is right for committed
weights: section 5.6 starts from the job instead and reaches a different hypothesis.

**Two findings about the bridge.** Both are ours, from our own code and not from a paper.
Neither is reviewed, and the first is not written out as a protocol.

1. **A Mersenne pair removes the projection.** BitZ proves a sum in the exponent of a
   generator of `F_{2^128}`'s multiplicative group, whose order `2^128 − 1` is a product of
   nine small primes. So it lifts the claim to the integers, reduces it modulo a random
   prime small enough that no column's sum wraps (below `2^128` over the row count), and
   buys back the projection's error with proof of work. The multiplicative group of
   `F_{2^127}` has prime order `2^127 − 1`. If the counting field is that Mersenne prime,
   which criterion 3 favours anyway, and the grand product runs in `F_{2^127}`, then
   exponents add modulo the counting field itself. There is no lift, no random prime, no
   primality test in the verifier, no projection error and no proof of work for it, and the
   split into rows and columns is no longer tied to soundness. Checked here: `2^127 − 1` is
   prime and `x^127 + x + 1` is irreducible over `F_2`, so reduction is by a trinomial.
   - *What it does not remove:* the grand product. On Qwen at 64 tokens that is 4.4 s of a
     10.2 s opening, and the projection 1.0 s.
   - *Open:* the commitment must open at points of `F_{2^127}`, a field with no tower and a
     degree that is not a power of two. Packing 127 bits to an element leaves one position in
     every 128 unused and changes the layout; packing 64 doubles the hashing. Every
     challenge in a 127-bit field is one bit short of the 128-bit one. And prior
     work may exist: a search result says the BitZ paper sets itself against an earlier
     scheme limited to this prime, which we have not found.
2. **The bridge's proof and verifier grow as the square root of the committed bits.** The
   proof carries one 16-byte sum per column, with as many columns as rows, and the verifier
   does one exponentiation per row and per column. The CPU backend's proof sizes on Qwen
   (1.77, 2.40, 2.49 and 3.55 MiB at 16, 32, 64 and 128 tokens) fit column sums of 0.5, 1, 1
   and 2 MiB, and its verify time (0.23 s to 0.62 s) grows by about the square root of 8.
   At 10^11 committed bits that is 4 to 8 MiB of column sums. This bears on criterion 5 and
   is to be measured, not inferred. A free split, as the Mersenne pair gives, trades proof
   bytes for verifier exponentiations but keeps their product.

**Next.** Write the hypothesis of 5.6 out as a protocol with its soundness argument, for
review; beside it the Mersenne pair, as the packed-bit alternative for the witness; read
Quasar, and the BitZ paper again for the earlier scheme; then the GPU rates of 5.7 for each
candidate's arithmetic and for the hash tree.

### 5.6 The commitment from first principles

Section 5.5 compares published schemes with the one we have. This section starts from the
job instead and assumes no scheme. It is a derivation of ours, written on 2026-11-25. Its
conclusion is a hypothesis, to be written out and sent for outside review before it decides
anything.

**The job.** Bind the prover to a vector of small integers. Later, for a point `r` drawn
after the binding, convince the verifier of one linear claim `Σ c_i·v_i = y` over the large
prime field the sumchecks run in, where the `c_i` are the tensor of `r`. Nothing else is
asked of a commitment here: one claim per proof, no zero knowledge, no recursion.

**Two kinds of data ask for different things.**

| | Weights | Witness |
|---|---|---|
| Size (7B model) | 6.7×10^9 values of about 8 bits | about 2×10^9 values per 1,000 tokens, plus lookups |
| Bound | once per model, offline | in every proof |
| Opened | in every proof, at a fresh point | in every proof |
| Must be proven in range | no: whatever was committed is the model, and a range can be shown once when it is committed | yes: remainders, digits and addresses are sound only in range |
| What its cost is | the opening, since the binding is paid once | the binding and the opening |

**The work nobody avoids.** One pass over the witness to bind it, and one pass over each
vector to form the claim: a multiply of a small value by a field element per value. All else
is the price of a design choice.

**How a prover can be bound.** Two ways are known at this scale.

- *Hash a redundant encoding.* The code must be linear in one characteristic, and that
  forces a choice. In characteristic 2 a bit costs a bit, but the claim is in the wrong
  field and a bridge pays for it, per bit, in every proof. Over a small prime there is no
  bridge, but an 8-bit value costs a 31-bit symbol. Over the integers the symbols grow
  (Zinc+). These are BitZ, the commodity stack and Zinc+: the three corners of one trade.
- *A compressing linear map that only short vectors can collide under* (a lattice
  commitment). The prover opens a short integer combination of its vector in the clear, and
  the verifier checks it against the commitment by linearity. What is opened is true over
  the integers, so it speaks every modulus at once: no bridge, no encoding, no hash tree.
  The modulus of the map is free of the claim's field and can be tiny (LaBinius uses primes
  of 14 and 15 bits, a GPU's own word size). The prices: a hardness assumption, not hashes
  alone; a commitment and proof that grow as the square root; and no range for free.

**Where the incumbent's habits do not follow from the job.**

1. *Paying per bit is the wrong unit for the weights.* A bridge runs over every committed
   bit in every proof: 5.4×10^10 bits of weights, against 6.7×10^9 weights. BitZ was chosen
   when the weights were public and only the witness was committed.
2. *One commitment and one opening* came with the packed-bit layout. Committed weights
   already make two commitments, one per model and one per proof (criterion 6 now says so),
   and nothing in the job says they must be the same scheme; only a measured gain justifies
   the second opening that two schemes cost.
3. *Range for free* is worth something only for the witness.

**The hypothesis.**

- **Weights under a linear-map commitment, opened in one shot.** Per proof: one pass forming
  the claim in the large field and one pass forming a short combination in machine words,
  with nothing recomputed from the binding. At 6.7×10^9 weights the proof would be a few
  megabytes and the verifier's work a few matrix products over a small modulus. That is at
  the floor of 5.7 for the weight pass.
- **Witness: undecided between two.** The same commitment with a range sumcheck over its
  bits, or packed bits with a bridge. Both pay a few field multiplications per bit for the
  range; which is cheaper on the GPU is a measurement.

**What is not known.**

- LaBinius proves this mechanism for claims modulo 2 only, and says its extraction extends
  to other primes under two conditions on the ring and the challenge set. A large prime is
  new work, and it is the piece to send for review.
- Parameters at 120 bits for 8-bit coefficients: the paper's are for bits at 100.
- Every rate on a GPU. Akita, a lattice scheme, lost to BitZ on our CPU by 14 to 15 times.
  The differences claimed here (a tiny modulus, a one-shot opening, binding done offline)
  must be measured, not argued.
- Whether the assumption is acceptable. Criterion 4 says provable, and the bar of 5.4 is
  security from standard hashes alone.

### 5.7 The floor, and measuring against it

A protocol that met every criterion of 5.5 perfectly would still have to do five things. Each
is a pass over data that can be timed with no protocol at all:

| Pass | Irreducible work | Grows with |
|---|---|---|
| Inference | Run the integer model | weights × tokens |
| Weight pass | About two ring multiply-adds per weight (a fold halves its input each round), for the matrix products and again to open the weight commitment | weights |
| Witness pass | The same fold over every activation value | activations |
| Commit pass | Encode and hash every committed bit once | committed bits over the code rate |
| Lookup pass | A table read and a few ring operations per lookup | lookups |

The verifier's floor is reading the prompt and output and checking a few hundred hash paths;
Magnetar's 600 KiB and 11 to 13 ms at 120 provable bits is the reference for a hash-based
proof.

There are two floors. The **hard floor** is every pass at memory bandwidth; nothing beats it.
The **arithmetic floor** is every pass at the measured rate of its cheapest operation on our
GPU, and is what a perfect protocol would reach.

We measure against it from the start:

1. **A floor suite**: one kernel per pass, on the GPU, with no protocol; the arithmetic
   kernels once per candidate ring of 5.5.
2. **A floor calculator**: from a model's shape, the token count and the bit widths, the
   count of weights, activations, lookups and committed bits, and so the floor per pass.
3. **A ratio in every gate**: each build of the prover reports its time per stage over that
   stage's floor. A ratio that rises is a regression.

This is the floor of the designs we know. A protocol that commits fewer bits or skips a pass
lowers the floor, and the calculator is revised when one does.

A first estimate for the 7B model at 1,000 tokens, from assumed rates (2×10^10 128-bit
multiply-adds a second, 2×10^11 word-sized ones, 10 GB a second of hashing; none measured):
about 3 s with a 128-bit ring and about 1.8 s with 64-bit words, with hashing the committed
bits the largest term in both. If that holds, committing fewer bits matters more than faster
arithmetic. The word-sized column is a bound, not a design: no sound protocol we have read
runs there (5.5).

## 6. Phases

Each phase ends at a gate. A phase's specs are written when it starts.

**Phase 0: measure and decide.** No protocol code.
- GPU instance up; the gate harness runs on it.
- The floor suite and calculator of 5.7. This is the first code in `gpu/`, and it turns
  every assumed rate in this plan into a measured one.
- The search of 5.5: each candidate read in full, then dropped on paper or prototyped.
- Kernel microbenchmarks on that GPU for what survives: `F_{2^128}` multiplication with
  `clmad`; wrapping 32- and 64-bit ring arithmetic; each candidate prime; encoding and the
  hash tree over packed bits; the BitZ grand product's inner loop; one sumcheck round over
  2^28 values in each.
- The baseline to beat on the same GPU: a small prime field with a FRI-family commitment, at
  the same sizes, from an existing implementation.
- The accuracy experiment of 5.3, on Qwen2-0.5B first.
- A cost model for Qwen2-0.5B and for the 7B target at 1,000 tokens: values per token,
  committed values, lookups, the batched weight opening, GPU memory at each step.
- The wider literature pass, recency first: Sable and FlashProve in full, GPU sumcheck work
  (zkDuel, arXiv 2610.37710; the DATE 2026 binary-tower paper), and whatever the candidates
  cite, with every number read sceptically.
- *Gate:* the floor measured for Qwen2-0.5B and the 7B model; every open decision of 5.2
  that phase 2 depends on (1, 2, 3, 5, 7, 8) made and written into a top-level protocol
  spec; a target rate stated as a multiple of the floor.

**Phase 1: the integer model.** Runs alongside phase 2.
- The numeric contract; the Llama block defined in integers; weights from safetensors; the
  calibration tool that produces integer weights.
- The inference engine: a CPU reference and the GPU engine, bit-identical, keeping the witness.
- *Gate:* identical outputs from both engines on Qwen2-0.5B; accuracy against the float model
  within the bound phase 0 set.

**Phase 2: protocol, reference prover, verifier.** CPU only, tiny models.
- Field, transcript and proof format, the relation layer and sumcheck engine, the lookup and
  range arguments, the commitment scheme, each layer's relation.
- The verifier crate and its audit document; the soundness sweeps; fuzzing.
- *Gate:* a tiny model and Qwen2-0.5B prove and verify on the CPU; every sweep site rejected.

**Phase 3: GPU kernels.**
- The backend trait's CUDA implementation, kernel by kernel, each tested against its CPU twin.
- *Gate:* the GPU prover's proof bytes equal the reference prover's on the phase 2 models;
  Qwen2-0.5B faster than the CPU backend's 2.7 s.

**Phase 4: scale.**
- The 7B model's configuration and weights; the weight commitment, streamed; scheduling under
  a memory bound (7B weights do not fit the GPU as field elements at once); long contexts.
- *Gate:* the 7B model end to end at 1,000 tokens at the phase 0 target rate.

**Phase 5: harden and open.** External audit of the verifier, public benchmarks, licence and
documentation for outside contributors. Zero knowledge and multiple GPUs start here as their
own tracks.

## 7. Spec map

About thirty specs reach the first target. More arrive with each new architecture, not with
each model.

| Area | Specs |
|---|---|
| Foundations | numeric contract and widths; field and extension; transcript and proof format; security parameters |
| Commitment | the scheme (encoding, Merkle tree, opening); the weight commitment and its batched opening; the witness layout |
| Protocol core | relation layer and sumcheck engine; batching and scheduling under a memory bound; lookup argument; range checks |
| Layers | linear and rescale; RMSNorm; attention scores and mask; softmax; RoPE; gated MLP and SiLU; residual; embedding; output head and decoding |
| Model | configuration and weight loading; calibration; CPU reference engine; GPU engine; accuracy evaluation |
| GPU | backend trait and CPU backend; CUDA kernels; memory manager; determinism |
| Verifier | the verifier crate; audit document; soundness sweeps; fuzzing |
| Operations | performance gate; CI with a GPU runner; release |

## 8. Risks

- **Accuracy at narrow widths.** If 8-bit weights or 16-bit activations cost too much
  perplexity on a 7B model, the field and cost model change. This is why the experiment is
  first.
- **GPU memory.** Seven billion weights as 64-bit field elements are 54 GB before any
  encoding. The prover must stream by matrix, which constrains how far layers can share a
  batched sumcheck.
- **Attention is quadratic.** At 1,000 tokens the attention scores are half a billion softmax
  lookups per proof under the causal mask, a billion if the full square is proven. The lookup
  argument's cost per entry is the number to watch.
- **Committed weights are opened in every proof.** One batched opening over every weight
  matrix costs at least a pass over all the weights. The phase 0 cost model prices it before
  the commitment scheme is chosen.
- **Two backends, one team.** The CPU backend is frozen in scope so it does not compete for
  time; the padded-lane gaps pinned on 2026-11-19 stay pinned there.
- **The search may end at the incumbent.** Three families have been read and dropped: proof
  systems over `Z_{2^k}`, integers committed under a code over the rationals, and the lattice
  commitments. If nothing beats packed bits with a prime field, the gain over the CPU backend
  comes from the GPU, narrower values and a cheaper bridge, not from a new core.
- **A ring that counts.** Our relations are integer products. A binary field alone cannot
  host them cheaply, so a binary commitment needs a bridge, and the bridge is our largest
  opening cost today.
- **Unmeasured GPU arithmetic.** Carry-less multiplication on the GPU is new (July 2026) and
  NVIDIA's figures are against bit-slicing, not against a prime field. Every rate in this
  plan's reasoning is measured in phase 0 before it decides anything.

## Sources

Read in full: Magnetar (Setty, ePrint 2026/2423); BitZ (ePrint 2026/2141, and
`specs/bitz-commitment.md`); Sumcheck bit packing (Frigo and shelat, 2026/2770). Read in
full with the proofs skimmed: Rotor (ePrint 2026/2175); Nenya (2026/2672); Zinc (2025/316);
Zinc+ (2026/855); Neo and SuperNeo (2026/242); LaBinius (2026/2103). Read as abstracts or
summaries only: Quasar (2026/1839); Sable (arXiv 2611.14083); zkDuel (arXiv 2610.37710);
NVIDIA, "Building faster cryptography with carryless multiplication in CUDA
13.3" (15 July 2026). Cited by Magnetar: BinarySpartan (2026/1656), ring switching (Diamond
and Posen, 2024/504), Ligerito (2025/1187). FlashProve and other systems' positions:
`docs/prover-plan.md` section 4.

## Appendix: the 7B target in numbers

Llama-2-7B's configuration: 32 layers, width 4,096, 32 heads, MLP width 11,008, vocabulary
32,000, RMSNorm, RoPE, gated SiLU MLP. Qwen2 and Mistral-7B use the same block, with grouped
key and value heads.

| Quantity | Size |
|---|---|
| Weights | 6.74 billion: 67 M (attention) + 135 M (MLP) per layer, plus 131 M each for the embedding and the output head |
| Weights padded to powers of two | about 8.9 billion |
| Weight storage | 27 GB at 32 bits, 13.5 GB at 16, 6.7 GB at 8 |
| Activation values per token | roughly 2 million across the 32 layers, attention scores excluded |
| Attention scores for the last token at 1,000 tokens of context | about 1 million (32 heads × 32 layers × 1,000 tokens attended to) |
| Total at 1,000 tokens | roughly 2 billion activation values, and 0.5 billion attention scores under the causal mask (1 billion for the full square) |
