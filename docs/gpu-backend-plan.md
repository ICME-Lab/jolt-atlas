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
prover (GPU)                   layer relations → batched sumchecks → one commitment opening
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

## 5. Decisions

### 5.1 Made (2026-11-25)

| Decision | Outcome |
|---|---|
| Weights | **Committed** once per model and opened in every proof. The verifier holds the commitment, not the weights |
| First model | **Qwen2-0.5B**: the same block as the 7B targets, and the CPU backend's 2.7 s to compare against. Which 7B follows is still open (5.2) |
| Where it lives | **`gpu/` in this repository**, its own workspace (4.1) |
| Ownership | **All code written here.** No dependency on this repository's crates, on upstream Jolt, or on outside proof-system crates |
| Not the commodity stack | **We do not build on a small prime field with a FRI-family commitment** (Goldilocks or a 31-bit field with WHIR, BaseFold or FRI). Sable, FlashProve and SP1 already do, so it differentiates nothing, and Magnetar (5.4) proves 24 to 66 times faster than SP1 Hypercube on the same 12 CPU cores, at 120 bits of provable security against SP1's 100. That stack is the baseline we measure against, not a candidate |

### 5.2 Open

Each has a recommendation and the measurement or check that settles it. None should be decided
by argument alone.

| # | Decision | Options | Recommendation | Settled by |
|---|---|---|---|---|
| 1 | Numeric representation | (a) 32-bit fixed point everywhere, as today; (b) per-tensor widths, e.g. 8-bit weights and 16-bit activations; (c) floating point | (b), with the widths chosen by experiment. (c) has no affordable proof. See 5.3 | Accuracy experiment, phase 0 |
| 2 | Commitment | Packed bits in `F_{2^128}` under a hash tree, opened with ring switching and Ligerito: what BitZ and Magnetar both do, and what we own; a lattice scheme | Packed bits, laid out as one polynomial with one opening. It prices a value by its bits and gives range checks for free, which is why BitZ beat Akita here by 14 to 15 times (PR #409). The unknown is binary-field arithmetic on a GPU, which has no carry-less multiply instruction | GPU rates for `F_{2^128}` multiplication by each known method (bit-sliced, tower, table, integer-multiply), for the encoding and the hash tree, and for the BitZ grand product's inner loop, phase 0 |
| 3 | Field of the sumchecks | A prime field, with BitZ's bridge from a commitment of bits to a claim over that field (which prime: Fp128 as today, a Mersenne prime, a smaller prime with an extension); `F_{2^128}` throughout, with integer arithmetic as bit constraints, as Magnetar does | A prime field. A sumcheck for a matrix product touches each weight once only in a field where integers add and multiply; as bit constraints each multiply-add costs a few hundred constraints (an estimate), and Qwen2-0.5B at 16 tokens is several billion multiply-adds against under half a billion weights. Which prime is open, and the bridge (the grand product, our largest opening cost today) is what the choice has to pay for | GPU sumcheck-round rates per candidate prime; the bridge's cost on the GPU; the widths of decision 1 (an 8-bit by 16-bit product over 11,008 terms needs about 38 bits) |
| 4 | The 7B model | Llama-2-7B; Mistral-7B; Qwen2.5-7B; Llama-3.1-8B | One config-driven block serves all four, so this is a licence question: Llama 2 is under Meta's community licence and gated; Mistral-7B and Qwen2.5-7B are Apache 2.0 | Licence review, before phase 4 |
| 5 | GPU kernels | CUDA through a Rust backend trait, with which toolchain (C++ kernels over FFI, or Rust-side code generation) | CUDA C++ kernels behind the trait; benchmark an existing proving library only as the bar to beat | Phase 0 |
| 6 | Statement | What is public (prompt, output, sampling seed); greedy or sampled decoding | Public prompt and output, greedy decoding first; one proof covers a whole sequence, since a causal model's forward pass over N tokens checks every next-token choice | Product owner |
| 7 | Security level | 128 bits as today; 100 bits as the Gemma result; proven or conjectured soundness for the commitment | Decide explicitly and state it next to every benchmark | Phase 0 spec |
| 8 | Padding | Pad every dimension to a power of two; lay tensors out without it | Do not assume power-of-two padding: Llama's MLP width 11,008 pads to 16,384, half as many weight cells again in every MLP matrix | Phase 0 cost model |

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

- **One commitment and one opening.** Every committed vector sits on an aligned subcube of
  one bit polynomial, so a claim on a part lifts to a claim on the whole by appending fixed
  coordinates, with no adapter sumcheck. `specs/bitz-commitment.md` already took this.
- **Commit only what nothing else establishes.** A segment another argument proves correct is
  read through an evaluation claim and never committed. For us: no intermediate tensor
  between layers is committed.
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
  constraints. Our workload is integer multiply-add, by the billion (open decision 3).
- **Its speed, to a GPU.** Every figure is from a CPU with a carry-less multiply
  instruction. The paper has no GPU measurement, and it measured SP1 on that CPU too, not on
  the GPUs SP1 is built for.

## 6. Phases

Each phase ends at a gate. A phase's specs are written when it starts.

**Phase 0: measure and decide.** No protocol code.
- GPU instance up; the gate harness runs on it.
- Kernel microbenchmarks on that GPU: binary-field multiplication by each known method;
  Reed–Solomon encoding and the hash tree over packed bits; the BitZ grand product's inner
  loop; one sumcheck round over 2^28 values in each candidate field of decision 3.
- The baseline to beat on the same GPU: a small prime field with a FRI-family commitment, at
  the same sizes, from an existing implementation.
- The accuracy experiment of 5.3, on Qwen2-0.5B first.
- A cost model for Qwen2-0.5B and for the 7B target at 1,000 tokens: values per token,
  committed values, lookups, the batched weight opening, GPU memory at each step.
- A recency-first literature pass: Magnetar and BinarySpartan in full, bit packing in
  sumcheck (ePrint 2026/2770), binary-field proving on GPUs, and the zkML systems of the last
  few months, with their numbers read sceptically.
- *Gate:* every open decision of 5.2 that phase 2 depends on (1, 2, 3, 5, 7, 8) made and
  written into a top-level protocol spec; a target rate.

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
- **Binary-field arithmetic on a GPU.** The packed-bit design rests on multiplication in
  `F_{2^128}`, which CPUs do in one instruction and GPUs do not have. Magnetar's and BitZ's
  speeds are CPU measurements. If no method reaches a useful rate, decision 2 reopens; this
  is the first number phase 0 produces.
- **A field that can count.** Our relations are integer products. Either the sumchecks stay
  in a prime field, whose cost on a GPU decision 3 measures, or integer arithmetic becomes
  binary constraints, whose cost for a matrix product nobody has priced for us.

## Sources

Magnetar: Setty, ePrint 2026/2423, read in full (sections 1.3, 2.2, 5.2 and 6 are what this
plan relies on). BinarySpartan: Setty, ePrint 2026/1656. Ring switching: Diamond and Posen,
ePrint 2024/504. Ligerito: Novakovic and Angeris, ePrint 2025/1187. Sumcheck bit packing:
Frigo and shelat, ePrint 2026/2770 (not yet read). BitZ: ePrint 2026/2141 and
`specs/bitz-commitment.md`. Sable, FlashProve and the positions of other systems:
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
