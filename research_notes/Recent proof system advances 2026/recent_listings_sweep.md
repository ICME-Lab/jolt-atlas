# Recent listings sweep: proof systems and verifiable inference, July to 9 November 2026

Method, in one paragraph. Sweep run on 2026-11-09. Sources browsed: IACR ePrint (`/complete/` listing down to
2026/2709, highest ID seen 2026/2808; the RSS feed; about 25 ePrint searches restricted to `submittedafter=2026-07-01`
by topic and by author: Thaler, Setty, Bünz, Fenzi), arXiv search and the cs.CR recent listing, GitHub (a16z/jolt pull
requests since late August via `gh`, new zkML repositories, PR search), Anthropic's roadmap page, press on FlashProve.
"Read level" for each item is one of: **abstract** (the ePrint/arXiv abstract page, read through a fetch tool that
paraphrases it; numbers are as that page states them), **listing** (search-result snippet only), **repo** (README or PR
description read directly), **secondary** (press or third-party description). No full PDFs were read. Nothing below was
taken from memory. Dates are ePrint "received" dates unless marked. All ePrint items are preprints unless a venue is
given.

Count: 47 dated items. 44 are dated July 2026 or later; 3 are June 2026 (marked); older work appears only as pointers.

## Which recent items could change the design of a zkML prover? (annotated list)

### Takeaway

The autumn 2026 literature has moved decisively toward "pay for the bits you have": BitZ, Limber, Mikros, Sumcheck bit
packing, Nenya, Rotor, LaBinius and Flock all attack the cost of committing small values, and two of them (BitZ,
Limber) do it for integers proved in a prime field, which is the construction `docs/research-directions.md` section 5
lists as "no known construction" (Option C). Separately, lattice work (tensor-train random projections, PikkuFold)
supplies a succinctly verifiable random-projection range check, which is the open design question in section 7.3.

### Cited Findings

Format per item: **Title**. Authors. Date. Status. Read level. Core idea. Numbers. *Take for jolt-atlas* (this last line
is my inference, not the paper's claim).

#### A. Pay-per-bit commitment, integers and bit-width-aware SNARKs

1. **BitZ: proofs and commitments in arbitrary rings through binary fields.** R. Bloemen (World Foundation), A. Garreta,
   L. Soukhanov, J. Wu (Nethermind), M. Kostrzewa (Reilabs), S. Londhe (Ivy). Received 2026-09-21, revised 2026-10-04.
   Preprint. Read: abstract. A hash-based multilinear PCS for polynomials with coefficients in an arbitrary finitely
   generated ring; values are decomposed into bits and committed over a binary field, and evaluation claims are proved
   over a target ring through a ring homomorphism. BitZ-SNARK commits over the integers and proves over a random prime
   field (Campanelli and Hall-Andersen fingerprinting). Cost depends mainly on coefficient bit size, and range checks
   come "almost for free". Numbers: 9 million 32-bit integer multiplications proved per second on a MacBook Air M5 (10
   threads, CPU), proofs under 200 kB; SHA-256 of 2 kB plus P-256 ECDSA in 32 ms single-threaded, 65 kB proof, 3.1 ms
   verifier. — [ePrint 2026/2141](https://eprint.iacr.org/2026/2141)
   *Take:* this is a concrete instance of Option C (bits in a binary hash commitment, arithmetic in a prime field) with
   range checks included; read the full paper before anything else in section 5.

2. **Limber: Low Overhead SNARKs for Integers from Any PCS.** J. Chen, L. Xia, B. Bünz (NYU), W. Nguyen (NYU, MSR).
   Received 2026-08-07, revised 2026-10-19. Preprint. Read: abstract. Builds on Zaratan (PKC 2026): integer relations are
   fingerprinted by reduction to a randomly sampled prime field. Limber is an integer "mod-PCS" with o(1) multiplicative
   commitment overhead that works over any standard field PCS, including small-field ones; combined with a PIOP for
   integer R1CS. Numbers: RSA accumulator 15-30x faster than circuit approaches; non-native Poseidon2 24x faster; cites
   9 of 27 collected critical ZK bugs as non-native-arithmetic related. — [ePrint 2026/1635](https://eprint.iacr.org/2026/1635)
   *Take:* PCS-agnostic route to committed integers with exact integer semantics; a candidate for "commit the
   accumulator once" (section 4.1) on top of Akita rather than replacing it.

3. **Mikros: Bitwidth-Aware SNARKs over Extensions of Small Fields.** M. Cong (Shenzhen Univ.). Received 2026-10-26.
   Preprint. Read: abstract. "Algebraic vector packing" packs a length-N base-field vector into N/κ elements of a
   degree-κ extension, making existing linear-time extension-field protocols bitwidth-aware; gives bitwidth-aware
   inner-product, Hadamard, PCS, and the first bitwidth-aware grand-product and permutation arguments. Does not rely on
   binary-field structure. Numbers: prover faster than Flock on batches of 256: 2.19x SHA-256, 2.55x Keccak, 1.85x
   BLAKE3. — [ePrint 2026/2646](https://eprint.iacr.org/2026/2646)
   *Take:* evidence that bit-width-proportional cost does not require characteristic 2; worth checking whether the
   packing applies to a prime base field with a 128-bit extension.

4. **Sumcheck bit packing.** M. Frigo, a. shelat (Google). Received 2026-11-04. Preprint. Read: abstract. Packs m bits
   into one GF(2^m) element while still letting the proof check linear claims on individual bits, using only the
   GF(2)-linearity of the field trace; soundness error 2^-m for arbitrary bit-linear claims; trace-zero elements make
   the reduction zero-knowledge over any hiding linear commitment. Prior constructions needed O(m^2) or O(m) quadratic
   constraints in the commitment; this needs only linear constraints. No benchmarks in the abstract. —
   [ePrint 2026/2770](https://eprint.iacr.org/2026/2770)
   *Take:* the cleanest statement of "committed bits, no Booleanity check, per-bit linear claims" (Magnetar's property);
   relevant if a binary-field bit commitment is paired with our prime-field sumchecks.

5. **Nenya: Efficient zkSNARKs for Integer Rings Z/2^kZ via Reverse Multiplication-Friendly Embeddings.** T. Zhang,
   N. Borisov, Y. Zhang (UIUC). Received 2026-10-28. IEEE S&P 2027 (accepted). Read: abstract. Native R1CS over Z/2^k
   reduced to Galois-ring constraints via RMFEs, with a code-based PCS supporting well-formedness checks and packing
   rotations. Numbers: 2^25-constraint data-parallel 32-bit computation in 4.8 s; about four orders of magnitude faster
   than the prior Z/2^k SNARK; only 1.06x slower than a field SNARK at equal constraint count; 12x faster than a
   field SNARK on mod-2^8 pixel filtering. — [ePrint 2026/2672](https://eprint.iacr.org/2026/2672)
   *Take:* machine-word wraparound semantics for free; interesting for int8/int32 models whose reference runtime wraps,
   less so for our saturating clamp.

6. **Rotor: SNARKs for Power-of-Two Rings.** C. Li, Y. Hu (SJTU), Y. Li, Z. Li (Ant), P. Zhu (Tsinghua), M. Dong, A. Li
   (Brevis). Received 2026-09-23, revised 2026-10-08. Preprint. Read: abstract. Works natively over Z_2^k with three
   precision levels for soundness; a sumcheck-style IOP with tensor queries, a Brakedown-style commitment from a
   weakened proximity gap, and "compatible commitments" linking ring witnesses to binary-field representations. The
   authors removed their evaluation numbers pending re-run after a parameter change. —
   [ePrint 2026/2175](https://eprint.iacr.org/2026/2175)
   *Take:* the "compatible commitments" idea is another cross-representation link (ring to binary field); no numbers
   to rely on yet.

7. **LaBinius: Fast Lattice-Based Binary Polynomial Commitment Scheme.** M. Osadnik (Aalto), G. Seiler (IBM Zurich).
   Received 2026-09-18. Preprint. Read: abstract. A lattice PCS that decouples commitment arithmetic (composite
   NTT-friendly modulus) from evaluation arithmetic (binary extension field), with no bitness checks; plugged into the
   Binius and Flock front ends; optional LaBRADOR compression. Numbers: single-core prover at most 1.2x the hash-based
   prover of each system; over 67,000 BLAKE3 compressions per second single-core; verification under 40 ms; proofs under
   100 KiB, about 5x smaller than Binius and Flock. — [ePrint 2026/2103](https://eprint.iacr.org/2026/2103)
   *Take:* shows a lattice commitment can carry bit-valued witnesses at near hash-PCS speed with small proofs; the
   strongest counterpoint to "lattice is not the fastest per bit" in section 5.

8. **Flock: Fast Proving for Batch Boolean Computations.** B. Bünz, W. Wang (NYU), R. Rothblum (Succinct). Received
   2026-06-28 (June), revised 2026-09-19. Preprint. Read: abstract. Hash-based SNARK for batches of the same Boolean
   R1CS circuit with new lincheck and zerocheck optimizations; implementation "heavily optimized with the help of coding
   agents". Numbers: single M4 Max core: 82k BLAKE3, 42k SHA-256, 30k Keccak per second; under 250x native overhead;
   over 9x faster than Binius64 on SHA-256. BinarySpartan and Mikros both benchmark against it. —
   [ePrint 2026/1329](https://eprint.iacr.org/2026/1329)
   *Take:* the baseline every binary-field system now cites; its batched-identical-circuit structure matches
   per-token or per-layer repetition.

9. **Speeding up Sumcheck using Algebraic Geometry.** R. Dalal, T. Hemo, R. Rothblum (Succinct), B. Bünz. Received
   2026-10-15. Preprint. Read: abstract. Generalizes sumcheck from polynomials to multiplication codes and builds a
   binary-preserving AG code (genus-95 curve over F_2) for the univariate-skip round: 64-bit messages map to binary
   product messages of length 222. Numbers: skip round 2.36x faster and whole zerocheck 1.65x faster at 2^32 constraints
   in Flock. — [ePrint 2026/2494](https://eprint.iacr.org/2026/2494)
   *Take:* binary-field specific; a pointer that the first sumcheck round is still being re-engineered.

10. **BinarySpartan** (S. Setty, [ePrint 2026/1656](https://eprint.iacr.org/2026/1656), listing date 2026-09-03) and
    **Magnetar** (S. Setty, [ePrint 2026/2423](https://eprint.iacr.org/2026/2423), listing date 2026-10-15). Read:
    listing. Already known to the team; recorded here for their ePrint IDs, which `docs/research-directions.md` lacks
    for Magnetar. — sources as linked.

#### B. Range checks, lookups, matmul

11. **Succinct Shortness Check Under a Few Kilobytes via Tensor Train Random Projections.** Z. Geng (Edinburgh),
    M. Plançon (IOG). Received 2026-09-22. Preprint. Read: abstract. Approximate range proofs in lattice systems use
    Johnson-Lindenstrauss projections: unstructured JL costs the verifier linear time, structured JL (RoK and Roll,
    Asiacrypt 2025) needs large committed projection vectors. Tensor Train Random Projection gives a projection matrix
    representable in Õ(λ) with an O(λ) projection vector; well-formedness of the projection is linearized with
    sumcheck. Numbers: verifier O(λ log m (log log m)^2), proof about 4 KB, as a drop-in approximate shortness check. —
    [ePrint 2026/2146](https://eprint.iacr.org/2026/2146)
    *Take:* directly addresses section 7.3's open question (a random vector whose extension the verifier evaluates
    succinctly, with an analysed distribution) and is already phrased for a lattice-plus-sumcheck stack.

12. **PikkuFold: Efficient Folding in a Few Kilobytes.** M. Osadnik. Received 2026-08-26, revised 2026-09-15. Preprint.
    Read: abstract. Lattice folding via layered random projections whose final output is small enough to send in the
    clear, so no in-protocol commitments beyond the fresh inputs; replaces Cyclo's range proofs. Includes a JL theorem
    for biased ternary matrices mod q with certified concrete constants. Numbers: 5.7 KB per fold step (vs at least
    30 KB Cyclo, 60 KB SALSAA), prover over 10x faster, millisecond verifier. —
    [ePrint 2026/1809](https://eprint.iacr.org/2026/1809)
    *Take:* the certified JL constants are what a soundness analysis of an approximate no-overflow check would cite;
    also a folding candidate for per-token accumulation.

13. **Celer: A Lookup Argument for Large-Scale Queries.** W. Qu, Y. Guo, Z. Xuan, J. Zhang (NUS), X. Liu (Zhejiang).
    Received 2026-07-16. CRYPTO 2026. Read: abstract. Targets the regime where the query vector m is far larger than
    the table n; the prover uses a number of commitments sublinear in m while field operations stay linear in m.
    Numbers: at m = 2^28, n = 2^16 (a workload "drawn from proving Llama language model inference") the prover is 25.9x
    faster than plookup and 9.4x faster than LogUp. — [ePrint 2026/1453](https://eprint.iacr.org/2026/1453)
    *Take:* exactly our regime (range checks and activations: huge query count, 2^8 to 2^16 tables) and exactly our
    bottleneck (commitment count); must be compared against Shout one-hot digits.

14. **Rarus: A Succinct and Efficient Range Proof for Polynomial-based Vector Commitment.** X. Yang, W. Qu, Y. Guo,
    J. Zhang (NUS). Received 2026-07-15. USENIX Security 2026. Read: abstract. Batch range proof with b-ary
    decomposition, a bivariate zero-test and an accelerated univariate sumcheck; O(1) proof and verifier; prover
    O(Nℓ/log(Nℓ)) group operations; supports arbitrary ranges [0, R). Numbers: 20x faster than Bulletproofs and
    Missileproof for 16,384 values in [0, 2^64). — [ePrint 2026/1440](https://eprint.iacr.org/2026/1440)
    *Take:* pairing-based, so not adoptable, but the b-ary decomposition with sub-linear group work is the same lever
    as choosing digit width under Akita.

15. **A Separation Principle for Lookup-Based zkML.** Min Jun Jo. Listing date 2026-07-15. Preprint. Read: listing.
    Argues per-lookup proving cost in Shout-style one-hot arguments depends on the access pattern, so exploiting an
    activation's mathematical structure cannot reduce it. Already on the team's list; included for completeness. —
    [ePrint 2026/1390](https://eprint.iacr.org/2026/1390)

16. **TenseUp: Doubly-Efficient Lookup Arguments from Tensor Codes and Applications to Dynamic SNARKs.** C. Pappas,
    D. Papadopoulos (HKUST), P. Akhiani, Y. Zhang (UIUC). Received 2026-10-10. Preprint. Read: abstract. Transparent,
    updatable, post-quantum lookup arguments where prover time is proportional to the number of lookups k and sublinear
    in table size, built from two-dimensional tensor codes; plus a compiler from static to dynamic SNARKs. No numbers
    in the abstract (claims parity with non-post-quantum schemes). — [ePrint 2026/2435](https://eprint.iacr.org/2026/2435)
    *Take:* relevant only if committed weights are treated as a large table; a dynamic SNARK could re-prove after a
    small prompt change.

17. **Post-Quantum Sublinear Lookup Arguments from Polynomial Preprocessing.** A. Basumata, C. Ganesh (IISc),
    V. Narayanan, S. Patranabis, N. Singh (IBM), S. Shaw. Received 2026-10-21. Preprint. Read: abstract. Kedlaya-Umans
    style preprocessing checked by Reed-Muller proximity testing, giving k·polylog(N) online prover for k lookups into a
    size-N table. Asymptotic only. — [ePrint 2026/2581](https://eprint.iacr.org/2026/2581)
    *Take:* theoretical; pointer only.

18. **SAK: Sparse Arguments of Knowledge, from Sparse Lookup Arguments.** A. Kothapalli (MSR), S. Sridhar (Berkeley),
    A. Zapico. Received 2026-09-21. Preprint. Read: abstract. Prover time O(n log n) in the number n of non-zero witness
    entries rather than the length N, for CCS, by using sublinear lookups to select only constraints touching non-zero
    entries. No benchmarks. — [ePrint 2026/2143](https://eprint.iacr.org/2026/2143)
    *Take:* conceptual support for "sub-quadratic attention" (section 8): pay only for attention pairs that survive
    the exp cutoff.

19. **LAMP: Linear Verification of Matrix Multiplication via Proximity Testing.** K. Lee, J. Kim (Kookmin), B. Han,
    H. Oh (Hanyang). Received 2026-07-31, revised 2026-11-05. Preprint. Read: abstract. Freivalds' check combined with
    proximity testing over Reed-Solomon codes: the prover commits encoded matrices and intermediate vectors, and a
    CP-SNARK circuit checks only t sampled codeword positions. Numbers: at k = 2^12, 9.68x fewer constraints and 5.81x
    faster proving than a Freivalds-in-circuit baseline; verification about 0.13 s. —
    [ePrint 2026/1571](https://eprint.iacr.org/2026/1571)
    *Take:* circuit-model work; Thaler's matmul sumcheck already beats it for us. Pointer.

20. **Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation.**
    B. Merbaum, M. A. Raeisi, W. Wang, C. Papamanthou, K. Sotiraki, F. Zhang (Yale). Received 2026-09-09. Preprint.
    Read: abstract. Not a SNARK: an information-theoretically sound verification protocol for delegated matrix-vector
    products with transparent preprocessing and batch verification, near-zero server overhead, and LPN-based masking
    for input privacy. Numbers on Qwen3-4B: up to 17x (online masks), 45x (precomputed) and 44x (verify-only) client
    throughput over local inference with one client thread. — [ePrint 2026/1955](https://eprint.iacr.org/2026/1955)
    *Take:* a competing design point: the client keeps doing the nonlinearities and only outsources linear algebra, so
    the server pays almost nothing; it removes the range-check problem by not having one.

#### C. zkML and inference-verification systems not on the named list

21. **Bits Under ZK-LLM: Evaluating Zero-Knowledge-Friendly Quantization for Verifiable Private LLM Inference.**
    T. Yoon, Y. Zhang, X. Liao. arXiv, submitted 2026-09-29. Preprint. Read: abstract. First systematic study of
    quantization for ZK inference across nine models (including Qwen2.5-14B and the MoE Qwen3-30B-A3B), varying weight
    precision, activation precision and lookup-table precision. Findings: activation precision matters far more than
    weight precision; lookup-table approximation of nonlinearities can become the main accuracy loss; the
    inverse-square-root lookup in RMSNorm is a recurring bottleneck and raising precision only there restores
    near-baseline quality; lowering bit-width or table size does not reduce end-to-end proving cost proportionally. —
    [arXiv 2609.36437](https://arxiv.org/abs/2609.36437)
    *Take:* direct evidence for and against sections 1 and 6: supports per-operator precision selection (our "rounding
    placement as a compiler problem") and warns that int8 alone will not give a proportional speedup.

22. **Seal, Then Sample: Sampled Layerwise Proofs for Verifiable LLM Inference from GPT-2 to 70B.** Y. Lim, S. Yong.
    arXiv, submitted 2026-09-23. Preprint. Read: abstract. Commit to boundary activations of every chunk of the trace
    before the verifier's challenge, then prove only a sampled subset of chunks plus those binding prompt and answer;
    concurrent requests are packed into one trace with a block-diagonal causal mask. Numbers: TinyLlama-1.1B, 7 of 47
    chunks costs 22.0% of the time and 6.8% of the proof of all 47; twelve packed requests in 181.9 s (6.5x less than
    separately); Llama-2-70B on a 2 TB CPU host: 163 chunks sealed, 5 proved, 4.34 MiB proof in 1,259 s, 46.3 s
    verification without weights; 84.8-84.9% argmax agreement with float over 334,705 WikiText-2 positions after an
    "LLM-aware observer" fixes a residual-stream bit-width loss. Stated limit: one bad chunk is caught with probability
    3/161 at 70B. Data: https://github.com/TrueOpen/slp-experiments — [arXiv 2609.27367](https://arxiv.org/abs/2609.27367)
    *Take:* commit-then-sample turns audit coverage into a dial over one commitment set; our layer-chunked proving
    could offer the same mode. The residual-stream bit-width finding is a data point for our 32-bit vs int8 question.

23. **SASH zk-inference-prover** (Singapore AI Safety Hub). Repo created 2026-09-26, last push 2026-10-19. Repo, no
    paper found. Read: repo README and benchmark adapter. A Circle-STARK prover (vendored library "nitroo", CUDA
    backend) for a quantized transformer; a sequence is proved as a grid of segments, each one layer applied to a block
    of S = 512 token positions with a context of earlier blocks, plus a boundary-grid check; table size flag
    `--table-bits 16`. Independent benchmark numbers (inferenceproofs.ai v0.1): Llama 3.1 8B, W8A8, batch 1, 15,760x
    overhead vs warm vLLM serving; 75.41 ms proving and 639.9 kB proof per token per Gbit of active weights. —
    [repo](https://github.com/sg-ai-safety-hub/zk-inference-prover); numbers from [inferenceproofs.ai](https://www.inferenceproofs.ai)
    *Take:* a second open full-LLM prover at W8A8; its segment-grid statement and per-edge binding table are a
    reference for how to state what a chunked proof binds.

24. **Inference Proofs: an open, neutral benchmark for provable inference** (ai-verification-observatory). Repo created
    2026-11-03; release v0.1. Repo and site. Read: repo README and site. Pinned models, reference hardware, named timing
    scopes and a "claim card" per system stating what the proof establishes (weights hidden, model bound, whole output
    checked, formally verified, security level, trusted party). 2 of 19 listed systems have numbers so far. FlashProve
    at commit 9e2bdd8f: 61.28x overhead vs vLLM serving at batch 24, 68.21 µs proving, 222.9 B proof and 198.7 µs check
    per token per Gbit of active weights. — [repo](https://github.com/ai-verification-observatory/inference-proofs),
    [site](https://www.inferenceproofs.ai)
    *Take:* this is the "like-for-like benchmark" of section 8, already existing; an adapter for jolt-atlas would put
    our small-proof, fast-verifier position on the same page. Its normalisation (per token per Gbit of active weights)
    is a ready-made metric.

25. **stwo-zig tensor frontend: provable integer inference (vs FlashProve).** T. Pender. PR opened 2026-11-04, open.
    Repo PR, self-reported, AI-generated code. Read: PR description. A multilinear Ligero-style PCS over M31 (flat and
    recursive), **virtual requant remainders**, one batched LogUp-GKR, claim batching, committed weights with a key
    proof, GQA attention, softmax, RMSNorm, RoPE, SwiGLU, top-1 MoE, greedy-token check, a zero-knowledge mode, and a
    Metal backend bit-identical to CPU. Numbers (M4 Max, self-reported): hiding a∘b=c at 2^20 on one core 0.305 s vs
    FlashProve CPU 8.63 s (28x); decoder layer over 1,024 tokens in 1.33 s on Metal; SwiGLU MLP over 1,024 tokens in
    0.49 s. Not shown: model-scale parity, Lean proofs, fidelity. —
    [PR](https://github.com/teddyjfpender/stwo-zig/pull/321)
    *Take:* a FlashProve-style architecture on an Apple laptop within days of release, over a 31-bit field; the
    "virtual requant remainders" claim is worth reading in the code since it is the thing we commit digits for.

26. **FlashProve ports.** `zkwasm/FlashProve-L4` (created 2026-11-04, TinyStories260K on an NVIDIA L4),
    `jp4g/flashprove-mlx` (created 2026-11-05, Apple Silicon), `whyagents/flashprove-consumer-gpu` (created 2026-11-09).
    Repos. Read: the third README; the first two confirmed to exist only. The consumer-GPU port reports 12 prover
    patches (+34/−28 lines), no verifier changes; layers at Llama 4 Maverick width (d = 5,120, 40 heads) commit in about
    43 s and prove in 3-6 s within 16 GB on an RTX 5070 Ti; batches of 64 requests at 9.3 GiB GPU peak; 770/770 proofs
    accepted by the unmodified upstream verifier. —
    [repo](https://github.com/whyagents/flashprove-consumer-gpu), [L4](https://github.com/zkwasm/FlashProve-L4),
    [mlx](https://github.com/jp4g/flashprove-mlx)
    *Take:* FlashProve's "datacentre GPU" limitation (our section 7.6) eroded within a week; treat laptop-class
    hardware as no longer a differentiator.

27. **Anthropic "inference verification" (not FlashProve).** Roadmap update dated 2026-10-19. First-party web page.
    Read: page. Anthropic reports a prototype that "runs on production inference and checks that outputs come from a
    specific set of model weights", plus signing of all internal token traffic; it renamed the goal from "provable
    inference" to "inference verification" because the former "suggests a narrower approach" than needed. No mechanism,
    paper or repo is given. Press describes FlashProve (repo created 2026-10-30, announced about 2026-11-02, with EPFL,
    MIT and StarkWare collaborators) as a separate project pursuing the same aim. —
    [Anthropic roadmap](https://www.anthropic.com/rsp/roadmap); press: [Help Net Security](https://www.helpnetsecurity.com/?p=390217)
    *Take:* the largest prospective customer for this technology says cryptographic proof is one option among several;
    the non-SNARK alternatives are the real competition for the production use case.

28. **Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference.** C. Gong,
    B. Liu, M. Li. arXiv, submitted 2026-07-30. Preprint. Read: abstract. With private committed weights, a ZK proof
    shows the output matches an architecture, not that a large model was run: "ghost" weights with collapsing algebraic
    structure pass the same circuit while the provider serves something close to a small model. No numbers in the
    abstract. — [arXiv 2607.28884](https://arxiv.org/abs/2607.28884)
    *Take:* a statement-level caveat for any committed-weights mode; public weights (our current setting) are immune.

29. **Private and Verifiable Outsourcing of Open-Weight LLM Inference.** K. Gupta (UMD, Google), J. Katz (Google),
    I. Miers (UMD). Received 2026-08-31. Preprint. Read: abstract. Two non-colluding malicious servers; client checks
    the result using information the model owner publishes with the weights. Numbers: 11-14x faster than SIGMA (PETS
    2024), no server overhead beyond running the model; a client runs Llama 2-70B with 179 MB of local state instead of
    140 GB. — [ePrint 2026/1849](https://eprint.iacr.org/2026/1849)
    *Take:* non-SNARK competitor with zero prover overhead under a two-server assumption.

30. **Agree on the Model, Verify the Inference: GKR Protocols for HND-Based Transformer Inference.** X. Liang, J. Li,
    R. Qin, Y. Lv. arXiv, submitted 2026-07-23. Preprint. Read: abstract. GKR over the polynomial core of
    Homomorphic-Nonhomomorphic Decomposition transformers; the verifier checks the transcript and weight openings and
    delegates expensive public evaluations to a signed worker (assumes prover and worker do not collude). No numbers in
    the abstract. — [arXiv 2607.21162](https://arxiv.org/abs/2607.21162)
    *Take:* an answer to "our verifier reads every weight": outsource the public-weight evaluation under a weaker trust
    model. Low priority.

31. **Communication-Efficient Verifiable Attention for LLM Inference (VeriAttn).** Z. Chen et al. arXiv, submitted
    2026-06-15 (June). Preprint. Read: abstract. TEE-based: attention runs on an untrusted GPU and the TEE only
    verifies. Numbers on Intel TDX: 2.60-3.38x faster than TSDP for 6k-token prefill, 3.86-5.42x for 10k-token
    decoding. — [arXiv 2606.16352](https://arxiv.org/abs/2606.16352)
    *Take:* the TEE baseline for the same use case; pointer.

32. **Interactive Proofs of Proximity for Model Evaluation.** G. Couteau, N. Melissaris, T. Paris. Received 2026-09-25.
    Preprint. Read: abstract. Doubly-sublinear IPPs letting an auditor certify statistics of a model (accuracy,
    fairness, calibration) with Õ(1/g) verifier queries and O(1/g^2) prover queries for gap g. Theory. —
    [ePrint 2026/2217](https://eprint.iacr.org/2026/2217)
    *Take:* a different statement (properties of a model, not one inference) that could sit above per-inference proofs.

33. **Azkaban: A Zero-Knowledge Abstract Analysis for Neural Networks** (S. Das et al., listing date 2026-09-10,
    [ePrint 2026/1967](https://eprint.iacr.org/2026/1967)) and **VERIF: ZK proof for IVF-Flat retrieval in RAG**
    (Z. Zhang et al., listing date 2026-08-26, [ePrint 2026/1802](https://eprint.iacr.org/2026/1802)). Preprints. Read:
    listing. ZK certification of DNN properties by abstract interpretation; ZK proof that RAG vector search followed
    the promised index. *Take:* adjacent statements (robustness certificates, retrieval) a zkML product may be asked for.

34. **arXiv cs.CR, week of 2026-11-03 to 11-09 (titles only).** "SciCheck: A Framework for Verifying the Integrity of
    Deep Learning Training" (2611.09889); "ReplayAudit: Auditing Outsourced LLM Fine-Tuning via Sealed Records and
    Selective Replay" (2611.09607); "Proving the Wrong Program: Miscompilations as a Threat to zkVM Guarantees"
    (2611.04701, Tan, Chaliasos, Donaldson); "Performance Bottlenecks in Computing ZK Proofs on a GPU" (2611.09569,
    abstract read: a parallel PlonK study of the GPU parallelism vs memory trade-off and multi-GPU/RDMA designs, no
    numbers in the abstract). Read: listing, except as noted. —
    [arXiv cs.CR recent](https://arxiv.org/list/cs.CR/recent), [2611.09569](https://arxiv.org/abs/2611.09569)
    *Take:* the miscompilation paper is the analogue of our ONNX-importer trust question (we prove the lowered IR, not
    the ONNX file).

#### D. Prover engineering: streaming, memory, GPU, fields, upstream Jolt

35. **a16z/jolt pull requests, 4-9 November 2026.** Repo. Read: PR list and six PR descriptions via `gh`. Items that
    bear on jolt-atlas:
    - **Streaming kernels** (#2373-#2376, open, 2026-11-08): opt-in streaming kernels for the instruction-input
      sumcheck, the later rounds of the Spartan outer and product sumchecks, and RAM (a 32-byte-per-event sparse log
      replacing per-cycle columns); merged `--memory-budget` accounting (#2372).
    - **Whiteout: zero-knowledge Jolt over Akita** (#2340-#2366, open, 2026-11-06/07): a masked group opening over
      Akita, a Σ-protocol over a lattice pad commitment, constant-time samplers; opt-in because dudect flags a timing
      residual. Previews at `a16z/jolt` and `LayerZero-Labs/akita` branches `whiteout/preview`.
    - **Production field change** (#2356, #2379, open, 2026-11-06/08): `Prime128Offset801F`, p = 2^128 − 1,780,711,455,
      replaces `Prime128OffsetA7F7`; Akita switches to the same field (LayerZero-Labs/akita#326, #387).
    - **Akita commit micro-optimizations** (#2387 reuse adjacent one-hot NTTs: commitment-stage instructions −1.67% on
      an 83M-row Ethereum guest; #2325 vectorized trace digit extraction; #2339 exact Fp128 product accumulation: 47-51%
      fewer cycles per coefficient in Akita packing, about 1% of single-worker prover cost; #2370 subring convolutions).
    - **Crate moves**: `jolt-field` renamed `jolt-algebra` (#2279, merged); Akita's ring, CRT-NTT and FFT kernels being
      upstreamed into it (#2283-#2287).
    - **Lattice precompiles spec** (#2307): a GF(2^128) relation proving Keccak-f on every table slot with a one-hot
      pointer link from each call; Ethereum block rows fall 34.62%.
    - **Recursion** (#2268-#2273): "recipe" tables for the recursive verifier.
    — [a16z/jolt pulls](https://github.com/a16z/jolt/pulls), e.g. [#2376](https://github.com/a16z/jolt/pull/2376),
    [#2363](https://github.com/a16z/jolt/pull/2363), [#2379](https://github.com/a16z/jolt/pull/2379),
    [#2387](https://github.com/a16z/jolt/pull/2387), [#2339](https://github.com/a16z/jolt/pull/2339),
    [#2307](https://github.com/a16z/jolt/pull/2307)
    *Take:* three things arrive from upstream whether we plan for them or not: a field modulus change that touches our
    vendored Fp128, a ZK mode for Akita, and a streaming-kernel pattern we can copy for memory-bounded proving. The
    Keccak precompile spec is a worked example of mixing a GF(2^128) relation into a prime-field Akita proof.

36. **Weir: Bounded-Memory Indexing and Proving with Sequential Storage.** K. Lee, J. Kim, H. Oh. Received 2026-10-30.
    Preprint. Read: abstract. A read-write streaming schedule for a Shockwave-derived (sumcheck plus Reed-Solomon PCS)
    protocol: O(B + log T) field RAM, the same proof bytes as the in-RAM prover, and a fused Split-Butterfly-Merge that
    removes n−1 materialized transform passes. Numbers: under a 64 MiB cap the streaming prover completes all tested
    inputs; largest point has 64x the constraints of the largest in-RAM point at 3.33 MB prover RSS, 9.51 minutes. Same
    group: **Sluice** (bounded-memory Groth16, [ePrint 2026/1758](https://eprint.iacr.org/2026/1758), listing only). —
    [ePrint 2026/2689](https://eprint.iacr.org/2026/2689)
    *Take:* disk-backed sumcheck tables with byte-identical proofs; the extreme end of memory-bounded proving.

37. **Succinct Arguments with (almost) Minimal Time and Space Overheads from Symmetric-Key Primitives.** A. Block,
    C. Geren, P. Soni. Received 2026-10-09. Preprint. Read: abstract. Streaming multilinear PCS from Brakedown- and
    BaseFold-style codes, giving hash-only zkSNARKs for time-T space-S RAM computations with Õ(T) prover time and
    Õ(S + √T) space, or polylog proofs with Õ(T^1.5) time. Theory. — [ePrint 2026/2433](https://eprint.iacr.org/2026/2433)
    *Take:* the asymptotic floor for a streaming hash-based commitment if strategy 1 of section 7.6 is taken.

38. **MamaBearZKP: A Holistic Co-design of Prime Fields and Proving Stacks for High-Throughput ZKP on Modern CPUs.**
    J. Zhang, Y. Guo, T. Lu, J. Zhang (NUS), H. Cheng. Received 2026-08-15, revised 2026-09-08. ACM CCS 2026. Read:
    abstract. A 49-bit prime p = 2^49 − 2^34 + 1 chosen for AVX-512 IFMA (52-bit multipliers), with a vectorization
    framework for sumcheck and FFT in HyperPlonk and DeepFold. Numbers: single-thread up to 42x on ZeroCheck and 21x
    end to end vs a Goldilocks baseline; up to 18x single-thread end to end over Plonky3 (AVX-512 BabyBear). —
    [ePrint 2026/1698](https://eprint.iacr.org/2026/1698)
    *Take:* a 49-bit field holds an int8 matmul accumulator (about 28 bits) with room to spare; relevant to any
    small-field, hash-PCS variant of our prover on x86. No Apple Silicon numbers.

39. **Chimera: A Hybrid GPU Backend for Sumcheck Acceleration in Zero-Knowledge Provers.** K. Farheen, N. Tsoutsos
    (Delaware). Received 2026-07-07, revised 2026-10-16. Preprint. Read: abstract. Keeps polynomial tables on the GPU,
    returns only transcript values, hands small late rounds back to the CPU, and chunks when data exceeds GPU memory;
    proof format unchanged. Numbers (BN254 Spartan): sumcheck region 7.99x at 2^17; full proving 1.27-1.64x at
    2^24-2^27; standalone quadratic sumcheck 2.94x faster than ICICLE CUDA at 2^22. —
    [ePrint 2026/1388](https://eprint.iacr.org/2026/1388)
    *Take:* realistic expectation for bolting a GPU onto an existing CPU sumcheck prover: under 2x end to end unless
    commitment also moves.

40. **GPU WHIR recursion in production zkVMs (repos).** LambdaClass `lambda_vm` PR #1010 (2026-09-28, draft): Ethereum
    block 25368371 proved in 32.62 s with pure-WHIR recursion on GPU (same block: 60.18 s with STARK recursion, PR
    #1009), using Gruen's split in GKR layers and grinding only before queries. Matter Labs `zksync-airbender` PR #584
    (2026-10-27): WHIR packing on GPU with an explicit per-proof "soundness mode" (pessimistic default, a separate
    Ethproofs mode). ZKM `Ziren` PR #569 (2026-10-21): binary WHIR engines in the recursion VM; **Ziren** paper
    (S. Duan et al., [ePrint 2026/2330](https://eprint.iacr.org/2026/2330), listing only: MIPS32 zkVM that has proved
    mainnet blocks). Read: PR descriptions (first two), titles (third). Self-reported. —
    [lambda_vm #1010](https://github.com/yetanotherco/lambda_vm/pull/1010),
    [airbender #584](https://github.com/matter-labs/zksync-airbender/pull/584),
    [Ziren #569](https://github.com/ProjectZKM/Ziren/pull/569)
    *Take:* WHIR has become the default recursion and commitment layer across zkVM teams in autumn 2026, with GPU
    kernels available in open repos.

#### E. Commitments, proximity gaps, folding (new items the specialist researchers may not have)

41. **Counterexamples to Beyond-Johnson Proximity Gaps over Binary Fields.** Q. Dao, S. Kominers, J. Thaler, K. Zheng.
    Received 2026-10-09, revised 2026-11-03. Preprint. Read: abstract. On additive (subspace) evaluation domains,
    proximity gaps below the Johnson threshold fail: explicit constructions with superpolynomially many exceptional
    challenges and superpolynomial list sizes, covering the length-2^27 Binius64 domain and the sparse domains of LeanVM
    and Flock; e.g. rate 1/2, length 2^22: 53.125% agreement from 50% with probability at least 2^-20 over a 192-bit
    field. Formally verified in Lean (ArkLib). — [ePrint 2026/2424](https://eprint.iacr.org/2026/2424)
    *Take:* binary-field hash PCS (Magnetar's substrate) cannot use capacity-regime parameters; prime-field
    Reed-Solomon keeps that option. A point for staying in a prime field.

42. **Reed-Solomon Codes Beyond Johnson: Efficient Decoding and Smaller Cryptographic Proofs.** Q. Dao, S. Kominers,
    J. Thaler. Received 2026-09-16, revised 2026-11-03. Preprint. Read: abstract. Explicit list-decoding and mutual
    correlated agreement bounds below the Johnson radius (first-derivative threshold, then up to capacity) for large
    characteristic, and an improved above-Johnson MCA error from O(n/(qη^5)) to O(n/(qη^3)) in every characteristic.
    Numbers: proof-size savings at unchanged security in deployed systems: ProveKit 79.4 KB (11.1%), ZisK 13.2 KB
    (4.63%), LambdaVM 59.8 KB (4.85%). Verified in ArkLib. Companion: "The One-and-a-Half Johnson Bound Is Tight for
    Proximity Gaps of General Linear Codes" ([ePrint 2026/1894](https://eprint.iacr.org/2026/1894), listing only). —
    [ePrint 2026/2056](https://eprint.iacr.org/2026/2056)
    *Take:* if we benchmark WHIR or Ligerito over a prime field (step 8), these are the provable parameters to use.

43. **Vector Commit-and-Reduce Schemes.** A. Baweja, P. Mishra, T. Mopuri, B. Namboothiry (Penn). Received 2026-11-05.
    Preprint. Read: abstract. Generalizes vector commitments so that opening is an interactive reduction; "Canopy"
    reduces Merkle opening claims to proximity claims that can be batched and accumulated. Numbers: Bolt-based SNARKs
    (Bolt: CRYPTO 2026) 50% smaller proofs and 3x fewer verifier hashes for 5-15% prover overhead; Reed-Solomon SNARKs
    20% smaller and 1.5x fewer hashes for about 20%; hash-based PCD with O(λ√log N) recursion overhead. —
    [ePrint 2026/2790](https://eprint.iacr.org/2026/2790)
    *Take:* shrinks exactly the cost (Merkle paths) that makes hash-based proofs large; narrows the proof-size
    advantage we attribute to lattices in section 7.6. I could not locate the Bolt paper itself.

44. **Maltese: Succinct Polynomial Commitment from Lattices.** K. Cheng, N. Tyagi (UW), W. Nguyen (MSR). Received
    2026-09-17. ASIACRYPT 2026. Read: abstract. Merkle tree over Ajtai hashes with folding and sumcheck-based
    norm-control reductions, under standard Module-SIS with polylog verifier. Numbers: 335 KB openings at N = 2^30,
    about 300x smaller than prior polylog-verifier lattice schemes. — [ePrint 2026/2067](https://eprint.iacr.org/2026/2067)
    *Take:* a peer-reviewed comparison point for Akita on assumptions and proof size.

45. **Akita ePrint record.** Q. Dao et al. [ePrint 2026/1983](https://eprint.iacr.org/2026/1983), listing date
    2026-10-15. Read: listing ("commit time tied to nonzero entries, suited to memory checking"). Recorded for the ID;
    the specialist covers it. Related lattice-folding listings, snippet only: **Sonata** (certifying canonical digit
    openings in high-arity lattice folding, [2026/2011](https://eprint.iacr.org/2026/2011), revised 2026-11-09),
    **LatticeBlindFold** ([2026/1857](https://eprint.iacr.org/2026/1857)), **Accordion** (non-recursive multi-folding
    for batches: 64 SHA-256 instances up to 2.5x faster than 64 Plonk proofs, abstract read,
    [2026/2160](https://eprint.iacr.org/2026/2160)), **Aerie** (Falcon signature aggregation by the Akita authors,
    [2026/2501](https://eprint.iacr.org/2026/2501)).

#### F. Assurance: formal verification, Fiat-Shamir, tooling

46. **Jolt-QED: Formally Verifying Bytecode Expansions In Lean.** A. Biswas, Q. Dao, D. Ross, J. Thaler. Received
    2026-09-22, revised 2026-10-02. Preprint. Read: abstract. Lean 4 proofs that Jolt's expanded instruction programs
    emulate RISC-V (against the Sail-derived Lean model): 51 of 58 proved equivalent, via a generator that runs the Rust
    expander and emits Lean. — [ePrint 2026/2157](https://eprint.iacr.org/2026/2157)
    *Take:* the template for proving our ONNX lowering and op semantics against a reference; with FlashProve's Lean
    IOP proofs and ArkLib-verified bounds (items 41, 42), machine-checked artefacts are becoming expected. Upstream
    also ships HOL Light proofs of the Fp128 assembly (PR #2379).

47. **Zippel: A Language, Compiler, Runtime, and Analyzer for ZK Proof Systems.** S. Angel, P. Mishra, A. Ozdemir et al.
    Received 2026-10-16. IEEE S&P 2027. Read: abstract. A DSL resembling paper pseudocode whose type system keeps
    prover and verifier transcripts in sync and applies Fiat-Shamir correctly, with automated static analyses for
    completeness and soundness; 35 proof systems implemented (Sumcheck, Spartan, Dory, HyperPlonk, ...), competitive
    with hand-written code on multi-core. Related: **It Takes Two: Proofs of Work for Fiat-Shamir** (Bünz, Chen, Guan,
    received 2026-10-03, abstract read: one PoW cannot be both "strong" and "non-amortizing" when the verifier is much
    cheaper than the prover, so grinding and anti-diagonalization need two PoWs,
    [ePrint 2026/2326](https://eprint.iacr.org/2026/2326)); **How to prove more false statements: Fiat-Shamir
    limitations on (generated) R1CS** (Fenzi, [ePrint 2026/1838](https://eprint.iacr.org/2026/1838), listing only). —
    [ePrint 2026/2517](https://eprint.iacr.org/2026/2517)
    *Take:* same goal as our "every sumcheck is a symbolic relation, the verifier is derived" design; the grinding
    result is relevant because our transcript grinds before each challenge.

### Inferences

- The "pay-per-bit in a prime field" question has changed status since `docs/research-directions.md` was written.
  BitZ and Limber both claim integer commitments whose cost tracks bit size with proofs over a (random) prime field;
  Mikros claims bit-width awareness without binary fields; LaBinius claims a lattice commitment for bit witnesses at
  near hash speed. None was read past the abstract, so whether any gives an *exact* per-value bound usable for a
  rounding remainder is unverified.
- Approximate range checks have two independent lineages that now meet: FlashProve's random sums (hash-based) and the
  lattice JL line (RoK and Roll, PikkuFold, tensor-train projections). The lattice line has explicit soundness
  constants and a succinct verifier, which is what section 7.3 says is missing.
- Celer is the only lookup paper found that names LLM inference as its motivating workload and optimizes the number of
  commitments, our dominant cost.
- The quantization study (item 21) and the Seal-then-Sample fidelity result (item 22) both suggest activation and
  residual-stream precision, not weight precision, is the binding constraint, and that narrower bit-width does not
  translate proportionally into prover time. This tempers the int8 thesis.
- Open implementations of full-LLM provers multiplied in six weeks (FlashProve, three ports, SASH, stwo-zig frontend),
  and a neutral benchmark exists. Laptop-class hardware and generality are weaker differentiators than they were on
  1 November; small proofs and fast verification remain unclaimed by these systems (FlashProve 18.78 MB, SASH
  639.9 kB per token per Gbit).
- Upstream Jolt is changing the 128-bit modulus and adding ZK over Akita; both will reach jolt-atlas through the
  vendored crates.

### Gaps

- No full texts were read; every technical claim above is from an abstract page, a listing snippet, or a README, as
  marked. Numbers are the authors' own.
- ePrint IDs 2026/1400-2700 were covered by keyword and author search, not by reading the whole chronological listing
  (the `/complete/` page only exposed 2026/2709-2808 to the fetch tool, and ePrint rate-limited at about ten requests).
  Items with unusual vocabulary may have been missed.
- Not reached: ECCC 2026, Google Scholar and Semantic Scholar "cited by" lists, CRYPTO/CCS/USENIX/Asiacrypt 2026
  accepted-paper pages (venue labels above come from ePrint metadata), ZK Summit and ZKProof programmes, snark.fast,
  better.codes, zk.golf.
- "Bolt" (CRYPTO 2026), cited by item 43, could not be located on ePrint by title search.
- No 2026 paper on integer-only LLM quantization without retraining was found; the search returned only I-LLM
  (arXiv 2405.17849, 2024), LLM.int8 and SmoothQuant. Item 21 is the nearest recent work.
- No September-November 2026 announcements from Lagrange, Polyhedra, EZKL or Giza were found by search.
- Anthropic's production "inference verification" prototype has no public technical description.
- `inferenceproofs.ai` lists 19 systems; only FlashProve and SASH had numbers and I did not retrieve the other 17 names.
- The a16z/jolt PR scan read descriptions for six PRs only and covers 4-9 November in detail; September-October PRs
  were not enumerated.

## Which ten items have the highest potential impact on jolt-atlas, and why?

### Takeaway

The top of the ranking is dominated by work that makes committing small integers cheap or makes proving smallness
approximate and succinct, since that is where 25 s of our 40 s GPT-2 proof goes; the next tier is evidence about
whether narrower quantization pays off, and what upstream and competitors have just shipped.

### Cited Findings

1. **BitZ** — claims the construction our document lists as non-existent: integers committed bitwise over a binary
   hash PCS, proved over a prime field, range checks nearly free, 9M 32-bit multiplications per second on a MacBook
   Air. — [ePrint 2026/2141](https://eprint.iacr.org/2026/2141)
2. **Tensor-train random projections** (with **PikkuFold**'s certified JL constants) — a succinctly verifiable
   approximate shortness check, about 4 KB, sumcheck-linearized, in a lattice setting. —
   [ePrint 2026/2146](https://eprint.iacr.org/2026/2146), [ePrint 2026/1809](https://eprint.iacr.org/2026/1809)
3. **Celer** (CRYPTO 2026) — lookup argument with commitments sublinear in the number of queries, benchmarked at
   m = 2^28, n = 2^16 from Llama inference, 9.4x over LogUp. — [ePrint 2026/1453](https://eprint.iacr.org/2026/1453)
4. **Bits Under ZK-LLM** — activation precision and the RMSNorm inverse-square-root table dominate accuracy; lower
   bit-width does not cut proving cost proportionally. — [arXiv 2609.36437](https://arxiv.org/abs/2609.36437)
5. **Limber** — integer commitments with o(1) overhead over any field PCS. —
   [ePrint 2026/1635](https://eprint.iacr.org/2026/1635)
6. **Mikros** and **Sumcheck bit packing** — bit-width-aware proving without binary fields; per-bit linear claims on
   packed bits with only linear constraints. — [ePrint 2026/2646](https://eprint.iacr.org/2026/2646),
   [ePrint 2026/2770](https://eprint.iacr.org/2026/2770)
7. **Upstream Jolt, November 2026** — modulus change to 2^128 − 1,780,711,455, Whiteout ZK over Akita, streaming
   kernels, Akita one-hot commit optimizations. — [a16z/jolt pulls](https://github.com/a16z/jolt/pulls)
8. **The new open inference provers and their benchmark** — stwo-zig tensor frontend (virtual requant remainders,
   M31, Metal, self-reported 28x over FlashProve's CPU prover on one kernel), SASH (Llama 3.1 8B W8A8), FlashProve on a
   16 GB consumer GPU, and inferenceproofs.ai's per-token-per-Gbit metric and claim cards. —
   [stwo-zig #321](https://github.com/teddyjfpender/stwo-zig/pull/321),
   [SASH](https://github.com/sg-ai-safety-hub/zk-inference-prover),
   [consumer GPU port](https://github.com/whyagents/flashprove-consumer-gpu),
   [benchmark](https://github.com/ai-verification-observatory/inference-proofs)
9. **LaBinius** — lattice PCS for bit witnesses within 1.2x of hash-based provers, proofs under 100 KiB, no bitness
   checks. — [ePrint 2026/2103](https://eprint.iacr.org/2026/2103)
10. **Dao-Kominers-Thaler proximity-gap pair** — beyond-Johnson gaps fail on binary additive domains but hold with
    explicit bounds in large characteristic, with measured proof-size savings. —
    [ePrint 2026/2424](https://eprint.iacr.org/2026/2424), [ePrint 2026/2056](https://eprint.iacr.org/2026/2056)

### Inferences

Reasons for the order (all my judgement):

1. BitZ: if its range guarantee is exact, it resolves section 5 with a construction that keeps our prime-field
   sumchecks; highest value per hour of reading.
2. TTRP: turns FlashProve's approximate range check, the technique our document calls the one "with the largest
   consequence for us", into something specified for our kind of stack; could remove 8 of 12 digits per einsum output
   without changing the commitment scheme.
3. Celer: a drop-in competitor to one-hot digits in our exact regime, peer reviewed, from the zkGPT/zkLLM group.
4. Bits Under ZK-LLM: cheap to read and may redirect the int8 plan toward per-operator precision before any tracer
   work is done.
5. Limber: the PCS-agnostic version of item 1; lower because no throughput number for plain integer arithmetic.
6. Mikros and bit packing: mechanisms rather than systems; they matter if we build option C ourselves.
7. Upstream Jolt: not research, but unavoidable integration work with soundness-relevant consequences (field change),
   plus free performance and a ZK mode.
8. New provers and benchmark: defines what we are compared against and where we can still claim a position (proof
   size, verifier time); the stwo-zig "virtual requant remainder" claim may be a transferable trick.
9. LaBinius: keeps a lattice route to bit commitments alive; binary-field evaluation limits direct reuse.
10. Proximity-gap pair: decides parameters, and argues for prime over binary fields, if we move to a hash-based PCS.

Runners-up: Seal-then-Sample (sampled audit mode), Nenya/Rotor (native Z/2^k), Vector Commit-and-Reduce (smaller
hash proofs), Weir (disk-backed proving), Jolt-QED (Lean template), Maverick and Gupta-Katz-Miers (non-SNARK
competitors), Hollow-LLM (statement caveat for committed weights).

### Gaps

- The ranking rests on abstracts. Items 1, 2, 3 and 5 need a full read to confirm (a) exact versus approximate
  bounds, (b) field and PCS requirements, (c) whether benchmarks include commitment time.
- No like-for-like numbers exist between any of these and Akita one-hot digit commitment; every speedup quoted is
  against the authors' own baselines.
