# Measurement plan

What to build and run, and in what order, to settle the open questions behind
`docs/prover-plan.md` (which condensed the earlier roadmap and research documents once Phases 1–3
had run; the item labels D1–D5, B1–B5, C1–C4, A1–A4 used below are kept there). Phases 1–3 are
done and their results sit under each gate. A cold build of this workspace takes about ten minutes, so the plan is organised around
builds, not around questions: each build is done once and answers as many questions as possible.

## 1. Ground rules

- **One Cargo invocation at a time**, across people and agents. Parallel invocations fight over
  the build lock.
- **Never benchmark while anything else is compiling or proving.** Timings on a busy machine are
  not usable. Reading and writing can overlap a build; measuring cannot.
- **Run built binaries directly** (`$CARGO_TARGET_DIR/release/examples/prove ...`), not through `cargo run`.
  `cargo run` re-checks the workspace every time and can trigger a rebuild after an unrelated edit.
- **Keep a copy of every binary that produced a number.** Copy it out of `target/` with the commit
  hash in its name. A later code change then needs no rebuild of the old version to compare.
- **Put instrumentation in the example, not the library, wherever possible.** Editing
  `jolt-atlas-core/examples/prove.rs` rebuilds only the example and its link step. Editing a
  library crate rebuilds that crate and everything above it. Under the `release` profile the
  link step is fat LTO over the whole program, so an example-only edit still costs about 12
  minutes (measured). Anything that is not a timing run goes through
  `--profile build-fast` (no LTO, incremental: 4 m 46 s cold, seconds after that); its binary
  lands in `$CARGO_TARGET_DIR/build-fast/examples/`.
- **Toggle variants with flags or environment variables, never with Cargo features.** Changing
  features rebuilds.
- **Batch library edits.** Decide every instrumentation point first, land them in one change,
  build once.
- **Release for timing, test profile for correctness, and keep them apart.** They are separate
  builds of about ten minutes each. Phases 1–3 need no test build at all.
- **No `--release` on tests** (repository rule); the test profile is already `opt-level = 3`.
- **Record every run** in one results file: commit, binary, command, machine state, wall time,
  stage split, proof size, verify time, peak memory.

Machine: M3, 11 cores, 18 GB. The target directory is `$CARGO_TARGET_DIR`
(`~/.cargo/shared_target`), not `target/`; the release profile has been built once (Phase 1), the
test profile not yet. Models present: `gpt2`, `gpt2-tiny`, `bge-small-en-v1.5`, `bge-tiny`, `qwen2-tiny`,
`op_gauntlet`. Full Qwen (2.5 GB) is not downloaded.

## 2. Phases

Each phase lists its build cost, what it runs, what it answers, and the decision it feeds. Do not
start a phase's build until the previous phase's gate is read.

### Phase 0: no build

Reading and writing only. Runs alongside Phase 1's build.

- Read BitZ in full (ePrint 2026/2141; needs a manual PDF download).
- Read FlashProve's Lean range-test theorem and write the soundness argument for the row check on
  a virtual tensor.
- Decide the exact contents of the Phase 2 census, so it is one edit.

### Phase 1: baseline (build 1: release `prove` example, cold)

No code changes. One build, then many runs.

```bash
cargo build --release -p jolt-atlas-core --example prove
cp $CARGO_TARGET_DIR/release/examples/prove bench/bin/prove-$(git rev-parse --short HEAD)
```

Runs, each with `--trace-terminal` and under `/usr/bin/time -l` for peak memory:

| Run | Purpose |
|---|---|
| `gpt2-tiny`, `qwen2-tiny`, `bge-tiny` | Smoke test; confirms the binary and the harness |
| `gpt2 --seq-len 16 --stats` | The reference number (40 s) with stage split, clamp widths and buckets |
| `gpt2 --seq-len 32`, `64`, then `128` | Scaling with context. Stop at the first length that exhausts memory |
| `bge-small-en-v1.5` | A second real model with a different op mix |
| `gpt2 --seq-len 16` three times | Run-to-run variance, so later differences can be judged |

Also build and run `dump_lowered_ops` for each model (small build in the same profile):

```bash
cargo build --release -p atlas-onnx-import --example dump_lowered_ops
$CARGO_TARGET_DIR/release/examples/dump_lowered_ops models/gpt2/network.onnx \
    batch_size=1 sequence_length=16 past_sequence_length=0
```

Three splits the decisions need, all already present as spans (checked before the build):

- witness generation (`ONNXProof::polynomial_map`) inside the commit stage;
- the opening stage as the reduction sumcheck plus the per-class Akita openings
  (`AkitaScheme::prove_rlc`, with one `commit_class` span per class);
- the deferred lookup sumchecks (`deferred_lookups::prove_all`) as their own total.

Answers: stage split and per-op spans on the current backend (roadmap measurement 1), lowered op
histograms (3), cost growth with context (5), peak memory, and the one number everything else
waits on: whether committing digits or running lookup sumchecks dominates per lookup on Akita.
On the July 2026 HyperKZG backend it was the sumchecks.

**Gate 1.** Which term dominates per lookup: commit, or lookup sumchecks? If sumchecks, the
roadmap's digit-savings table is an upper bound and items that remove sumcheck instances (B1, B2,
C2) move ahead of items that only remove digits. Also: what share of the opening stage is per-class
openings (this bounds the one-commitment layout).

#### Phase 1 results (2026-11-10, commit 25c7218f1, binary `bench/bin/prove-25c7218f1`)

Build 1 took 11 m 52 s cold. The existing spans already separate witness generation
(`ONNXProof::polynomial_map`) from the Akita commit (`AkitaScheme::batch_commit_all`), the
reduction sumcheck from the per-class openings (`AkitaScheme::prove_rlc`), and the deferred
lookups (`deferred_lookups::prove_all`), so no span edit was needed. Logs are in `bench/runs/`;
`bench/spans.py` aggregates them; `bench/phase1.sh` is the run table.

| Run | prove s | witness | commit | IOP (deferred) | reduction | openings | verify s | proof MiB | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-tiny / qwen2-tiny / bge-tiny | 2.8 / 3.2 / 2.9 | | | | | | 0.16–0.20 | 0.7–1.0 | 0.5–0.6 |
| gpt2, 16 tok | 40.2 | 0.8 | 23.6 | 5.7 (3.1) | 1.8 | 5.0 | 2.24 | 2.38 | 6.0 |
| gpt2, 32 tok | 74.2 | 1.1 | 49.2 | 8.0 (4.3) | ~2 | 7.3 | 2.76 | 2.90 | 8.4 |
| gpt2, 64 tok | 145.1 | 1.7 | 104.0 | 13.4 (7.3) | ~3 | 11.0 | 3.41 | 3.60 | 10.4 |
| gpt2, 128 tok | 285.9 | 2.8 | 207.0 | 25.5 (14.4) | 5.0 | 20.5 | 5.12 | 5.15 | 11.2 |
| bge-small, 16 tok | 16.9 | 0.4 | 4.8 | 3.9 (1.8) | 1.9 | 5.4 | 1.44 | 3.88 | 2.0 |

Run-to-run variance on `gpt2 --seq-len 16`: 40.24, 40.50, 39.55, 39.81 s (±1.2%). Differences
under 3% are noise. The stages do not sum to `prove`: 3 s at 16 tokens and 23 s at 128 are
unspanned (trace execution inside `prove`, `Prover::new`, `finalize_proof`); the Phase 2
example edit times `model.trace(inputs)` separately.

**Gate 1 verdict: commit dominates, by a wide margin, and it is one-hot digits.** On GPT-2 at 16
tokens the Akita commit is 59% of the prover and the three `OneHot num_vars=24, 64 slots` classes
alone are 20.9 s of it: 192 polynomials of 2^20 K=16 digits, about 104 ns per committed digit.
Witness generation is 2% (the README's "25 s witness + commit" is 0.8 s + 23.6 s). The IOP is
14% and the deferred bucket sumchecks (the clamp split sumchecks and the Booleanity chunk
checks; the buckets run no read-raf and no ra-virtualization) are 3.1 s of that. The opening stage is 17%, of which 72% is the per-class Akita
openings and 28% the reduction sumcheck over 1,833 claims. Scaling is linear in context
(1.84x, 1.95x, 1.97x per doubling), the commit share rising to 72% at 128 tokens; 128 tokens fits
in 11.2 GB.

The July HyperKZG profile does not transfer: on Akita the digit-savings table in the roadmap is the
first-order lever, not an upper bound. Items that remove committed digits (D2, then C1–C3, B1's
read-raf digits, D1 on wide rows) move ahead of items that only remove sumcheck instances.

Where the digits come from (`--stats` and the Phase 2 census, GPT-2 16 tok): 14 packed clamp
buckets of 2^20 rows, each committing **8 output chunks** of `OUT = out + 2^31` per row
(`clamp_split.rs`: 112 polynomials) plus `w/4` slack chunks for its static width `w ∈ {40, 48,
56, 64}` that are empty off saturated rows (168 polynomials, 152 of them entirely empty), and 9
remainder buckets at 14 bits (4 chunks) plus one at 28 (7) = 39. An earlier version of this
paragraph counted `w/4` clamp digits per element from the static width; the committed count is
8 per element whatever the width, and empty columns cost nothing to commit (Phase 3). Of the
non-empty committed digit rows, the clamp output chunks are 71%, the remainders 22%, every
per-node polynomial together 7%.

bge-small is the other regime: 4,902 commitments across 27 small one-hot classes (2^19 × 64),
and the opening stage (7.4 s, 44%) exceeds the commit (4.8 s). Per-class opening cost scales
with the class count, so the one-commitment layout matters for models with many small buckets.

Lowered-op histograms (`bench/runs/lowered-*.txt`): GPT-2 1,579 nodes, 97 einsums (73
`mk,kn->mn`, 24 `abmk,abnk->abmn`), 183 Add, 147 Mul, 344 Broadcast, 176 Reshape, 84 MoveAxis,
25 each of Rsqrt / MeanOfSquares / Sum / Sub, 12 each of SoftmaxLastAxis / Tanh / Cube / Slice /
IsNan / Iff(×24); the mask is lowered as 12 Slice → Iff pairs over constants.

### Phase 2: census (build 2: example-only rebuild)

One edit to `prove.rs`, extending `--stats`. Everything here is computable from the model and the
trace without proving, so the runs take seconds.

- **Committed digits, and bits, by producing op and by row width**, split into overflow digits,
  remainder digits and lookup address digits. Static, from node shapes. Bits as well as digits,
  because the BitZ option prices by bit.
- **Rows by width**, and how many overflow digits sit on rows wider than 80, 112, 176, 1,024 and
  4,096. This sizes the row check (roadmap D1) and its union-bound term.
- **Width audit per one-hot family**: the encoded width against the range the value can reach
  (the method of jolt-atlas PR #258). Any family wider than its reachable range is a zero-risk
  digit saving.
- **Value magnitudes from the trace**: per einsum output and per residual-stream tensor, the
  largest absolute value against its static clamp bound; the saturation count per bucket; the
  spread of `max − x` in softmax rows.
- **Tracer clamp-width sweep** (no prover build): per-site power-of-two shifts with a per-op clamp
  width of 16, 20, 24 and 32 bits on GPT-2, qwen2-tiny and bge-small, scored by perplexity ratio
  and top-5 agreement against float on text windows, plus the same fidelity measurement of the
  current scheme. Decision rule proposed by the reading: the smallest digit count at perplexity
  ratio ≤ 1.01 and top-5 ≥ 0.96. This decides roadmap D2's first role (8 → 3–4 clamp digits).
- **Biased layers**: how many einsum nodes feed a constant Add, and their output sizes (C1).
- **Activation inputs**: how many activation nodes read an einsum output directly (C2).
- **Slice and Concat alignment**: whether any misaligned case survives lowering (B4).

Run on `gpt2` at 16 and 64 tokens, `bge-small-en-v1.5` and `qwen2-tiny`.

Answers: roadmap measurements 2 and 4, and the share of the 25 s each roadmap item can touch.

**Gate 2.** The main decision point.

- If the clamp-width sweep lands GPT-2-class models at 12–16 bits, per-op clamp widths (D2) take
  most of the overflow digits exactly, and the row check (D1) is left the wide residual-stream
  rows of larger models.
- If most remaining overflow digits sit on wide rows, D1 goes to Phase 4.
- If they sit on narrow rows (attention pairs, short sequences), D1 pays little and the fusions
  (C1–C3) move ahead of it.
- If static clamp bounds are far above observed magnitudes, compile-time bounds or declared widths
  are enough to make D1 output-preserving.

#### Phase 2 results (2026-11-10, `prove --census`, binary `bench/bin/prove-census-fast`)

The census is a `--census` mode of the `prove` example: it executes the trace and prints the
committed-column census, the clamp audit, the softmax spread and the fusion candidates without
proving (3 s on GPT-2 at 16 tokens, 13 s at 64). Text inputs are truncated to the longest
power-of-two token prefix. Outputs: `bench/runs/census-*.txt`. Runs: GPT-2 on synthetic ids
(16 tokens) and on three texts (prose 64, Python 64, finance 32 tokens), bge-small on prose
(64), qwen2-tiny synthetic (16). The numbers below are stable across the three texts to within
a few percent; GPT-2 figures are the 64-token prose run.

**Where the committed digits are (GPT-2, 64 tokens).** Counted from the polynomials the prover
commits (`global_clamp::bucket_witnesses` plus the per-node list), in digit rows (one row = one
cycle of one K = 16 one-hot polynomial): clamp output chunks 112 polynomials, 2^28.8 rows; clamp
slack chunks 168 polynomials, 2^29.4 rows but 152 of them entirely empty and 2^5 nonzeros in
all (the two saturated elements); remainder chunks 39 polynomials, 2^27.2 rows; every per-node
polynomial together 2^25.8. Since empty columns are free to commit (Phase 3), the paid digit
rows split **71% clamp output chunks (8 per element), 22% remainders (4 per element), 7%
everything else**. Of the 7%, the Tanh lookups (`ActivationClampRaD` 96 + `ActivationSmallRaD`
60) are 70% and the softmax families 30%; rsqrt, mean-of-squares and gather are under 0.3%
together. bge-small has the same shape (Erf for Tanh; 1,256 output-chunk polynomials in 157
small buckets, 1,828 slack polynomials all empty).

**Static vs needed clamp width.** Static widths are 40–64 bits. On real text the traced
quotients need 13–29 bits: residual-stream Adds 17–23, einsum outputs 16–24, Muls 15–23; the
27–29-bit nodes are the LayerNorm chain (Sum, Sub, MeanOfSquares, Add: 19–20 nodes each, i.e.
one per norm) and the Cubes (24–28, one at 33 bits). Exactly one node saturates today (node 436,
Cube, 2 of 262,144 elements). The static width sets only the slack chunks, which are empty and
free; what a declared width can shrink is the 8 output chunks, to `⌈w'/4⌉`:

| per-op width w' | GPT-2 nodes / elements over | output-chunk rows vs today | bge-small nodes over | vs today |
|---|---|---|---|---|
| per-node observed | — | 1.47x | — | 1.58x |
| 24 | 112 / 486 | 1.29x | 0 | 1.33x |
| 28 | 39 / 67 | 1.14x | 0 | 1.14x |
| 32 | 1 / 2 (the saturating Cube) | 1.00x | 0 | 1.00x |

(Nodes over w' keep 8 chunks; the ratio is over all clamp output-chunk rows. An earlier version
of this table measured against `w/4` digits per element and showed 1.5–2.2x; that baseline was
wrong.)

**Rows.** 92% of GPT-2's clamp output-chunk rows sit on rows at least 1,024 wide and 63% on rows at least
4,096 wide (the MLP); bge-small has 48% on rows of 1,024 and nothing wider.

**Softmax.** On GPT-2 text, 48–50% of live lanes have `max − x ≥ 2^17` (8.0 at scale 2^14):
half of the exp lookups sit at the saturated end of the table. On bge-small it is 8–24% per
layer.

**Fusion candidates.** 72 einsum → constant-Add (bias) pairs on GPT-2 (2^22.8 output elements,
about 12% of the clamp output-chunk rows, C1); none on bge-small or qwen2-tiny. No activation reads an einsum
directly on any of the three (the bias Add sits between), so C2 is empty until C1 lands. Every
Slice is aligned (B4 has nothing to do on these models). LayerNorm is not fused: each norm is
nine lowered ops (Sum, Mul, Sub, MeanOfSquares, Add, Rsqrt, Mul, Mul, Add), eight of them
clamped, and holds the widest quotients in the model.

**Trace execution** is 3.0 s at 16 tokens and 12.8 s at 64 (the unspanned gap in Phase 1),
7–9% of the prover and growing linearly; the tracer's i64 accumulations are worth a profile
of their own.

**Gate 2 verdict.** The sweep's premise was wrong twice over. At scale 2^14 the honest
quotients need 17–29 bits, so a 12–16-bit per-op width is a change of scale, not of clamp
width (FlashProve's 12-bit activations are at a far coarser scale); and the clamp buckets
already commit 8 output chunks per element regardless of the static width (the width only sizes
the empty slack chunks), so a declared width can only take 8 chunks to `⌈w'/4⌉`: **1.3–1.5x on
the clamp output chunks** at the widths that keep today's outputs (uniform 24 with 112 nodes
excepted, or per-node observed), about 1.2–1.3x on all committed digit rows. It is exact by
construction (every quotient but the saturating Cube stays inside its observed width on these
inputs; the margin on unseen inputs is a tracer-side assertion under partial semantics), so the
accuracy decision rule is retired, and D2 drops to a companion item. Since 92% of the output
chunks sit on rows at least 1,024 wide, the **row check (D1) is the larger lever**: at 1–2
amortised digits per element it takes those rows down roughly an order of magnitude; and the
encoding of whatever remains is Phase 3's question. C1 (bias fusion) is worth 12% of the clamp
output chunks on GPT-2 and nothing on the encoder models; a fused LayerNorm would remove the
widest eight clamps per norm and is a new roadmap item.

### Phase 3: commitment microbenchmarks (build 3: `atlas-pcs` only)

Independent of Phases 1 and 2 and of the prover. It can be built any time nothing else is
compiling, and run any time nothing else is measuring.

An ignored test or example in `atlas-pcs` that commits, at GPT-2 volume (the column sizes from the
Phase 1 `--stats` output):

- K = 16 one-hot digit columns, as today, with witness generation timed separately;
- dense digit columns with values in {0, 1}, [0, 4) and [0, 16), plus the cost of the degree-(2b−1)
  zero-check term in a batched sumcheck (roadmap D3, SuperNeo's design);
- **the BitZ arm**: build the BitZ proof-of-concept (`albert-garreta/BitZ-PoC`, a scratch crate
  outside the workspace with its own `target/`) and run its commitment at 2^30 bits with a
  128-bit prime as the target ring so the extra round triggers, at 100 and 128 bits. Threshold
  from the reading: under 3 s single-threaded and more than 5x below the isolated Akita commit of
  2^28 one-hot digits means write the option-C spec.

Also in this phase, all as small-crate microbenchmarks:

- **Virtual-remainder grand product**: 2^24 virtual 16-bit remainders checked (a) with today's
  four one-hot digits plus Booleanity, ra-virtualization and the read check, (b) with a Quark
  grand product over committed multiplicities (Celer's Protocol 1 without the Pippenger side), on
  Fp128 + Akita, recording the witness, commit and sumcheck split and the soundness bound. This
  decides roadmap D4 and, independently of Phase 1, whether commit or sumcheck dominates one lookup.
- **Fp128 sumcheck kernels**: the small-value prover and univariate skip on `Σ eq·a·b` over 2^20
  i32 values; delayed reduction and the one-reduction small-by-big fold (Fp128 versions of ePrint
  2026/587's Tables 4 and 6); the balanced product kernel on 8- and 16-factor products. These
  price roadmap A1, A2 and group E before anyone writes them into the prover.

Record time per committed value, per nonzero and per bit, and peak memory.

Answers: whether dense tiny digits beat one-hot digits on Akita (D3); whether BitZ's 0.52 ns per
bit holds on our machine with the extra round (D5, research option C); whether the remainder
digits can go (D4); and which sumcheck kernels pay on a two-limb field.

**Gate 3.** Three outcomes decide the strategy:

- If BitZ meets its threshold, the lattice-only end state is a decision to be made explicitly
  against that number (`docs/prover-plan.md` section 5), not a default.
- If dense tiny cells are several times cheaper per value than one-hot digits, D3 is the
  lattice-side path and moves up.
- If neither, option B (one-hot digits, fewer of them) is the end state and the remaining work is
  the fusions, D2 and the sumcheck kernels that measured above 1.3x.

#### Phase 3 results so far (2026-11-10)

**Akita commit microbenchmark** (`atlas-pcs/examples/commit_bench.rs`, release, 4 m 28 s to
build; binary `bench/bin/commit_bench`; outputs `bench/runs/commit-bench-*.txt`). One 4-bit
digit per row, 64 columns, committed four ways; per-digit costs are flat from 2^16 to 2^20 rows.

| arm (same 2^26 digits) | cells | witness | commit | ns / digit | ns / key bit | vs one-hot |
|---|---|---|---|---|---|---|
| K = 16 one-hot (today) | 2^30 | 0.14 s | 9.6 s | 143 | 36 | 1x |
| K = 16 one-hot, 24 of 64 columns empty | 2^30 | 0.14 s | 6.1 s | 144 per nonzero row | — | — |
| dense u8 in [0, 16), 1 column | 2^26 | 0.12 s | 0.81 s | 12 | 3.0 | **11.8x** |
| dense u8 in [0, 4), 2 columns | 2^27 | 0.18 s | 2.5 s | 37 | 9.4 | 3.8x |
| dense Boolean, 4 columns | 2^28 | 0.23 s | 3.8 s | 57 | 14 | 2.5x |

A fifth arm, added after the correction to the census: 64 one-hot columns of which 24 are
entirely empty (the clamp buckets' slack chunks) commits in 6.06 s against 9.66 s for 64 full
columns, i.e. 40/64 of it: **an empty column costs nothing**, and Akita's one-hot cost is per
nonzero row (~144 ns), not per cell of the K × rows object. Peak RSS 8.8 GB for the arms in one
process; repeat runs of the dense arms vary by ±30% (base-4 1.7–2.5 s, Boolean 3.8–4.6 s), the
one-hot arm by 1%. The one-hot arm reproduces Phase 1's per-class cost (~104 ns per committed
row inside the prover, where a share of the rows are padding).

Reading: a committed one-hot row costs 144 ns; a dense value costs 12 ns; so the one-hot
encoding is paying about 12x for the 16 cells it spreads one digit over, and sparsity across
rows buys nothing because every row has a digit. The catch is the range proof: a dense column
in [0, 16) has no cheap constraint (a degree-15 zero-check, or a lookup, which is one-hot
again), so the usable arms are base-4 digits with a degree-4 zero-check, 5 with eq (SuperNeo's design,
3.8–5.7x on the commit) or Booleans with Booleanity (2.1–2.5x). For the clamp output chunks (71%
of paid rows) and the remainders (22%) a base-4 decomposition needs no lookup machinery:
committed digits, one zero-check, and the linear recomposition, replacing today's one-hot
output chunks under the split sumcheck and Booleanity chunk check (`clamp_split.rs`; the
buckets already run no read-raf). The split sumcheck itself keeps its shape. The row check (D1)
then applies to the decomposed form the same way. For the address digits of real table lookups
(Tanh, softmax, 7% of rows) the one-hot column can stay, or be virtualised from the digits.

Not yet run: the virtual-remainder grand-product test (D4), the Fp128 sumcheck kernels
(A1, A2, E), and the **BitZ arm**. The BitZ proof-of-concept is checked out at
`/Users/kg/swe/icme/scratch/BitZ-PoC` (toolchain 1.98.1, own target dir); building and running
downloaded code is outside what the agent may do unattended, so it is run by hand:

```bash
cd /Users/kg/swe/icme/scratch/BitZ-PoC
CARGO_TARGET_DIR=$PWD/target RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --locked --example bitz_bench
CARGO_TARGET_DIR=$PWD/target RAYON_NUM_THREADS=1 target/release/examples/bitz_bench 28 --reps 3
CARGO_TARGET_DIR=$PWD/target target/release/examples/bitz_bench 30 --reps 3
```

2^28 bits is one GPT-2 one-hot class (2^26 digits × 4 key bits); the threshold is under 3 s
single-threaded at 2^30 and more than 5x below the Akita one-hot commit of the same digits
(9.6 s for 2^26 digits above, so ~38 s for 2^28 digits). The example samples its own
transcript prime (`min(113, 127 − t − 1)` bits); the extra round for a fixed 128-bit target
ring is not exercised by it and would be an addition on our side.

**Gate 3, lattice side: decided.** Dense tiny digits are several times cheaper per digit than
one-hot digits on Akita (3.8x for base 4, 2.5x for bits, before the sumcheck side, which also
gets simpler for range checks). D3 is the lattice-side path and moves up, with D1 on top of it.
The end state was decided on 2026-11-10 without the BitZ arm: Akita stays, BitZ is kept in view
(`docs/prover-plan.md` section 5). The arm above is optional and off the critical path.

### Phase 4: prototypes (builds 4 and 5: release and test, both full)

Only what Gates 2 and 3 selected. This is the first phase that changes protocol code, so it is the
first that needs the test build.

- Prototype on a branch, one item per branch. The row check's prototype (if selected) is a
  standalone relation test on one GPT-2 einsum output of width 3,072: the random matrix drawn
  after the witness batch, the projection digits as a second batch through the existing one-hot
  range check, the eq-weighted linking relation, and a negative test showing the kernel-steering
  attack when the matrix is drawn before the remainder commitment.
- Iterate on a small model: `gpt2-tiny` or a probe model, with a filtered test run
  (`cargo nextest run -p jolt-atlas-core -E 'test(<name>)'`). Build the test profile once and keep
  iterating inside it; do not switch to release until the prototype passes.
- When it passes: one release build, then the Phase 1 run table on the new binary against the
  saved baseline binary.
- Before any merge: the zoo and soundness filters, then one real-model run, as the repository
  rules require.

### Phase 5: pin bump (full rebuild of both profiles)

Bumping the Jolt and Akita pins changes dependencies, so every profile rebuilds from cold. Do it
on its own branch when no prototype is in flight, and re-run the Phase 1 table for a before and
after. It may be worth doing before Phase 4 if Gate 1 shows commit and field kernels dominate,
since upstream's kernel work since the pin lands there.

## 3. Build budget

| Build | Profile | Trigger | Phases served |
|---|---|---|---|
| 1 | release, `prove` example | once, cold | 1 |
| 2 | release, example only | one edit to `prove.rs` | 2 |
| 3 | `atlas-pcs` test or example, plus a scratch crate | once | 3 |
| 4 | test profile, core | first protocol prototype | 4 |
| 5 | release, full | prototype passes | 4 |
| 6, 7 | release and test, cold | pin bump | 5 |

Phases 0–3 cost two full builds' worth of compile time in total and produce every number the
strategy decision needs. Nothing in them changes protocol code or needs the test suite.

## 4. What each open question waits on

| Question | Settled by |
|---|---|
| Does commitment dominate per lookup on Akita, or do the lookup sumchecks? | Phase 1 (span split), Phase 3 (grand-product test) |
| How many clamp digits do per-op widths remove exactly, at what accuracy? | Phase 2 (tracer sweep) |
| How much does the row check remove on what remains? | Phase 2 (row census) |
| Can the row check be output-preserving without a semantics change? | Phase 2 (magnitudes); the proof is written |
| Do the bias, activation and attention fusions pay? | Phase 2 |
| Dense tiny digits against one-hot on Akita? | Phase 3 |
| Does BitZ's per-bit cost hold on our machine with the extra round? | Phase 3 |
| Can the remainder digits go (virtual-key lookup)? | Phase 3 |
| Which Fp128 sumcheck kernels pay? | Phase 3 |
| How does cost grow with context? | Phase 1 |
| What do upstream's post-pin kernels give us? | Phase 5 |
| Lattice-only end state, or a hash-committed digit layer? | Decision after Gate 3 |

Phase 0's reads are done. Not yet reconciled: the July 2026 HyperKZG GPT-2 figure (about 10 s in
PR #258) against today's 40 s; Phase 1 should record sequence length and model file for every run
so one workload can be named.

After Gate 3, the roadmap and the research document can be condensed into a single plan with
measured numbers in place of estimates.
