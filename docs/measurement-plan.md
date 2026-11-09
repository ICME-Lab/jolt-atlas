# Measurement plan

What to build and run, and in what order, to settle the open questions in
`docs/optimization-roadmap.md` and `docs/research-directions.md` before condensing them into one
plan. A cold build of this workspace takes about ten minutes, so the plan is organised around
builds, not around questions: each build is done once and answers as many questions as possible.

## 1. Ground rules

- **One Cargo invocation at a time**, across people and agents. Parallel invocations fight over
  the build lock.
- **Never benchmark while anything else is compiling or proving.** Timings on a busy machine are
  not usable. Reading and writing can overlap a build; measuring cannot.
- **Run built binaries directly** (`target/release/examples/prove ...`), not through `cargo run`.
  `cargo run` re-checks the workspace every time and can trigger a rebuild after an unrelated edit.
- **Keep a copy of every binary that produced a number.** Copy it out of `target/` with the commit
  hash in its name. A later code change then needs no rebuild of the old version to compare.
- **Put instrumentation in the example, not the library, wherever possible.** Editing
  `jolt-atlas-core/examples/prove.rs` rebuilds only the example and its link step. Editing a
  library crate rebuilds that crate and everything above it.
- **Toggle variants with flags or environment variables, never with Cargo features.** Changing
  features rebuilds.
- **Batch library edits.** Decide every instrumentation point first, land them in one change,
  build once.
- **Release for timing, test profile for correctness, and keep them apart.** They are separate
  builds of about ten minutes each. Phases 1–3 need no test build at all.
- **No `--release` on tests** (repository rule); the test profile is already `opt-level = 3`.
- **Record every run** in one results file: commit, binary, command, machine state, wall time,
  stage split, proof size, verify time, peak memory.

Machine: M3, 11 cores, 18 GB. `target/` is currently empty, so the first build of each profile is
cold. Models present: `gpt2`, `gpt2-tiny`, `bge-small-en-v1.5`, `bge-tiny`, `qwen2-tiny`,
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
cp target/release/examples/prove bench/bin/prove-$(git rev-parse --short HEAD)
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
```

The existing spans do not separate three things the decisions need, so if the first runs show
they are merged, Phase 1 gets one span-only edit to `prove.rs` (an example-only rebuild):

- witness generation separated from the Akita commit inside the 25 s stage;
- the opening stage split into the reduction sumcheck and the per-class Akita openings, with the
  class count;
- the lookup sumchecks (read-raf, Booleanity, ra-virtualization) as their own total.

Answers: stage split and per-op spans on the current backend (roadmap measurement 1), lowered op
histograms (3), cost growth with context (5), peak memory, and the one number everything else
waits on: whether committing digits or running lookup sumchecks dominates per lookup on Akita.
On the July 2026 HyperKZG backend it was the sumchecks.

**Gate 1.** Which term dominates per lookup: commit, or lookup sumchecks? If sumchecks, the
roadmap's digit-savings table is an upper bound and items that remove sumcheck instances (B1, B2,
C2) move ahead of items that only remove digits. Also: what share of the opening stage is per-class
openings (this bounds the one-commitment layout).

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
  against that number (`research-directions.md` section 6), not a default.
- If dense tiny cells are several times cheaper per value than one-hot digits, D3 is the
  lattice-side path and moves up.
- If neither, option B (one-hot digits, fewer of them) is the end state and the remaining work is
  the fusions, D2 and the sumcheck kernels that measured above 1.3x.

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
