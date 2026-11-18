# CLAUDE.md

## Project Overview

jolt-atlas is a zkML prover: it proves that an ONNX neural network was executed correctly on
given inputs. It is a lookup-centric fork of the Jolt zkVM (a16z/jolt) by ICME Labs. Models
are imported from ONNX, executed as a deterministic fixed-point trace, and proven with
sumcheck-based protocols over multilinear polynomial commitments.

## Agent Workflow

- Carry an authorized request through implementation and verification. Resolve routine choices
  from repository conventions; ask only when missing information would materially change
  correctness or scope. Continue independent work while an answer is pending.
- User instructions take precedence over skill guidance. If a skill blocks requested work, cite
  its file and instruction and explain the conflict.
- Read relevant code before asking about repository facts. Run Cargo commands sequentially,
  including across agents; parallel Cargo invocations fight over the build lock.
- Iterate narrow, verify wide. While editing, compile with `cargo check -p <crate> --tests` and
  run only the affected crate and filter (`cargo nextest run -p <crate> -E 'test(<name>)'`).
  Run `cargo nextest run --workspace` once, as the final check before reporting; each workspace
  run rebuilds and relinks every crate's test binaries at `opt-level = 3`.
- Run expensive work on the benchmark machine, not the laptop, when one is configured
  (`ssh atlas-bench true` succeeds, or `scripts/remote.sh up` starts it): the workspace test
  run, the soundness sweeps, the real-model tests, release builds, and every timing run. The
  laptop is in use while agents work, so its timings are not usable and a ten-minute build
  there blocks the user. `scripts/remote.sh run '<command>'` copies the working tree
  (uncommitted changes included) and runs the command there; see "Remote runs" below. Narrow
  `cargo check` and single-filter test runs stay local.
- Lead with the result. Report the checks actually run, their results, and any remaining blocker.
- Before opening or updating a PR, read `CONTRIBUTING.md` and run `/clean-branch`.
- Changes of 500+ non-trivial lines start with a spec in `specs/` (`/new-spec`, then
  `/analyze-spec`, then `/implement-spec`). Specs are frozen at merge; the code is the source of
  truth for current behavior.

## Essential Commands

CI (`.github/workflows/rust.yml`) sets `RUSTFLAGS=-D warnings`, so every warning fails CI.

```bash
# Format and lint (mirror CI)
cargo fmt --all --check
RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets

# Tests: dev profile, nextest (`.config/nextest.toml` holds the profiles). Never pass --release
# to tests; the release recompile after every edit costs minutes. Release is for the example
# below. The workspace run covers unit tests, the op-surface table, the soundness sweeps, the
# checked-in zoo (op_gauntlet + tiny GPT-2 / Qwen2 / BERT stand-ins) end to end, and the
# importer's golden hashes, in about a minute once built. While iterating, check and run the one
# crate and filter you touched; keep the workspace run for the final check.
cargo check -p jolt-atlas-core --tests                       # type-check only, no codegen
cargo nextest run --workspace
cargo nextest run -p jolt-atlas-core -E 'test(op_surface)'   # one case per op variant
cargo nextest run -p jolt-atlas-core -E 'test(soundness)'    # mutation / tamper / forgery sweeps
cargo nextest run -p jolt-atlas-core -E 'test(zoo)'          # end-to-end models
cargo nextest run -p jolt-atlas-core -E 'test(<name>)'
JOLT_SOUNDNESS_EXHAUSTIVE=1 cargo nextest run -p jolt-atlas-core -E 'test(soundness)'  # every site (nightly)

# Full-zoo import gates (decode / analyze / import) over whatever is in models/. #[ignore] tests
# that CI runs explicitly after the workspace run.
cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'not test(golden_zoo_big)'

# Real-model prove & verify (the heavy CI job: main, nightly, PRs labelled `ci:heavy`). Download
# the ONNX files first (one-time, needs .venv with Python 3.11; installs the pinned toolchain
# from scripts/requirements-models.txt):
#   python -m venv .venv && source .venv/bin/activate && python scripts/download_gpt2.py
#   python scripts/download_bge_small_en_v1_5.py && python scripts/download_qwen.py   # qwen: 2.5 GB
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::gpt2$/)'
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::bge_small_en_v1_5$/)'
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::qwen$/)'
cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'test(golden_zoo_big)'

# Pinned open issues: #[ignore]d `known_gap_*` tests that fail today and turn green when fixed.
cargo nextest run -p jolt-atlas-core --run-ignored ignored-only -E 'test(known_gap)'

# The example (release): proves and verifies any model directory; also the benchmark driver.
cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen2-tiny
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --trace-terminal
cargo run --release -p jolt-atlas-core --example prove -- --model models/op_gauntlet --schedule  # the verification schedule, one line per step
cargo run --release -p atlas-onnx-import --example dump_lowered_ops -- <model.onnx>

# Regenerating the tiny fixtures (changes weights → re-pin atlas-onnx-import/tests/golden_zoo.rs)
.venv/bin/python scripts/gen_tiny_models.py
```

### Remote runs

`scripts/remote.sh` drives a remote machine through an SSH alias (`atlas-bench` by default,
`ATLAS_REMOTE` to override); the machine's address and credentials live in the developer's own
`~/.ssh/config`, never in the repository. It copies the tree with rsync, so nothing needs
pushing, and leaves gitignored paths on the remote alone (`target/`, `bench/bin/`, models).

```bash
scripts/remote.sh up                                    # start the machine (it powers off when idle)
scripts/remote.sh setup                                 # once: compiler, rustup, cargo-nextest
scripts/remote.sh models                                # once: copy models/gpt2 (other dirs as arguments)
scripts/remote.sh run 'cargo nextest run --workspace'   # sync, run, wait
scripts/remote.sh bg sweeps 'JOLT_SOUNDNESS_EXHAUSTIVE=1 cargo nextest run -p jolt-atlas-core -E "test(soundness)"'
scripts/remote.sh tail sweeps                           # follow a detached run
scripts/remote.sh down                                  # stop it when done for the day

# A performance gate: release `prove` at each commit, interleaved runs, per-stage medians and
# the change against the first commit; exits 1 outside +3% time or +5% peak memory.
scripts/remote.sh run 'bench/gate.sh <base> <commit>... --runs 3'
scripts/remote.sh fetch bench/runs/gate                 # copy the raw logs back
```

Timings are comparable only between commits measured on the same machine in the same session;
`docs/prover-plan.md`'s tables are from the remote and are not comparable to a laptop's.

The remote gives huge pages only to memory that asks for them
(`/sys/kernel/mm/transparent_hugepage/enabled` is `madvise`), and the prover streams over
multi-gigabyte buffers. `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` in front of a `prove` run makes
the allocator ask: measured on 2026-11-18, Qwen 9.6 s to 7.5 s and GPT-2 3.1 s to 2.6 s on 16
cores and Qwen 5.8 s to 4.9 s on 32, with up to 10% more peak memory and verify unchanged. `bench/gate.sh` does not set it, so compare
runs with it only against runs with it. It has no effect on macOS.

## Architecture

### Crate Map

Workspace members, in dependency order:

- **common** — shared constants and parallel helpers.
- The proof-system toolkit inherited from Jolt, in dependency order:
  - **atlas-algebra** — `field` (the Solinas field taken from upstream `jolt-field`: the
    `Ring`/`Field` spine, unreduced products, packing, extensions and the `solinas` kernels with
    their assembly under `asm/`; `JoltField`, which adds the unreduced-product ladder and
    challenge types on top; `field::fp128::Fp128`, the 128-bit Solinas field, is the one
    implementation), `poly` (the one polynomial library: multilinear
    operand polynomials, `EqPolynomial`, the Gruen `split_eq`, the `univariate` round messages
    `UnivariatePoly`/`CompressedPoly`, opening points) and `utils`. Fields and polynomials share
    a crate because `MultilinearPolynomial`'s `From` impls overlap unless `JoltField` is local.
  Upstream a16z/jolt is a reference, not a dependency: `docs/upstream-inventory.md` records
  the provenance of every crate taken from it.
  - **atlas-transcript** — the Fiat-Shamir transcript (taken from upstream `jolt-transcript`):
    one `Transcript` trait; the prover uses a `ProvingTranscript` (spongefish `ProverState`,
    proof-of-work grinding behind the `prover` feature), which sends values (`send_scalar`,
    `send_scalars`, `send_serializable`) as prover messages into its NARG string, and the
    verifier a `VerifyingTranscript` over that string, whose reads (`read_scalar`,
    `read_scalars`, `read_serializable`) absorb what they return. The NARG string is the
    proof: every commitment, claim, round message, proof-of-work nonce and the joint opening
    proof, in the order the verifier reads them. `words` is the deployed format's word
    encodings the sumcheck engine absorbs. The challenge codec and grind
    pricing are `atlas-algebra::field::challenge`.
  - **atlas-lookup-tables** — the prefix-suffix lookup tables.
  - **atlas-claims** — the symbolic-relation layer: `SymbolicSumcheck` (a relation value's
    id, rounds, degree and input and output expressions), the `Expr` algebra, `Int` (the
    integer coefficient ring of an expression stated once for every field; `Expr::cast` reads
    it in a field) and the claim-data traits, owned by atlas (taken from upstream
    `jolt-claims` without its zkVM protocol graph).
  - **atlas-sumcheck** — the batched sumcheck engine (`engine`: `prove_batch`, `ProveRounds`,
    `verify_compressed`, taken from upstream `jolt-sumcheck`; round messages are sent and read
    through the transcript), the relation core (`relation`), the opening
    accumulators (`opening_proof`: an append binds an id to a point and sends or reads its
    claim), evaluation reduction, the one-hot and shout subprotocols,
    `config`. As
    upstream's protocol crates do, it ships both sides; prover items live in each module's
    `prover` submodule, which the verifier may not name (CI path check).
  - **atlas-pcs** — the `CommitmentScheme` interface (one batch commitment, one joint
    opening) and its one scheme, `bitz` (`specs/bitz-commitment.md`): a commitment to a
    vector of bits packed into `F_{2^128}`, Reed–Solomon encoded and Merkle-committed, with one
    claim over Fp128 proven against it by a projection onto a random prime, a grand product
    (`gkr`), ring switching and a Ligerito opening. `bitz::scheme` lays every committed
    polynomial out as bit planes of that one vector and turns the proof's joint opening into
    one such claim. Written from the papers, no external proof-system dependency. Its example
    `bitz_bench` is the scheme's benchmark.
  No BN254, curve, arkworks or Akita crate is in the resolved dependency graph (checked in
  CI). Every sumcheck is a relation: the verifier runs
  `relation::verify_relation{,_batch}` over `RelationVerifier`s, the prover
  `relation::prover::prove_relation{,_batch,_batch_parallel}` over `RelationWitness`es, both
  driving `atlas_sumcheck::engine` (`prove_batch`, `verify_compressed`), which grinds before
  each challenge. A `RelationWitness` extends the engine's `ProveRounds`, so
  every prover computes its own round messages (`UnivariatePoly`) over
  `GruenSplitEqPolynomial`, while its relation supplies the degree, input claim and opening
  appends; Gruen rounds go through
  `atlas_algebra::poly::split_eq::SplitEqUnreducedFold`, which takes the relation at the endpoint the
  round otherwise recovers by division.
- **atlas-trace** — deterministic fixed-point IR, op semantics, and the trace engine. Owns the
  behavioral contract below.
- **atlas-onnx-import** — tract-free ONNX frontend: protobuf decode (prost, vendored proto),
  static shape inference, whitelist constant folding, rewrite passes, weight quantization, and
  lowering to the `atlas-trace` IR. Golden output hashes for the model zoo live in `tests/`;
  cargo-fuzz targets and their regression corpus in `fuzz/` (see `fuzz/README.md`).
- **jolt-atlas-verifier** — the proof and all verification: `ONNXProof` (with
  `ONNXProof::verify`), a newtype over the bytes of the prover's transcript with no type
  parameter and no format of its own, and `onnx_proof` — the verification schedule (`schedule.rs`: a
  model's `Schedule` of `Step`s over relation values, compiled from the model alone by
  `Schedule::compile` when the preprocessing is constructed; each op's `OpPlan::plan` writes
  its node's steps to a `ScheduleBuilder`, the only way an op is verified, and the steps run
  in waves of ready nodes with one batched sumcheck per wave; the committed
  polynomials are `Schedule::committed`), the one op
  relation type (`relation.rs`: `OpRel`, a value each op builds with a constructor function
  in its `ops` module; `OpRelKind` names the relation and the ops it may serve), the table
  reads (`lookup.rs`: `TableRead`, `SmallTableDef`, `RaEncoding`), the packed range buckets
  and their checks (`range_bucket.rs`, `range_check.rs`), the fused rescale's verifier side,
  the `Verifier` state (its interpreter `Verifier::run`, one loop over the schedule's steps;
  its accumulator and transcript are crate-private, with `&mut` accessors behind the
  `test-util` feature),
  `verify`/`verify_with` (read the commitments, run the schedule, read the joint opening
  proof, require the bytes to end), and the shared and verifier preprocessing (the verifier's
  holds the schedule); also
  `node_view` and `utils`. The verifier reaches the proof only through its transcript's
  sequential reads, in schedule order. No prover dependency and no prover code: the crate is
  Trace-free (nothing in it names `atlas_trace::Trace` or its witness types; witness
  construction is in `jolt-atlas-core`). CI checks it builds in isolation, `clippy.toml`
  disallows the trace types and the prover items of the vendored and atlas protocol crates, and
  CI rejects `prover`/`witness` module paths and `Trace` in its sources. `docs/verifier.md` is
  the audit reading order and the trust-perimeter statement.
- **jolt-atlas-core** — the prover. `onnx_proof` mirrors the verifier crate's module tree: each
  module glob-re-exports its verifier counterpart and adds the prover side
  (`ops::prove_schedule` walks the model's schedule with the relations' witnesses, the
  prover's one loop over its steps; the prover side of an `OpRel` is a
  wildcard-free match on its `OpRelKind` in `ops/mod.rs`, giving the engine's witnesses
  (`op_rel_witnesses`), a tie's claims (`op_rel_tie_claims`) or a specialized prover
  (`op_rel_prover`), and that of a table read is `lookup::table_read_prover`),
  with `prover.rs`, `witness.rs`, `preprocessing.rs` (prover setup and the schedule) and
  `types.rs`.
  `prove` and the free `verify` (adding the test-only lockstep debug comparison) are re-exported
  at the crate root. The whole test suite is under `onnx_proof/testing/`; the `prove` example is
  in `examples/`.

Pipeline: ONNX file → `atlas-onnx-import` (decode, infer, fold, lower) → `atlas-trace` model
→ execute to a trace → `jolt-atlas-core` preprocess, commit, prove → verify against the model's
public IO. The proof crosses the prover/verifier boundary as a `jolt-atlas-verifier` type.

No arkworks crate is in the graph (CI checks `Cargo.lock`): commitments, opening proofs, setups
and transcript messages encode through `atlas_algebra::serialize::Canonical` (`common::serialize`),
byte for byte the arkworks compressed encoding the format was defined with.

### The atlas-trace Behavioral Contract

- A node's value is an integer at a level `j` of one global power-of-two scale: `q` represents
  `q / 2^(j·scale)`. A level-1 value is an `i32`. A deferred product (einsum, mul) or a sum of
  one keeps its exact accumulation as a wide value (`i128`) at the sum of its operands' levels,
  with a magnitude bound below `2^125` computed at compile time from the model's constants;
  `Rescale` takes it back to level 1 by one Euclidean split, quotient in `i32`, remainder
  captured. The importer places one rescale per chain of products
  (`atlas_trace::defer`, `specs/deferred-rescale.md`).
- Fused-rescale ops (einsum, mul, square, cube, mean-of-squares) accumulate in `i64`, split
  `acc = quotient·D + remainder` (Euclidean, `remainder ∈ [0, D)`), and emit the quotient,
  which must lie in `i32`. Quotient and remainder are captured in the trace for the prover.
- Division-like ops capture their advice the same way: ScalarConstDiv its floor-division
  remainder, Rsqrt its division quotient and both remainders, Sin and Cos the period reduction
  their table indexes by.
- Arithmetic is exact or absent: a sum or difference (each pairwise step of Add and Sub), an
  axis sum, a negation or a rescaled quotient outside `i32` is an execution error, never a
  clamped value.
- Softmax over a row is the output table read at `min(c − x, top)`, where the shift `c` is the
  smallest integer at or above the row's maximum whose sum-table row sum does not exceed the sum
  scale (`atlas_trace::semantics::softmax`). The shift, the addresses and the two row gaps are
  captured in the trace.
- Nonlinear ops are lookup tables generated by pure integer arithmetic in `atlas_trace::semantics`,
  versioned by `SemVersion`. The prover materializes its tables by calling the same functions, so
  tracer and prover agree bit for bit by construction. No platform floating point anywhere.
- Dimensions are padded per-dimension to powers of two at compile time; every op declares how it
  behaves under padding.

### Prover/Verifier Invariants

- The verifier never trusts the prover for anything the trace semantics define. Soundness is
  tested generically in `onnx_proof/testing/soundness.rs`: every value the verifier reads from
  a finished proof is mutated (`proof_mutation_sweep`, its sites taken from the read log of a
  recording transcript, `testing/recording.rs`), the trace is tampered before proving
  (`witness_tamper_sweep`, via `prove_from_trace`), and every opening claim is forged
  at the accumulator (`opening_forgery_sweep`, via the test-feature hook
  `atlas_sumcheck::opening_proof::forgery`). A new op or witness column is covered once it
  appears in one of the probe models in `onnx_proof/testing/models.rs`; a new read is a
  mutation site by construction. Prover-side `#[cfg(test)]` self-checks use `probe_assert!` so
  a tampered proof still reaches the verifier.
- The verifier rejects, never panics, and reads the proof exactly once, front to back.
  `jolt-atlas-verifier` denies `unwrap`/`expect`/`panic!`/`unreachable!`/`todo!`/indexing at
  the crate root with no `#[allow]` outside tests; a short read, a non-canonical field
  element, a frame longer than the bytes left and a public IO of the wrong shape are `Err`s;
  every read is absorbed by the read itself; how many values are read and how long each is
  comes from the schedule, never from the proof; a claim exists only from the append that
  binds it; `verify` ends by requiring the end of the bytes. Every `mutate::Site` must be
  rejected with an `Err` (no tolerance list), as must the proof cut at each read boundary or
  extended by a byte; every relation kind is in the schedule of some checked-in model
  (`relation_kinds_all_exercised`), and `fuzz_regressions_replay` replays the `verify` fuzz
  target's driver (`jolt-atlas-core/fuzz/`, run by hand, not in CI).
- Every `atlas_trace::Op` variant has a row in `onnx_proof/testing/op_surface.rs`; `OpKind::of`
  is a wildcard-free match, so adding a variant fails to compile until the table declares a
  stance (a proving case, an `#[ignore]` with its reason, or an expected rejection).
- Protocol-level changes (sumcheck ordering, transcript absorption, claim lifecycle, opening
  reduction) need the zoo tests in `onnx_proof/testing/zoo.rs` (every e2e also verifies the
  proof again from its bytes, and requires those bytes plus one to be rejected) and at least
  one real-model prove & verify run.
- An open gap is pinned as an `#[ignore]`d `known_gap_*` test whose message states the failure;
  it must fail today and turn green when fixed. Never allowlist a sweep finding silently.
- Lookup table generation has one owner in `atlas-trace`. Never regenerate or approximate a
  table on the prover side.
- Every sumcheck is a symbolic relation (an `atlas_claims` `SymbolicSumcheck` over atlas
  openings), and the relation is the source of truth: the verifier is derived from it
  (`RelationVerifier`; no hand-written verifier exists), and a prover is either the engine's
  `RelationProver` or a specialized `RelationWitness` that computes only its round messages.
  The relation core (`Relation`, `RelationVerifier`, the opening/derived/coefficient ids, the
  batched-sumcheck drivers) is `atlas_sumcheck::relation`; subprotocol relations (one-hot
  checks, read-raf, range checks, opening reduction) take their setup from the caller. An op
  relation is an `OpRel` value built by a constructor function, whose doc states its S ⊆ R
  argument; it is bound to its ops by `OpRelKind::serves`, checked by
  `relation_kinds_match_their_dispatched_op_kind`. `docs/pull-requests.md` §4 is the
  checklist for a new relation.

## Development Guidelines

- Performance is the product. Profile before optimizing; the sumcheck inner loops dominate
  prover time, and small regressions in `atlas-algebra/src/poly` multiply across every round.
- Prefer `x.evaluate_mle(&r)` over building a polynomial from a clone just to evaluate it.
- Tests assert against independent ground truth: the trace semantics, golden hashes, or
  algebraic properties. Never a second implementation of the rule under test.
- Before PR handoff, remove development-only probes, ad-hoc timing, commented-out code, and
  `#[ignore]` or `TODO` without a stated reason. `docs/pull-requests.md` has the full list;
  `/clean-branch` audits a branch against it.
- Comments describe the code as it is, not the change that produced it. Change narration
  belongs in commit messages.
- Math variables use `non_snake_case` (`log_T`, `log_K`); `jolt-atlas-core` allows it crate-wide.
