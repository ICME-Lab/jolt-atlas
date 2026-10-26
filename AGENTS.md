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
# importer's golden hashes, in about a minute.
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
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::gpt2_dory$/)'
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::bge_small_en_v1_5$/)'
cargo nextest run -p jolt-atlas-core --profile heavy --run-ignored ignored-only -E 'test(/zoo::qwen$/)'
cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'test(golden_zoo_big)'

# Pinned open issues: #[ignore]d `known_gap_*` tests that fail today and turn green when fixed.
cargo nextest run -p jolt-atlas-core --run-ignored ignored-only -E 'test(known_gap)'

# The example (release): proves and verifies any model directory; also the benchmark driver.
cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen2-tiny      # or models/gpt2 --pcs dory
cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --trace-terminal
cargo run --release -p atlas-onnx-import --example dump_lowered_ops -- <model.onnx>

# Regenerating the tiny fixtures (changes weights → re-pin atlas-onnx-import/tests/golden_zoo.rs)
.venv/bin/python scripts/gen_tiny_models.py
```

## Architecture

### Crate Map

Workspace members, in dependency order:

- **common** — shared constants and parallel helpers.
- **third_party/** — upstream a16z/jolt crates vendored whole at recorded revs
  (`third_party/PINS.md`): `jolt-field`, `jolt-transcript`, `jolt-utils`, `jolt-poly`,
  `jolt-sumcheck`.
- **joltworks** — the proof-system toolkit inherited from Jolt: `field`, `poly` (multilinear
  polynomials and commitment schemes), `subprotocols` (sumcheck, evaluation reduction),
  `lookup_tables`, `msm`, `transcripts`, `config`. `transcripts` runs on upstream
  `jolt-transcript`'s split surface: the prover uses a `ProvingTranscript` (spongefish
  `ProverState`, proof-of-work grinding behind the `prover` feature), the verifier a
  `VerifyingTranscript` over the proof's NARG string. `field::JoltField` extends upstream
  `jolt_field::JoltField` (ring/field methods resolve to upstream's) and adds the
  unreduced-product ladder and challenge types. The BN254 scalar is upstream `jolt_field::Fr`
  (`field::bn254`; its ladder runs on the private arkworks kernel in `field::ark`, and
  `ark_bn254::Fr` itself is not a `JoltField`); `field::fp128` is the 128-bit Solinas field for
  the lattice backend. `BatchedSumcheck`/`Sumcheck` drive upstream `jolt-sumcheck`'s engine
  (`prove_batch`, `verify_compressed`) through `subprotocols::upstream_sumcheck`, which presents
  our transcript as upstream's facade (grinding before each challenge). `SumcheckInstanceProver`
  extends upstream's `ProveRounds`, so every prover instance computes its own round messages
  (upstream `UnivariatePoly`) over upstream's `GruenSplitEqPolynomial`; Gruen rounds go through
  `poly::split_eq_poly::SplitEqUnreducedFold`, which takes the relation at the endpoint the
  round otherwise recovers by division.
- **atlas-trace** — deterministic fixed-point IR, op semantics, and the trace engine. Owns the
  behavioral contract below.
- **atlas-onnx-import** — tract-free ONNX frontend: protobuf decode (prost, vendored proto),
  static shape inference, whitelist constant folding, rewrite passes, weight quantization, and
  lowering to the `atlas-trace` IR. Golden output hashes for the model zoo live in `tests/`;
  cargo-fuzz targets and their regression corpus in `fuzz/` (see `fuzz/README.md`).
- **jolt-atlas-verifier** — the proof object and its wire format: `ONNXProof`,
  `ReducedOpeningProof`, `Claims`, `ProofId`/`ProofType`, and `proof_serialization`. Every proof
  field is private (`from_parts` + `&` accessors; `&mut` behind the `test-util` feature). No
  prover dependency; CI checks it builds in isolation. Verification logic still lives in core
  until the per-op verifiers move.
- **jolt-atlas-core** — the prover and the verification logic. `onnx_proof` is the entry point:
  the free functions `prove` / `verify` (re-exported at the crate root), with `prover.rs`,
  `verifier.rs`, `preprocessing.rs`, `witness.rs`, and `types.rs`. Per-op protocols live under
  `onnx_proof/ops/`; shared machinery under `clamp_lookups`, `clamp_split`, `fused_rebase`,
  `global_clamp`, `deferred_lookups`, `op_lookups`, `range_checking`, `neural_teleport`. The
  whole test suite is under `onnx_proof/testing/`; the `prove` example is in `examples/`.

Pipeline: ONNX file → `atlas-onnx-import` (decode, infer, fold, lower) → `atlas-trace` model
→ execute to a trace → `jolt-atlas-core` preprocess, commit, prove → verify against the model's
public IO. The proof crosses the prover/verifier boundary as a `jolt-atlas-verifier` type.

Arkworks comes from the fork `a16z/arkworks-algebra`, branch `dev/twist-shout`, pinned in the
root `Cargo.toml`.

### The atlas-trace Behavioral Contract

- Quantized values are `i32` at one global power-of-two scale: `q` represents `q / 2^scale`.
- Fused-rescale ops (einsum, mul, square, cube, mean-of-squares) accumulate in `i64`, split
  `acc = quotient·D + remainder` (Euclidean, `remainder ∈ [0, D)`), and emit
  `saturating_clamp_i32(quotient)`. Quotient and remainder are captured in the trace for the prover.
- Nonlinear ops are lookup tables generated by pure integer arithmetic in `atlas_trace::semantics`,
  versioned by `SemVersion`. The prover materializes its tables by calling the same functions, so
  tracer and prover agree bit for bit by construction. No platform floating point anywhere.
- Dimensions are padded per-dimension to powers of two at compile time; every op declares how it
  behaves under padding.

### Prover/Verifier Invariants

- The verifier never trusts the prover for anything the trace semantics define. Soundness is
  tested generically in `onnx_proof/testing/soundness.rs`: every field of a finished proof is
  mutated (`proof_mutation_sweep`), the trace is tampered before proving
  (`witness_tamper_sweep`, via `prove_from_trace`), and every opening claim is forged
  at the accumulator (`opening_forgery_sweep`, via the test-feature hook
  `joltworks::poly::opening_proof::forgery`). A new op or witness column is covered once it
  appears in one of the probe models in `onnx_proof/testing/models.rs`; a new proof field needs a
  `mutate::Site`. Prover-side `#[cfg(test)]` self-checks use `probe_assert!` so a tampered proof
  still reaches the verifier.
- Every `atlas_trace::Op` variant has a row in `onnx_proof/testing/op_surface.rs`; `OpKind::of`
  is a wildcard-free match, so adding a variant fails to compile until the table declares a
  stance (a proving case, an `#[ignore]` with its reason, or an expected rejection).
- Protocol-level changes (sumcheck ordering, transcript absorption, claim lifecycle, opening
  reduction) need the zoo tests in `onnx_proof/testing/zoo.rs` (every e2e also round-trips the
  proof through serialization) and at least one real-model prove & verify run.
- An open gap is pinned as an `#[ignore]`d `known_gap_*` test whose message states the failure;
  it must fail today and turn green when fixed. Never allowlist a sweep finding silently.
- Lookup table generation has one owner in `atlas-trace`. Never regenerate or approximate a
  table on the prover side.

## Development Guidelines

- Performance is the product. Profile before optimizing; the sumcheck inner loops dominate
  prover time, and small regressions in `joltworks/poly` multiply across every round.
- Prefer `x.evaluate_mle(&r)` over building a polynomial from a clone just to evaluate it.
- Tests assert against independent ground truth: the trace semantics, golden hashes, or
  algebraic properties. Never a second implementation of the rule under test.
- Before PR handoff, remove development-only probes, ad-hoc timing, commented-out code, and
  `#[ignore]` or `TODO` without a stated reason. `docs/pull-requests.md` has the full list;
  `/clean-branch` audits a branch against it.
- Comments describe the code as it is, not the change that produced it. Change narration
  belongs in commit messages.
- Math variables use `non_snake_case` (`log_T`, `log_K`); `jolt-atlas-core` allows it crate-wide.
