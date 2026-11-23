# Spec: Generic Einsum

| Field      | Value        |
| ---------- | ------------ |
| Author(s)  | @AntoineF4C5 |
| Created    | 2026-10-27   |
| Status     | proposed     |
| PR         |              |
| Depends on | —            |
| Supersedes | —            |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The einsum prover only accepts an allowlist of equations: four strings in the importer's
`CANONICAL_EINSUMS`, mapped by `EINSUM_REGISTRY` (`jolt-atlas-core/src/utils/dims.rs`) onto three
per-equation layouts (`mk_kn_mn`, `k_nk_n`, `abmk_abnk_abmn`) behind an `EinsumLayout` trait. The
tracer already executes any einsum; the restriction is the prover's alone, and it costs us three
ways. The importer must rewrite graphs into an accepted spelling (a `MoveAxis` before a batched
`…kn` operand, two `Reshape`s around unit-batch projections, a kept transpose on the left
operand). Some mappings are odd: `mk,nk->mn` is proven by the `k,nk->n` layout, so only `m == 1`
works. And every new contraction shape needs a hand-written layout. This spec replaces the
allowlist with one prover that derives each letter's role (batch, contracted, free) from the
equation, accepts any two-operand contraction regardless of spelling or axis order, and names the
points and coordinates it manipulates consistently so the protocol is easy to follow.

## Intent

### Goal

Prove every two-operand einsum whose summed letters appear in both operands through one
role-driven contraction sumcheck in `jolt-atlas-core/src/onnx_proof/ops/einsum/contraction.rs`.
Lower ONNX `MatMul` into one such einsum, and `Gemm` into one such einsum followed by an `Add`
when it has a bias.

Boundaries:

- **Introduced** (all in `jolt-atlas-core`):
  - `Contraction`: the letter roles of an equation, **batch** (in both operands and the output),
    **contracted** (in both operands, not the output) and **free** (in one operand and the
    output), plus `log_batch` and the per-operand fold tables.
  - `TensorPoint` (`utils/tensor_point.rs`): an MLE point over a tensor, split per axis; axis `i`
    owns `log2(dims[i])` consecutive coordinates.
  - Sumcheck variable order `[batch axes…, contracted axes…]`; degree 3 when `log_batch > 0`,
    degree 2 otherwise.
- **Removed:**
  - `jolt-atlas-core`: the `EinsumLayout` trait, `EqSchedule`, `ops/einsum/{mk_kn_mn, k_nk_n,
abmk_abnk_abmn}.rs`, and from `utils/dims.rs`: `EINSUM_REGISTRY`, `EinsumConfig`,
    `DimExtractor`, the `extract_*_dims` functions, `lookup_einsum_config`, `EinsumDims`.
  - `atlas-onnx-import`: `CANONICAL_EINSUMS`, the `atlas.Einsum` lowering arm, and the
    `matmul_gemm_to_einsum` pass.
- **Unchanged:** `atlas-trace` (including `EinsumEq`); the einsum's fused-rebase stages, clamp and
  remainder range checks; `ProofType::EinsumMatmul`; the `EinsumProver::sumcheck` /
  `EinsumVerifier::sumcheck` entry points `zk.rs` calls.

### Invariants

Relied on from the `atlas-trace` behavioral contract:

- **Fused rescale.** Einsum accumulates in `i64` and splits `acc = quotient·2^S + remainder`; the
  contraction sumcheck proves `Σ L·R = acc(r_output)` and the `fused_rebase` stages prove the
  rest. Held by: the existing fused-rebase op_surface rows and saturation rows
  (`einsum_mk_kn_mn_saturates_{pos,neg}`).
- **Per-dimension power-of-two padding.** Every letter owns `log2(padded dim)` MLE variables,
  which is what lets a point split per axis. Held by: the compiler-side padding in `atlas-trace`;
  non-power-of-two einsum rows stay `#[ignore]`d.

Introduced or changed:

- **A. Claims match the trace.** For every accepted equation, prover and verifier agree and the
  contraction sumcheck reduces the trace's `acc(r_output)` to operand openings at the points the
  letter roles define. Held by: op_surface proving rows (existing and new, see Acceptance
  Criteria) and the zoo e2e tests.
- **B. Refactor commits keep proofs valid.** At commits 1–3 of the Execution plan, the existing
  op_surface and zoo tests pass unchanged. Held by: CI on each commit (`cargo nextest run
--workspace`).
- **C. Unsupported equations are rejected, never proven.** Three or more operands, or a letter
  summed in only one operand, panic with a stated message when the `Contraction` is built, on both
  prover and verifier side (they build the same `Contraction`). Held by: `#[should_panic]`
  op_surface rejection rows.
- **D. Soundness of the batched, permuted path.** A malicious prover cannot open an operand at a
  point other than the one the letter roles define, nor cancel the batch `eq` factor. Held by: a
  probe model in `onnx_proof/testing/models.rs` containing a batched einsum with batch,
  contracted and free letters, at least one of them not in leading position, run under
  `proof_mutation_sweep`, `witness_tamper_sweep` and `opening_forgery_sweep`. Today the only
  probe einsum is `mk,kn->mn` (`attention_like`), so this path is not swept.
- **E. zk does not regress.** `zk.rs` does not compile on `main`. `cargo check -p
jolt-atlas-core --features zk` must fail only in `zk.rs`, with the same errors as on `main`.
  Correctness of the einsum's `#[cfg(feature = "zk")]` claim constraints is review-held.

### Non-Goals

- Non-power-of-two einsum; its op_surface rows stay `#[ignore]`d.
- Repeated letters within an operand (diagonals, traces); `atlas-trace` already rejects them.
- A letter summed in only one operand (e.g. `ab,bc->c` summing `a`): rejected, not proven.
- Three or more operands: rejected.
- Importing ONNX's own `Einsum` operator.
- Repairing `zk.rs`.
- The fused rebase, clamp and remainder range checks, including the open `i64` accumulator
  overflow question.
- Any change to `atlas-trace`.
- Changing what `slice_through_projection` rewrites.
- Prover speedups: the target is no regression.

## Evaluation

### Acceptance Criteria

Code shape:

- [ ] `ops/einsum/contraction.rs` holds the only einsum contraction sumcheck. `EinsumLayout`,
      `EqSchedule`, `ops/einsum/{mk_kn_mn, k_nk_n, abmk_abnk_abmn}.rs`, `EINSUM_REGISTRY`,
      `EinsumConfig`, `DimExtractor`, `extract_*_dims`, `lookup_einsum_config`, `EinsumDims`,
      `CANONICAL_EINSUMS` and `matmul_gemm_to_einsum` no longer exist (grep).

Supported equations:

- [ ] op_surface proving rows pass for every existing einsum row and for `mk,nk->mn` with
      `m = 2` (replacing `einsum_mk_nk_mn_m2_rejected`), `km,kn->mn`, `bmk,kbn->nbm`, `ab,ab->a`,
      `ab,b->a`, and `k,k->`.
- [ ] op_surface rejection rows (three operands; a letter summed in only one operand) are
      `#[should_panic]` with the stated message.
- [ ] A unit test on the contraction sumcheck params asserts `degree() == 2` for `amk,ank->amn`
      with `a = 1, k = 8` (`log_batch = 0`) and `degree() == 3` with `a = 2`.
- [ ] Unit tests: `TensorPoint` splits a point per axis; for random operands and points, the
      folded operand evaluated at `r_sumcheck` equals the operand's MLE evaluated at its opening point
      (two independent computations).

Soundness:

- [ ] The probe model of invariant D exists and the three sweeps pass with it.

Refactor safety:

- [ ] At commits 1–3, the existing op_surface and zoo tests pass unchanged.

Importer:

- [ ] `lower.rs` lowers `MatMul` to one `Einsum`, and `Gemm` to one `Einsum` followed by an `Add`
      when it has a bias, with `transA` / `transB` expressed in the equation (`km` / `nk`) and 1-D
      `MatMul` operands supported (`[k]·[k,n]`, `[m,k]·[k]`, `[k]·[k]`). Non-float outputs are
      rejected. No
      `Reshape` or `MoveAxis` is inserted around them, and no producer `Transpose` is merged: a
      `Transpose` stays a `MoveAxis`.
- [ ] `slice_through_projection` matches `MatMul` / `Gemm` and still rewrites GPT-2's QKV slices
      (importer zoo gates and golden hashes).
- [ ] Every re-pinned golden hash is listed, with its reason, in the commit message that re-pins it.

CI and real models:

- [ ] `cargo nextest run --workspace`, the importer's ignored zoo gates, and the real-model GPT-2
      (Dory) and Qwen prove & verify pass.
- [ ] `cargo check -p jolt-atlas-core --features zk` fails only in `zk.rs`, with the same error
      set as `main`.
- [ ] IOP proving time and total prove time on GPT-2 (Dory) and Qwen are within ±5% of `main`
      (see Performance).

### Testing Strategy

Existing tests that must keep passing:

- `cargo nextest run --workspace`: op-surface table, soundness sweeps, checked-in zoo
  (op_gauntlet, tiny GPT-2 / Qwen2 / BERT), importer golden hashes.
- `cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'not test(golden_zoo_big)'`
  and `-E 'test(golden_zoo_big)'`: the full-zoo import gates.
- The heavy real-model jobs `zoo::gpt2_dory` and `zoo::qwen` (protocol-level change: commit 4
  changes the sumcheck degree for size-1 batches, commit 8 changes the imported graphs).

New tests:

- op_surface proving and rejection rows listed in Acceptance Criteria.
- Unit tests for `TensorPoint` and the fold, against the operand's MLE evaluation.
- A unit test on the contraction sumcheck params' `degree()` (2 without batch bits, 3 with).
- The batched einsum probe model (invariant D).
- `atlas-onnx-import` unit tests: `Gemm` with `transA` / `transB` yields the expected equation; a
  batched `MatMul` yields `…mk,…kn->…mn` with no added node; 1-D operands yield `k,kn->n`,
  `mk,k->m` and `k,k->`; an integer `MatMul` is rejected; `slice_through_projection` still fires
  on a `MatMul` + bias projection.

Proof format: no new proof field, so no new `mutate::Site`; the round polynomials of the
contraction sumcheck are already mutation sites whatever their degree.

zk: CI does not build the feature, and the crate does not compile with it on `main`. The
compile-only check of invariant E is run by hand; the einsum constraints are review-held.

### Performance

Benchmark: `cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs
dory --trace-terminal` and the same on `models/qwen`, `main` versus the branch tip. Each side is
the median of 3 runs on the same machine, back to back, with no other heavy process; the compared
times are the `ONNXProof::iop` and `ONNXProof::prove` stages `--trace-terminal` reports.

- **IOP proving:** roughly flat. The generic fold does the same single pass and the same
  small-scalar multiplications as the per-equation folds; the extra per-entry cost (two table
  reads and an add) is negligible next to the field multiplication. GPT-2 loses the two
  `Reshape` proofs around each projection, a slight decrease. Budget: within ±5% of `main`.
- **Total prove time:** within ±5% of `main`.
- **Witness + commitment, proof size, verify:** slightly lower (fewer nodes); not bounded.
- **Setup, opening proof, PCS prove, peak memory:** unchanged (set by the largest committed
  polynomial, which the einsum does not change); not bounded.

## Design

### Architecture

```
ONNX MatMul / Gemm ──lower.rs──▶ Op::Einsum(EinsumEq)  [+ Op::Add for a Gemm bias]
                                        │  atlas-trace: unchanged
                                        ▼
ops/einsum/mod.rs   EinsumProver::sumcheck / EinsumVerifier::sumcheck   (zk.rs entry points)
        │  builds a Contraction from the node's EinsumEq and padded operand dims
        ▼
ops/einsum/contraction.rs
  Contraction            roles, log_batch, fold, operand opening points, batch eq factor
  ContractionParams / ContractionProver / ContractionVerifier
                         the sumcheck engine (message loop, eq caching, claim plumbing)
        │  uses
        ▼
utils/tensor_point.rs    TensorPoint: an MLE point split per tensor axis
```

`jolt-atlas-core` owns all new code: `TensorPoint` describes how this crate lays tensors out as
MLEs, which `joltworks` knows nothing about.

**Vocabulary** (used identically in code, comments and the module doc):

- Roles: `batch`, `contracted`, `free`. Every letter of an accepted equation has exactly one.
- `r_output`: the output opening point, a `TensorPoint` over the output's axes (batch and free
  letters).
- `r_sumcheck`: the sumcheck challenges, a `TensorPoint` over the virtual axes
  `[batch…, contracted…]`.
- Batch and contracted letters are each ordered as they appear in the left operand.

**The claim.** With `X'` the operand `X` folded over its free axes,

```
acc(r_output) = Σ_{s ∈ [batch, contracted]} eq(r_output.batch, s.batch) · L'(s) · R'(s)
X'(s)         = Σ_f eq(r_output[free axes of X], f) · X[index(s, f)]
```

and after the last round the verifier checks `eq(r_output.batch, r_sumcheck.batch) · L(p_L) ·
R(p_R)`, where operand `X`'s opening point `p_X` takes, axis by axis in `X`'s order,
`r_output.axis(·)` for a free letter and `r_sumcheck.axis(·)` for a batch or contracted letter.

**Fold.** Tensors are row-major, so an entry's flat index is `Σ_axis coordinate · stride`. The
sumcheck axes and the free axes of an operand are disjoint, so
`index(s, f) = offset_sumcheck[s] + offset_free[f]`, both tables built once from the operand's
per-axis strides. The fold is one parallel pass over `s`, summing `eq_free[f] · x[base +
offset_free[f]]` with small-scalar multiplication; `eq_free` is `EqPolynomial::evals` over the
free axes' coordinates of `r_output`, enumerated in the same axis order as `offset_free`.

**Rejection.** `Contraction::new` panics with a stated message on three or more operands or on a
letter summed in only one operand, as today's `m == 1` rejection does.

**`atlas-onnx-import`.** `lower.rs` gains `MatMul` and `Gemm` arms. Operands of ranks `r` and `s`
get letters `…mk` (or `km` with `transA`) and `…kn` (or `nk` with `transB`), batch letters shared
when both operands carry them. A 1-D `MatMul` operand follows numpy semantics: a 1-D left operand
is `k` (no `m`), a 1-D right operand is `k` (no `n`), and the missing letter is absent from the
output. A `Gemm` bias is emitted as an `Add` after the `Einsum`. The new arms reject non-float
outputs, as the removed `atlas.Einsum` arm and `matmul_gemm_to_einsum` did: an einsum always
rebases by `2^scale`. Shapes the equation cannot express (a batch axis of size 1 in only one
operand) are rejected as today.
`passes.rs` drops `matmul_gemm_to_einsum` from `run_all`; `slice_through_projection` matches
`MatMul` / `Gemm` and reads the weight's `n` axis from `transB` instead of the einsum letters.

### Alternatives Considered

- **Per-equation layouts with importer normalization** (the current design). Every new shape
  needs a layout, and the importer inserts `Reshape` / `MoveAxis` nodes and merges `Transpose`s
  so graphs fit the allowlist, so the IR no longer mirrors the ONNX graph.
- **Port the native BlindFold stack's `Contraction` as is** (`native_einsum.rs`, see References).
  Its per-bit indexing (`Coordinate::Shared(i)` / `Fixed(j)`) is hard to follow; its batch selector is
  materialized over the whole sumcheck domain instead of `2^log_batch` entries; its message loop
  is serial and does not use the round hint; and it lives in the `joltworks` BlindFold stack on
  the deleted `atlas-onnx-tracer`. Its role analysis and single-pass fold are kept.
- **Keep the `EinsumLayout` trait and `EqSchedule` (`High` / `Low`).** It would have one
  implementation, and since the prover chooses the variable order, batch-first (`High`) always
  applies.
- **Keep merging a producer `Transpose` into the einsum.** The `MoveAxis` it would save only
  permutes evaluation-point coordinates, so the saving is negligible, and lowering would have to
  inspect other ONNX nodes.
- **Fold by first copying the operand into `[sumcheck…, free…]` order.** The simplest fold loop,
  but it copies the operand (256 MB for GPT-2's padded LM-head weight).
- **Fold with per-bit index tables** (`Contraction::partial` in `native_einsum.rs`). Same cost as per-axis offsets, but
  bit-level; kept only as the fallback optimization for `offset_free` if the benchmark shows a
  cost.
- **Skip the fold for an operand with no free letter.** The saving is a fraction of one sumcheck
  round on the smaller operand.

## Documentation

- `README.md`: "the lowered op inventory and the canonical einsum equations" becomes "…and the
  einsum equations".
- `atlas-onnx-import/src/lower.rs` module doc: the "gates einsum equations to the canonical set"
  bullet is replaced by how `MatMul` / `Gemm` lower.
- `op_surface.rs`: the einsum section comment describes the role combinations and rejections the
  rows cover.
- `ops/einsum/contraction.rs` module doc: defines, in this order, the three roles, the two points
  and their per-axis split, the claim, the fold and the operand opening points, with the names
  the code uses and no symbol before its definition.
- Comments in removed code go with it; no remaining comment names a removed identifier.

## Execution

Commits follow `docs/pull-requests.md`: one concern per commit, each compiling and green.

**Part A — prover (commits 1–6).** Goal: one role-driven contraction sumcheck proves every
two-operand einsum whose summed letters appear in both operands. The importer is untouched, so
its graphs and the existing proofs stay valid throughout.

| #   | Commit                                                                                                                                                                                                                                                                                       | Kind          |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 1   | `git mv dot.rs contraction.rs`; rename `EinsumDot{Params,Prover,Verifier}` to `Contraction{Params,Prover,Verifier}`                                                                                                                                                                          | mechanical    |
| 2   | Add `TensorPoint` (with unit tests), the letter roles and the per-axis fold as one `EinsumLayout` implementation; delete the three layout files. The registry stays only as the accepted-equation gate (scaffolding, announced in the message, removed in 5). Existing tests pass unchanged. | refactor      |
| 3   | Remove `EinsumLayout` and `EqSchedule`: the single implementation merges into the engine; the schedule becomes `log_batch`; update the einsum zk constraints. Existing tests pass unchanged.                                                                                                 | mechanical    |
| 4   | Degree-2 sumcheck when `log_batch = 0`, including the zk constraints                                                                                                                                                                                                                         | protocol      |
| 5   | Accept any two-operand contraction: delete the `utils/dims.rs` registry, add the rejection panics and the new op_surface rows                                                                                                                                                                | wider support |
| 6   | Add the batched einsum probe model (invariant D)                                                                                                                                                                                                                                             | test          |

**Part B — importer (commits 7–8).** Goal: `MatMul` / `Gemm` lower one-to-one to `Einsum` (plus
`Add` for a `Gemm` bias) instead of being rewritten into an allowlist of spellings, which Part A
made unnecessary.

| #   | Commit                                                                                                                                                              | Kind     |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 7   | Remove `CANONICAL_EINSUMS`                                                                                                                                          | behavior |
| 8   | Lower `MatMul` / `Gemm` in `lower.rs`, remove `matmul_gemm_to_einsum`, make `slice_through_projection` match `MatMul` / `Gemm`, re-pin golden hashes if they change | behavior |

The batch-then-contracted order, each in left-operand letter order, keeps `abmk,abnk->abmn`
identical to today's layout, which commits 2–3 rely on. The message loop of today's engine is kept
as is.

Benchmark numbers (Performance) go in the PR description, not in commits.

## References

- Prior art: `joltworks/src/poly/commitment/dory/native_einsum.rs` on branch
  `review/native-operators` (commits `decfcdb2`, `31e3b796`): a role-based contraction for the
  native/BlindFold stack, source of the role analysis and single-pass fold idea.
- `docs/pull-requests.md`: the branch standard the Execution plan follows.
- Upstream `a16z/jolt`: no counterpart (no tensor contractions).
