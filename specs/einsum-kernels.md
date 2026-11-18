# Spec: Einsum kernels

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-18                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | —                              |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

A matrix product is paid for three times in a proof: the tracer computes it in integers, the
prover folds both operands at the output point before the contraction sumcheck, and the
verifier evaluates the weights at the point the sumcheck leaves. The sumcheck between them is
small: it runs over the contraction axis only, a few thousand entries. The three passes over
the operands are where the time is, and each is a plain loop that streams memory it could keep
in cache.

Measured on the benchmark machine (32 cores, 2026-11-18, Qwen):

| | 16 tokens | 64 tokens |
|---|---|---|
| prove | 6.9 s | 19.9 s |
| trace execution | 1.43 s | 3.42 s |
| of which einsum | 1.06 s | 2.54 s |
| op-batch witness build (the einsum folds are most of it) | 0.34 s | 0.37 s |
| verify | 0.88 s | 0.93 s |
| of which outside the batched sumchecks (constants evaluated at a point) | about 0.5 s | about 0.5 s |

The tracer's `mk,kn->mn` kernel computes one output row at a time, and for each row reads the
whole weight matrix: Qwen's vocabulary projection streams a 1 GB matrix once per token. That
pass does 4.3·10^9 multiply-adds in 0.44 s at 16 tokens on 16 of the 32 cores, about 1.6 ns
each, where a kernel that keeps its operands in cache does one in a fraction of a nanosecond.

This spec replaces the three passes with kernels written for the memory hierarchy, and makes
the prover's fold and the verifier's evaluation one kernel.

## Intent

### Goal

Make the three operand passes of an einsum (tracer contraction, prover fold, verifier
evaluation of a constant operand) cache-blocked, unreduced where they sum field products, and
shared where they are the same computation.

- `atlas-trace/src/op/einsum.rs`: `scaled_rows_acc_i64` becomes a tiled kernel that reads each
  weight row once per column tile and applies it to every output row.
- `atlas-algebra/src/poly`: one kernel for `Σ_h eq(r, h)·v[h]` over rows of `i32` values, with
  the eq table split into a high and a cache-sized low part and each run summed unreduced
  (`SmallAccumS`). It is what `mk_kn_mn.rs` does today for its right operand.
- `jolt-atlas-core/src/onnx_proof/ops/einsum/`: every layout's fold (`mk_kn_mn` left,
  `k_nk_n`, both operands of `abmk_abnk_abmn`) uses that kernel in row-major order.
- `jolt-atlas-verifier`: the evaluation of a constant tensor at a point uses the same kernel.

### Invariants

- **The trace is bit-identical.** Integer sums in `i64` are exact and associative for `i32`
  operands at the sizes a contraction has, so the order of accumulation does not change any
  output, remainder or error. Mechanism: the importer's golden output hashes
  (`atlas-onnx-import/tests/golden_zoo.rs`), the proof digest pins (`testing/digests.rs`), and
  a tracer unit test that compares the tiled kernel against the generic contraction on random
  operands of shapes that are not multiples of the tile, including a row of zeros.
- **An accumulation that leaves `i64` is still not silent.** The kernel keeps today's
  arithmetic (`i64` products of `i32` operands summed in `i64`); the bound that makes this
  exact is the contraction length times `2^62`, stated where the kernel is. Mechanism:
  review-held for the stated bound; unchanged behavior is covered by the hashes above.
- **The fold and the verifier's evaluation are the multilinear extension.** The shared kernel
  returns `Σ_h eq(r, h)·v[h]` for every `r` and every `v`, zeros included. Mechanism: a unit
  test in `atlas-algebra` against `evaluate_mle` on random and all-zero rows, at lengths below,
  at and above the low table's size; `relation_prover_equiv_contraction` (the dot engine's
  messages against the relation's reference messages) for every layout.
- **Nothing in the protocol moves.** No relation, schedule, transcript message or committed
  polynomial changes. Mechanism: the schedule pins and the proof digest pins pass without
  re-pinning.
- **The verifier still does not panic and takes no shortcut on a constant.** The kernel in the
  verifier crate is safe indexing-free code under the crate's deny list. Mechanism: the crate's
  lints and the isolation build in CI.
- **`atlas-trace`'s semantics module stays free of `rayon` and floats.** The tracer kernel
  lives in `op/einsum.rs`, which already uses `rayon` behind its feature. Mechanism:
  `semantics::guard::no_float_outside_tests`.

### Non-Goals

- Changing the contraction sumcheck, its round order, or its relation. The rounds are a small
  part of the cost.
- A different arithmetic for the tracer (narrower integers by operand width, floating point
  with a correction). Possible later; this spec keeps `i64` accumulation.
- Skipping padded extents by the model's unpadded shape. The kernels skip zero scales and zero
  runs as they meet them; knowing the padding statically is a separate change to what the
  model records.
- The einsum patterns that are not contractions (`m_an_a1nm`-style ties) and the other ops'
  tracer kernels.
- A GPU kernel.

## Evaluation

### Acceptance Criteria

- [ ] On Qwen at 16 and at 64 tokens, the time the tracer spends in einsum nodes is at most a
      quarter of the baseline (0.27 s and 0.64 s), measured with a per-op timer on the
      benchmark machine in one session with the baseline.
- [ ] The golden output hashes and the proof digest pins pass unchanged.
- [ ] The tiled kernel equals the generic contraction on random operands for shapes around the
      tile boundaries (unit test), and takes the same time or less on a single output row.
- [ ] Every einsum layout's fold calls the shared kernel; no layout reads an operand with a
      stride across rows.
- [ ] The op-batch witness build on Qwen at 16 tokens is at most 0.25 s (0.34 s today).
- [ ] The verifier's time outside the batched sumchecks on Qwen at 16 tokens is at most 0.3 s
      (about 0.5 s today), with the proof bytes unchanged.
- [ ] `bench/gate.sh` passes on GPT-2 and Qwen against the base commit, and prove time on Qwen
      at 16 tokens is at least 10% lower.

### Testing Strategy

- Existing, unchanged: `cargo nextest run --workspace` (zoo end to end, soundness sweeps, digest
  and schedule pins), the `atlas-onnx-import` zoo gates and golden hashes.
- New in `atlas-trace`: the tiled kernel against the generic contraction (the existing
  `einsum_acc_i64` fallback) over random shapes and zero rows.
- New in `atlas-algebra`: the eq-weighted row sum against `evaluate_mle`.
- `relation_prover_equiv_contraction` already covers each layout's messages.
- One real-model prove and verify of GPT-2 and of Qwen on the benchmark machine. The heavy CI
  job covers them as today.

### Performance

Benchmark: `prove` on `models/qwen` at `--seq-len 16` and `--seq-len 64`, and `bench/gate.sh`
on GPT-2 and Qwen, on the benchmark machine in one session.

- Trace execution: einsum from 1.06 s to about 0.25 s at 16 tokens and from 2.54 s to about
  0.6 s at 64 (estimate from the kernel's arithmetic, not a measurement). That is about 0.8 s
  of 6.9 s at 16 tokens and 1.9 s of 19.9 s at 64.
- Sumchecks: the op-batch build from 0.34 s to about 0.2 s.
- Verify: from 0.88 s to about 0.65 s.
- Commit, openings, proof size: unchanged.
- Peak memory: unchanged or lower (the tracer no longer copies a permuted operand it can read
  in place).

## Design

### Architecture

**The tracer kernel.** `out[i][c] = Σ_j a[i][j]·w[j][c]` for `m` rows of activations and a
`k × n` weight matrix. Today the loops are `i, j, c`: each output row is a pass over all of
`w`. The tiled kernel takes the columns in tiles of `C` and, per tile, runs `j` over the weight
rows and `i` over the output rows:

```
for each column tile [c0, c1)            in parallel
    for j in 0..k
        w_run = w[j][c0..c1]             C·4 bytes, read once from memory
        for i in 0..m
            s = a[i][j]; skip if s == 0
            out[i][c0..c1] += s · w_run  the same run from L1, m times
```

`C` is chosen so that the output tile (`m·C·8` bytes) stays in the second-level cache. The
weights are read from memory once per einsum whatever the token count; the output tile and the
weight run stay in cache. The inner loop is a widening multiply-add the compiler vectorises as
it does today. Threads own disjoint column tiles, so the kernel uses every core at any `m`
(today it has one task per output row, sixteen at sixteen tokens).

**The fold kernel.** For a row `v` of `2^a` values and a point `r`, split `r` into high and
low parts, build `eq_high` and `eq_low` (the low table at most 512 entries), and sum each run
of `v` against `eq_low` unreduced, multiplying by `eq_high` once per run; an all-zero run
costs a comparison. `mk_kn_mn`'s right fold is this already. The left fold of `mk,kn->mn`, the
fold of `k,nk->n` and both folds of `abmk,abnk->abmn` are sums over the other axis, column by
column with one reduction per element; they become row-major passes that add a row scaled by
one eq value into an unreduced accumulator per column.

**The verifier.** A constant operand's claim is its multilinear extension at the operand
point, computed over all of its cells. That is the fold kernel with the whole tensor as one
row.

### Alternatives Considered

- **A general blocked matrix product with packing, as a BLAS does.** More speed on square
  products; here `m` is the token count (16 to 128) against `k` and `n` in the thousands, so
  tiling the long axis is most of the gain at a fraction of the code.
- **Transposing the weights once per model and caching them.** Removes the permutation copy
  for the layouts that need one, but not the streaming; kept as a follow-up.
- **Stacking the identical layers into one contraction.** That is `digits-as-trace` stage 3
  and changes the protocol; it does not make any one fold cheaper.
- **Float matrix products checked against the integer result.** Rejected: the trace contract
  has no platform floating point.

## Documentation

- `docs/prover-plan.md`: the measured rows, and section A's tracer and einsum items rewritten
  as landed.
- No user-facing change.

## Execution

1. Baseline on the benchmark machine with a per-op tracer timer (temporary), Qwen at 16 and 64
   tokens and GPT-2 at 64.
2. `atlas-trace`: the tiled kernel behind `scaled_rows_acc_i64`, its test. Gate.
3. `atlas-algebra`: the eq-weighted row-sum kernel and its test; `mk_kn_mn`'s right fold moves
   onto it with no change in behavior.
4. The remaining folds onto the kernel, one commit per layout. Gate.
5. The verifier's constant evaluation onto the kernel. Gate, verify time.
6. Measurements and `docs/prover-plan.md`.

Each step is independently mergeable; step 2 carries most of the gain.

## References

- `jolt-atlas-core/src/onnx_proof/ops/einsum/dot.rs`: the contraction's prover.
- `atlas-trace/src/op/einsum.rs`: the tracer's contraction kernels.
- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008): the
  blocking argument.
- Thaler, "Time-Optimal Interactive Proofs for Circuit Evaluation" (2013), section 5: the
  matrix-product sumcheck the fold serves.
