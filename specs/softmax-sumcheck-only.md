# Spec: Softmax sumcheck-only

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @AntoineF4C5                   |
| Created     | 2026-11-03                     |
| Status      | proposed                       |
| PR          | #404                           |
| Depends on  | —                              |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`SoftmaxLastAxis` is the only op whose proof sends witness data in the clear. For every softmax
row, the prover writes three values (`max_k`, `exp_sum_q`, `argmax_k`) into the transcript. The
verifier evaluates these vectors directly and derives `inv_sum = ⌊S²/exp_sum_q⌋` itself. That
breaks the pattern every other op follows, where the verifier sees only sumcheck messages and
opening claims. It also makes proof size and verifier work grow with the number of rows, and
lets a malicious proof crash the verifier, because an out-of-range `argmax_k` is used as an
index without a bounds check. The same protocol has two soundness gaps:

- Nothing proves `max_k ≥ x`. The clamp lookup accepts a negative `z = max_k − x` and floors it
  to 0, so any lane can pose as the row maximum.
- The quotients `out` and `exp_q` are never bit-decomposed. Each
  `quotient·S + remainder = accumulator` relation therefore holds for `S` different field values
  per lane, and only one of them is the integer the trace defines.

This spec rebuilds the softmax protocol so every intermediate is either a virtual claim proven by
sumcheck or a committed one-hot, and closes both gaps.

Notation used throughout: the softmax input is `rows × N`, with `k ∈ [0, rows)` the row index
and `j ∈ [0, N)` the lane index; `S = 2^s` at scale `s`.

## Intent

### Goal

Stop sending softmax witness vectors to the verifier in the clear. Today `max_k`, `exp_sum_q`
and `argmax_k` are smuggled through `opening_claims` as zero-variable "openings", one per row.
Instead, `SoftmaxLastAxis` is proven with only sumchecks, opening claims at protocol-derived
points, and committed one-hots, like every other op. The rebuilt protocol also binds every
intermediate to the integer the trace semantics define, which closes the `max_k ≥ x` gap and the
unbounded-quotient gap.

Boundaries:

- **Modified:** `jolt-atlas-core/src/onnx_proof/ops/softmax_last_axis/` (all stages; see
  Architecture), plus the softmax polynomial ids in `common`, where the per-row ids
  `SoftmaxSumOutput`, `SoftmaxMaxOutput` and `SoftmaxMaxIndex` are removed. Two shared files
  follow the id changes: `onnx_proof/witness.rs` (committed witnesses for the softmax one-hots)
  and `onnx_proof/deferred_lookups.rs` (the deferred softmax batch, which loses the clamp
  lookup).
- **Unchanged:** `atlas-trace`. The prover derives the encoding-only values (the accumulators
  `out_acc` and `exp_acc`, `slack`, `IND`, `inv_sum`'s remainder `r_inv` and gap `d`, one-hot
  indices) from `SoftmaxWitness`, the same way `clamp_split` derives its `slack` and `IND`.
- **Left in place:** `joltworks`. `SoftmaxClampTable` and its prefix/suffix types become unused
  by softmax but are not removed.

### Invariants

**Relies on** (from the `atlas-trace` behavioral contract):

- **Lookup-table semantics:** `exp_lut_decomposed` stays the only source of `lut_hi` and
  `lut_lo`, and `z_bound = |lut_hi padded|·B` (`B` the digit base) is a power of two. Held by
  tracer and prover calling the same `atlas_trace` function.
- **Global scale:** `S = 2^s`, with `out_acc, exp_acc ∈ [0, S²]`, so each value fits in `2s+1`
  bits (its one-hot uses `2s+2` address bits, the top one always 0).
- **Padding:** masked softmax (`active_len < N`) stays rejected. Held by the op-surface row
  `softmax_masked_rejected`.
- **The softmax quotient/remainder split** (Euclidean, remainder in `[0, S)`), so `out` and
  `exp_q` are the high bits of `out_acc` and `exp_acc` exactly as the trace computes them.

**Invariants and the mechanism holding each:**

- **A. No clear vectors.** No softmax witness value reaches the verifier except as a claim that
  a sumcheck or a committed one-hot proves. *Review-held*, with the compiler as backstop: the
  per-row ids are deleted from `common`, so reintroducing them is a visible change.
- **B. The max is real.** A trace whose pivot is a non-maximal lane, with everything downstream
  recomputed consistently from it, is rejected. *Mechanism:* `known_gap_softmax_non_max_pivot`
  (see Testing Strategy), which lands failing and flips to passing.
- **C. Saturation is sound on both branches.** `IND = 0 ⇒ z = z_c`; `IND = 1 ⇒
  z_c = z_bound − 1` and `z ≥ z_bound − 1`; a negative `z` is unsatisfiable. *Mechanism:* the
  `attention_like` probe with saturated softmax lanes (asserted at build time) and the widened
  `softmax_random_4x16` op-surface row, both under the existing soundness sweeps.
- **D. The quotients are bound.** `out` and `exp_q` are the high bits of one-hots of `out_acc`
  and `exp_acc`, so each is a unique integer in `[0, S]`. *Review-held* for non-integer
  forgeries, which neither a trace tamper nor the `+1` opening forgery can produce;
  `opening_forgery_sweep` covers the `out_acc` and `exp_acc` claims.
- **E. `inv_sum = ⌊S²/exp_sum⌋` is bound** through the row-sum and division zero-checks and the
  `inv_sum`, `r_inv` and `d` one-hots. Depends on D: `exp_sum` is a bounded integer only because
  each `exp_q` is. *Mechanism:* the existing `inv_sum` and `exp_sum_q` fields of
  `witness_tamper_sweep`.
- **F. The verifier never panics on a malformed proof.** *Mechanism:* the existing
  `proof_mutation_sweep`. The out-of-range `argmax_k` index disappears by construction.
- **G. Prover and verifier agree on honest traces.** *Mechanism:* the op-surface softmax rows,
  the checked-in zoo end to end, and a real-model GPT-2 run, since this is a protocol-level
  change.
- **H. Softmax proves at any scale.** The `MODEL_SCALE` restriction is lifted, because
  `z_bound` is computed at runtime from `node.scale`. *Mechanism:* `softmax_model_scale_2x8`
  moves to scale 12, and the GPT-2 run (scale 12) checks it end to end.
- **I. Every sent claim is proven.** Each claim the prover sends between stages is the input
  claim of a later sumcheck or a PCS opening. *Review-held*; the stage layout below names the
  proving step for each.
- **J. Tampered traces reach the verifier.** The prover never panics while deriving `slack`,
  `r_inv`, `d` or the one-hot indices from an inconsistent witness. It computes them with
  wrapping arithmetic, truncated to the one-hot's address width, and commits to the result, so
  the verifier's relations reject the trace. A prover panic is only a "refusal", which the sweeps
  count as inconclusive, so B and E rely on J. *Mechanism:* `known_gap_softmax_non_max_pivot`
  asserts `Outcome::Rejected`, and the `inv_sum`/`exp_sum_q` tampers of `witness_tamper_sweep`
  must not appear among its refusals.

### Non-Goals

- **Masked softmax** (`active_len < N`). It stays rejected by prover and verifier. Adding the
  `[j < active_len]` selector is a separate change.
- **Removing unused `joltworks` machinery.** `SoftmaxClampTable` and its softmax-specific
  prefix/suffix types stay in place even though softmax no longer uses them.
- **Extending `clamp_split`.** The saturation relation is dedicated to softmax and doesn't join
  the packed clamp buckets. Alternatives Considered explains why.
- **Quotient-binding audits of other ops.** Div, Rsqrt and ScalarConstDiv may share the
  "remainder range-checked, quotient never decomposed" pattern. Checking them is a separate
  issue.
- **Changing softmax semantics.** No change to `atlas-trace`, the exp tables, `SemVersion`, or
  any output value. Every trace proves exactly the same values as before.
- **The `zk` feature beyond keeping it compiling.** The BlindFold constraints in the touched
  sumchecks get updated so `--features zk` builds, but there are no new zk tests. CI doesn't
  build that feature.

## Evaluation

### Acceptance Criteria

Every criterion is ticked on this spec's PR.

- [ ] `send_auxiliary_vectors` and its verifier counterpart are gone. The per-row ids
  `SoftmaxSumOutput`, `SoftmaxMaxOutput` and `SoftmaxMaxIndex` no longer exist in `common`, and
  the softmax verifier reads no witness value except claims proven by a sumcheck or a committed
  one-hot (review).
- [ ] `argmax_k` is proven through a committed one-hot whose booleanity and per-row Hamming
  weight are checked. `max_k` is a virtual claim, sent at `r2_k` and proven by `MaxIndicator`.
  `exp_sum` is not a protocol polynomial: it is written as `d + 1 + r_inv`.
- [ ] `inv_sum` is committed only as a one-hot of an `s+1`-bit value (`inv_sum ∈ [0, S]`).
  `Σ_j exp_q = d + 1 + r_inv` and `(d + 1 + r_inv)·inv_sum + r_inv = S²` are proven by
  zero-checks. `r_inv ≥ 0` and `d ≥ 0` are proven by unsigned-identity one-hots of
  `log_N + s`-bit values each, using the static bound `exp_sum ≤ N·S`. Every one-hot read uses
  an even address width (see Architecture, Definitions).
- [ ] The softmax clamp lookup (`SoftmaxClampRaD`, `SoftmaxSignificanceClampOperands`) is
  replaced by the dedicated saturation relation: `z = z_c + slack`,
  `IND·(z_c − (z_bound − 1)) = 0`, `IND` boolean and equal to the Hamming weight of the slack
  chunks, `z_c = z_hi·B + z_lo`.
- [ ] `out` and `exp_q` are proven as the high bits of one-hot decompositions of `out_acc` and
  `exp_acc` over `2s+2` address bits (top bit always 0), replacing the `SoftmaxRemainderRaD` and
  `SoftmaxExpRemainderRaD` range checks.
- [ ] `known_gap_softmax_non_max_pivot` lands in an earlier commit, failing as an `#[ignore]`d
  pin, and passes un-ignored after the rebuild.
- [ ] The `attention_like` probe has saturated softmax lanes, asserted at build time, and
  `softmax_random_4x16` draws inputs whose spread exceeds `z_bound`. Both land before the
  protocol change and pass before and after it.
- [ ] The `MODEL_SCALE` debug assertions in `SoftmaxLastAxis::prove`/`verify` are removed, and
  `softmax_model_scale_2x8` proves at scale 12.
- [ ] `cargo nextest run --workspace` passes: op-surface, soundness sweeps, and zoo.
- [ ] The real-model GPT-2 prove and verify passes (`zoo::gpt2_dory`, heavy profile), run
  locally.
- [ ] `cargo build -p jolt-atlas-core --features zk` passes.
- [ ] Clippy with `-D warnings` and `cargo fmt --check` are clean.
- [ ] The PR description reports GPT-2 (Dory) prove time, verify time, proof size and peak RSS
  on `main` and on the branch, measured with the `prove` example under `/usr/bin/time -v` on the
  same machine (see Performance).

### Testing Strategy

**Existing tests that must keep passing**

- `cargo nextest run --workspace`. That covers the op-surface table (`softmax_model_scale_2x8`,
  `softmax_random_4x16`, `softmax_masked_rejected`), the soundness sweeps over the
  `attention_like` probe (`proof_mutation_sweep`, `witness_tamper_sweep` including every
  `SOFTMAX_FIELDS` column, `opening_forgery_sweep`), and the checked-in zoo end to end:
  `op_gauntlet`, plus the tiny GPT-2, Qwen2 and BERT stand-ins, each round-tripping its proof
  through serialization.
- The `atlas-onnx-import` zoo gates and golden hashes are unaffected: `atlas-trace` and the
  importer don't change.

**Test edits** (land before the protocol change and pass on both sides of it)

- `attention_like` gets saturating softmax lanes (a large bias entry per row) and a build-time
  assertion that some lane has `z ≥ z_bound`. Probe models have no golden hash, so the edit costs
  nothing else.
- `softmax_random_4x16` draws inputs whose spread exceeds `z_bound` (today ±2¹⁵, while
  `z_bound = 2¹⁸` at scale 14).

**Test edit landing with the saturation commit**

- `softmax_model_scale_2x8` moves to scale 12. It cannot land earlier, because the current
  protocol only supports `MODEL_SCALE`.

**New test**

- `known_gap_softmax_non_max_pivot` in `soundness.rs`. It lowers an input's max lane to the
  second-largest value, runs the real `SoftmaxLastAxis` evaluation on that input, sets
  `argmax_k` to the second-largest lane, puts the original input back into the witness and the
  input node, and proves with `prove_from_trace` on a single-softmax model (so the tampered
  output is the public output). Setting `argmax_k` keeps `max_k = x[argmax_k]` true after the
  restore whatever the trace's tie-breaking, so `MaxIndicator` accepts and only the `max_k ≥ x`
  check can reject. The max lane then has `z < 0`, which today's clamp floors to 0. The test asserts that the prover returns a proof and
  the verifier rejects it (`Outcome::Rejected`); a prover panic fails the test (invariant J). It
  lands `#[ignore]`d, failing today because the proof verifies, and is un-ignored by the
  saturation commit.

**Proof-format coverage:** no new `mutate::Site`. The new claims and commitments are reached by
the generic `Site::Claim`, `ClaimRemoved` and commitment sites.

**Real model:** `zoo::gpt2_dory` (heavy profile), run locally. It is the protocol-level
real-model check and also covers scale 12 end to end.

**`zk`:** `cargo build -p jolt-atlas-core --features zk` only. There are no zk tests, so the
BlindFold constraint updates in the touched sumchecks are review-held.

### Performance

**Benchmark:** the `prove` example on GPT-2 with Dory, on `main` and on the branch, on the same
machine, under `/usr/bin/time -v` for peak RSS:

```bash
/usr/bin/time -v cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory --trace-terminal
/usr/bin/time -v cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory --trace-terminal --seq-len 128
```

At the default `--seq-len 16`, each attention softmax is only 12 heads × 16 × 16 lanes, so
softmax is a small share of the total. The `--seq-len 128` run makes it a visible share. Compare
the end-to-end stages, proof size, peak RSS, and the `SoftmaxLastAxis::prove`/`verify` spans.
No numeric budget.

**Expected movement per softmax node** (scale `s`, 4-bit one-hot chunks, `rows × N` lanes):

| Removed | Added |
|---|---|
| clamp one-hot, 32 bits (8 chunks), and its 32-round prefix-suffix read-raf | slack one-hot, 32 address bits (8 chunks), nonzero only on saturated lanes |
| `R` range-check one-hot, `s` bits (`⌈s/4⌉` chunks) | `out_acc` one-hot, `2s+2` bits (`⌈(2s+2)/4⌉` chunks) |
| `r_exp` range-check one-hot, `s` bits (`⌈s/4⌉` chunks) | `exp_acc` one-hot, `2s+2` bits (`⌈(2s+2)/4⌉` chunks) |
| 3·rows clear values, and the verifier's O(rows) work evaluating them | argmax, `inv_sum`, `r_inv` and `d` one-hots, and sumchecks over `rows` cycles (negligible) |

- **Witness + commitment:** slightly up for softmax, since the remainder one-hots roughly double
  in width.
- **IOP proving:** down for softmax; the 32-round read-raf is gone.
- **Verify:** expected to improve. The 32-round read-raf verifier and the O(rows) clear-vector
  evaluations both disappear.
- **Proof size:** about neutral.
- **Unchanged:** key generation and every non-softmax node.

## Design

### Architecture

Everything lives in `jolt-atlas-core/src/onnx_proof/ops/softmax_last_axis/` and `common`.
`atlas-trace`, the importer and `joltworks` don't change.

#### Definitions

- `B` is the base splitting `z_c` into digits; `z_bound` is the exp table's input bound.
- `z = max_k − x ≥ 0`, the distance of each lane from its row maximum.
- `z_c = min(z, z_bound − 1) = z_hi·B + z_lo`, the clamped distance, as two digits.
- `slack = z − z_c`, the part cut off by the clamp: a committed one-hot of a 31-bit value.
- `IND ∈ {0, 1}` flags saturated lanes (`IND = 1` exactly when `slack ≠ 0`). It is the Hamming
  weight of the slack chunks.
- `e(k, j)`, the argmax one-hot: 1 at `j = argmax_k`, 0 elsewhere. Address `j`, cycle `k`,
  committed as a single chunk with `log_k_chunk = log N`, like `GatherSmall`'s one-hot
  (`ops/gather/small.rs`). `N` is a power of two (masked softmax is rejected), so every address
  is a real lane. 4-bit chunks would leave padded addresses `j ≥ N` that a Hamming weight of 1
  doesn't exclude: an argmax there makes `max_k = 0` pass for a row of negative inputs.
- `exp_acc = exp_hi·exp_lo = exp_q·S + r_exp` and `out_acc = exp_q·inv_sum = out·S + R`, each a
  one-hot over `2s+2` address bits (top bit always 0). `Hi(a)` reads the quotient (`exp_q` or
  `out`) off the address.
- `inv_sum = ⌊S²/exp_sum⌋` and `r_inv = S² − exp_sum·inv_sum`, the division remainder.
- `d = exp_sum − 1 − r_inv`, the **remainder gap**. `d ≥ 0` proves `r_inv < exp_sum`, which makes
  `inv_sum` the floor and not just *a* quotient.
- `exp_sum := d + 1 + r_inv`. It is not a protocol polynomial: it has no id, no claim, and is
  written out wherever a relation needs it. (Its multilinear extension is `d + 1 + r_inv` by
  linearity.)
- **Even address widths.** The identity range check rejects an odd number of address bits
  above 2 (`default_phases` in `joltworks/src/subprotocols/identity_range_check.rs`), so every
  one-hot here is read over `even(w)` address bits, where `w` is its
  value's width and `even(w)` rounds up to an even number; the extra top bit is always 0. This
  gives `2s+2` for `out_acc` and `exp_acc`, 32 for `slack`, `even(s+1)` for `inv_sum`, and
  `even(log N + s)` for `r_inv` and `d`.
- `r_k` is the row part of a point `r`, `r_a` an address part. `τ_sum`, `τ_div` and `τ_sat` are
  the zero-check points of 1b, 1c and 2c. `γ` is a scalar combining terms inside one instance,
  drawn per instance. All are drawn by the verifier before the stage that uses them.

#### Stage layout

`BatchedSumcheck` front-pads shorter instances, so they bind the last challenges. Every
`log(rows·N)`-round instance binds the lane variables first, so a `log rows`-round instance lands
on the row part of the batch point. One-hot reads bind address variables first and cycle
variables last (`identity_range_check.rs`), so a read batched with `log(rows·N)`-round instances
outputs `ra(r_a, r)` at the batch point `r`. Within each stage, every claim is therefore at the
stage's point, its row part, or the point extended by address challenges.

**Before stage 1:** the prover sends `out_acc(r0)`, proven by 1a and 1d. `out(r0)` is the node's
reduced output claim.

**Stage 1** (inline): point `r1`

| # | Relation | Rounds | Needs | Produces |
|---|---|---|---|---|
| 1a | `RecipMult`: `Σ_{k,j} eq(r0,(k,j))·exp_q·inv_sum(k) = out_acc(r0)` | log(rows·N) | `out_acc(r0)` (sent) | `exp_q(r1)`, `inv_sum(r1_k)` |
| 1b | Row sum: `Σ_{k,j} eq(τ_sum,k)·[exp_q − (d + 1 + r_inv)/N] = 0` | log(rows·N) | — | `exp_q(r1)`, `d(r1_k)`, `r_inv(r1_k)` |
| 1c | Division: `Σ_k eq(τ_div,k)·[(d + 1 + r_inv)·inv_sum + r_inv − S²] = 0` | log rows | — | `d`, `r_inv`, `inv_sum` at `r1_k` |
| 1d | `out_acc` read: `Σ_{a,t} eq(r0,t)·ra_out(a,t)·(Id(a) + γ·Hi(a)) = out_acc(r0) + γ·out(r0)` | (2s+2) + log(rows·N) | `out_acc(r0)` (sent), `out(r0)` (node) | `ra_out(r1_a, r1)` |

In 1b, the `1/N` weight puts the per-row term inside the lane sum, so `d` and `r_inv` land on
`r1_k` rather than on a second point.

**Before stage 2:** the prover sends `exp_acc(r1)`, proven by 2a and 2b.

**Stage 2** (inline): point `r2`

| # | Relation | Rounds | Needs | Produces |
|---|---|---|---|---|
| 2a | `Mult`: `Σ_t eq(r1,t)·exp_hi·exp_lo = exp_acc(r1)` | log(rows·N) | `exp_acc(r1)` (sent) | `exp_hi(r2)`, `exp_lo(r2)` |
| 2b | `exp_acc` read: `Σ_{a,t} eq(r1,t)·ra_exp(a,t)·(Id(a) + γ·Hi(a)) = exp_acc(r1) + γ·exp_q(r1)` | (2s+2) + log(rows·N) | `exp_acc(r1)` (sent), `exp_q(r1)` (1a) | `ra_exp(r2_a, r2)` |
| 2c | Saturation: `Σ_t eq(τ_sat,t)·[IND·(z_c − (z_bound − 1)) + γ·IND·(1 − IND)] = 0` | log(rows·N) | — | `IND(r2)`, `z_hi(r2)`, `z_lo(r2)` |

**Link, before stage 3:** the prover sends `slack(r2)` (proven by 4c) and `max_k(r2_k)` (proven
by 3a). The verifier derives `x(r2) = max_k(r2_k) − z_hi(r2)·B − z_lo(r2) − slack(r2)`, the
upstream node's first claim on `x`.

**Stage 3** (inline): point `r3`

| # | Relation | Rounds | Needs | Produces |
|---|---|---|---|---|
| 3a | `MaxIndicator`: `Σ_{k,j} eq(r2_k,k)·e(k,j)·x(k,j) = max_k(r2_k)` | log(rows·N) | `max_k(r2_k)` (sent) | `e(r3)`, `x(r3)` |

Stage 3 must be inline because it claims `x`, the upstream node's output, which is reduced when
the node walk reaches that node. It comes after stage 2 because its eq point is `r2_k`.

**Stage 4** (deferred, batched across nodes): point `r4`

| # | Relation | Rounds | Needs | Produces |
|---|---|---|---|---|
| 4a | `exp_hi` Shout: `Σ_{a,t} eq(r2,t)·ra_hi(a,t)·(Val_hi(a) + γ·Id(a)) = exp_hi(r2) + γ·z_hi(r2)` | log\|lut_hi\| + log(rows·N) | 2a, 2c | `ra_hi(r4_a, r4)` |
| 4b | `exp_lo` Shout: the same shape with `lo` | log\|lut_lo\| + log(rows·N) | 2a, 2c | `ra_lo(r4_a, r4)` |
| 4c | Slack read: `Σ_{a,t} eq(r2,t)·ra_slack(a,t)·(Id(a) + γ·1) = slack(r2) + γ·IND(r2)` | 32 + log(rows·N) | sent, 2c | `ra_slack(r4_a, r4)` |
| 4d | `inv_sum` read: `Σ_{a,k} eq(r1_k,k)·ra_inv(a,k)·Id(a) = inv_sum(r1_k)` | even(s+1) + log rows | 1a, 1c | `ra_inv(r4_a, r4_k)` |
| 4e | `r_inv` read: the same shape | even(log N + s) + log rows | 1b, 1c | `ra_rinv(r4_a, r4_k)` |
| 4f | `d` read: the same shape | even(log N + s) + log rows | 1b, 1c | `ra_d(r4_a, r4_k)` |

In 4c, `Σ_a ra_slack(a, t)` is the Hamming weight of the slack chunks, so the `γ·1` term ties
`IND` to the slack one-hot in the same read.

**Stage 5** (deferred one-hot batch): point `r5`

| # | Check | Needs |
|---|---|---|
| 5a | Booleanity and Hamming weight = 1 on `ra_out`, `ra_exp`, `ra_hi`, `ra_lo`, `ra_inv`, `ra_rinv`, `ra_d` | `ra` claims from 1d, 2b, 4a, 4b, 4d, 4e, 4f |
| 5b | Booleanity only on `ra_slack` (its Hamming weight is `IND`, tied in 4c) | 4c |
| 5c | Booleanity and Hamming weight = 1 on `e`, in their own instances (`log_k_chunk = log N`) | `e(r3)` from 3a |

**Claims per polynomial**

| Polynomial | Point(s) | Kind |
|---|---|---|
| `out` | r0 | node output |
| `out_acc` | r0 | virtual, sent; value of a committed one-hot |
| `exp_acc` | r1 | virtual, sent; value of a committed one-hot |
| `exp_q` | r1 | virtual |
| `inv_sum`, `r_inv`, `d` | r1_k | values of committed one-hots |
| `exp_hi`, `exp_lo`, `z_hi`, `z_lo`, `IND` | r2 | virtual |
| `slack` | r2 | virtual, sent; value of a committed one-hot |
| `max_k` | r2_k | virtual, sent |
| `x` | r2, r3 | upstream output; both claims reduced by the upstream node |
| `ra_*`, `e` | their stage of origin, plus the one-hot batch | committed |

`x` is the only polynomial claimed at two points.

#### Module layout (`ops/softmax_last_axis/`)

| File | Change | Content |
|---|---|---|
| `mod.rs` | reworked | Stages 1–3, the claims sent between stages, the link check, and the stage 4–5 instances handed to the deferred batches. `send_auxiliary_vectors` is removed. |
| `recip_mult.rs` | kept | 1a. Input claim becomes `out_acc(r0)`; `inv_sum` becomes an output claim instead of being computed by the verifier. |
| `exp_sum.rs` → `row_sum.rs` | renamed (`git mv`) | 1b. |
| `division.rs` | new | 1c. |
| `exponentiation/mult.rs` | kept | 2a. Input claim becomes `exp_acc(r1)`. |
| `saturation.rs` | new | 2c. Consumes `z_c(r)` as `z_hi·B + z_lo`. |
| `max.rs` | kept | 3a. `e` is the committed argmax one-hot instead of being built by the verifier; input claim is the sent `max_k(r2_k)`. |
| `rc.rs` → `reads.rs` | reworked | One-hot read providers for 1d, 2b (`Id + γ·Hi`) and 4c–4f. |
| `significance_clamp.rs` | deleted | Replaced by `saturation.rs` and the slack read 4c. |
| `witness.rs` | reworked | Prover-derived values: `out_acc`, `exp_acc`, `inv_sum`, `r_inv`, `d`, `slack`, `IND`, and the one-hot indices. |

#### Polynomial ids (`common/src/lib.rs`)

**Removed:**

- Virtual: `SoftmaxSumOutput`, `SoftmaxMaxOutput`, `SoftmaxMaxIndex`, `SoftmaxExpSum`,
  `SoftmaxRecipMultRemainder`, `SoftmaxRemainderRa`, `SoftmaxExpRemainder`,
  `SoftmaxExpRemainderRa`, `SoftmaxClampWitness`, `SoftmaxClampRa`.
- Committed: `SoftmaxRemainderRaD`, `SoftmaxExpRemainderRaD`, `SoftmaxClampRaD`.

**Kept:** `SoftmaxExpQ`, `SoftmaxExpHi`, `SoftmaxExpLo`, `SoftmaxZHi`, `SoftmaxZLo`,
`SoftmaxZHiRa`, `SoftmaxZLoRa` (virtual); `SoftmaxZHiRaD`, `SoftmaxZLoRaD` (committed).

**Added.** Each chunked one-hot has a virtual `…Ra` and a committed `…RaD(node, chunk)`. The
single-chunk argmax one-hot is committed as `…Ra(node)`, like `GatherRa`:

| Quantity | Value id (virtual) | One-hot |
|---|---|---|
| `out_acc` | `SoftmaxOutAcc` | `SoftmaxOutAccRa` / `SoftmaxOutAccRaD` |
| `exp_acc` | `SoftmaxExpAcc` | `SoftmaxExpAccRa` / `SoftmaxExpAccRaD` |
| `inv_sum` | `SoftmaxInvSum` | `SoftmaxInvSumRa` / `SoftmaxInvSumRaD` |
| `r_inv` | `SoftmaxInvRemainder` | `SoftmaxInvRemainderRa` / `SoftmaxInvRemainderRaD` |
| `d` | `SoftmaxInvGap` | `SoftmaxInvGapRa` / `SoftmaxInvGapRaD` |
| `slack` | `SoftmaxSlack` | `SoftmaxSlackRa` / `SoftmaxSlackRaD` |
| `e` (argmax) | — | `SoftmaxArgmaxRa(node)`, committed directly (one chunk, no `…RaD`) |
| `max_k` | `SoftmaxMaxK` | — |
| `IND` | `SoftmaxSatInd` | — |

Committed families go from 5 (`R`, `r_exp`, clamp, `z_hi`, `z_lo`) to 9 (`out_acc`, `exp_acc`,
`slack`, `z_hi`, `z_lo`, argmax, `inv_sum`, `r_inv`, `d`). The first three are 1:1 replacements
of today's wide families; the four new ones are `rows`-sized, smaller than any lane-sized one-hot
by a factor of `N`.

### Alternatives Considered

**1. Proving `max_k ≥ x` (gap 1).** Today nothing rules out `z = max_k − x < 0`: the clamp
lookup silently floors negative `z` to 0. Chosen: the dedicated flag relation
(`z = z_c + slack`, `IND·(z_c − (z_bound − 1)) = 0`), reusing the `z_hi`/`z_lo` digit one-hots
and adding a `slack` one-hot of a 31-bit value.

| Alternative | Why not |
|---|---|
| Extend `clamp_split` (`onnx_proof/clamp_split.rs`), the shared saturating-clamp module, to clamp `z` | It is fixed to i32 (`OUT_LOG_K = 32`). Its symmetric low-saturation branch accepts negative inputs, so gap 1 comes back, and offsetting the input doesn't help. Its output chunks would also duplicate the `z_hi`/`z_lo` digits. |
| Keep today's 32-bit clamp lookup, with an unsigned-identity raf in place of the signed one | Also closes gap 1, but keeps a 32-bit lookup on every lane. The flag relation pays only the sparse 31-bit slack. |

**2. Bit-decomposing `out` and `exp_q` (gap 2).** Today `out·S + R = exp_q·inv_sum` (and likewise
for `exp_q`) holds for `S` field solutions, because neither quotient is decomposed. Chosen: one
one-hot per accumulator (`out_acc`, `exp_acc`), read as `Id + γ·Hi`, which yields the quotient's
claim and leaves the remainder implicit.

| Alternative | Why not |
|---|---|
| Two-phase rescale: prove `out·S + R = out_acc` in one sumcheck and the accumulation in a separate one | Needs an extra stage. The single read gives the quotient in the same sumcheck. |

**3. Range-checking the division remainder.** Once `exp_sum` stops being sent in the clear,
`inv_sum = ⌊S²/exp_sum⌋` needs `0 ≤ r_inv < exp_sum` proven. Chosen: identity one-hots on `r_inv`
and on `d = exp_sum − 1 − r_inv`, each of a `log N + s`-bit value.

| Alternative | Why not |
|---|---|
| `UnsignedLessThan` lookup, as Div uses (`ops/div.rs`) | A 64-bit interleaved lookup per row, larger than two `log N + s`-bit one-hots. |

**4. Binding `max_k` without sending it in the clear.** Chosen: `max_k` stays virtual;
`MaxIndicator` proves it from the committed argmax one-hot, so `x` is claimed twice (`r2` and
`r3`).

| Alternative | Why not |
|---|---|
| Commit `max_k` so `x` is claimed once | Adds a committed polynomial. The upstream node's evaluation reduction merges two `x` claims more cheaply. |
| Prove the argmax is the first occurrence on ties | Softmax output depends only on the max value, so a tie can't produce a second output. First-occurrence argmax belongs to an ArgMax op. |

**5. Binding `exp_sum` without sending it in the clear.** Chosen: `exp_sum` is not a protocol
polynomial; it is written as `d + 1 + r_inv` inside the row-sum and division zero-checks.

| Alternative | Why not |
|---|---|
| Keep a separate `ExpSum` sumcheck | `exp_sum` would need claims at two points: one fixed before stage 1, one at `r1_k`. |

**Future work** (not in this spec):

- Pack the `inv_sum`, `r_inv` and `d` one-hots (all `rows`-sized) into one bucket, like
  `global_clamp`'s, bringing the committed families from 9 to 7.
- A single one-hot over `z_c` instead of `z_hi` and `z_lo`, read by both exp tables over half the
  address variables each. `saturation.rs` consumes only a `z_c(r)` claim so this stays open.
- A single one-hot over `z` (31 bits) with partially structured exp tables, removing `slack` and
  `IND`, at the cost of 31-round reads and new `joltworks` tables.

## Documentation

- `softmax_last_axis/mod.rs` module doc: rewritten. It drops the auxiliary-vector paragraph,
  `TODO(#218)`, the `MODEL_SCALE`-only note and the HackMD link (which describes the old claim
  flow). It describes the stage layout in short, and defines `d` (the remainder gap and why
  `d ≥ 0` is checked) and `exp_sum` (the expression `d + 1 + r_inv`, not a polynomial).
- `row_sum.rs`, `division.rs`, `saturation.rs`, `reads.rs`: each gets a short module doc stating
  the relation it proves.
- `README.md`, `docs/` and `atlas-trace/MIGRATION.md`: no change. The README only mentions
  softmax in its introduction, `docs/` doesn't cover it, and the migration notes describe the
  unchanged trace side.

## Execution

Suggested commit sequence. Each commit leaves a complete, sound protocol that passes the
workspace tests. Any commit may be split further during implementation if its scope turns out
too large.

1. **Pin the gaps.** Saturating `attention_like` with its build-time assertion, the widened
   `softmax_random_4x16`, and `known_gap_softmax_non_max_pivot` landing `#[ignore]`d and failing.
2. **Accumulator one-hots (gap 2).** The `out_acc` and `exp_acc` one-hots with the `Id + γ·Hi`
   reads (1d, 2b) replace the `R` and `r_exp` range checks. `RecipMult` and `Mult` take
   `out_acc(r0)` and `exp_acc(r1)`.
3. **Saturation flag (gap 1).** `saturation.rs` (2c), the slack one-hot and read (4c), and the
   link `x(r2) = max_k − z_hi·B − z_lo − slack`. `significance_clamp.rs` is deleted and the
   `MODEL_SCALE` restriction lifted; `softmax_model_scale_2x8` moves to scale 12. The pin turns
   green and its `#[ignore]` is removed. Until commit 5, the link still takes `max_k` from the
   clear vector.
4. **Division.** `row_sum.rs` (1b, `git mv` from `exp_sum.rs`), `division.rs` (1c), and the
   `inv_sum`/`r_inv`/`d` one-hots and reads (4d–4f). The `exp_sum` clear vector and
   `SoftmaxExpSum` go.
5. **Argmax.** The committed argmax one-hot `e`, `MaxIndicator` moved to stage 3 on `r2_k`, and
   `max_k(r2_k)` sent. `send_auxiliary_vectors`, the `max_k`/`argmax` vectors and `TODO(#218)`
   go; the `mod.rs` doc is rewritten.

After commit 5: run `zoo::gpt2_dory` locally, then the Performance benchmark on `main` and on
the branch, and report the numbers in the PR description.

## References

- Issue #218, "restructure softmax prover to avoid sending aux vectors". This spec closes it.
