# Spec: Softmax without auxiliary vectors

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`, `specs/post-loop-batches.md` |
| Supersedes  | issue #218                     |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`SoftmaxLastAxis` over `F` rows of `N` lanes is the one op whose proof is not polylogarithmic in
its size and whose verifier does work linear in it. The prover sends three vectors of `F`
values in the clear, the row maximum `max_k`, the row sum of exponentials `exp_sum_q` and the
position of the maximum `argmax_k`, as `3F` opening claims at the empty point. The verifier
decodes them, performs `F` integer divisions to obtain `inv_sum[k] = ⌊S² / exp_sum_q[k]⌋`,
evaluates four `F`-entry multilinear extensions, builds an `F·N` indicator table and evaluates
it. These vectors are also why softmax has its own `Open`, `Tie` and relation enums in the
plan and a helper that re-decodes field elements into integers. Each of the three can be
derived inside the protocol from polynomials that are committed or already bound. This spec
does that: the proof of a softmax node carries no per-row values, and its verifier does no
per-row work.

## Intent

### Goal

Remove `VirtualPoly::{SoftmaxSumOutput, SoftmaxMaxOutput, SoftmaxMaxIndex}` and everything that
reads them, replacing each auxiliary vector by a relation:

| Auxiliary vector | Replaced by |
|---|---|
| `argmax_k` (public indicator `e`) | a committed one-hot `e` over `(row, lane)`: the address of the maximum in each row, chunked and checked like any Shout read address |
| `max_k` | nothing sent. `max_k(r_lead)` is defined as `x(r) + z(r)` from the operand and clamp-witness openings at one point `r = (r_lead, r_lane)`, and proven equal to `Σ_{k,j} eq(r_lead, k)·e(k,j)·x(k,j)` |
| `exp_sum_q` and the verifier's divisions | a committed row quotient `inv_sum` and remainder `r_div` with the division relation `Σ_k eq(τ, k)·(inv_sum[k]·exp_sum[k] + r_div[k]) = S²`, and `0 ≤ r_div < exp_sum` by the range buckets |

Introduced or modified:

- `jolt-atlas-verifier`: `ops/softmax_last_axis/` (`mod.rs` plan and stages; `max.rs` over a
  committed `e`; `recip_mult.rs` with `inv_sum` as an opening; `exp_sum.rs`; a new
  `division.rs`; `SoftmaxShape::aux` and `inv_sum_evals` deleted), `plan.rs`
  (`SoftmaxOpen::Aux`, `SoftmaxOpen::ExpSum` and `SoftmaxTie::ExpSum` deleted;
  `SoftmaxTie::OperandLink` becomes the definition of the max claim).
- `common`: three `VirtualPoly` variants deleted; `CommittedPoly::SoftmaxArgmaxRaD(node, d)`
  added; `VirtualPoly::{SoftmaxArgmaxRa, SoftmaxInvSum, SoftmaxDivRemainder}` added.
- `jolt-atlas-core`: the prover mirrors; witnesses from `SoftmaxWitness` (`argmax_k`,
  `inv_sum`, `exp_sum_q` are already in the trace; `r_div = S² − inv_sum·exp_sum_q` is
  computed by the prover).
- `atlas-trace`: no change.

### Invariants

Relied on, not changed:

1. **Softmax semantics.** `SoftmaxLastAxis::eval` (`atlas-trace/src/op/softmax.rs`) and the
   decomposed exp table from `semantics::tables::exp_lut_decomposed`: `z = max − x`, the
   significance clamp, `exp_q = ⌊exp_hi·exp_lo / S⌋`, `inv_sum = ⌊S² / Σ exp_q⌋`,
   `out = ⌊exp_q·inv_sum / S⌋`. Outputs do not move. *Mechanism:* the importer's golden hashes
   are not re-pinned.
2. **The untouched stages.** The exp product, the exp-digit lookups, the significance clamp
   and the two `2^scale` remainders keep their relations. *Mechanism:* review on the diff.
3. **Unmasked only.** A masked softmax (`active_len` below the last dimension) is still
   rejected. *Mechanism:* the existing plan check.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. The maximum is an element of the row.** `e` is one-hot in every row (Booleanity and
  Hamming weight one, by the standard one-hot check), so `M[k] = Σ_j e(k,j)·x(k,j)` is some
  `x[k, j*]`. *Mechanism:* the one-hot check in the `OneHot` batch; a `witness_tamper_sweep`
  site that sets two lanes of `e` in a row, and one that sets none.
- **B. `z` is the distance to that element, in every lane.** The max relation's input claim is
  `x(r) + z(r)` at a point `r` drawn after `e` and the digits of `z` are committed, and its sum
  is multilinear in `r_lead` and constant in `r_lane`; equality at a random `r` gives
  `x[k,j] + z[k,j] = M[k]` for all `k, j`. *Mechanism:* review-held, stated in `max.rs`;
  a tamper site that shifts `z` by one in a single lane.
- **C. That element is the maximum.** `z[k,j] ≥ 0` for all lanes, because `z` is the unsigned
  value of a 32-bit one-hot address and `x`, `M` are bounded `i32`s, so `M[k] ≥ x[k,j]`. With A
  and B this is `M[k] = max_j x[k,j]`. The analysis step confirms the reading of `z` as an
  unsigned address value against `significance_clamp.rs` before implementation.
  *Mechanism:* a tamper site that points `e` at a non-maximal lane, which forces a negative
  `z` in the maximal lane.
- **D. Division.** An accepted proof implies `inv_sum[k]·exp_sum[k] + r_div[k] = S²` and
  `0 ≤ r_div[k] < exp_sum[k]` for every row, hence `inv_sum[k] = ⌊S² / exp_sum[k]⌋` and
  `exp_sum[k] ≥ 1`. `inv_sum` and `r_div` are bound to committed digits through the range
  buckets, so they are fixed before `τ` is drawn; `inv_sum ∈ [0, 2^30)` and
  `exp_sum < 2^31` keep the product below the field modulus. *Mechanism:* tamper sites
  `inv_sum ± 1` with `r_div` adjusted to keep the identity; the range-bucket sites of
  `specs/one-range-check.md`.
- **E. `exp_sum` is the row sum.** The division relation's `exp_sum(ρ′)` claim is the input
  claim of the existing exp-sum relation `exp_sum(ρ′) = Σ_{k,j} eq(ρ′, k)·exp_q(k,j)`.
  *Mechanism:* the opening-forgery sweep over the new claim.
- **F. No per-row proof data and no per-row verifier work.** The number of opening claims of a
  softmax node does not depend on `F`, and the verifier's softmax code contains no loop over
  rows. *Mechanism:* a test comparing the claim counts of two softmax probes with different
  `F`; review for the second half.

### Non-Goals

- Masked softmax.
- The fused attention design of `docs/prover-plan.md` item 6 (scores emitted as `z`, one table
  for range, exp and mask). This spec keeps the op's arithmetization.
- Any change to the exp table, its decomposition, or `SemVersion`.
- Supporting a scale other than `MODEL_SCALE`.
- Proof-size or prover-time targets beyond the bounds below.

## Evaluation

### Acceptance Criteria

- [ ] The analysis of invariant C (the reading of `z`) is recorded in the PR.
- [ ] `VirtualPoly::{SoftmaxSumOutput, SoftmaxMaxOutput, SoftmaxMaxIndex}`,
      `SoftmaxOpen::Aux`, `SoftmaxTie::ExpSum`, `SoftmaxShape::aux` and `inv_sum_evals` do not
      exist; the `TODO(#218)` comments are gone.
- [ ] Invariant F: the claim-count test passes; no softmax verifier function iterates over
      rows.
- [ ] Invariants A, B, C and D: each named tamper site is rejected with `Err`.
- [ ] The three sweeps pass over the probe models with no tolerance entry; one `mutate::Site`
      per new proof field.
- [ ] `relation_specs_match_their_dispatched_op_kind` passes; the softmax disposition in
      `testing/relations.rs` is no longer `Deferred`.
- [ ] `proof_types_all_exercised` passes.
- [ ] The importer's golden hashes are untouched; `testing/digests.rs` is re-pinned in its own
      commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the softmax op-surface rows
(including ties between lanes, where any maximal lane is a valid `e`); the three sweeps; the
zoo with its serialization round-trip; `fuzz_regressions_replay`, its corpus regenerated in
the digest commit.

New tests: the tamper sites of invariants A–D; the claim-count test; an op-surface row with a
row whose lanes are all equal, and one with `exp_sum` at its smallest value; if a specialized
prover is written for the max or division relation, a `relation_prover_equiv_*` test against
the engine's.

CI coverage: the workspace job; the heavy job (GPT-2 has 12 softmax nodes; bge-small's
attention has them at a different shape). The `zk` feature is not touched.

### Performance

Benchmark: `gpt2 --seq-len 16` and `--seq-len 64`, and `bge-small-en-v1.5`, against
`bench/bin/prove-c65753bfa`. Softmax grows with the square of the sequence length, so the
64-token run is the one that can show a difference.

Expected: proof size down by `3F` field elements per softmax node, less the new sumcheck
proofs; verify time down; prover time flat (one extra `log F`-round sumcheck, one one-hot
family with `⌈log₂ N / 4⌉` chunks over `F` rows, and row-sized range-bucket members).

Bounds: prove within 3% of baseline on all three runs; verify no higher; proof size no larger
at 16 tokens and smaller at 64; peak memory no higher.

## Design

### Architecture

Plan of a softmax node, with `r0` the reduced output point:

```text
Reduce
Relation[SoftmaxDivision]               τ drawn over rows
    Σ_k eq(τ, k)·(inv_sum[k]·exp_sum[k] + r_div[k]) = S²                   → inv_sum(ρ′), exp_sum(ρ′), r_div(ρ′)
Open(RecipMultRemainder)                at r0
Batch[SoftmaxStage1]
    recip-mult   S·out(r0) + R(r0) = Σ eq(r0, ρ)·exp_q(ρ)·inv_sum(ρ_k)      → exp_q(r1), inv_sum(r1_k)
    exp-sum      exp_sum(ρ′) = Σ eq(ρ′, ρ_k)·exp_q(ρ)                       → exp_q(r1)
Open(ExpRemainder)                      at r1
Batch[SoftmaxStage2]
    exp product  S·exp_q + r_exp = Σ eq·exp_hi·exp_lo                       → exp_hi, exp_lo at r2
Open(ZHiLo), Open(x)                    at r2
Defer{Lookups}   exp-hi, exp-lo, significance clamp                         (appends z at r2)
Tie(OperandLink) defines c = x(r2) + z(r2)
Relation[SoftmaxMax]
    c = Σ_{k,j} eq(r2_lead, k)·e(k,j)·x(k,j)                                → e(ρ″), x(ρ″)
Defer{OneHot}    one-hot checks of exp-hi, exp-lo, clamp and e
```

The division relation runs first so that exp-sum, whose input claim it produces, stays in the
stage-1 batch with recip-mult: the two have equal rounds and share the `exp_q` point `r1`, as
they do today. `inv_sum` is opened at two points (`ρ′` and `r1_k`), each a member of its range
bucket's value sumcheck, as is
`r_div(ρ′)` and the slack `exp_sum(ρ′) − 1 − r_div(ρ′)`. The operand `x` receives two claims
(`r2` and `ρ″`), which its evaluation reduction merges as it merges any two consumers.

`e` is the read address of "read row `k` at its maximal lane": committed as
`SoftmaxArgmaxRaD` chunks, virtualised to `SoftmaxArgmaxRa` and checked by
`shout::ra_onehot_verifiers` in the `OneHot` batch, with no new one-hot machinery.

### Alternatives Considered

- **Keep the vectors.** Rejected: linear proof and verifier, and the only relations in the
  crate that depend on decoded proof values.
- **`⟨e, z⟩ = 0` in place of the max relation** (each row of `z` has a zero). Rejected: it
  shows attainment but not that `x + z` is the same in every lane; the max relation gives both
  from one sumcheck.
- **A per-row product `Π_j z[k,j] = 0` by a GKR tree, with no commitment.** Rejected: more
  prover work and the same gap.
- **Check lane-independence of `x + z` by opening both at a second lane point.** Rejected: a
  second opening of `z`, which is a lookup's raf.
- **Leave `inv_sum` virtual.** Unsound: an uncommitted polynomial is not fixed before the
  division challenge.
- **Fused attention now.** Rejected here: it changes the arithmetization and needs its own
  spec; this change is orthogonal and smaller.

## Documentation

- `docs/verifier.md`: the softmax paragraph of "Plans" is rewritten (no aux vectors).
- Module docs of `ops/softmax_last_axis/` in both crates, with the argument of invariants A–D.
- `docs/upstream-inventory.md`: the softmax note that waits on #218.
- `README.md`: the benchmark table if a row moves outside noise.

## Execution

Suggested order: (1) the division relation with `inv_sum` and `r_div` in the range buckets,
`exp_sum_q` removed; (2) the committed `e` and the max relation, `argmax_k` and `max_k`
removed; (3) deletions; (4) the digest re-pin. Each step leaves the tree provable.

## References

- `jolt-atlas-verifier/src/onnx_proof/ops/softmax_last_axis/` (current protocol);
  `atlas-trace/src/op/softmax.rs` (semantics); `atlas-trace/src/trace.rs` (`SoftmaxWitness`).
- Issue #218; the protocol note linked from the module header.
- `research_notes/Op proving efficiency survey/softmax.md`.
- `docs/prover-plan.md` §3 item 6 (fused attention, out of scope).
