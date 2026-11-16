# Spec: Log-partition softmax

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/one-range-check.md`, `specs/post-loop-batches.md` |
| Supersedes  | `specs/softmax-aux-removal.md`; `docs/prover-plan.md` §6, "prover-chosen softmax shifts" |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`SoftmaxLastAxis` computes `max`, `z = max − x`, an exponential as the floored product of two
512-entry tables, a row sum, a floored reciprocal and a floored product. Its proof commits 28
cells per entry (14 one-hot chunks for the clamp and the two exp digits, 14 digits for two
remainders), sends three per-row vectors in the clear, and has the verifier divide once per
row. Simulated against float at scale 2^14, single entries are off by up to 80 ulp and wide
rows lose up to 5% of their probability mass to the floors.

Softmax is `exp(x_j − L)` with `L` the log of the row's partition sum. Every published
shift-normalised protocol (zkLLM, DeepProve) uses that form with a prover-chosen `L` and a
tolerance band on the row sum, so the output is not a function of the input. This spec uses
the same form and pins the shift: `c` is the smallest integer shift whose row sum does not
exceed the scale. Each entry is then one table read at address `c − x`. There is no maximum,
no argmax, no reciprocal, no per-entry division or remainder, and nothing sent in the clear.
The minimality of `c` is checked by a second read, of the table shifted by one address,
through the same committed lookup address, which costs no commitment.

## Intent

### Goal

Replace the semantics and the proof of `SoftmaxLastAxis`, under a new `SemVersion`, by the
log-partition form.

Semantics, per row of `N` lanes, in integers. `S = 2^scale`, `S' = S·2^8`, `D` the table's
address width (18 at scale 14), `top = 2^D − 1`:

```text
T_out[u] = round(S ·e^(−u/S))        u ∈ [0, 2^D)      the output table
T_sum[u] = round(S'·e^(−u/S))        u ∈ [0, 2^D)      the same function, 8 more bits
f(c)     = Σ_j T_sum[min(c − x_j, top)]
c        = min { c ≥ max_j x_j : f(c) ≤ S' }
y_j      = T_out[min(c − x_j, top)]
```

`D` is the smallest width at which `T_sum` is zero on its two top entries. Execution fails
with a `KernelError` if `c − min_j x_j ≥ 2^32`.

The shift is pinned by the 22-bit sum and the output read from the 14-bit table because
pinning it by the output table's own sum makes the largest entries absorb the accumulated
rounding of the small ones (simulated: up to 11 ulp at `N = 1024`).

Proof, per softmax node of `F` rows:

| Object | Shape | Kind |
|---|---|---|
| `c` | per row | committed, dense |
| clamp address `u = c − x` | per entry | committed one-hot chunks of a 32-bit address (8 chunks), as today's clamp |
| table address `ũ = min(u, top)` | per entry | committed one-hot chunks of a `D`-bit address (5 chunks) |
| `g = S' − f(c)` and `g⁻ = f⁻ − S' − 1` | per row | range claims in the remainder buckets |

with `f⁻ = Σ_j T_sum⁻[ũ_j]`, where `T_sum⁻[0] = S' + 1` and `T_sum⁻[u] = T_sum[u − 1]`.

Per entry the proof commits 13 one-hot cells and no digits, against 28 cells today.

Introduced or modified:

- `atlas-trace`: the next `SemVersion`; `op/softmax.rs` (the kernel above; `SoftmaxWitness`
  reduced to `c` per row and `u` per entry); `semantics/tables.rs` (`T_out`, `T_sum` from
  `exp_neg_q62`; `exp_lut_decomposed` removed with its last reader); the behavioral contract.
- `jolt-atlas-verifier`: `ops/softmax_last_axis/` rewritten (one module: the plan, the address
  link, the read relation); `plan.rs` (`SoftmaxOpen`, `SoftmaxTie`, `SoftmaxRelation` lose
  their stage variants; the softmax-specific enums disappear if their remaining members fit
  the generic ones); `global_clamp.rs` (two `Remainder` variants over a node's row domain,
  replacing `SoftmaxRecipMult` and `SoftmaxExp`; a bucket member whose domain is the node's
  rows, not its cells); `types.rs` (`SoftmaxStage1`, `SoftmaxStage2` retired).
- `common`: the softmax `CommittedPoly` and `VirtualPoly` variants replaced by the objects of
  the table above.
- `jolt-atlas-core`: the prover mirror.
- `atlas-onnx-import`: golden hashes re-pinned.

### Invariants

Relied on, not changed:

1. **Lookup machinery.** The prefix-suffix read-raf over a 32-bit address (the clamp), the
   Shout read over a small table, the one-hot checks, and the two post-loop batches.
   *Mechanism:* their existing tests; the softmax rows of the op-surface table.
2. **Range claims.** A remainder bucket proves a member's value is in `[0, 2^w)`.
   *Mechanism:* `specs/one-range-check.md`.
3. **Unmasked only.** A masked softmax (`active_len` below the last dimension) is still
   rejected. *Mechanism:* the existing plan check and `softmax_masked_rejected`.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. The output is a function of the input.** For every row there is exactly one `c`
  satisfying the definition, so `R` is single-valued and the trace is reproducible.
  *Mechanism:* `f` is nonincreasing in `c` (each `T_sum[min(c − x_j, top)]` is) and reaches
  zero; a tracer unit test compares the kernel's `c` with a linear scan from `max_j x_j`.
- **B. S ⊆ R.** From an accepted proof:
  1. the one-hot checks make `u` an integer in `[0, 2^32)` and `ũ` an integer in `[0, 2^D)`
     per entry, with `ũ = min(u, top)` (the clamp table) and the reads equal to table entries;
  2. the address link, at a random point, gives `u_kj = c_k − x_kj` in the field for every
     entry; both sides are integers of magnitude below `2^33`, so the equation holds over the
     integers and `c_k ≥ x_kj` for every lane;
  3. the read relation, at a random row point, gives `g_k = S' − f(c_k)` and
     `g⁻_k = f⁻_k − S' − 1` for every row; all sums are below `N·(S' + 1)`, far from the
     modulus; the range claims give `f(c_k) ≤ S'` and `f⁻_k > S'`;
  4. `f⁻_k` is `f(c_k − 1)` when every `u_kj ≥ 1`, and exceeds `S'` by construction when some
     `u_kj = 0`, that is when `c_k = max_j x_kj`. Either way no smaller admissible shift has
     `f ≤ S'`, since `f` is nonincreasing. So `c_k` is the defined shift and `y = T_out[ũ]`
     is the defined output.
  *Mechanism:* the relation docs carry this argument (`docs/pull-requests.md` §4.2);
  `witness_tamper_sweep` over a probe model with the softmax witness fields as sites; forged
  tests for `c + 1`, `c − 1` and a lane above `c`.
- **C. Nothing per row in the clear, no per-row verifier work.** The proof of a softmax node
  has `O(log)` field elements beyond its sumchecks, and the verifier performs no division and
  builds no table of `F` or `F·N` entries. *Mechanism:* the three aux `VirtualPoly` variants
  do not exist; a test on the proof's claim count for two row counts.
- **D. One table evaluation per table per proof.** The verifier evaluates the multilinear
  extensions of `T_out`, `T_sum` and `T_sum⁻` at most once each per proof, not once per softmax
  node, or the verify-time bound of the Performance section is met without that.
  *Mechanism:* the gate.
- **E. Accuracy.** Against the float softmax of the same integer logits, every entry is within
  2 ulp and every row sum within `N/2 + 2` of `S`. *Mechanism:* a tracer test over random rows
  at `N ∈ {16, 64, 256, 1024}` and logit spreads from 0.5 to 6, with `f64` as the reference.
- **F. Version binding.** Outputs change, so the importer's golden hashes and the digest pins
  are re-pinned, each in its own commit. *Mechanism:* the golden-hash tests.

### Non-Goals

- Fused attention (`docs/prover-plan.md` item C3): the score einsum emitting `c − score` so
  its committed digits serve as the lookup address. This spec keeps softmax a separate op.
- Narrowing the 32-bit clamp address, or fusing the clamp into the table read (item B2).
- Masked softmax and the causal mask.
- A piecewise-linear exp table. `T_out` and `T_sum` are plain rounded tables; the
  piecewise-linear tables emit 8 unrounded fractional bits and would bring back a per-entry
  rounding check.
- An output scale other than `S`.
- Any relaxed statement: no tolerance band, no prover-chosen value that the proof does not pin.

## Evaluation

### Acceptance Criteria

- [ ] Invariant A: the kernel's `c` equals a linear scan's on random and adversarial rows (all
      lanes equal, one dominant lane, spread beyond `top`).
- [ ] Invariant E holds on the stated grid; the test prints the measured maxima.
- [ ] A row with `c − min x ≥ 2^32` fails to execute with a `KernelError` naming the node.
- [ ] `VirtualPoly::{SoftmaxSumOutput, SoftmaxMaxOutput, SoftmaxMaxIndex}`,
      `SoftmaxShape::aux`, `inv_sum_evals`, the exp digit tables and the two softmax remainder
      claims do not exist; `ProofType::{SoftmaxStage1, SoftmaxStage2}` are retired.
- [ ] The softmax plan is: reduce, open `c` and the clamped address, the address link, two
      `Defer`s to `Lookups`, two `Defer`s to `OneHot`; no in-loop sumcheck.
- [ ] A softmax node's proof commits 13 one-hot chunk polynomials per entry and none of the
      digit columns it has today (checked on the committed-polynomial list of a probe model).
- [ ] Invariant B: the three forged tests are rejected; the sweeps pass over a probe model
      with a softmax, with no tolerance entry.
- [ ] Invariant C: the claim-count test passes.
- [ ] The softmax op-surface rows pass, including a one-row and a one-lane case.
- [ ] The golden hashes and `testing/digests.rs` are re-pinned, each in its own commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the three sweeps; the zoo with its
serialization round-trip; `proof_types_all_exercised`; the importer's zoo gates.

Changed tests: the softmax unit tests of `op/softmax.rs` are rewritten for the new kernel; the
`Softmax` sites of the witness tamper sweep follow the new witness fields; the golden hashes.

New tests: the linear-scan and accuracy tests of invariants A and E; the forged-shift tests
and the claim-count test; an op-surface row with a row whose spread exceeds `top`.

CI coverage: the workspace job; the importer job; the heavy job for GPT-2 and bge-small. The
`zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the
commit this lands on, on the benchmark machine. Softmax is about 1.5% of committed rows on
GPT-2 at 16 tokens and grows with the square of the sequence length, so the PR also reports
one GPT-2 run at 64 tokens before and after.

Expected: commit and openings down by softmax's share times roughly one half (13 cells per
entry for 28); the IOP stage down (two in-loop batches and four relations per node removed);
proof size down by `3F` field elements and two sumchecks per node. Verify time down by the
per-row work, up by three evaluations of a `2^D`-entry table (invariant D).

Tracer: the shift is found by bisection, about 33 evaluations of `f` per row.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better on both
models at 16 tokens; peak memory within 5%; proof size no larger.

## Design

### Architecture

```text
node turn:
    Reduce                               y(r0), r0 = (r0_row, r0_lane)
    Open     c at r0_row (committed); ũ at r0 (virtual); g, g⁻ at τ (virtual), τ a row challenge
    Tie      address link: the clamp lookup's operand is c(r0_row) − x(r0)
    Defer    Lookups   clamp read-raf:  ũ(r0) = Clamp[u],  u(r0) = c(r0_row) − x(r0)
    Defer    Lookups   table read (below)
    Defer    OneHot    the clamp address's one-hot checks
    Defer    OneHot    the table address's one-hot checks
post-loop:
    g and g⁻ in the remainder buckets, members over the node's row domain
```

The table read is one sumcheck over the table address and the node's cells:

```text
y(r0) + γ·(S' − g(τ)) + γ²·(g⁻(τ) + S' + 1) + γ³·ũ(r0)
  = Σ_{a, k, j} ra(a, k, j) · ( eq(r0, (k,j))·(T_out[a] + γ³·a)
                               + eq(τ, k)·(γ·T_sum[a] + γ²·T_sum⁻[a]) )
```

The row sums need no sumcheck of their own: they are the same read with the cell weight
`eq(τ, k)`, constant along the lane. All three table reads and the address read share one
committed `ra`, so the relation produces one `ra` claim.

The operand claim `x(r0)` is appended in the node's turn, at the reduced output point.

The range claims `g` and `g⁻` are the first whose domain is a node's rows; a bucket member's
size becomes a property of the claim, not of the node (`global_clamp::BucketNode`,
`range_claim_bits`).

For invariant D the read relation binds the cell variables before the address variables, so
every softmax read in the `Lookups` batch ends at the same address point and the verifier
caches the three table evaluations. If the existing small-table prover binds the address
first, this needs its order changed; settle it before the rest of the prover.

### Alternatives Considered

- **`specs/softmax-aux-removal.md`.** Keeps today's semantics and replaces the three aux
  vectors by a committed argmax one-hot, a row reciprocal and a division relation. Rejected:
  it keeps 28 cells per entry and today's accuracy, and adds commitments.
- **The reciprocal design with one exp table** (the `sc3` proposal: `m − x` as the lookup
  address, the maximum attained read off the address-zero slice, one reciprocal per row).
  Sound and better than today, but it keeps a per-entry quotient range check and remainder,
  about 28 cells per entry in our encodings, and floors that lose up to 5% of a wide row's
  mass.
- **Shift normalisation with a tolerance band** (zkLLM, DeepProve). Rejected: the output is
  not a function of the input.
- **Pinning the shift by the output table's own sum.** Rejected on accuracy (Goal).
- **Normalisation fused into the value einsum** (zkGPT). Not needed: the log-partition form
  has no normalisation step. Fused attention remains a later item.
- **Base-2 range reduction or digit-product exp tables.** Rejected: products of rounded
  entries are what costs today's accuracy, and published measurements find lookup cost
  follows lookup count and address width, not table size.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the softmax clause of the behavioral contract.
- `docs/verifier.md`: the "Plans" section drops softmax as the stress case.
- `docs/prover-plan.md`: §6 "Not pursuing" loses "prover-chosen softmax shifts" (the shift
  here is pinned); the softmax and C3 lines of item 6 are rewritten.
- `docs/op-roadmap.md`: the exp op's note on the multiplicative split.
- Module docs of the softmax modules in both crates, with the S ⊆ R argument.
- `README.md`: the benchmark table and the accuracy statement.

## Execution

Suggested order: (1) the kernel, tables and tracer tests under the new `SemVersion`, with the
golden hashes re-pinned; (2) the row-domain range claim; (3) the read relation and the plan,
verifier then prover; (4) deletion of the stages, aux vectors and digit tables; (5) the digest
re-pin and the gate.

If `specs/fused-gelu.md` or `specs/one-clamp.md` lands in the same period, share one
`SemVersion` bump.

## References

- `atlas-trace/src/op/softmax.rs`, `atlas-trace/src/semantics/{tables,exp}.rs`.
- `jolt-atlas-verifier/src/onnx_proof/ops/softmax_last_axis/`, `global_clamp.rs`,
  `atlas-sumcheck/src/{shout,ps_shout}/`.
- zkLLM (arXiv 2404.16109, §5: the unverified log-sum-exp shift and the row-sum band);
  DeepProve (ePrint 2026/1112); zkGPT (ePrint 2025/1184: normalisation fused into the value
  product); NanoZK (arXiv 2603.18046); "Bits Under ZK-LLM" (arXiv 2609.36437) and "A Separation
  Principle for Lookup-Based zkML" (ePrint 2026/1390) on what lookup cost depends on.
- The accuracy simulation behind the Summary and invariant E: scale 2^14, 40 random rows per
  cell, `N ∈ {16, 64, 256, 1024}`, Gaussian logits with σ from 0.5 to 6.
