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
rows lose up to 5% of their probability mass to the floors. A softmax whose width is not a
power of two is not provable at all, which limits every model to power-of-two sequence
lengths.

Softmax is `exp(x_j − L)` with `L` the log of the row's partition sum. Every published
shift-normalised protocol (zkLLM, DeepProve) uses that form with a prover-chosen `L` and a
tolerance band on the row sum, so the output is not a function of the input. This spec uses
the same form and pins the shift: `c` is the smallest integer shift whose row sum does not
exceed the scale. Each entry is then one table read at address `c − x`. There is no maximum,
no argmax, no reciprocal, no per-entry division or remainder, and nothing sent in the clear.
The minimality of `c` is checked by a second read, of the table shifted by one address,
through the same committed lookup address, which costs no commitment. Padded lanes are
excluded by a public selector, so a softmax of any width is provable.

## Intent

### Goal

Replace the semantics and the proof of `SoftmaxLastAxis`, under `SemVersion::V2`, by the
log-partition form, for any `active_len` up to the padded width.

Semantics, per row, in integers. `A` is the set of active lanes (`j < active_len`),
`S = 2^scale`, `S' = S·2^8`, `D` the table's address width (18 at scale 14), `top = 2^D − 1`:

```text
T_out[u] = round(S ·e^(−u/S))        u ∈ [0, 2^D)      the output table
T_sum[u] = round(S'·e^(−u/S))        u ∈ [0, 2^D)      the same function, 8 more bits
f(c)     = Σ_{j ∈ A} T_sum[min(c − x_j, top)]
c        = min { c ≥ max_{j ∈ A} x_j : f(c) ≤ S' }
y_j      = T_out[min(c − x_j, top)]   for j ∈ A,   0 otherwise
```

`D` is the smallest width at which `T_sum` is zero on its two top entries. Execution fails
with a `KernelError` naming the node if `c` does not fit in `i32`, or if `c − x_j` leaves the
signed 32-bit range on any lane of the padded row, active or not.

The shift is pinned by the 22-bit sum and the output read from the 14-bit table because
pinning it by the output table's own sum makes the largest entries absorb the accumulated
rounding of the small ones (simulated: up to 11 ulp at 1024 lanes).

Proof, per softmax node of `F` rows and `N` padded lanes:

| Object | Shape | Kind |
|---|---|---|
| `c` | per row | committed, dense `i32` |
| clamp address `u = c − x` | per entry | committed one-hot chunks of a signed 32-bit address (8 chunks), as today's clamp |
| table address `ũ = Clamp[u]` | per entry | committed one-hot chunks of a `D`-bit address (5 chunks) |
| `g = S' − f` and `g⁻ = f⁻ − S' − 1` | per row | range claims in the remainder buckets |

with `f⁻ = Σ_{j ∈ A} T_sum⁻[ũ_j]`, where `T_sum⁻[0] = S' + 1` and
`T_sum⁻[u] = T_sum[u − 1]`. `Clamp` is today's `SoftmaxClampTable`: a negative address maps to
0, an address above `top` to `top`.

Per entry the proof commits 13 one-hot cells and no digits, against 28 cells today.

Introduced or modified:

- `atlas-trace`: `SemVersion::V2`, the default of `CompileOptions`; `op/softmax.rs` (the kernel
  above; `SoftmaxWitness` reduced to `c` per row and `u` per entry); `semantics/tables.rs`
  (`T_out`, `T_sum` from `exp_neg_q62`; `exp_lut_decomposed` removed); the behavioral
  contract.
- `atlas-sumcheck`: `shout::ReadRafRelation` reads a random combination of several explicit
  tables through one `ra` (it reads one `Vec<i32>` today).
- `jolt-atlas-verifier`: `ops/softmax_last_axis/` rewritten (the plan, the sum reduction, the
  address link); `plan.rs` (the softmax `Open`, `Tie` and relation enums reduced to the
  members below); `global_clamp.rs` (two `Remainder` variants over a node's rows, replacing
  `SoftmaxRecipMult` and `SoftmaxExp`; a member's domain size comes from the claim, which
  touches `BucketNode::log_t`, `remainder_buckets`, `remainder_inputs` and the prover's
  `range_claim_bits`); `types.rs` (`SoftmaxStage1`, `SoftmaxStage2` retired, `SoftmaxSums`
  added); a per-proof cache of table evaluations in `Verifier`; `verify` requires
  `model.version() == SemVersion::V2`.
- `common`: the softmax `CommittedPoly` and `VirtualPoly` variants replaced by the objects of
  the table above.
- `jolt-atlas-core`: the prover mirror; the `prove` example no longer truncates text to a
  power-of-two length.
- `atlas-onnx-import`: golden hashes re-pinned.

### Invariants

Relied on, not changed:

1. **Lookup machinery.** The prefix-suffix read-raf over a signed 32-bit address (the clamp),
   the address-only Shout read over an explicit table, the one-hot checks, and the two
   post-loop batches. *Mechanism:* their existing tests; the softmax op-surface rows.
2. **Range claims.** A remainder bucket proves a member's value is in `[0, 2^w)` for even `w`.
   *Mechanism:* `specs/one-range-check.md`.
3. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. The output is a function of the input.** For every row there is exactly one `c`
  satisfying the definition. *Mechanism:* `f` is nonincreasing in `c` and reaches zero; a
  tracer test compares the kernel's `c` with a linear scan from the row maximum.
- **B. S ⊆ R.** From an accepted proof, for every row:
  1. the one-hot checks make `u` an integer in `[−2^31, 2^31)` and `ũ` an integer in
     `[0, 2^D)` per entry, with `ũ = Clamp[u]` and every read a table entry;
  2. the address link, at a random point, gives `u_kj = c_k − x_kj` in the field for every
     entry; both sides are integers of magnitude below `2^33`, so it holds over the integers;
  3. the sum reduction and the read give `g_k = S' − f_k` and `g⁻_k = f⁻_k − S' − 1`, with
     `f_k`, `f⁻_k` the sums of the reads over the active lanes; all sums are below
     `N·(S' + 1)`, far from the modulus; the range claims give `f_k ≤ S'` and `f⁻_k > S'`;
  4. if every active lane has `u ≥ 0`, then `c_k ≥ x_kj` on the active lanes, `f_k = f(c_k)`,
     and `f⁻_k` is `f(c_k − 1)`, or exceeds `S'` by construction when some `u = 0`. Since `f`
     is nonincreasing, no smaller admissible shift has `f ≤ S'`: `c_k` is the defined shift
     and the reads are the defined output;
  5. if some active lane has `u < 0`, its clamped address is 0 and it contributes `S'` to
     `f_k`, so `f_k ≤ S'` forces every other active lane's `T_sum` read, hence its `T_out`
     read, to zero. That lane is then the row's strict maximum, the defined shift is its
     logit, and the defined output is `S` on that lane and zero elsewhere, which is what was
     read. The committed `c` is not the defined shift in this case, and `c` is not an output.
  *Mechanism:* the relation docs carry this argument (`docs/pull-requests.md` §4.2);
  `witness_tamper_sweep` over a probe model with the softmax witness fields as sites; forged
  tests for `c + 1`, for `c − 1`, and for a shift below an active lane's logit with a second
  nonzero lane (all rejected), and for a shift below a dominant lane's logit with every other
  lane at zero (accepted, with the honest output).
- **C. Padded lanes carry nothing.** The output is zero on lanes at or beyond `active_len`,
  and those lanes do not enter either row sum; their logits are read only to form an
  addressable `u`. *Mechanism:* the public lane selector in the sum reduction's weights; an
  op-surface row with nonzero values forced into padded input lanes, proving the same output.
- **D. Nothing per row in the clear, no per-row verifier work.** *Mechanism:* the three aux
  `VirtualPoly` variants do not exist; a test on the proof's claim count for two row counts.
- **E. One evaluation per table per proof.** The softmax reads of all nodes end at one address
  point (address-only reads of equal width are tail-aligned in `Lookups`), and the verifier
  generates and evaluates each of `T_out`, `T_sum`, `T_sum⁻` once per proof. *Mechanism:* the
  cache; a test counting evaluations on a model with two softmax nodes of different shapes.
- **F. Accuracy.** Against the float softmax of the same integer logits, every entry is within
  2 ulp and every row sum within `N/2 + 2` of `S`. *Mechanism:* a tracer test over random rows
  at 16, 64, 256 and 1024 lanes and logit spreads from 0.5 to 6, with `f64` as the reference.
- **G. Version binding.** The verifier rejects a model that is not `SemVersion::V2` with
  `Err`. Outputs change, so the importer's golden hashes and the digest pins are re-pinned,
  each in its own commit. *Mechanism:* a verifier test; the golden-hash tests.

### Non-Goals

- Fused attention (`docs/prover-plan.md` item C3): the score einsum emitting `c − score` so
  its committed digits serve as the lookup address.
- Narrowing the 32-bit clamp address, or fusing the clamp into the table read (item B2).
- Any special handling of the causal mask: masked scores arrive as very negative logits and
  read zero.
- A softmax over an axis other than the last, or with a mask that is not a lane prefix.
- A piecewise-linear exp table. `T_out` and `T_sum` are plain rounded tables.
- An output scale other than `S`.
- Any relaxed statement: no tolerance band, no prover-chosen value the proof does not pin
  where it can change an output.
- Keeping `SemVersion::V1` provable.

## Evaluation

### Acceptance Criteria

- [ ] Invariant A: the kernel's `c` equals a linear scan's on random and adversarial rows (all
      lanes equal, one dominant lane, spread beyond `top`, `active_len = 1`).
- [ ] Invariant F holds on the stated grid; the test prints the measured maxima.
- [ ] A row whose `c` or `c − x` leaves its range fails to execute with a `KernelError`.
- [ ] `VirtualPoly::{SoftmaxSumOutput, SoftmaxMaxOutput, SoftmaxMaxIndex}`,
      `SoftmaxShape::aux`, `inv_sum_evals`, the exp digit tables and the two softmax remainder
      claims do not exist; `ProofType::{SoftmaxStage1, SoftmaxStage2}` are retired.
- [ ] The softmax plan is the one of the Architecture section: one in-loop sumcheck, two
      `Defer`s to `Lookups`, two to `OneHot`.
- [ ] A softmax node commits 13 one-hot chunk polynomials per entry, one dense per-row column,
      and none of the digit columns it has today (checked on a probe model's committed list).
- [ ] Invariant B: the four forged tests behave as stated; the sweeps pass over a probe model
      with a softmax, with no tolerance entry.
- [ ] Invariant C: `softmax_masked_rejected` becomes a proving row; a row with a width that is
      not a power of two built with padding proves; the forced-padding row proves.
- [ ] Invariant D: the claim-count test passes. Invariant E: the evaluation-count test passes.
- [ ] Invariant G: a `V1` model is rejected by `verify`.
- [ ] GPT-2 proves and verifies at a sequence length that is not a power of two (24 tokens). If
      another op rejects that length, the PR names the op and this criterion is reported as
      blocked by it.
- [ ] The golden hashes and `testing/digests.rs` are re-pinned, each in its own commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass; the importer's zoo gates pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the three sweeps; the zoo with its
serialization round-trip; `proof_types_all_exercised`; the importer's zoo gates.

Changed tests: the unit tests of `op/softmax.rs` are rewritten for the new kernel;
`SOFTMAX_FIELDS` of the witness tamper sweep follows the new witness (`c`, `u`);
`softmax_masked_rejected`; the golden hashes.

New tests: those named by invariants A to G.

CI coverage: the workspace job; the importer job; the heavy job for GPT-2 and bge-small. The
`zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the
commit this lands on, on the benchmark machine. Softmax is about 1.5% of committed rows on
GPT-2 at 16 tokens and grows with the square of the sequence length, so the PR also reports
one GPT-2 run at 64 tokens before and after.

Expected: commit and openings down by softmax's share times roughly one half (13 cells per
entry for 28); the IOP stage down (two in-loop batches of two relations become one sumcheck,
three reads become one); proof size down by `3F` field elements per node. Verify time down by
the per-row work and by the table cache.

Tracer: the shift is found by bisection, about 33 evaluations of `f` per row.

Bounds (the gate's defaults): prove and verify within 3% of the reference or better on both
models at 16 tokens; peak memory within 5%; proof size no larger.

## Design

### Architecture

`a` is the public lane selector, `a(j) = 1` for `j < active_len`; its multilinear extension
is evaluated in `O(log N)`.

```text
node turn:
    Reduce                                y(r0), r0 = (r0_row, r0_lane)
    Open      g, g⁻ at r0_row (virtual)
    Relation  SoftmaxSums, proof SoftmaxSums: over the node's cells, degree 3
                y(r0) + γ·(S' − g(r0_row)) + γ²·(g⁻(r0_row) + S' + 1)
                  = Σ_{k,j} a(j)·( eq(r0,(k,j))·v(k,j) + eq(r0_row,k)·(γ·t(k,j) + γ²·t⁻(k,j)) )
              produces the read values v, t, t⁻ at one cell point r1
    Open      c at r1_row (committed); ũ at r1 (virtual)
    Tie       address link: the clamp lookup's operand is c(r1_row) − x(r1); appends x(r1)
    Defer     Lookups   table read at r1:
                v + β·t + β²·t⁻ + β³·ũ  =  Σ_a ra(a, r1)·(T_out[a] + β·T_sum[a] + β²·T_sum⁻[a] + β³·a)
    Defer     Lookups   clamp read-raf at r1:  ũ(r1) = Clamp[u],  u(r1) = c(r1_row) − x(r1)
    Defer     OneHot    the table address's one-hot checks
    Defer     OneHot    the clamp address's one-hot checks
post-loop:
    g and g⁻ in the remainder buckets, members over the node's rows
```

`v`, `t`, `t⁻` are the reads of `T_out`, `T_sum`, `T_sum⁻` at the table address; the node's
output is `a·v`.

The range claims: `g ∈ [0, 2^24)` and `g⁻ ∈ [0, 2^w)` with `w` the even width covering
`N·(S' + 1)`. Their openings are appended in the node's turn at `r0_row`, as
`SoftmaxExpSum` is today.

The table read is the existing address-only read with its table generalized to a
`β`-combination; it has `D` rounds, so every softmax node's read is bound in the last `D`
rounds of the `Lookups` batch and all share one address point (invariant E).

`c` is committed as the other dense signed columns are (`TeleportNodeQuotient`).

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
- **One read relation over address and cells with two cell weights,** in place of the sum
  reduction plus an address-only read. Rejected: no existing read sums over cells with a
  custom weight for an explicit table, and its address point would differ between nodes of
  different shapes.
- **An unsigned 32-bit address.** Not needed: step 5 of invariant B covers the negative
  address with the existing signed read.
- **Normalisation fused into the value einsum** (zkGPT). Not needed: the log-partition form
  has no normalisation step.
- **Base-2 range reduction or digit-product exp tables.** Rejected: products of rounded
  entries are what costs today's accuracy, and published measurements find lookup cost
  follows lookup count and address width, not table size.
- **Changing `SemVersion::V1` in place.** Rejected: the contract versions the tables, and the
  verifier should refuse a model compiled under the old semantics.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the softmax clause of the behavioral contract.
- `docs/verifier.md`: the "Plans" section drops softmax as the stress case; the version check.
- `docs/prover-plan.md`: §6 "Not pursuing" loses "prover-chosen softmax shifts" (the shift
  here is pinned); the softmax and C3 lines of item 6 are rewritten.
- `docs/op-roadmap.md`: the exp op's note on the multiplicative split.
- Module docs of the softmax modules in both crates, with the S ⊆ R argument.
- `README.md`: the benchmark table, the accuracy statement, and the sequence-length note.

## Execution

Suggested order: (1) `SemVersion::V2`, the kernel, tables and tracer tests, with the golden
hashes re-pinned; (2) the row-domain range claim; (3) the multi-table read and the table
cache; (4) the sum reduction and the plan, verifier then prover; (5) deletion of the stages,
aux vectors and digit tables; (6) the digest re-pin, the non-power-of-two run and the gate.

If `specs/fused-gelu.md` or `specs/one-clamp.md` lands in the same period, do the golden
re-pins together.

## References

- `atlas-trace/src/op/softmax.rs`, `atlas-trace/src/semantics/{tables,exp}.rs`,
  `atlas-trace/src/id.rs` (`SemVersion`).
- `jolt-atlas-verifier/src/onnx_proof/ops/softmax_last_axis/`, `global_clamp.rs`;
  `atlas-sumcheck/src/shout/mod.rs` (`ReadRafRelation`), `atlas-sumcheck/src/ps_shout/`;
  `atlas-lookup-tables/src/clamp.rs` (`SoftmaxClampTable`);
  `atlas-sumcheck/src/relation/batch.rs` (tail alignment).
- zkLLM (arXiv 2404.16109, §5: the unverified log-sum-exp shift and the row-sum band);
  DeepProve (ePrint 2026/1112); zkGPT (ePrint 2025/1184); NanoZK (arXiv 2603.18046); "Bits
  Under ZK-LLM" (arXiv 2609.36437) and "A Separation Principle for Lookup-Based zkML" (ePrint
  2026/1390) on what lookup cost depends on.
- The accuracy simulation behind the Summary and invariant F: scale 2^14, 40 random rows per
  cell, 16 to 1024 lanes, Gaussian logits with σ from 0.5 to 6.
