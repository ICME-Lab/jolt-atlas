# Spec: GELU as one op

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/post-loop-batches.md`   |
| Supersedes  | `docs/op-roadmap.md` tier 1 (GELU); `docs/prover-plan.md` §3 item 6, "GELU as one table op (C4)" |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

GELU reaches the prover as a chain. GPT-2 exports the tanh form,
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, which lowers to a Cube, four Muls, two Adds and
a Tanh; bge-small exports the erf form, `0.5·x·(1 + erf(x/√2))`, which lowers to an Erf with
Muls and an Add around it. Every one of those ops but the table read is a clamped or
fused-rescale node with its own committed digits, each rounds, and the Cube is the one node in
the zoo and the real models that saturates (two elements of GPT-2), which is what keeps
`specs/one-clamp.md` from landing.

GELU is a function of one input, so it can be one lookup. The obstacle named in the roadmap is
its linear tail: above the table's domain the output is `x`, not a constant. This spec removes
the obstacle with an identity both forms satisfy: `GELU(x) − ReLU(x)` is an even function that
vanishes outside `|x| < 4.4`. So

```text
GELU(x) = ReLU(x) + H(clamp(x))
```

where `H` is a small table that is zero at both ends of its domain. `ReLU` and the activation
clamp are tables the prover already reads through a 32-bit address, and both can be read
through the same committed address. The op costs what a Tanh costs today, and the chain around
it disappears.

## Intent

### Goal

Add `Op::Gelu { form }`, with `form` the erf or the tanh form, lowered by the importer from
the two exported subgraphs and proven as two reads of one 32-bit address plus one small-table
read.

Semantics, in integers, with `S = 2^scale`, `B = 2^ACTIVATION_BOUND` and the activation
clamp `clamp(x) = min(max(x, −B), B − 1)`:

```text
G_form[c] = round(S · gelu_form(c / S))            c ∈ [−B, B)
H_form[c] = G_form[c] − max(c, 0)
y         = max(x, 0) + H_form[clamp(x)]
```

`H_form` is zero at both ends of the domain (checked numerically at scale 14: it is nonzero
only for `|c|/S < 4.35` in the erf form and `< 4.17` in the tanh form, with minimum −2786), so
`y = x` above the domain and `y = 0` below it. The output needs no clamp: `|y| ≤ |x|`.

Proof, per element:

| Read | Table | Address |
|---|---|---|
| `relu(x)` | `ReluTable<32>` | the 32-bit address `x`, committed once as 8 one-hot chunks |
| `clamp(x)` | `ActivationClampTable<32>` | the same address |
| `H[clamp(x)]` | the new small table | the 18-bit address `clamp(x)`, 5 one-hot chunks |

and the zero-round identity `y = relu(x) + H[clamp(x)]` at the reduced output point. 13 one-hot
cells per element, as a Tanh or Erf node today, and no digits.

Introduced or modified:

- `atlas-trace`: `Op::Gelu { form: GeluForm }` in `op/activation.rs`;
  `semantics::tables::TableKind::{GeluErf, GeluTanh}` holding `H`, generated from `erf_q62`
  and the tanh kernel; the behavioral contract's nonlinear-op clause is unchanged.
- `atlas-onnx-import`: a `gelu_fusion` pass in `passes.rs`, before `pow_to_square_cube`,
  matching both subgraphs; `lower.rs`; golden hashes re-pinned for GPT-2, bge-small and the
  tiny stand-ins that contain a GELU.
- `jolt-atlas-verifier`: `ops/gelu.rs` (the plan, the layout, the tie);
  `ops/activation_clamped/` (the small-table read takes its input claim from the caller, so
  it can be `y − relu(x)`); `plan.rs` (an `OpLookup` variant for the ReLU read of a GELU's
  operand, an `OpActivation` variant per form, one `OpTie`).
- `jolt-atlas-core`: the prover mirror; `op_surface.rs`, `op_kind.rs` and a probe model.

### Invariants

Relied on, not changed:

1. **The clamped-activation protocol.** The activation clamp read-raf, the small-table read
   and their one-hot checks, as Tanh, Erf and Sigmoid use them. *Mechanism:* their op-surface
   rows and the sweeps over `tables_and_layout`.
2. **Table ownership.** `H` is generated in `atlas_trace::semantics` by integer arithmetic and
   the prover materializes it by calling the same function. *Mechanism:* the contract; the
   table's unit test.
3. **Post-loop batches.** Reads go to `Lookups`, one-hot checks to `OneHot`.
4. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. The identity.** For both forms and every `c` in the domain, `G_form[c] − max(c, 0)` is
  zero at `c = −B` and at `c = B − 1`, and `G_form` is within 1 ulp of the real function.
  *Mechanism:* a tracer test over the whole table against an `f64` reference, asserting the
  two end values and the maximum error.
- **B. S ⊆ R.** The one-hot checks make the 32-bit address an integer `x`, the two reads give
  `relu(x)` and `clamp(x)` for that same `x` (one committed address), the small-table read
  gives `H[clamp(x)]`, and the tie gives `y = relu(x) + H[clamp(x)]` as polynomials, hence on
  every element. The address is tied to the operand by the read's address claim, as for ReLU.
  *Mechanism:* the relation docs; `witness_tamper_sweep` with a Gelu node in a probe model.
- **C. The importer fuses only GELU.** The pass rewrites a subgraph only when its structure
  and its constants (`0.5`, `1`, `√2`, `√(2/π)`, `0.044715`, the exponent 3) match one of the
  two forms to float precision, and no interior value has another consumer. Anything else
  lowers as today. *Mechanism:* importer tests with a perturbed constant and with a tapped
  interior node, both left unfused.
- **D. No GELU chain survives in the real models.** After the pass GPT-2 has no Cube and no
  Tanh, and bge-small no Erf. *Mechanism:* the lowered-op inventories in the PR.
- **E. No saturation in GPT-2.** The census's saturation count for GPT-2 on its benchmark
  inputs is zero. *Mechanism:* the census; this discharges invariant G of
  `specs/one-clamp.md`.
- **F. Accuracy does not regress.** Against the float GELU of the same integer input, the
  fused op's error is at most 1 ulp; the chain's is larger. *Mechanism:* the test of invariant
  A; the PR reports the chain's measured maximum for comparison.

### Non-Goals

- SiLU. `SiLU(x) − ReLU(x)` is even and vanishing too, so the same op shape covers it, but it
  decays more slowly (about `|x| < 14` at scale 14) and needs a wider table. Separate change.
- The piecewise-linear tables of `atlas-lookup-tables/src/piecewise_linear.rs`, and the
  vector-valued prefix state they need. `H` is a plain rounded table of `2^18` entries.
- Fusing the activation clamp into the table address (`docs/prover-plan.md` item B2), or
  reading the table through an upstream op's committed digits (C2).
- A half-size table indexed by `|x|`.
- Replacing the Tanh, Erf and Sigmoid ops, which remain for models that use them directly.
- `specs/one-clamp.md` itself.

## Evaluation

### Acceptance Criteria

- [ ] Step 1 of Execution: a census of the cells and the prover time the GELU chains cost
      today on GPT-2 and bge-small is in the PR, with the expected saving.
- [ ] Invariant A: the table test passes for both forms and prints the maximum error.
- [ ] `Op::Gelu` has a row per form in `op_surface.rs`, including inputs at and beyond both
      ends of the domain, `i32::MAX` and `i32::MIN`, and a non-power-of-two size.
- [ ] A probe model contains a Gelu node; the three sweeps pass with no tolerance entry.
- [ ] Invariant C: the two negative importer tests pass; a positive test per form asserts one
      `Gelu` node replaces the subgraph.
- [ ] Invariant D: the lowered inventories of GPT-2 and bge-small.
- [ ] Invariant E: the census's saturation count for GPT-2 is zero.
- [ ] A Gelu node commits exactly the 13 one-hot chunk polynomials of the table above.
- [ ] `relation_specs_match_their_dispatched_op_kind` and `proof_types_all_exercised` pass.
- [ ] The golden hashes and `testing/digests.rs` are re-pinned, each in its own commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass; the importer's zoo gates pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the Tanh, Erf, Sigmoid, ReLU and
Clamp rows; the sweeps; the zoo with its serialization round-trip; the importer's zoo gates.

New tests: those of invariants A and C; the op-surface rows; the probe model.

Changed: the golden hashes of every zoo model with a GELU; `scripts/gen_tiny_models.py` is not
changed, so the tiny models' weights stay.

CI coverage: the workspace job; the importer job; the heavy job for GPT-2 and bge-small. The
`zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the
commit this lands on, on the benchmark machine.

Expected: prove time down on both. Per GELU element GPT-2 loses one Cube, four Muls and two
Adds, each with clamp digits and most with a remainder, and keeps 13 one-hot cells; bge-small
loses the Muls and the Add around its Erf. `docs/prover-plan.md` puts the Tanh lookups
themselves at about 3% of rows; the chain's share is what step 1 measures, and the expected
numbers are written into the PR before the implementation is measured. Verify time and proof
size down with the node count.

Bounds: prove, verify and peak memory no worse than the reference on either model; proof size
no larger.

## Design

### Architecture

```text
node turn:
    Reduce                               y(r0)
    Open     relu(x) and clamp(x) at r0 (virtual)
    Tie      y(r0) = relu(x)(r0) + h(r0), which defines the small-table read's input claim h(r0)
    Defer    Lookups   small-table read of H at address clamp(x), input claim h(r0)
    Defer    Lookups   read-raf of ReluTable   at the 32-bit address, operand x(r0)
    Defer    Lookups   read-raf of ActivationClampTable at the same address, operand x(r0)
    Defer    OneHot    the 32-bit address's one-hot checks, once
    Defer    OneHot    the table address's one-hot checks
```

The two 32-bit reads are two instances of the existing prefix-suffix read-raf over the same
committed chunks. They have the same rounds and sit in the same batch, so they end at the same
point and produce the same `ra` claim; the one-hot checks run once. A read-raf that reads two
tables in one instance would halve that prover work and is not required.

The operand claim `x(r0)` is appended once, in the node's turn.

`H` has signed entries; the small-table read already handles signed tables (Tanh, Erf).

The importer pass matches on the normalized graph, where the exporter's `Pow(x, 3)`, `Div` by
a constant and `Mul` by its reciprocal are still distinguishable, and accepts both operand
orders of each commutative node.

### Alternatives Considered

- **A table with a linear tail in its decomposition** (the roadmap's sketch). Rejected: it
  needs a new prefix-suffix table; the identity with ReLU needs none.
- **The piecewise-linear table.** Rejected for now: it needs the prefix-suffix sumcheck to
  carry a folded knot vector, and its unrounded output needs a rounding check per element.
- **Bound the Cube's input with a clamp in the importer.** Removes the saturation but adds a
  lookup per element and keeps the chain.
- **Keep the chain and fuse only the affine parts.** Rejected: the Cube and its saturation
  stay.
- **One op covering GELU and SiLU now.** Deferred: SiLU needs a wider table and Qwen's census.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: no contract change; the op list where it appears.
- `docs/op-roadmap.md`: tier 1 marks GELU done and restates SiLU with the identity; the "linear
  tail" open item is removed.
- `docs/prover-plan.md`: item 6's C4 line; the saturation note of item 5.
- `docs/verifier.md`: the clamped-activation paragraph names Gelu.
- `specs/one-clamp.md` is not edited (it is a proposal and names this as a prerequisite).
- `README.md`: the supported-op list and the benchmark table.

## Execution

Suggested order: (1) the census of what the chains cost; (2) the tables and `Op::Gelu` in the
tracer with their tests; (3) the proof, verifier then prover, with op-surface rows and the
probe model; (4) the importer pass and the golden re-pin; (5) the digest re-pin and the gate.

The op and its tables are additions, so this spec needs no `SemVersion` change of its own; the
golden hashes move because the importer lowers GELU differently. If
`specs/log-partition-softmax.md` lands in the same period, do the two golden re-pins together.

## References

- `atlas-trace/src/op/activation.rs`, `atlas-trace/src/semantics/tables.rs` (`erf_q62`,
  `tanh_entry`).
- `jolt-atlas-verifier/src/onnx_proof/ops/{activation_clamped,relu}.rs`,
  `atlas-lookup-tables/src/{relu,clamp}.rs`.
- `atlas-onnx-import/src/passes.rs` (`norm_fusion`, `pow_to_square_cube` as pattern passes).
- `docs/op-roadmap.md` tier 1; `docs/prover-plan.md` §3 items 5 and 6; `specs/one-clamp.md`
  invariant G.
- `bench/runs/lowered-gpt2.txt`, `bench/runs/lowered-bge-small-en-v1.5.txt` (12 Cube and 12
  Tanh nodes in GPT-2; 12 Erf nodes in bge-small).
