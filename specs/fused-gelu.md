# Spec: GELU and SiLU as one op

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | `specs/post-loop-batches.md`   |
| Supersedes  | `docs/op-roadmap.md` tier 1; `docs/prover-plan.md` §3 item 6, "GELU as one table op (C4)" |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

The smooth activations reach the prover as chains. GPT-2 exports the tanh form of GELU,
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, which lowers to four Muls, a Cube, two Adds and
a Tanh. bge-small exports the erf form, `0.5·x·(1 + erf(x/√2))`, which lowers to three Muls,
an Add and an Erf. Qwen's SiLU, `x·σ(x)`, lowers to a Sigmoid and a Mul. Every arithmetic node
of those chains is clamped, most carry a rescale remainder, each rounds, and GPT-2's Cube is
the one node in the zoo and the real models that saturates, which is what keeps
`specs/one-clamp.md` from landing. By count, the GELU chains are about 5.5 million of GPT-2's
12.8 million clamped elements at 16 tokens: about 167 committed cells per activation element.

Each of these is a function of one input, so it can be one lookup. The obstacle named in the
roadmap is the linear tail: above the table's domain the output is `x`, not a constant. All
three functions satisfy `f(x) − f(−x) = x`, so `f(x) − ReLU(x)` is even, and it vanishes
outside a bounded range. Hence

```text
f(x) = ReLU(x) + H(clamp(x))
```

with `H` a small table that is zero at both ends of its domain. `ReLU` and the activation
clamp are tables the prover already reads through a 32-bit address, and both can be read
through one committed address. The op costs what a Tanh node costs today, 13 cells per
element, and the chain around it disappears.

## Intent

### Goal

Add `Op::SmoothRelu { form }`, with `form` one of `GeluErf`, `GeluTanh`, `Silu`, lowered by the
importer from the three exported subgraphs and proven as two reads of one 32-bit address plus
one small-table read.

Semantics, in integers, with `S = 2^scale`, `B = 2^b` the form's clamp bound and
`clamp(x) = min(max(x, −B), B − 1)`:

```text
G_form[c] = round(S · f_form(c / S))               c ∈ [−B, B)
H_form[c] = G_form[c] − max(c, 0)
y         = max(x, 0) + H_form[clamp(x)]
```

| Form | `b` at scale 14 | `H` nonzero for `|c|/S` below | Minimum of `H` |
|---|---|---|---|
| `GeluErf` | 17 (`ACTIVATION_BOUND`) | 4.35 | −2785 |
| `GeluTanh` | 17 | 4.17 | −2786 |
| `Silu` | 18 | 12.96 | −4562 |

(checked numerically in `f64`; `H` is zero at `−B` and at `B − 1` for each form). The output
needs no clamp: `|y| ≤ |x|`.

Proof, per element:

| Read | Table | Address |
|---|---|---|
| `relu(x)` | `ReluTable<32>` | the signed 32-bit address `x`, committed once as 8 one-hot chunks |
| `clamp(x)` | the clamp table at bound `b` | the same address |
| `H[clamp(x)]` | the form's small table | the `(b + 1)`-bit address `clamp(x)`, 5 one-hot chunks |

and the zero-round identity `y = relu(x) + H[clamp(x)]` at the reduced output point. 13 one-hot
cells per element and no digits, for every form.

Introduced or modified:

- `atlas-trace`: `Op::SmoothRelu { form: SmoothReluForm }` in `op/activation.rs`;
  `semantics::tables::TableKind::{GeluErf, GeluTanh, Silu}` holding `H`;
  `semantics/coeffs.rs` gains `1/√2` and `√(2/π)` as Q62 literals with their libm
  cross-checks; `TableCache` grows by three slots.
- `atlas-onnx-import`: a `smooth_relu_fusion` pass in `passes.rs`, before
  `pow_to_square_cube`, matching the three subgraphs; `lower.rs`; golden hashes re-pinned for
  every zoo model.
- `atlas-lookup-tables`: a clamp table alias at bound 18 for SiLU.
- `jolt-atlas-verifier`: `ops/smooth_relu.rs` (the plan, the layout, the tie);
  `neural_teleport/small_table.rs` (the read's value claim becomes a trait item, defaulting to
  the reduced node output); `op_lookups` (a ReLU read whose chunks and `ra` are the clamp's
  and whose value is a new advice opening); `plan.rs` (`OpLookup`, `OpActivation` and `OpTie`
  variants); `node_view::node_committed_poly_num_vars`; `common` (the advice openings).
- `jolt-atlas-core`: the prover mirror; `op_kind.rs`, `op_surface.rs`, `relations.rs`, the
  sweep's elementwise list, `model_builder.rs`, and a probe model.

### Invariants

Relied on, not changed:

1. **The clamped-activation protocol.** The clamp read-raf, the small-table read and their
   one-hot checks, as Tanh, Erf and Sigmoid use them. *Mechanism:* their op-surface rows and
   the sweeps over `tables_and_layout`.
2. **Table ownership.** `H` is generated in `atlas_trace::semantics` by integer arithmetic and
   the prover materializes it by calling the same function. *Mechanism:* the contract; the
   `no_float_outside_tests` guard; the table tests.
3. **Existing tables.** No entry of an existing table changes, so `SemVersion` does not.
   *Mechanism:* `atlas-trace/tests/golden_tables.rs` for the existing kinds.
4. **Post-loop batches.** Reads go to `Lookups`, one-hot checks to `OneHot`.
5. **Verifier hygiene.** *Mechanism:* the crate lints, the CI path checks, the mutation sweep
   with no tolerance entry, the drain check.

New:

- **A. The identity.** For each form and every `c` in its domain, `H_form` is zero at
  `c = −B` and at `c = B − 1`, and `G_form` is within 1 ulp of the real function.
  *Mechanism:* a tracer test over each whole table against an `f64` reference, asserting the
  end values and the maximum error; golden hashes for the three tables.
- **B. S ⊆ R.** The one-hot checks make the 32-bit address an integer `x`; the two reads give
  `relu(x)` and `clamp(x)` for that same `x`, because they read one committed address; the
  small-table read gives `H[clamp(x)]`; the tie gives `y = relu(x) + H[clamp(x)]` as
  polynomials, hence on every element. The address is tied to the operand by the read's
  operand claim, as for ReLU. *Mechanism:* the relation docs; `witness_tamper_sweep` with a
  node of each form in a probe model.
- **C. The importer fuses only these functions.** The pass rewrites a subgraph only when its
  structure and its constants match one of the three forms (constants to `f32` precision: the
  exporters write `0.044715`, `√(2/π)` and `√2` as `f32`), and no interior tensor has another
  consumer (`IGraph::use_counts`). Anything else lowers as today. *Mechanism:* importer tests
  with a perturbed constant and with a tapped interior tensor, both left unfused.
- **D. No chain survives in the real models.** After the pass GPT-2 has no Cube and no Tanh,
  bge-small no Erf, and Qwen no Sigmoid feeding a Mul. *Mechanism:* the lowered-op inventories
  in the PR.
- **E. No saturation in GPT-2.** The census line `saturated elements: N of M` reads zero for
  GPT-2 on its benchmark inputs (it reads 2 today, both on a GELU's Cube). *Mechanism:* the
  census; this discharges invariant G of `specs/one-clamp.md`.
- **F. Accuracy does not regress.** Against the float function of the same integer input, the
  op's error is at most 1 ulp. *Mechanism:* the test of invariant A.
- **G. Shared address, one claim.** The two 32-bit reads have equal round counts and sit in
  one batch, so they produce the same `ra` opening at the same point; the prover's two appends
  carry the same claim. *Mechanism:* the accumulator rejects a re-append at a different point;
  a test that the two reads' produced points are equal.

### Non-Goals

- The piecewise-linear tables of `atlas-lookup-tables/src/piecewise_linear.rs`, and the
  vector-valued prefix state they need. `H` is a plain rounded table.
- Fusing the clamp into the table address (`docs/prover-plan.md` item B2), or reading the
  table through an upstream op's committed digits (C2).
- A read-raf that reads two tables in one instance.
- A half-size table indexed by `|x|`.
- Other activations (Mish, HardSwish, Swish with a slope other than one).
- Removing the Tanh, Erf and Sigmoid ops, which remain for models that use them directly.
- `specs/one-clamp.md` itself.

## Evaluation

### Acceptance Criteria

- [ ] Invariant A: the table test passes for the three forms and prints the maximum error;
      the three tables have golden hashes at scales 8, 14 and 15.
- [ ] `Op::SmoothRelu` has rows per form in `op_surface.rs`: random inputs, inputs at and
      beyond both ends of the domain, `i32::MAX` and `i32::MIN`, and a non-power-of-two size.
- [ ] A probe model contains a node of each form; the three sweeps pass with no tolerance
      entry.
- [ ] Invariant C: the two negative importer tests pass; a positive test per form asserts one
      `SmoothRelu` node replaces the subgraph.
- [ ] Invariant D: the lowered inventories of GPT-2, bge-small and qwen2-tiny.
- [ ] Invariant E: the census's saturation count for GPT-2 is zero.
- [ ] A node commits exactly the 13 one-hot chunk polynomials of the proof table.
- [ ] Invariant G: the equal-points test passes.
- [ ] `relation_specs_match_their_dispatched_op_kind` and `proof_types_all_exercised` pass.
- [ ] The golden hashes and `testing/digests.rs` are re-pinned, each in its own commit.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets` and `cargo nextest run --workspace` pass; the importer's zoo gates pass.
- [ ] GPT-2 and bge-small prove and verify under the `heavy` profile; Qwen in the CI heavy job.
- [ ] The Performance section's measurements are in the PR.

### Testing Strategy

Existing tests that must keep passing: the workspace run; the Tanh, Erf, Sigmoid, ReLU and
Clamp rows; the sweeps; the zoo with its serialization round-trip; the importer's zoo gates;
the existing table goldens.

New tests: those of invariants A, C and G; the op-surface rows; the probe model.

Changed: the golden hashes of every zoo model; `scripts/gen_tiny_models.py` is not changed, so
the tiny models' weights stay.

CI coverage: the workspace job; the importer job; the heavy job for GPT-2, bge-small and Qwen.
The `zk` feature is not touched.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16` and on `bge-small-en-v1.5`, against the
commit this lands on, on the benchmark machine; one `prove` run of `qwen2-tiny` before and
after.

Estimate, derived from the lowered chains and the per-op costs, not measured:

| Model | Chain per element | Cells today | Cells after |
|---|---|---|---|
| GPT-2 | 4 Mul, 1 Cube, 2 Add, 1 Tanh | about 167 (16 clamp digits per clamped node, 7 remainder digits per Mul, 14 for the Cube, 13 for the Tanh) | 13 |
| bge-small | 3 Mul, 1 Add, 1 Erf | about 98 | 13 |
| Qwen | 1 Mul, 1 Sigmoid | about 36 | 13 |

On GPT-2 at 16 tokens the chains hold 7 of the clamped nodes per MLP element, about 5.5 million
of 12.8 million clamped elements, so the clamp and remainder buckets should shrink by more
than a third. `--census` attributes bucket rows by family, not by op, so the gate is the
measurement.

Expected: commit, IOP and openings all down on GPT-2 and bge-small; verify time and proof size
down with the node count (about 190 fewer nodes in GPT-2).

Bounds: prove, verify and peak memory no worse than the reference on either model; proof size
no larger. The PR states the measured change per stage beside the estimate.

## Design

### Architecture

```text
node turn:
    Reduce                               y(r0)
    Open     relu(x) and clamp(x) at r0 (virtual advice)
    Tie      y(r0) = relu(x)(r0) + h(r0): defines the small-table read's value claim h(r0)
    Defer    Lookups   small-table read of H at address clamp(x), value claim h(r0)
    Defer    Lookups   read-raf of ReluTable  over the 32-bit address, operand x(r0)
    Defer    Lookups   read-raf of the clamp table over the same address, operand x(r0)
    Defer    OneHot    the 32-bit address's one-hot checks, once
    Defer    OneHot    the table address's one-hot checks
```

The small-table read is today's `SmallTableRelation` with its value claim taken from a trait
item: the reduced node output for Tanh, Erf and Sigmoid, and `y(r0) − relu(x)(r0)` here. Its
table width is the form's (`b + 1`: 18 for the GELUs, 19 for SiLU), which is 5 chunks either
way.

The two 32-bit reads are two instances of the prefix-suffix read-raf. The ReLU instance uses a
lookup helper whose committed chunks, `ra` and operand opening are the clamp helper's, and
whose value is the `relu(x)` advice opening. Both have `32 + log T` rounds (invariant G).

The operand claim `x(r0)` is the one opening both reads name; it is appended in the node's
turn.

Kernels: `G` for the erf form uses `erf_q62`, saturating outside its valid range as
`erf_entry` does; the tanh form builds tanh from `exp_neg_q62` as `tanh_entry` does, with the
argument `√(2/π)·(x + 0.044715·x³)` formed in Q62 (`0.044715` as the rational `8943/200000`);
SiLU uses `exp_neg_q62` directly. Rounding is to nearest, ties upward, with a signed helper.

The importer pass matches on the normalized graph, where constants are edge payloads still in
float, `Pow(x, 3.0)` has not yet become `atlas.Cube` and bge's `Div` by `√2` has not yet become
a `Mul`. The three patterns, as exported:

```text
GeluTanh   Mul(x, 0.5) · Add(Tanh(Mul(Add(x, Mul(Pow(x, 3), 0.044715)), √(2/π))), 1)
GeluErf    Mul(Mul(x, Add(Erf(Div(x, √2)), 1)), 0.5)
Silu       Mul(x, Sigmoid(x))
```

It accepts either operand order of each commutative node, and for the erf form a `Mul` by
`1/√2` in place of the `Div`. The fused node is an `atlas.SmoothRelu` node with a `form`
attribute, lowered by `lower.rs` like `atlas.Rsqrt`.

### Alternatives Considered

- **A table with a linear tail in its decomposition** (the roadmap's sketch). Rejected: it
  needs a new prefix-suffix table; the identity with ReLU needs none.
- **The piecewise-linear table.** Rejected for now: it needs the prefix-suffix sumcheck to
  carry a folded knot vector, and its unrounded output needs a rounding check per element.
- **Bound the Cube's input with a clamp in the importer.** Removes the saturation but adds a
  lookup per element and keeps the chain.
- **Keep the chain and fuse only the affine parts.** Rejected: the Cube and its saturation
  stay.
- **GELU only, SiLU later.** Rejected at analysis: the proof is identical, and SiLU costs one
  table, one clamp bound and one pattern.
- **Three ops.** Rejected: one op with a form keeps one plan, one layout and one row group.

## Documentation

- `atlas-trace/src/lib.rs` and `CLAUDE.md`: the op where the nonlinear ops are listed; no
  contract change.
- `docs/op-roadmap.md`: tier 1 marked done with the identity; the "linear tail" item and the
  "whether tier 1 pays for itself" open decision removed; the note on the missing
  `scripts/gen_constants.py` that `coeffs.rs` cites.
- `docs/prover-plan.md`: item 6's C4 line; the saturation note of item 5.
- `docs/verifier.md`: the clamped-activation paragraph names the op and the shared address.
- `specs/one-clamp.md` is not edited (it is a proposal and names this as a prerequisite).
- `README.md`: the supported-op list and the benchmark table.

## Execution

Suggested order: (1) the constants, kernels, tables and `Op::SmoothRelu` in the tracer with
their tests; (2) the proof, verifier then prover, with op-surface rows and the probe model;
(3) the importer pass and the golden re-pin; (4) the digest re-pin, the inventories, the
census and the gate.

The op and its tables are additions, so this spec needs no `SemVersion` change of its own; the
golden hashes move because the importer lowers these activations differently. If
`specs/log-partition-softmax.md` lands in the same period, do the two golden re-pins together.

## References

- `atlas-trace/src/op/activation.rs`, `atlas-trace/src/semantics/{tables,exp,coeffs}.rs`
  (`erf_q62`, `tanh_entry`, `exp_neg_q62`), `atlas-trace/src/op/mod.rs` (`TableCache`).
- `jolt-atlas-verifier/src/onnx_proof/ops/{activation_clamped,relu,tanh}.rs`,
  `neural_teleport/small_table.rs`, `op_lookups/mod.rs`; `atlas-lookup-tables/src/{relu,clamp}.rs`;
  `atlas-sumcheck/src/opening_proof/mod.rs` (re-append rule).
- `atlas-onnx-import/src/{passes,graph,lower,quantize}.rs`.
- `docs/op-roadmap.md` tier 1; `docs/prover-plan.md` §3 items 5 and 6; `specs/one-clamp.md`
  invariant G.
- The exported subgraphs, read from `models/gpt2`, `models/bge-small-en-v1.5` and
  `models/qwen2-tiny`: 12 identical tanh-form chains, 12 identical erf-form chains, and SiLU
  as `Sigmoid` then `Mul`.
