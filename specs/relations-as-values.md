# Spec: Relations as values, one table read

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-16                     |
| Status      | implemented                    |
| PR          |                                |
| Depends on  | `specs/one-clamp.md` (the clamp split, the exact flag and `clamp_lookups/` are gone; this spec does not convert them) |
| Supersedes  | `specs/verifier-audit-simplification.md`'s "each op relation is bound to its op by `type Spec`" mechanism |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every sumcheck is a symbolic relation and the verifier is derived from it. That part of the
design is settled. What an auditor still has to read around it is not. A relation is a Rust
type: each of the op relations in `jolt-atlas-verifier` implements `SymbolicSumcheck` (eight
associated types), `Produces`, `Relation`, `AtlasRelation` and a marker (`EngineRelation` or
`TieRelation`), gets its `SetupRelation` from a macro, carries a `PhantomData` spec where it
serves several ops, and is bound to its op by one of three macros. `plan.rs` then holds two closed enums,
`OpRelation` and `OpTie`, with one arm per type, and most arms call the same function. In
`ops/mul.rs` the protocol is six facts (rounds, degree, the input and output expressions, the
produced openings, the incoming point); the rest of the file is the encoding.

The lookups repeat the pattern one level up. "Read table `T` at address `A`, value `V`" is
expressed through an operands trait with one implementor per family, a provider generic over
the operands, a table type parameter, a marker type per small table, and five one-hot
encoding types. `ops/sin.rs` and `ops/cos.rs` differ only in names, as do `tanh.rs`,
`sigmoid.rs` and `erf.rs`; `ops/smooth_relu.rs` adds a sixth operands type for a read that
shares another read's address chunks.

Two ties are not relations at all. The trig period reduction (`TeleportInputRelation`) and
softmax's address link (`SoftmaxTie::AddressLink`) each have a hand-written `verify` that
appends an operand opening and compares claims, beside thirteen ties that go through
`verify_tie`.

One module is also misnamed. `neural_teleport/` was written for a technique from the
literature, rescaling an activation's input to shrink its lookup domain, which the prover no
longer uses: Tanh, Erf and Sigmoid clamp their input and read a small table. What the module
holds today is the generic small-table relation (read by the activations as well as the trig
ops), the period reduction `x = P·q + r` of Sin and Cos, their right-shift downscale, and two
index helpers. The period reduction is part of the `atlas-trace` contract and stays; the
name, and the grouping it implies, do not.

This spec makes a relation a value. One concrete type, `OpRel`, carries the six facts; each
op is a function that builds one. One `TableRead` value and one `RaEncoding` value replace
the lookup layers. Nothing about the protocol changes: every proof is byte-identical.

## Intent

### Goal

Replace the per-relation types of `jolt-atlas-verifier` with one relation value type built by
a constructor function per op, and the lookup operand, provider, marker and encoding types
with one table-read value and one one-hot encoding value, leaving every transcript byte
unchanged.

Introduced:

- `onnx_proof::relation::OpRel<F>`: the one op relation type. Fields: `kind: OpRelKind`,
  `node: ComputationNode`, `rounds`, `degree`, `input` and `output` (`RelationExpr<F>`),
  `produced: Vec<OpeningId>`, `incoming: Incoming`, `draws: Vec<RelationChallenge>`,
  `setup_opens: Vec<SetupOpen>`, `binding: BindingOrder`, and `custom: Custom<F>`.
- `Incoming`: where the incoming point comes from. `ReducedOutput` (the node's reduced output
  point) or `Fresh` (`rounds` challenges drawn from the transcript, as Div and Rsqrt do).
- `draws`: the challenges the setup draws after the incoming point, one scalar each, in order.
- `SetupOpen { id, range }`: an advice opening appended after the draws, at a sub-range of the
  incoming point (Gather's index operand; a small table's clamped input).
- `Custom<F>`: three optional `fn` pointers for what is not the default.
  `points(&OpRel<F>, &RelationSetup<F>, &[F])` (default: the bound point, or the incoming
  point for a tie); `derive(&OpRel<F>, RelationDerived, &RelationSetup<F>, &[F])` (default:
  `Eq` only); `finish_setup(&OpRel<F>, &dyn OpeningAccumulator<F>, &mut RelationSetup<F>)`
  (default: none), which derives further challenge entries from the drawn ones and has no
  transcript.
- `OpRelKind`: a closed enum, one variant per op relation, the prover's dispatch key. It maps
  to the diagnostic `RelationId` and declares the ops it may serve (`OpRelKind::ops`).
- `onnx_proof::lookup::TableRead`: `{ node, table: PrefixSuffixTable, value, operand, cycle,
  ra, log_k }`, the one description of a prefix-suffix read. `PrefixSuffixTable` is a closed
  enum over the `atlas-lookup-tables` tables the verifier reads.
- `onnx_proof::lookup::SmallTableDef`: `{ kind: TableKind, vars, signed, ra, address, value }`,
  the one description of a materialized small table.
- `onnx_proof::lookup::RaEncoding`: `{ committed, ra_source, r_cycle_source, params }`, the
  one implementor of `atlas_sumcheck::shout::RaOneHotEncoding` in the crate.

Modified:

- `atlas-claims`: `SymbolicSumcheck::id` takes `&self`; `new` and `Shape` are removed (no
  generic caller exists).
- `plan.rs`: `OpRelation` keeps one variant per kind of verifier construction (`Op(OpRel)`,
  `TableRead`, `OneHot(RaEncoding)`, the two GatherSmall one-hot stages, softmax's three-table
  read, the bucket value and digit relations). `OpTie` is deleted: `Step::Tie` holds an
  `OpRel`, verified by `verify_tie`. The two hand-written ties become `OpRel` ties: the trig
  period reduction (`input(r) = P·q(r) + rem(r)`, producing the operand opening at the
  reduced output point) and softmax's address link (`u(r1) = c(r1_row) − x(r1)`, producing
  the operand opening at the reads' point, with `Custom::points` giving the shift its row
  coordinates). `OpLookup`, `OpActivation` and `OpTrig` are deleted.
- `jolt-atlas-core`: `RelationWitnesses` and `TieClaims` become functions that match on
  `OpRelKind` with no wildcard; `LookupWitness` becomes a function of `TableRead`.

Renamed and moved (the `neural_teleport/` module is dissolved in both crates):

- The small-table relation and its prover move to `onnx_proof::lookup`.
- The period reduction, the downscale and the Sin and Cos plans move to one trig op module,
  `ops/trig.rs`.
- The index helpers `usize_to_n_bits` and `n_bits_to_usize` move to `utils`.
- `atlas-trace`: `TeleportAdvice` becomes `TrigPeriodAdvice`, `NodeAux::Teleport` becomes
  `NodeAux::TrigPeriod`, `Trace::teleport` becomes `Trace::trig_period`.
- `common`: `CommittedPoly::TeleportNodeQuotient` becomes `TrigPeriodQuotient` and
  `VirtualPoly::TeleportRemainder` becomes `TrigPeriodRemainder`, each at its variant's
  present position.
- `jolt-atlas-verifier`: `TeleportInputRelation` becomes the constructor `trig_period`
  (`OpRelKind::TrigPeriod`), `OpOpen::TeleportQuotient` becomes `OpOpen::TrigPeriodQuotient`.

Deleted: the per-relation types and their trait impls; `AtlasRelation`, `EngineRelation`,
`TieRelation`; `impl_plan_op!`, `bind_relation!`, `atlas_relation_types!`;
`OpTie`, `SoftmaxTie` and the two hand-written `verify` methods; `LookupOperandsTrait`,
`LookupRaPolys` and their implementors (`SmoothReluOperands` among them); `OpLookupProvider`,
`OpLookupEncoding`, `SmallTableRaEncoding`, `CosRaEncoding`, `SinRaEncoding`,
`GatherRaEncoding`, softmax's `ReadRaEncoding`; the `SmallTable` and `SmallActivationTable`
traits and their marker types.

### Invariants

Relied on, not changed:

1. **The `atlas-trace` behavioral contract**, all of it. No semantics, table or padding rule
   is touched, including the period reduction of Sin and Cos. *Mechanism:* the only diff
   under `atlas-trace/` is the three renames listed in the Goal; `atlas-trace/tests/` and
   the importer's golden hashes are not edited by this PR.
2. **Table generation has one owner.** A `SmallTableDef` names a `TableKind` and calls
   `atlas_trace::semantics::tables::table`, as the markers do. *Mechanism:* review; the
   golden table tests in `atlas-trace/tests/`.
3. **Verifier hygiene.** No panic, Trace-free, no prover item. *Mechanism:* the crate-root
   lints, `clippy.toml`, the `verifier-isolation` CI job.

Preserved, with the mechanism restated:

4. **The protocol is unchanged.** Every proof and the final transcript state are
   byte-identical to the parent commit's. *Mechanism:* the four pins in `testing/digests.rs`
   are not edited by this PR; the zoo; the three sweeps.
5. **The relation is the source of truth.** No hand-written verifier is added: an `OpRel` is
   verified by `RelationVerifier` or `verify_tie`, which evaluate its expressions.
   *Mechanism:* the compiler (`OpRel` has one `Relation` impl and no `verify` method, and
   `Step::Tie` holds nothing but an `OpRel`). If a hand-written tie cannot be converted with
   its transcript bytes unchanged, it stays as the one named exception and the PR states why.
6. **Completeness of the specialized provers.** *Mechanism:* the `relation_prover_equiv_*`
   tests, retargeted at `OpRel` values, with the same reference messages.

New:

- **A. Op binding by construction.** The relation built for a node is one its op may use.
  *Mechanism:* constructors are called only from the wildcard-free `Op` match
  (`dispatch_operator!`), so a new op fails to compile until it has a plan; and the test
  `relation_kinds_match_their_dispatched_op_kind`, which replaces
  `relation_specs_match_their_dispatched_op_kind` and checks every dispatched
  `OpRelKind` against `OpRelKind::ops` on the probe models and `op_gauntlet`.
- **B. A setup's transcript interaction is data.** What a relation's setup draws is
  `incoming` and `draws`; what it appends is `setup_opens`. *Mechanism:* the compiler:
  `Custom::finish_setup` has no transcript parameter, and `OpRel`'s one `SetupRelation` impl
  is the only code that touches the transcript for an op relation.
- **C. No captured state.** `Custom`'s members are `fn` pointers, not closures. *Mechanism:*
  the compiler (the field types).
- **D. Prover dispatch is exhaustive.** *Mechanism:* the compiler: the matches on `OpRelKind`
  in `jolt-atlas-core` have no wildcard arm.
- **E. One description per table read.** No type in `jolt-atlas-verifier` other than
  `RaEncoding` implements `RaOneHotEncoding`, and no trait in the crate is implemented once
  per lookup family. *Mechanism:* the acceptance greps below; review afterwards.
- **F. Each constructor documents its S ⊆ R argument**, where the type's doc comment has it
  today. *Mechanism:* review-held, as today.
- **G. The renames move no byte.** A renamed `CommittedPoly` or `VirtualPoly` variant keeps
  its encoding and its place in the sorted commitment order. *Mechanism:* the variants keep
  their positions in the enums; the digests of `op_gauntlet` and `qwen2-tiny`, if a Sin or
  Cos node survives lowering in either, and otherwise the op-surface rows `sin_*` and
  `cos_*` with the zoo's serialization round trip. The implementer records which of the two
  holds in the PR description.

### Non-Goals

- The subprotocol relations in `atlas-sumcheck` (Booleanity, Hamming weight,
  RA virtualization, read-raf, prefix-suffix read-raf, digit range, opening reduction). They
  keep their types and their `into_parts` setups.
- The plan and step structure, the deferral queue and `OpLayout`
  (`specs/verification-schedule.md`).
- The proof format (`specs/stream-proof.md`).
- Any change of transcript order, including making the three open-around-setup orders
  uniform. `setup_opens` and `TableRead` reproduce today's order exactly.
- Removing `Sin` and `Cos`, or changing their period reduction, downscale or tables. Whether
  any target model keeps a runtime trig node after constant folding is a separate question.
- Monomorphizing the verifier over `Fp128`.
- Pruning `RelationDerived` and `RelationChallenge` beyond variants nothing constructs.

## Evaluation

### Acceptance Criteria

- [ ] `testing/digests.rs` is not modified, and its four tests pass.
- [ ] `grep -rn 'SymbolicSumcheck for' jolt-atlas-verifier/src` finds `OpRel` and nothing
      else.
- [ ] `grep -rn 'RaOneHotEncoding for' jolt-atlas-verifier/src` finds `RaEncoding` and
      nothing else.
- [ ] `AtlasRelation`, `EngineRelation`, `TieRelation`, `impl_plan_op!`, `bind_relation!`,
      `atlas_relation_types!`, `LookupOperandsTrait`, `LookupRaPolys`, `SmoothReluOperands`,
      `OpLookupProvider`, `OpLookupEncoding`,
      `OpLookup`, `OpActivation`, `OpTrig`, `SmallTable` and `SmallActivationTable` do not
      exist in either crate.
- [ ] `jolt-atlas-verifier/src` contains no `PhantomData`.
- [ ] `op_lookups/` does not exist in either crate: table reads have one module,
      `onnx_proof::lookup`, and it is the file `lookup.rs` in each crate, not a directory
      holding a lone `mod.rs`.
- [ ] `SmallTableDef::for_node` is called once per relation built: the derived values and the
      one-hot encoding read the definition from the relation, not by dispatching on the
      node's operator again.
- [ ] `ops/sin.rs`, `ops/cos.rs`, `ops/tanh.rs`, `ops/sigmoid.rs` and `ops/erf.rs` do not
      exist; the trig ops share `ops/trig.rs` and the clamped activations one file.
      `ops/smooth_relu.rs` keeps its plan, layout and table definitions and holds no
      operands type.
- [ ] `OpTie` and `SoftmaxTie` do not exist; `grep -rn 'fn verify' jolt-atlas-verifier/src/
      onnx_proof/ops` finds no tie.
- [ ] The directory `onnx_proof/neural_teleport/` does not exist in `jolt-atlas-verifier` or
      `jolt-atlas-core`, and `grep -rni teleport` over the workspace's sources, `CLAUDE.md`,
      `README.md` and `docs/` finds nothing outside `specs/` and the historical note in
      `docs/prover-plan.md`.
- [ ] Invariant G: the two `common` variants are renamed in place; `atlas-trace/tests/` and
      `atlas-onnx-import/tests/golden_zoo.rs` are not modified.
- [ ] `OpRelation::verifiers` has one arm per variant and no arm per table, activation or
      trig function.
- [ ] Invariant A: `relation_kinds_match_their_dispatched_op_kind` passes; the matches on
      `OpRelKind` in `jolt-atlas-core` have no `_` arm.
- [ ] Invariant B: `Custom::finish_setup`'s signature has no transcript parameter.
- [ ] `relation_prover_equiv_engine`, `_dense`, `_mean_of_squares` and `_contraction` pass
      against `OpRel` values.
- [ ] The three sweeps pass with no tolerance entry; `proof_types_all_exercised` passes.
- [ ] `cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --workspace
      --all-targets`, `cargo check -p jolt-atlas-verifier` and `cargo nextest run --workspace`
      pass.
- [ ] The bounds of the Performance section hold.

### Testing Strategy

Existing tests that must keep passing unchanged: the digests; the zoo with its serialization
round trip; `proof_mutation_sweep`, `witness_tamper_sweep`, `opening_forgery_sweep`; the
op-surface table; `layout_ties_hold_on_padded_models`; `fuzz_regressions_replay`.

Retargeted tests (same assertions, new spelling): the `relation_prover_equiv_*` family;
`fused_product_reads_its_output_and_remainder`, `raw_product_reads_a_multi_consumer_reduction`
and `produced_opening_joins_a_shared_operand_reduction` in `testing/relations.rs`; the
`*_rounds_where_eq_vanishes` tests in `testing/rounds.rs`.

Replaced: `relation_specs_match_their_dispatched_op_kind` by
`relation_kinds_match_their_dispatched_op_kind`.

New tests: none beyond the replacement. The digests are the test of this refactor; a test
that restated a constructor's fields would be a second implementation of the rule.

CI coverage: the workspace job. The change is not protocol-level (invariant 4), so the heavy
job is not required; it runs on `main` as usual. No proof format change, so no new
`mutate::Site`.

### Performance

Benchmark: `bench/gate.sh` on `gpt2 --seq-len 16`, against this spec's parent commit, on the
benchmark machine.

Expected: no stage moves. The table enum is matched once per relation, outside every round
loop, and dispatches to the same generic `atlas-sumcheck` functions; the sumcheck inner loops
and `atlas-algebra/src/poly` are not touched. Bounds: the gate's defaults (prove and verify
within 3%, peak memory within 5%); proof size equal to the byte.

## Design

### Architecture

Only `jolt-atlas-verifier`'s `onnx_proof` and its mirror in `jolt-atlas-core` change, plus
the two-line trim of `SymbolicSumcheck` in `atlas-claims`. `atlas-sumcheck`'s relation core
(`Relation`, `SetupRelation`, `RelationVerifier`, `DynRelationVerifier`, the batch drivers)
is used as it is: `OpRel` is one more implementor.

`Mul` before: a `ProductRelation<S>` type, a `ProductShape` trait with four impls, and seven
trait impls. After:

```rust
/// S ⊆ R: … (the argument that is on `ProductRelation` today)
pub fn product<F: JoltField>(node: &ComputationNode, kind: OpRelKind, factors: &[usize]) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    OpRel {
        kind,
        node: node.clone(),
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: factors.len() + 1,
        input: fused_rebase::input_expression(node),
        output: factors.iter().fold(derived(RelationDerived::Eq), |product, &position| {
            product * opening(RelationOpening::Id(ids.nodeio(Target::Input(position))))
        }),
        produced: distinct(factors).map(|position| ids.nodeio(Target::Input(position))).collect(),
        incoming: Incoming::ReducedOutput,
        ..OpRel::engine()
    }
}
```

`OpRel`'s single `SetupRelation::setup` reads the incoming point as `incoming` says, draws
`draws` in order, appends `setup_opens`, then runs `custom.finish_setup` if present. Its
`Relation::opening_points` and `derive` call `custom.points` and `custom.derive` or fall back
to the trait defaults. Model-only preconditions that today sit in a `setup` (the divisor check
of `ScalarConstDiv`, the reduction check of `MeanOfSquares`) move into the constructor, which
returns `Result`.

Relations whose points or derived values are their own (Broadcast, Slice, Concat, MoveAxis,
Gather, MeanOfSquares, the public-value tie, the contraction, the axis sum, softmax's sums,
the small table) keep that code as named free functions beside the constructor and name them
in `Custom`. The rest use the defaults.

`Plan`, `Step` and `OpRelation` gain the field parameter `F`, since an `OpRel` holds
expressions over `F`.

A prefix-suffix read is verified by one function that takes a `TableRead`, appends its
operand opening at the `cycle` opening's point, and matches `table` once to call
`ps_read_raf_verifier` with the table type. The one-hot checks of any read are
`shout::ra_onehot_verifiers(&RaEncoding, ..)`.

On the prover side, `jolt-atlas-core` keeps its specialized `RelationWitness` provers and the
generic `RelationProver`; what selects one is `rel.kind`.

### Alternatives Considered

- **Closed enums for point rules and derived rules** instead of `fn` pointers, so a plan is
  fully printable data. Rejected for this spec: about half the op relations have their own
  point or derived computation (the einsum layouts, the concat blocks, softmax's closed
  forms), so the enums would hold one variant per relation and their evaluators would be the
  same code moved away from the op it belongs to. Locality was judged the better audit
  property. `OpRelKind` already makes a plan printable by name.
- **Boxed closures.** Rejected: a closure can capture state, which is what `plan.rs`
  forbids, and it cannot be compared or named.
- **Keep the types, generate the impls with a derive or a larger macro.** Rejected: the
  reader still has to expand the macro to see what a relation is, and the enum arms stay.
- **One enum with a `match` per trait method.** Rejected: it scatters an op's protocol over
  seven matches.
- **Change `atlas-sumcheck`'s `Relation` trait into a struct too.** Rejected as scope: the
  subprotocol relations have provers generic over table and polynomial types in the round
  loops, and converting them risks the inner loops for no audit gain in the verifier crate.
- **Normalize the open-around-setup order now.** Rejected: it would change the transcript
  and give up the byte-identity check on a refactor of this size.

## Documentation

- `docs/verifier.md`: reading order step 3 (the relation core), step 4 (the lookup machinery
  no longer lists `neural_teleport/`; "the trig teleport" becomes "the trig period
  reduction") and the "Plans" paragraph on what a step holds; the extension-trait list in
  "Trust perimeter".
- `atlas-trace/src/lib.rs` and `trace.rs`: the doc comments that name the advice.
- `docs/pull-requests.md` §4: the checklist for a new relation becomes "write a constructor,
  add an `OpRelKind`, declare its ops, state S ⊆ R".
- `CLAUDE.md`: the last bullet of "Prover/Verifier Invariants" (the `type Spec` binding and
  the test name) and the `jolt-atlas-core` crate-map sentence on extension traits.
- `docs/op-roadmap.md`: "what adding an op requires".
- Module docs of `onnx_proof/relation.rs`, `plan.rs`, `ops/mod.rs` and the new
  `onnx_proof/lookup` module, in both crates.

## Execution

Each commit leaves the digests green.

1. `atlas-claims`: `id(&self)`, drop `new` and `Shape`.
2. `OpRel`, `Incoming`, `Custom`, `OpRelKind` and the single trait impls, beside the existing
   types; `Plan<F>`.
3. Convert the default-shape relations (product, select, the copies and accumulations, the
   two divisions, the constant division, Rsqrt), then the custom ones, one op family per
   commit, moving the prover's extension-trait impl into the `OpRelKind` match in the same
   commit.
3a. The two hand-written ties as `OpRel` ties, each in its own commit, then `OpTie` deleted.
4. `SmallTableDef` (the three smooth-ReLU tables included) and the merge of the trig and
   clamped-activation files; the small-table
   relation moves to `onnx_proof::lookup`, the period reduction and downscale to
   `ops/trig.rs`, and `neural_teleport/` is deleted in both crates.
4a. The trig-period renames in `atlas-trace`, `common` and the two proof crates, in one
   commit that changes names only.
5. `TableRead`, `PrefixSuffixTable`, `RaEncoding`; delete the operand, provider and encoding
   types. The smooth ReLU's `ReLU` read is a `TableRead` whose `ra` names the clamp read's
   committed chunks. `op_lookups/` is then empty and is deleted, and `lookup/mod.rs` becomes
   `lookup.rs` in both crates.
6. Delete the traits, macros and enums of the acceptance list; retarget the tests; docs.

## References

- `jolt-atlas-verifier/src/onnx_proof/{relation.rs, plan.rs, ops/mod.rs, ops/mul.rs,
  op_lookups/mod.rs, neural_teleport/small_table.rs, ops/activation_clamped/mod.rs}`.
- `atlas-sumcheck/src/relation/{mod.rs, batch.rs}`; `atlas-sumcheck/src/shout/mod.rs`
  (`RaOneHotEncoding`, `ra_onehot_verifiers`); `atlas-sumcheck/src/ps_shout/mod.rs`
  (`ps_read_raf_verifier`).
- `atlas-claims/src/symbolic.rs`.
- `jolt-atlas-core/src/onnx_proof/{relation.rs, ops/mod.rs}`,
  `testing/{relations.rs, rounds.rs, digests.rs}`.
- `specs/verifier-audit-simplification.md` (plans; every op on a plan),
  `specs/atlas-claims.md` (the owned relation layer), `specs/one-clamp.md`,
  `specs/verification-schedule.md`, `specs/stream-proof.md`.
