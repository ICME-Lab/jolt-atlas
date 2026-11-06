# Spec: Single Gather

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @AntoineF4C5                   |
| Created     | 2026-11-06                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | #406                           |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Gather has two provers, chosen by dictionary size at import: `GatherSmall` up to 2^16 rows (after
power-of-two rounding), and `GatherLarge` above that. `GatherSmall`'s execution sumcheck ends on a
virtual `NodeOutputRa` claim that no later check uses: no `RaSumcheck` links it to the committed
`GatherRa`, and Booleanity and Hamming weight open that polynomial at their own points. A malicious
prover can therefore choose any Gather output and solve for the `ra` claim that makes the sumcheck
pass. A full `proof.verify` on `main` accepted this forgery on a 16-row dictionary, with output
`[7777, 7777]` for a row holding `[1, 2]`, and rejected the same forgery on `GatherLarge`. The small
path also panics on a one-row dictionary. The chunked flow `GatherLarge` uses has neither problem,
now that #406 handles dead addresses. This spec moves every dictionary onto that flow and deletes
the small path, leaving a single `Gather` op with one protocol, which is also what the unpadded
track builds on.

## Intent

### Goal

Replace `Op::GatherSmall` and `Op::GatherLarge` with a single `Op::Gather`, proven for every
dictionary size, including one row, by the existing read-raf execution sumcheck followed by the
`shout` chunked one-hot checks (`RaVirtual` and Booleanity with its Hamming term), in
`jolt-atlas-core/src/onnx_proof/ops/gather.rs`.

Notation: `log_K` is the base-2 log of the dictionary's row count rounded up to a power of two,
`log_T` the base-2 log of the padded index count, and `d` the number of 4-bit address chunks.

### Invariants

From the `atlas-trace` behavioral contract, this relies on the padding clause: `dict_len` stays the
raw size, the prover rounds it up, and `PadBehavior::Structural` does not change.

- **A. The output is bound to the dictionary.** A prover cannot get an output other than
  `dict[index]` accepted, including by solving the `ra` claim from the final sumcheck claim. Checked
  on the branch by the forged-output test: accepted at its pin, rejected from the fix on. After that
  test is removed, **review-held**: `Op::Gather` has one prover, and that prover runs `shout`'s
  one-hot checks on the claim its execution sumcheck produces.
- **B. Hot addresses are live.** No lane can be hot at a dead address, at any dictionary size. Held
  by `one_hot_checks_reject_hot_dead_address`, with `log_k = 0` added.
- **C. One-row dictionaries prove.** With `log_K = 0` there is one chunk with all 4 bits dead, so the
  index must be 0 and the execution sumcheck runs zero rounds, with no special case in the verifier.
  Held by the dict 1 row in `op_surface` and the `shout.rs` tests.
- **D. Trace semantics are unchanged.** Same outputs for the same inputs, at every size. Held by
  `kernel_goldens` and the `golden_zoo` hashes, with no pinned value changed.
- **E. Prover and verifier agree on the commitment shape.** The chunk count and `num_vars` come only
  from `dict_len` and the index dims, and `node_committed_poly_num_vars` returns
  `LOG_K_CHUNK + log_T`. Held by the `op_surface` rows with dictionaries under 16 rows, where
  `log_K + log_T` would be too low, and by the zoo.
- **F. Other one-hot users keep their protocol.** Allowing `log_K = 0` and removing the `Option` on
  `linear` change nothing for callers with `log_K > 0`. Held by the workspace run and the zoo, and
  **review-held** for the absence of a behavior change.
- **G. The proof carries one Gather protocol.** No `RaHammingWeight` entry and no `GatherRa`
  commitment. Held by the compiler, since the variants are gone, and by `proof_mutation_sweep` over
  the probe model.

### Non-Goals

- Non-power-of-two Gather. The `gather_non_pow2` row stays ignored; zero-hot one-hots for padded
  lanes belong to the unpadded track.
- Tuning chunk width for small dictionaries. Every Gather uses the default 4-bit chunks.
- Fixing `zk.rs`, which does not compile on `main`. Its Gather branches are updated to stay
  consistent.

## Evaluation

### Acceptance Criteria

Soundness:

- [ ] The forged-output test is accepted on the commit that pins it and rejected from the fix commit
  onward. Its removal in the last commit is the only change to it after that.
- [ ] `one_hot_checks_reject_hot_dead_address` passes with `log_k = 0` added.
- [ ] `proof_mutation_sweep`, `witness_tamper_sweep` and `opening_forgery_sweep` pass over the probe
  model, with Gather on the chunked flow.

Every dictionary size:

- [ ] `op_surface` has passing rows for dictionaries of 1, 2, 4, 16, 32, 65536 and 65537 rows. The
  dict 1 row is no longer ignored.
- [ ] `gather_non_pow2` stays ignored, with its stated reason.
- [ ] For Gather, `node_committed_poly_num_vars` returns `LOG_K_CHUNK + log_T`.

One protocol:

- [ ] `Op::GatherSmall`, `Op::GatherLarge` and `GATHER_SMALL_LIMIT` are gone, and `Op::Gather` is the
  only variant.
- [ ] The prover lives in `ops/gather.rs`, and the `ops/gather/` folder is gone.
- [ ] These are gone: `hamming_weight.rs`, `hamming_booleanity.rs`, `ProofType::RaHammingWeight`,
  `CommittedPoly::GatherRa`, `VirtualPoly::HammingWeight`, the `SumcheckId` variants `HammingWeight`,
  `RamHammingBooleanity` and `RamHammingWeight`, `SmallBooleanitySumcheckProver`, the `I` parameter of
  `BooleanitySumcheckProver`, and the `Option` on `BooleanitySumcheckParams::linear`.

No regressions:

- [ ] `golden_zoo` and `kernel_goldens` pass with no change to their pinned values.
- [ ] `cargo nextest run --workspace` passes, including the zoo round trips.
- [ ] GPT-2 proves and verifies with Dory.
- [ ] GPT-2 with Dory is at most 5% slower end to end than the base commit, measured on one machine.
- [ ] `--features zk` shows no new errors in the changed files, compared against the base commit.
- [ ] The PR description has before/after prove time, verify time and proof size.

### Testing Strategy

New tests:

- `op_surface`: the rows above under `OpKind::Gather`. The dict 1 row loses its `known_gap_` prefix
  and `#[ignore]`.
- `shout.rs`: `log_k = 0` added to the existing accept and dead-address tests.
- `soundness.rs`, branch only: the forged-output test, `known_gap_*` when added, un-ignored at the
  fix, and removed with its hook in the last commit.

The probe model at `onnx_proof/testing/models.rs` (dict 16) puts the chunked flow under the three
sweeps with no new model. The proof gains no new field, so no new `mutate::Site` is needed. Because
the Gather proof's shape changes, the sweeps also run once with `JOLT_SOUNDNESS_EXHAUSTIVE=1`.

CI jobs:

- `cargo nextest run --workspace`: unit tests, `op_surface`, the sweeps, and the checked-in zoo with
  its serialization round trips.
- The `atlas-onnx-import` zoo gates and golden hashes, unchanged.
- The real-model job (GPT-2, BGE, Qwen) through the `ci:heavy` label, the only place real BGE and Qwen
  run for this change.

CI does not build the `zk` feature and `zk.rs` does not compile on `main`, so the `--features zk`
errors are compared against the base commit.

### Performance

The limit is at most 5% slower end to end on
`cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory --trace-terminal`,
against a baseline on the same machine, with per-stage numbers reported. `qwen2-tiny`, `gpt2-tiny` and
`bge-tiny` are reported with no limit: their timings are noisy, and a dict 2 slowdown is expected.

Expected movement, from counting rounds and table sizes, not measured:

| Model | Stage | Direction | Why |
|---|---|---|---|
| GPT-2 | IOP proving | slightly down | The embedding's one-hot checks go from tables over `2^16` addresses to `d` tables of 16 entries, plus a short `RaSumcheck`. The execution sumcheck does not change. |
| GPT-2 | Verify | slightly down | One-hot rounds drop from about `2·log_K + 2·log_T` to about `4 + 2·log_T`. |
| GPT-2 | Witness + commitment | about flat | Four commitments instead of one, but Gather has few lanes next to the rest of the model. |
| GPT-2 | Proof size | about flat | Fewer rounds, three more commitments per large Gather. |
| `qwen2-tiny` | IOP proving | slightly up | Dict 2 Booleanity goes from `1 + log_T` to `4 + log_T` rounds, and `RaSumcheck` is added. |

`max_num_vars` can only go down, so the opening proof should not get worse.

## Design

### Architecture

The pipeline does not change shape. One op replaces two, and small dictionaries move onto the flow
large ones already use. Proving order for every `Op::Gather`:

```text
Execution sumcheck (read-raf over log_K rounds)
  → NodeOutputRa(r_address, r_cycle), dict claim, index claim
One-hot checks, batched (RaOneHotChecks)
  RaVirtual: NodeOutputRa = Π_d ra_d at (r_address, r_cycle)
  Booleanity + Hamming over live addresses, on every ra_d
```

**`atlas-trace`.** `Op::Gather { axis, dict_len }` replaces both variants, with a plain `OpSpec` impl
instead of `impl_gather!`. `gather_eval`, `gather_shape` and `PadBehavior::Structural` do not change.

**`atlas-onnx-import`.** `lower.rs` emits `Op::Gather` for every dictionary, and `GATHER_SMALL_LIMIT`
goes. `fixtures.rs` reads the single variant.

**`common`.** `CommittedPoly::GatherRa` and `VirtualPoly::HammingWeight` go. `GatherRaD(node, chunk)` is
the only Gather commitment.

**`joltworks`.**

- `OneHotParams::from_config_and_log_K` uses `d = max(1, ⌈log_K / log_k_chunk⌉)`.
  `compute_r_address_chunks` pads the address to `d·log_k_chunk` bits, so an empty address gives one
  all-zero chunk. `shout::ra_onehot_params` already derives `dead_bits = d·log_k_chunk − log_K`, which
  is 4 here, so the selector leaves only address 0 live.
- `BooleanitySumcheckParams::linear` becomes a plain `LinearTerm`, and `BooleanitySumcheckProver` drops
  its `I` parameter, always using `u8`.
- `hamming_weight`, `hamming_booleanity` and the three Hamming `SumcheckId` variants go.

**`jolt-atlas-core`.**

- `onnx_proof/ops/gather.rs` holds `GatherParams`, `GatherProver`, `GatherVerifier`, `GatherRaEncoding`
  and the `OperatorProofTrait` impl.
- `witness.rs` builds `GatherRaD` from `dict_len`.
- `node_view.rs` returns `LOG_K_CHUNK + log_T`, no longer reading `physical_outdims` for Gather.
- `ProofType::RaHammingWeight` goes.
- `zk.rs` keeps one Gather branch per direction.

There are no new types or traits. The shared code stays in `joltworks`, and Gather only supplies an
`RaOneHotEncoding`.

### Alternatives Considered

**Fix `GatherSmall` in place by linking its `ra` claim to the commitment.** Open the committed
`GatherRa` at the execution point, instead of a virtual claim. This is the smallest fix for the
forgery, and dictionaries with very few rows may prove slightly faster because there is no `RaVirtual`
and fewer address rounds. Rejected because it keeps two protocols for one operation: the one-row panic
needs its own fix, and the unpadded track would have to add zero-hot support to both.

**Merge the other way, using a single chunk for every dictionary.** One committed one-hot over all
`log_K` bits would remove `RaVirtual`. Rejected because `GatherRa`'s indices are `u16`, which cannot
address Qwen's 151936-row embedding, and widening them still leaves a polynomial over `log_K + log_T`
variables, which is what chunking exists to avoid.

**Handle one-row dictionaries in the importer.** `repeat_to_gather` is what produces them, so it could
lower that case to an identity, or the op could reject them. Rejected because the selector already
handles `log_K = 0` with `d = max(1, …)` and a small padding change, whereas an importer rule would
leave `Op::Gather` with a size it cannot prove.

## Documentation

`README.md`, `docs/` and `CLAUDE.md` need no changes: none of them names the Gather variants, and the
proving commands stay the same. The README's benchmark table is not re-measured, because it is for
HyperKZG on a different machine.

In-code documentation follows the code: the `Op::Gather` doc comment in `atlas-trace/src/op/mod.rs`, the
`GatherRa` row of the table in `common/src/lib.rs`, the `op_surface` comments about
`GATHER_SMALL_LIMIT`, and the comment in `scripts/gen_tiny_models.py` about `GatherLarge`, reworded to
say dictionaries above 2^16 rows are covered by the single-op table and the real-Qwen job. Weights do
not change, so nothing is regenerated.

## Execution

Take the baseline first, on the commit this branch starts from: `models/gpt2 --pcs dory --trace-terminal`,
plus `qwen2-tiny`, `gpt2-tiny` and `bge-tiny`. Then one concern per commit, each passing the workspace
run:

1. **Pin the forged Gather output.** A temporary `known_gap_*` test in `soundness.rs`, `#[ignore]`d
   because the verifier accepts the forgery today. Its hook sits in the shared
   `GatherProver::initialize`: a test-only flag and one adjusted `index_onehot` entry, so that the sum
   matches the tampered output.
2. **Support zero-bit addresses.** `from_config_and_log_K` and `compute_r_address_chunks`, with
   `log_k = 0` added to the `shout.rs` tests.
3. **Prove every dictionary through the chunked one-hot.** Both variants share one flow, `small.rs` goes,
   and `node_committed_poly_num_vars` changes. The forgery test loses its `known_gap_` prefix and
   `#[ignore]`, as does the dict 1 row, and the new rows are added. Both `Op` variants still exist at
   this point, because `every_op_kind_has_a_declared_case` and `every_case_contains_its_declared_op`
   require a row for each.
4. **Merge into `Op::Gather`.** A mechanical rename across crates, moving the code into `ops/gather.rs`.
   No behavior change.
5. **Delete what is left unused.** The list from the Acceptance Criteria, including the `I` parameter and
   the `Option` on `linear`.
6. **Remove the forgery test and its hook.**

Commit 3 is the protocol change, so the zoo and the GPT-2 run apply from there on. The before/after
numbers are taken at the tip.

## References

- [#406](https://github.com/ICME-Lab/jolt-atlas/pull/406): the address selector this spec depends on.
  Without it, moving small dictionaries onto the chunked flow would expose them to the dead-address
  forgery, and `log_K = 0` would have no way to make all its addresses dead.
