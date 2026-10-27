# Spec: Dims Disambiguate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @AntoineF4C5                   |
| Created     | 2026-10-26                     |
| Status      | proposed                       |
| PR          |                                |
| Depends on  | —                              |
| Supersedes  | —                              |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

"Dims" has no single meaning in `atlas-trace`. When a model is compiled with padding, `pad_graph`
overwrites every node's `output_dims` with power-of-two dims, and `apply_padding` rewrites op
state to match: Reshape and Broadcast replace their `shape` with padded dims, Constant pads its
tensor, SoftmaxLastAxis records the raw live width in `active_len`, MeanOfSquares records
`padded_count` (Slice and Concat only validate). Padding is optional (`CompileOptions::pad_pow2`): the importer turns it
on, `ModelBuilder::build()` leaves it off. The same field therefore holds raw dims in one model
and padded dims in another, and its name does not say which. The prover inherits the confusion:
the reshape protocol's `input_raw_dims` / `output_raw_dims` hold padded dims in practice;
`SoftmaxLastAxis.active_len`, `MeanOfSquares.padded_count` and `IoDims` exist only to stash raw
values next to padded ones; the power-of-two derivation is written in several places. The
ambiguity has already produced a soundness-relevant bug (prover reading padded tensor dims while
the verifier expected raw ones), and it stands in the way of later reshape and padding work. This
spec gives each dims notion one meaning: stored dims are raw, padded dims are derived. Raw is the
source of truth because the mapping is one-way: padded dims are always derivable from raw ones
(`next_power_of_two` per axis), raw dims cannot be recovered from padded ones.

## Intent

### Goal

Every dims value stored in the `atlas-trace` IR (node `output_dims` and every op field) holds raw
dims, and every padded dim is derived on demand from raw dims through one owner function.

- `atlas-trace`: `pad_graph` no longer writes `output_dims`; `apply_padding` no longer rewrites op
  fields, except `SoftmaxLastAxis.active_len` (see Non-Goals); `MeanOfSquares.padded_count` and
  `IoDims` are removed; one `padded_dims(raw)` function owns the raw→padded derivation; a
  model-wide layout notion states whether tensors are physically padded, `physical_dims(node)`
  resolves a node's physical dims from it, and `EvalCtx` carries it with a helper
  `physical(&raw)` that structural ops call in `eval`.
- `atlas-onnx-import`: holds no padding logic beyond the Gather Small/Large cost heuristic, which
  is computed from raw dims.
- `jolt-atlas-core`: every node-dims read resolves raw or physical dims explicitly; no proof-domain
  decision reads `tensor.dims()`.

### Invariants

The feature relies on the `atlas-trace` behavioral contract's padding clause (dimensions padded
per-dimension to powers of two at compile time; every op declares its padding behavior): physical
padding is unchanged, only what is stored as dims changes. The clause's wording changes to say
that IR dims are raw and padded dims are derived (see Documentation). The scale, fused-rescale
and lookup-table clauses are untouched. The relation "node dims == tensor dims", held today by
`check_shapes` re-run after padding, becomes "tensor dims == `physical_dims(node)`".

- **A. Proofs are byte-identical.** For a fixed model and input, the serialized proof is the
  same before and after this change. A wrong raw/physical choice at any prover or verifier site
  changes the transcript and is caught here. Mechanism: the temporary test
  `temp_proof_hash_pinned` (checked-in zoo models, hashes pinned from `main`), run by the
  workspace CI job at the PR head; plus a manual before/after run on real GPT-2 (`--pcs dory`)
  reported in the PR description.
- **B. Stored dims are raw.** After `Model::compile`, every node's `output_dims` equals the
  dims shape inference gave it before padding, and the model's input/output dims accessors
  return the same values `IoDims` held. Mechanism: a new test over every checked-in zoo model and
  every probe model in `onnx_proof/testing/models.rs`.
- **C. One owner for raw→padded.** `padded_dims` is the only place that maps raw dims to padded
  dims; every other site calls it. Mechanism: review-held.
- **D. No proof-domain shape comes from a tensor.** The prover and verifier take num_vars,
  selectors and split points from node dims (raw or `physical_dims`) or from the op's own raw
  fields resolved through `padded_dims`, never from `tensor.dims()`. Reading `tensor.dims()` to rebuild a witness tensor in the same physical
  layout is allowed. Mechanism: review-held, backed by A.
- **E. The importer knows nothing of padding.** `atlas-onnx-import` contains no padding logic
  except the Gather Small/Large choice. Mechanism: review-held.
- **F. A malicious prover gains no freedom.** Every dimension remains part of the preprocessed
  model the verifier holds; the prover chooses none. Mechanism: the existing soundness sweeps
  (`proof_mutation_sweep`, `witness_tamper_sweep`, `opening_forgery_sweep`), unchanged.

### Non-Goals

- **Bug fixes.** Bugs or rejected cases found along the way are recorded as `#[ignore]`d
  `known_gap_*` tests, not fixed. The PR changes no proven or rejected behavior.
- **Removing physical padding.** Tensors stay physically padded in production; only the stored
  dims change.
- **Removing `SoftmaxLastAxis.active_len`.** Masked softmax exists only for padding: when the
  last axis widens, lanes beyond the raw width are masked. Computing that in `eval` needs the
  node's raw input width, which `OpSpec::eval` does not receive (it sees only the physical
  operand tensors and model-wide `EvalCtx`). Changing the `eval` API, or dropping physical
  padding so tensors carry raw dims, is a separate change; until then `apply_padding` keeps
  setting `active_len`, and the proof layer keeps rejecting masked softmax
  (`softmax_masked_rejected`).
- **Padded-lane inertness.** The KNOWN GAP documented on `pad_graph` (whether padded lanes stay
  inert through each op) is unchanged.
- **Compatibility with previously serialized `Model`s.** `AtlasSharedPreprocessing` serializes
  the model; the meaning of its `output_dims` and op fields changes. The model is not absorbed
  into the transcript, so proof bytes are unaffected (A).
- **Performance work.** No intended prover or verifier speed change.
- **`zk` behavior.** Its call sites are converted like the rest; nothing about ZK changes.

## Evaluation

### Acceptance Criteria

- [ ] `pad_graph` does not write `output_dims`; B's test passes on every checked-in zoo model and
      every probe model.
- [ ] No op rewrites a field in `apply_padding` except `SoftmaxLastAxis.active_len`; Constant
      still pads its tensor data (physical layout, not dims).
- [ ] `MeanOfSquares.padded_count` (op field and importer computation) and `IoDims` are removed;
      the model's input/output dims accessors keep their signatures.
- [ ] One `padded_dims` function owns raw→padded; the existing power-of-two derivations
      (`pow2_dims`, `pow2_padded_num_output_elements`, tensor padding) call it.
- [ ] `Model` holds a model-wide layout (raw or padded) and exposes `physical_dims(node)`;
      `EvalCtx` carries the layout and a `physical(&raw)` helper (delegating to `padded_dims`);
      Reshape and Broadcast derive their physical target through it in `eval`.
- [ ] Shape checking compares tensor dims against `physical_dims`; `annotate_clamp_widths` uses
      physical dims.
- [ ] Every node-dims read in `jolt-atlas-core` resolves raw or physical explicitly; the
      prover's proof-domain decisions in Reshape, Broadcast and Gather read node dims, not
      `tensor.dims()`.
- [ ] `atlas-onnx-import` holds no padding logic except the Gather Small/Large heuristic.
- [ ] `temp_proof_hash_pinned` passes unchanged at the PR head; the manual GPT-2 / Dory
      before/after run shows identical proof hashes, reported in the PR description.
- [ ] The importer's golden output hashes pass without re-pinning.
- [ ] `cargo nextest run --workspace` passes with no test deleted or newly ignored; each existing
      `known_gap_*` test still fails, or the PR description records which one turned green.
- [ ] `cargo build -p jolt-atlas-core --features zk --all-targets` builds.
- [ ] The behavioral contract in `atlas-trace/src/lib.rs` and its copy in `CLAUDE.md` state that
      IR dims are raw and padded dims are derived.

### Testing Strategy

Must keep passing, unmodified: the workspace suite (unit tests, op-surface table, soundness
sweeps, checked-in zoo); the `atlas-onnx-import` zoo gates and golden hashes; the
`known_gap_*` set (each still failing, or recorded); the heavy job's GPT-2 / BGE / Qwen prove &
verify (PR labelled `ci:heavy`).

New tests:

- **`temp_proof_hash_pinned`** (A), temporary. Proves each checked-in zoo model (op_gauntlet,
  tiny GPT-2 / Qwen2 / BERT) with fixed inputs and asserts the hash of the serialized proof
  against constants pinned from a run on `main`. Its doc comment states it is removed by a
  follow-up PR. It assumes proving is deterministic for a fixed model and input; the implementer
  checks this first and fixes any seed (e.g. PCS setup) the test needs.
- **Raw dims after compile** (B). The input/output accessor assertions hold on `main` (they read
  `IoDims`) and land with the baseline; the node-level assertions land with the commit that
  stops `pad_graph` writing `output_dims`.
- **Padded op-surface rows.** Every op-surface row builds unpadded today, so no op-level case
  runs the path this spec rewrites (raw ≠ padded). Add `build_padded()` variants of the existing
  non-power-of-two rows for Reshape, Broadcast, Slice, Concat and MeanOfSquares; Constant is
  covered by any padded model. SoftmaxLastAxis gets a padded row whose non-power-of-two axis is a
  leading one (e.g. raw `[3, 8]`): a non-power-of-two last axis sets `active_len` and is
  rejected, which the row's comment states by pointing at `softmax_masked_rejected`. They pass on
  `main` and land with the baseline.

No probe model is added: the padded op-surface rows and the zoo exercise the changed path, and
the sweeps' tampering does not depend on how dims are stored.

`zk`: CI does not build it. The implementer runs `cargo build -p jolt-atlas-core --features zk
--all-targets` locally and states the result in the PR description.

### Performance

No measurable change. Proofs are byte-identical, so sumcheck, commitment and opening work is
unchanged; the only added cost is deriving padded dims on demand (a few `next_power_of_two` calls
per node, outside the inner loops). Benchmark: `cargo run --release -p jolt-atlas-core --example
prove -- --model models/gpt2 --pcs dory --trace-terminal` on `main` and on the PR head. Criterion:
no README stage (witness + commitment, IOP proving, opening proof, verify) regresses beyond
run-to-run noise (±5%), and proof size is identical. Results go in the PR description.

## Design

### Architecture

All changes live in `atlas-trace`, with call-site updates in `atlas-onnx-import` and
`jolt-atlas-core`; `joltworks` is untouched.

```
Graph (raw dims)                          Model
  │  validate, check_shapes (raw)           graph: raw dims in nodes and op fields
  ├─ per-op padding validation ────────►    layout: Raw | Padded        (model-wide)
  │    (e.g. coherent-reshape check;        meta: clamp widths
  │     no mutation)
  └─ annotate_clamp_widths (physical dims)

padded_dims(raw)     = next_power_of_two per axis          (single owner)
physical_dims(node)  = match layout { Raw => raw, Padded => padded_dims(raw) }

exec    EvalCtx carries the layout and physical(&raw); Reshape and Broadcast
        call it on their raw shape field in eval;
        the output check compares tensor.dims() with physical_dims(node)
prover  each read picks raw node dims or physical_dims(node) explicitly;
        MeanOfSquares reads its raw count field and the padded count from physical dims;
        SoftmaxLastAxis still reads active_len (see Non-Goals)
```

- **Layout is model-wide.** `pad_pow2` applies to the whole model, so the layout is one value on
  `Model`, not a per-dims tag. Raw and padded dims are plain `Vec<usize>`.
- **Padded op params are derived in `eval`.** `apply_padding` keeps only its validation role
  (the coherent-reshape and element-count checks) plus setting `active_len`. Reshape and
  Broadcast hold their raw output shape as a field, so `ctx.physical(&self.shape)` gives the
  physical target from the field and the model-wide layout alone; a raw-valid expansion stays
  valid after padding because `padded_dims` maps 1 to 1 and equal extents to equal extents.
  `EvalCtx` gains only model-wide data; the `OpSpec::eval` signature is unchanged.
- **Clamp widths use physical dims.** `clamp_bound` depends on dims (einsum's contraction length);
  computing it on raw dims would change clamp widths and the proof.
- **One code path for both modes.** With raw dims stored in both modes, padded and unpadded
  models differ only in `physical_dims` (identity versus `padded_dims`). Unit tests built with
  `ModelBuilder::build()` exercise the same dims-resolution code as production, which is why
  `pad_pow2` stays optional.

### Alternatives Considered

- **Store both raw and padded.** This is in effect today's design: padded fields plus
  `active_len`, `padded_count` and `IoDims` stashing raw values. Rejected: two sources of truth
  that must agree, and nothing checks that they do.
- **`Raw` / `Padded` newtypes.** Would make a raw/physical mix-up a compile error, but changes the
  type of every op field and every dims read site, while the confusion that matters is resolved
  at one accessor (`physical_dims`). D stays review-held, backed by A.
- **Cache padded op params at compile time.** Rejected: deriving them in `eval` costs a handful of
  `next_power_of_two` calls per structural node per execution, negligible next to proving, and a
  cache would reintroduce stored padded values.

Upstream `a16z/jolt` has no tensor shapes and nothing comparable.

## Documentation

- `atlas-trace/src/lib.rs`, behavioral contract padding clause: "Dimensions stored in the IR are
  raw. When a model is compiled with padding, tensors are physically padded per-dimension to
  powers of two, and padded dims are always derived from raw ones; every op declares how it
  behaves under padding."
- `CLAUDE.md`: the matching line in "The atlas-trace Behavioral Contract".
- Doc comments that describe padded storage: `Node::output_dims`, `Model::graph`, the `model.rs`
  module doc, `OpSpec::apply_padding`, `PadCtx`.

No `README.md` or `docs/` change: no user-facing surface, CLI or benchmark changes.

## Execution

Each commit compiles, passes the workspace suite, and keeps `temp_proof_hash_pinned` green.

1. **Pin the baseline.** Add `temp_proof_hash_pinned` (hashes from `main`), B's input/output
   accessor assertions, and the padded op-surface rows.
2. **Route every node-dims read through one ambiguous accessor**
   (`raw_or_padded_output_dims()`), and make `IoDims` private behind accessors. A pure rename:
   the diff shows every site that must decide.
3. **Introduce the vocabulary.** The layout enum, `padded_dims` as single owner (existing
   derivations call it), `physical_dims(node)`. No behavior change.
4. **Split the accessor.** `pad_graph` temporarily keeps the per-node raw dims it already
   collects as a stash on `Model` (generalizing `IoDims` from boundary nodes to every node);
   `raw_output_dims()` reads the stash, `physical_dims()` reads stored `output_dims`. Each site
   chooses against correct values while storage is still padded. Proof-domain `tensor.dims()`
   reads move to node dims here.
5. **Store raw node dims.** `pad_graph` stops writing `output_dims`; `physical_dims` derives from
   raw; the stash and `IoDims` are deleted and the accessors read nodes. The post-padding shape
   check becomes `infer_shape(physical inputs) == physical_dims(node)`, which holds because op
   fields are still padded. `annotate_clamp_widths` uses physical dims. B's node-level assertions
   land.
6. **Store raw op fields.** `apply_padding` becomes validation only (plus `active_len`);
   `EvalCtx` carries the layout and `physical(&raw)`; Reshape and Broadcast derive their target
   in `eval`; `padded_count` (op and importer) is deleted. Shape checking reduces to `check_shapes` on raw dims plus the
   executor's physical output check.
7. **Docs.** Contract clause, `CLAUDE.md`, stale doc comments.

Follow-up PR: remove `temp_proof_hash_pinned`.

## References

- PR #378 (closed): the prior attempt on the `atlas-onnx-tracer` crate. Source of the
  ambiguity-first accessor sequence and of the `original_{input,output}_dims` → accessors →
  removal lineage that `IoDims` follows here.
