# WIP: generic einsum prover

Implementation of [`specs/generic-einsum.md`](specs/generic-einsum.md) (PR #402). Remove this file
before merging.

The branch is based on `main` at `329b21ef` and may need a rebase.

## Status

The spec's two parts are committed. Part A (prover) proves every two-operand einsum through one
role-driven contraction sumcheck. Part B (importer) lowers `MatMul` and `Gemm` directly to
`Einsum`. Final QA has not been done; see [Remaining work](#remaining-work).

The history has 10 implementation commits instead of the spec's 8, because the spec's commit 3 was
split so that the trait removal stays mechanical:

| Spec commit | Branch commits                                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------------------------------- |
| 1           | `rename the dot engine to contraction`                                                                                |
| 2           | `prove einsums through one role-driven layout`                                                                        |
| 3           | `read the bound eq from its polynomial`, `drop EinsumLayout and EqSchedule`, `keep r_output as a NamedPoint in the params` |
| 4           | `use a degree-2 sumcheck for size-1 batches`                                                                          |
| 5           | `accept any two-operand contraction`                                                                                  |
| 6           | `sweep a batched, permuted einsum for soundness`                                                                      |
| 7           | `stop gating einsum equations to a canonical set`                                                                     |
| 8           | `lower MatMul and Gemm directly to einsum`                                                                            |

Two things differ from the spec's wording. Points are `NamedPoint`s, which are `TensorPoint`s with
one einsum letter per axis. `has_batch()` is named `has_batch_bits()`, since a size-1 batch letter
adds no sumcheck variable.

## Checks run on the last commit

- `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets`: clean.
- `cargo nextest run -p atlas-onnx-import`: passes, including the golden hashes, which needed no
  re-pin.
- `cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'not test(golden_zoo_big)'`:
  passes.
- `cargo nextest run -p jolt-atlas-core -E 'test(zoo)'`: passes (checked-in zoo, prove and verify).

The full `cargo nextest run --workspace` last passed on an earlier version of
`accept any two-operand contraction`. Later commits ran only scoped tests: the einsum and soundness
tests (the sweeps pass with the new `batched_einsum` probe model), then the importer and zoo tests
above.

## Remaining work

1. `cargo nextest run --workspace` on the last commit.
2. `cargo nextest run -p atlas-onnx-import --run-ignored ignored-only -E 'test(golden_zoo_big)'`.
3. The heavy real-model runs `zoo::gpt2_dory` and `zoo::qwen` (see `CLAUDE.md`).
4. `cargo check -p jolt-atlas-core --features zk`: it must fail only in `zk.rs`, with the same error
   set as `main`. This matched on the prover commits but was not re-run after the importer commits.
5. Benchmark: the `prove` example with `--pcs dory --trace-terminal` on `models/gpt2` and
   `models/qwen`, median of 3 runs per side. `ONNXProof::iop` and `ONNXProof::prove` must stay
   within ±5% of `main`.
6. `/clean-branch`, then set the spec's `Status` to `implemented` in the last change, and delete
   this file.

## Design choices not visible in the spec

- `matmul_equation` (`atlas-onnx-import/src/lower.rs`) spells out the three 1-D cases literally and
  names batch axes from one `abcdefgh` list. A 1-D operand against a batched one, such as
  `[k]·[a,k,n]`, is rejected; the spec does not ask for it and `main` rejects it too.
- `slice_through_projection` keeps its `loop` / `break value` search. Extracting it into a helper
  would be a separate refactor.
