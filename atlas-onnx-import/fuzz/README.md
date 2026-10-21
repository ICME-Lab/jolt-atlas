# Fuzzing atlas-onnx-import

libFuzzer harness (via [cargo-fuzz]) for the import pipeline. The `import`
target feeds arbitrary bytes through `ModelProto::decode` and then the same
stages as `load_graph`: graph build → normalize → analyze → DCE → input
classification → rewrite passes → lower. `ImportError` returns are expected;
panics, arithmetic-overflow aborts, OOMs, and timeouts are findings.

## Setup

```sh
cargo install cargo-fuzz   # once
```

cargo-fuzz needs a nightly toolchain (the repo pins stable), so run it with
`+nightly`.

## Seed the corpus

Coverage is far better starting from real models:

```sh
cd atlas-onnx-import
mkdir -p fuzz/corpus/import
for f in ../models/*/network.onnx; do
  cp "$f" "fuzz/corpus/import/$(basename "$(dirname "$f")").onnx"
done
```

(Skip the large ones if you care about iteration speed; `reject_concat`,
`reject_slice`, and `op_gauntlet` are small and op-dense.)

## Run

```sh
cd atlas-onnx-import
cargo +nightly fuzz run import -- -timeout=10 -rss_limit_mb=4096 -max_len=262144
```

Crashing inputs land in `fuzz/artifacts/import/`. Reproduce one with:

```sh
cargo +nightly fuzz run import fuzz/artifacts/import/crash-<hash>
```

and minimize it with:

```sh
cargo +nightly fuzz tmin import fuzz/artifacts/import/crash-<hash>
```

## Known findings

Crashers worth keeping are checked in under `fuzz/regressions/`; re-run them
with `cargo +nightly fuzz run import fuzz/regressions/<file>`.

- `crash-8cc58347…`: a Reshape whose target dims are huge positive `i64`s
  drives the unchecked `usize` products in `reshape_dims`
  (`src/shape_fns.rs`) to a multiply-with-overflow abort. Unfixed; found on
  the harness's first 60-second run.

## Notes

- `fuzz/` is its own cargo workspace, so the sanitizer/nightly build never
  touches the main workspace; `cargo build` / `cargo test` at the repo root
  are unaffected.
- The harness points external-data tensors at a nonexistent directory, so
  fuzzed models can't read files from disk; those take the clean `Io` error
  path.
- Inputs that ask for absurdly large tensors can trip the libFuzzer RSS
  limit. Those OOMs are real findings (the importer accepted the shape), not
  harness noise.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
