# Fuzzing the verifier

libFuzzer harness (via [cargo-fuzz]) for `jolt_atlas_verifier::onnx_proof::verify`. The
`verify` target takes an honest proof of the checked-in op gauntlet
(`models/op_gauntlet`), lets the fuzzer mutate its serialized bytes and the public IO, and
verifies the result. The verifier must return on every input: `Err` is the expected verdict;
a panic, an arithmetic-overflow abort, an OOM or a timeout is a finding. The input format and
the driver live in `jolt_atlas_core::onnx_proof::fuzz`, shared with the regression replay.

Not run in CI. Run it after a change to the verifier, the proof format or a relation.

## Setup

```sh
cargo install cargo-fuzz   # once
```

cargo-fuzz needs a nightly toolchain (the repo pins stable), so run it with `+nightly`.

## Seed the corpus

The target proves the model once at startup (a few seconds in release) and can write the
honest input as the seed:

```sh
cd jolt-atlas-core
mkdir -p fuzz/corpus/verify
ATLAS_FUZZ_WRITE_SEED=fuzz/corpus/verify/honest \
  cargo +nightly fuzz run verify -- -runs=0
```

## Run

```sh
cd jolt-atlas-core
cargo +nightly fuzz run verify -- -timeout=60 -rss_limit_mb=8192 -max_len=8388608
```

Crashing inputs land in `fuzz/artifacts/verify/`. Reproduce one with
`cargo +nightly fuzz run verify fuzz/artifacts/verify/crash-<hash>` and minimize it with
`cargo +nightly fuzz tmin verify fuzz/artifacts/verify/crash-<hash>`.

## Regressions

A crasher worth keeping is checked in under `fuzz/regressions/` after the fix. The test
`fuzz_regressions_replay` (`cargo nextest run -p jolt-atlas-core -E 'test(fuzz_regressions)'`)
replays every file there, plus a fixed set of truncations and byte flips of the honest input,
through the same driver on every CI run, so a fixed crash stays fixed.

## Findings log

None yet.

## Notes

- `fuzz/` is its own cargo workspace, so the sanitizer/nightly build never touches the main
  workspace.
- An input is tied to the op gauntlet and to the proof format: after a protocol-breaking
  change, re-seed the corpus; old regression inputs then exercise only the deserializer.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
