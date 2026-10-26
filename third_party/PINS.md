# Vendored upstream crates

Crates adopted from [a16z/jolt](https://github.com/a16z/jolt) are vendored whole under
`third_party/` at a recorded upstream rev and wired in through `[workspace.dependencies]` in the
root `Cargo.toml`. Files are never copied piecemeal into our own crates. A pin bump is a
deliberate PR that links the upstream diff between the old and new rev. `specs/adoption-program.md`
owns the adoption order; `docs/upstream-inventory.md` records the divergence each stage resolves.

| Crate | Upstream rev | Date | Adopted at | Local delta |
|---|---|---|---|---|
| `jolt-field` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 2 (bumped from `0e3faf07` at stage 3) | Manifest only (below) |
| `jolt-transcript` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 3 | Manifest only (below) |
| `jolt-utils` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 4a | Manifest only (below) |
| `jolt-poly` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 4b | Manifest only (below) |
| `jolt-sumcheck` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 4b | Manifest only (below) |
| `jolt-claims` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-claims-derive` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-openings` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-crypto` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-riscv` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-lookup-tables` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |
| `jolt-r1cs` | `f3265b5a9981f0c1359f2b63008824616f0c82fa` | 2026-10-24 | stage 5 (spike) | Manifest only (below) |

## `jolt-field`

`src/`, `asm/` and `build.rs` are byte-identical to `crates/jolt-field` at the pinned rev.
The manifest differs from upstream's:

- workspace dependency inheritance is resolved to explicit versions, with the arkworks crates
  pointing at the same `a16z/arkworks-algebra` `dev/twist-shout` branch the root `Cargo.toml`
  pins, so the graph holds one `ark-bn254`/`ark-ff` and `jolt_field::Fr` converts to and from
  our arkworks types;
- the optional `allocative` dependency (and dev-dependency) is `0.3` instead of upstream's
  `=0.3.6`, so the root `[patch]` pin of `allocative` covers it and the graph keeps one
  `allocative` (two copies make arkworks' `Allocative` impls target the wrong one);
- the `benches/`, `examples/`, `fuzz/` and `tests/` targets are not vendored (their
  dev-dependencies are upstream-workspace-only).

Stage 2 vendored `0e3faf07`, the rev the Akita integration branch (`akita-adapter`) pins.
Stage 3 bumped it to `f3265b5a` so that `jolt-field` and `jolt-transcript` (whose codecs and
proof-of-work are newer than `0e3faf07`) share one rev; the upstream diff is
`git diff 0e3faf07 f3265b5a -- crates/jolt-field`. The bump added the optional `bytemuck`
dependency and the `binary`, `bytemuck` and `fp64-proof-linkage` features, carried into the
manifest as upstream declares them. `akita-adapter` must take the same bump when it rebases.

## `jolt-transcript`

`src/` is byte-identical to `crates/jolt-transcript` at the pinned rev. The manifest resolves
workspace inheritance to explicit versions (arkworks on our pinned fork branch, `jolt-field` by
path to the vendored copy) and drops the `benches/`, `fuzz/` and `tests/` targets and the
`[lints] workspace = true` table; `num-bigint` stays as the dev-dependency of the in-source unit
tests. It brings `spongefish =0.8.0`, `blake2 0.11.0-rc.6` and `digest 0.11` from crates.io,
alongside the `blake2 0.10` our own code uses.

## `jolt-utils`

`src/` is byte-identical to `crates/jolt-utils` at the pinned rev. The manifest resolves
workspace inheritance to explicit versions (`jolt-field` dev-dependency by path) and drops the
`[lints] workspace = true` table.

## `jolt-poly`

`src/` is byte-identical to `crates/jolt-poly` at the pinned rev. The manifest resolves workspace
inheritance to explicit versions (internal crates by path; `allocative` relaxed to `0.3` as for
`jolt-field`) and drops the `benches/`, `fuzz/` and `tests/` targets.

## `jolt-sumcheck`

`src/` is byte-identical to `crates/jolt-sumcheck` at the pinned rev. The manifest resolves
workspace inheritance and drops the `examples/`, `fuzz/` and `tests/` targets. Upstream's optional
`jolt-crypto` and `jolt-r1cs` dependencies are not vendored, so the `committed` and `r1cs`
features are declared empty (the source's `cfg(feature = ...)` gates stay valid) and must stay
off; `parallel` no longer forwards to `jolt-crypto`. They come back when those crates are vendored.

## `jolt-claims` and its dependency cone

`jolt-claims`, `jolt-claims-derive`, `jolt-openings`, `jolt-crypto`, `jolt-riscv`,
`jolt-lookup-tables` and `jolt-r1cs` vendor `src/` byte-identical to `crates/<name>` at the
pinned rev. Atlas uses only `jolt-claims`' relation layer (`Expr`, `SymbolicSumcheck`); the rest
of the cone is there because `jolt-claims` depends on it unconditionally (its zkVM protocol
modules use the RISC-V crates, its lattice packing uses `jolt-openings`). The manifests differ
from upstream's:

- workspace dependency inheritance is resolved to explicit versions, with the arkworks crates on
  the root `Cargo.toml`'s `dev/twist-shout` branch;
- `jolt-claims` takes `jolt-openings` with default features off, so `jolt-crypto`'s `bn254`
  feature (and with it `ark-bn254`/`ark-ec`) stays off in the resolved graph;
- `jolt-lookup-tables`' `field-inline` feature drops its forward to the `tracer` dev-dependency;
- `jolt-claims` and `jolt-claims-derive` are workspace members with their dev-dependencies
  (`bincode`, `rand_chacha`, `prettyplease`, ...) so their upstream tests run in the workspace
  suite; the other five are `exclude`d from the workspace, because their upstream tests need
  crates not vendored here (`tracer`, `jolt-dory`), and the `benches/`, `fuzz/` and `tests/`
  targets of all seven are not vendored.
