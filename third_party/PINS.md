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
