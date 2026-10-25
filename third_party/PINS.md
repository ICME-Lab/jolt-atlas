# Vendored upstream crates

Crates adopted from [a16z/jolt](https://github.com/a16z/jolt) are vendored whole under
`third_party/` at a recorded upstream rev and wired in through `[workspace.dependencies]` in the
root `Cargo.toml`. Files are never copied piecemeal into our own crates. A pin bump is a
deliberate PR that links the upstream diff between the old and new rev. `specs/adoption-program.md`
owns the adoption order; `docs/upstream-inventory.md` records the divergence each stage resolves.

| Crate | Upstream rev | Date | Adopted at | Local delta |
|---|---|---|---|---|
| `jolt-field` | `0e3faf07f7ebb4a074e0ad58bcea29fc1dda521b` | 2026-09-15 | stage 2 | Manifest only (below) |

## `jolt-field`

`src/`, `asm/` and `build.rs` are byte-identical to `crates/jolt-field` at the pinned rev.
The manifest differs from upstream's:

- workspace dependency inheritance is resolved to explicit versions, with the arkworks crates
  pointing at the same `a16z/arkworks-algebra` `dev/twist-shout` branch the root `Cargo.toml`
  pins, so the graph holds one `ark-bn254`/`ark-ff` and `jolt_field::Fr` converts to and from
  our arkworks types;
- the `benches/`, `examples/`, `fuzz/` and `tests/` targets are not vendored (their
  dev-dependencies are upstream-workspace-only).

The pin is the one the Akita integration branch (`akita-adapter`) vendors, so the two lines of
work share a single `jolt-field`.
