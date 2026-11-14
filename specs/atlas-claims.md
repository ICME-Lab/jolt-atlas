# Spec: atlas-claims — own the relation layer, drop the RISC-V cone

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/adoption-program.md` (stage 5: the relation layer is adopted), `specs/verifier-audit-simplification.md` (every sumcheck is a relation) |
| Supersedes  | `specs/adoption-program.md` stage 5's "vendor `jolt-claims` and its cone" decision; `third_party/PINS.md`'s "files are never copied piecemeal into our own crates" rule, for `jolt-claims` |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Every atlas sumcheck is a `SymbolicSumcheck` over atlas openings, and that trait, its `Expr`
algebra and the claim-data traits are the only things atlas takes from upstream's
`jolt-claims`: about 1,100 lines in six modules with no dependency outside `jolt-field`,
`jolt-poly` and `jolt-utils`. The other 18,000 lines of the vendored crate are the zkVM's
protocol stage graph (`protocols/`, `r1cs.rs`), and they are why the workspace carries a RISC-V
ISA (`jolt-riscv`), the RISC-V instruction lookup tables (`jolt-lookup-tables`) and an R1CS
builder (`jolt-r1cs`): some 23,000 lines that no atlas crate names, that every build compiles,
that `docs/verifier.md` must list inside the trust perimeter, and that make `jolt-claims` the one
vendored crate whose upstream tests need crates we do not have. The byte-identical vendoring
rule (`third_party/PINS.md`) forbids pruning them in place. This spec replaces the vendored
crate with an atlas-owned `atlas-claims` crate holding exactly the relation layer, and removes
the RISC-V cone from the repository. It is the first step of the program the user set on
2026-11-14: joltworks exists so that atlas owns its proof-system code and takes what is best
from upstream as a reference, rather than vendoring upstream whole; the long-run state has no
`third_party/` at all (see Non-Goals for what is deferred).

## Intent

### Goal

Add the workspace crate `atlas-claims` (the relation layer copied from `jolt-claims` at
`f3265b5a`: `claims`, `claim_data`, `symbolic`, `ops`, `formula_error`, `sumcheck`), retarget
every `jolt_claims::` import in `atlas-sumcheck`, `jolt-atlas-verifier`, `jolt-atlas-core` and
`jolt-akita` to it, and delete `third_party/jolt-claims`, `jolt-claims-derive`, `jolt-riscv`,
`jolt-lookup-tables` and `jolt-r1cs`, with transcripts and proofs byte-identical.

### Invariants

- **Behavior-preserving.** Every checked-in model's proof bytes and transcript are unchanged:
  the digest pins in `jolt-atlas-core/src/onnx_proof/testing/` match before and after.
  Mechanism: `cargo nextest run -p jolt-atlas-core -E 'test(digest)'` in the workspace test job.
  The relation layer is pure symbolic algebra (no transcript, no field arithmetic beyond
  `Expr` evaluation), so a semantic change in the copy shows up as a digest change or a
  verifier rejection, not as a silent drift.
- **The copied modules are upstream's modules.** Each file in `atlas-claims/src/` is a copy of
  the named `crates/jolt-claims/src/<file>` at `f3265b5a`, with the local changes limited to:
  the crate name and doc links, removal of the `jolt_claims_derive` re-exports, removal of
  items whose only users were `protocols/` (listed in the PR), and the test port below.
  Mechanism: review-held. The PR description carries, per file, the upstream path and
  `git diff --no-index` against the vendored copy, and `docs/upstream-inventory.md` records
  the provenance row so a later upstream pull has a diff base. There is no CI check that a
  copied file matches upstream; the vendoring rule that provided one is what this spec retires
  for this crate, by the user's decision.
- **The in-source tests of the copied modules keep passing** (today: `claims` 5, `symbolic` 3,
  `claim_data` 4, `ops` 1). Tests that name zkVM protocol ids (`claim_data`'s
  `sumcheck_challenges_tests` imports `protocols::jolt`) are ported to local test-only id
  enums; tests over `jolt_field::Fr` (BN254) are ported to a field the normal graph already
  resolves (`jolt-field`'s Solinas field, feature `solinas`, as `jolt-akita` uses it), so
  `atlas-claims` enables no `bn254` feature in any dependency table. Mechanism:
  `cargo nextest run -p atlas-claims` in the workspace test job; the BN254 graph check covers
  the normal/build edges (CI "No BN254 curve crates"); the dev-edge claim is review-held.
- **No RISC-V, lookup-table or R1CS crate in any member's graph.** Mechanism: a new step in the
  "Verifier isolation" CI job runs `cargo tree -e normal,build,dev --prefix none` for every
  workspace member and fails on a line starting with `jolt-riscv `, `jolt-lookup-tables `,
  `jolt-r1cs ` or `jolt-claims `. (`jolt-sumcheck`'s `r1cs` feature stays declared empty, as
  `PINS.md` records; the `cfg` gates in its source remain valid and off.)
- **Verifier isolation holds as before.** `jolt-atlas-verifier` still builds alone, still
  resolves no `jolt-atlas-core`, and `jolt-atlas-verifier/clippy.toml` needs no new entry: the
  relation layer has no prover items (it is transcript- and witness-free; `docs/verifier.md`
  already lists it as such). Mechanism: the existing CI job and the `clippy.toml` list.
- **`jolt-akita`'s commitment behavior is unchanged.** Its two uses of the zkVM geometry are
  the 64-column trace one-hot cap (`MAX_ONE_HOT_TRACE_COLUMNS`, checked in
  `trace_onehot/source.rs`) and the schedule-key admission list in `schedules::one_hot_keys`,
  which today derives the admitted column widths by enumerating RV64 bytecode/RAM shapes.
  After the change the constant lives in `jolt-akita` and the admitted widths are an explicit
  sorted constant equal to the set the enumeration produces today, with the derivation kept as
  a comment. Mechanism: the digest tests (the Akita class openings are in the proof bytes) and
  a `jolt-akita` unit test asserting the constant list equals the enumeration's output,
  committed with the enumeration code inlined in the test (so the test carries the old
  derivation and the shipped code carries only the list). `atlas-pcs` commits 64-column K=16
  one-hot classes (`ONE_HOT_COLUMNS`), so the list must contain 64; the test pins it.
- **The `atlas-trace` behavioral contract is untouched.** Nothing in this spec reads or changes
  the fixed-point scale, the quotient/remainder split, the lookup tables or the padding rules;
  the trace crate is not in the diff. Mechanism: `git diff --stat` on the PR shows no
  `atlas-trace/` path (review-held), and the op-surface and soundness tests run unchanged.
- **A malicious prover gains nothing.** The verifier's relation evaluation (`RelationVerifier`
  over `Expr`) is the same code under a new crate name; soundness coverage is the unchanged
  sweeps in `onnx_proof/testing/soundness.rs`. No new proof field, no new `mutate::Site`.

### Non-Goals

- **`jolt-openings` and `jolt-crypto` stay vendored in this PR.** They are the PCS-side cone
  (`CommitmentScheme`, `PrefixPackedLayout`, `Commitment`), used by `atlas-pcs` and `jolt-akita`,
  not by the relation layer. Owning them is the next step of the program, together with
  `jolt-akita`, in a follow-up spec.
- **No de-duplication of the proof-system toolkit.** `atlas-algebra/poly` and `jolt-poly`,
  `atlas-sumcheck` and `jolt-sumcheck`, `atlas-transcript` and `jolt-transcript`, our `JoltField`
  extension and `jolt-field` all coexist today, which is the duplication joltworks set out to
  remove and vendoring reintroduced. Phases 3 to 6 of `docs/roadmap.md` take each pair in
  turn, keep one implementation, and retire `third_party/`. This spec only removes code that
  nothing uses.
- **No change to any relation, op plan, proof field or wire format.** The crate rename is the
  whole surface change for atlas code.
- **No upstream pin bump.** `atlas-claims` is copied from the rev already vendored
  (`f3265b5a`); pulling newer upstream claim-layer changes is a later, separate PR.
- **No new lint policy for `atlas-claims` beyond what the copied `lib.rs` already carries**
  (`forbid(unsafe_code)` and the panic-discipline `deny` list); it is not in the verifier's
  deny-at-root set (that set is `jolt-atlas-verifier`'s own), and `clippy.toml` is not extended.

## Evaluation

### Acceptance Criteria

- [ ] `atlas-claims` is a workspace member at the repository root beside `atlas-sumcheck`,
      with `src/{lib,claims,claim_data,symbolic,ops,formula_error,sumcheck}.rs`, no
      `protocols/`, `r1cs.rs`, `lattice.rs` or `twist/`, and dependencies only on `jolt-field`,
      `jolt-poly`, `jolt-utils`, `derive_more`, `serde`, `thiserror` (and `rayon` if `ops`
      still needs it after the copy). `cargo nextest run -p atlas-claims` passes.
- [ ] No source file under `atlas-*`, `jolt-atlas-*`, `common` or `third_party/jolt-akita`
      contains `jolt_claims` (grep), and `Cargo.toml`/`Cargo.lock` resolve no `jolt-claims`.
- [ ] `third_party/jolt-claims`, `third_party/jolt-claims-derive`, `third_party/jolt-riscv`,
      `third_party/jolt-lookup-tables` and `third_party/jolt-r1cs` are deleted; the root
      `Cargo.toml` `exclude` list keeps only `jolt-crypto` and `jolt-openings`; `PINS.md` has
      no rows for the five crates.
- [ ] The "Verifier isolation" CI job has the new graph step and it passes; it fails on a
      branch that re-adds a `jolt-riscv` path dependency to any member (checked once by hand
      in the PR, recorded in the PR description).
- [ ] The digest pins in `jolt-atlas-core/src/onnx_proof/testing/` are unchanged in the PR diff
      and `cargo nextest run -p jolt-atlas-core -E 'test(digest)'` passes.
- [ ] `jolt-akita` has the admitted-width constant and its equality test; the test passes.
- [ ] CI's Format, Clippy (`RUSTFLAGS=-D warnings`), Verifier isolation and Test jobs are
      green on the PR; the heavy job (GPT-2 / BGE / Qwen prove & verify) is green on `main`
      after merge or on the PR with the `ci:heavy` label.
- [ ] `docs/verifier.md`'s trust-perimeter table, `docs/upstream-inventory.md`, `PINS.md`,
      `CLAUDE.md` and `AGENTS.md` crate maps name `atlas-claims` and no deleted crate.
- [ ] `cargo build --timings` for `cargo check -p jolt-atlas-verifier` (cold, `cargo clean`
      first) is recorded before and after in the PR description, and the "after" is not slower.

Later phases (not this PR; `docs/roadmap.md` is the program):

- Phase 2 `specs/own-pcs-adapter.md` (`jolt-akita`, `jolt-openings`, `jolt-crypto` into
  `atlas-pcs`); phase 3 `specs/own-sumcheck-engine.md`; phase 4 `specs/own-transcript.md`;
  phase 5 `specs/own-poly.md` (one polynomial library, `jolt-utils` folded in); phase 6
  `specs/own-field.md` (one field crate, `third_party/` and `PINS.md` deleted); phase 7
  `specs/drop-arkworks.md`.

### Testing Strategy

Existing coverage carries the change: `cargo nextest run --workspace` (unit tests, the
op-surface table, the soundness sweeps, the checked-in zoo end to end with proof
round-trips, the digest pins) runs in the Test job; the Verifier isolation job runs the graph
checks; the heavy job proves the real models. `atlas-claims`' own tests are the ported
in-source tests of the copied modules. New tests: the `jolt-akita` admitted-width equality test;
the new CI graph step. No `zk` feature exists in this tree. No `mutate::Site` changes (no proof
field changes). No `op_surface.rs` rows change (no op changes). The `atlas-onnx-import` zoo
gates are unaffected (no importer or trace change) and need not run.

### Performance

Prover and verifier time are unchanged by construction (same code, new crate name; digests
prove the transcript is identical), so the `prove` example is not a gate here; the final
report quotes one `--trace-terminal` run on `models/gpt2-tiny` as a sanity check only. The
measured quantity is build time: the cone is roughly 23,000 lines compiled at `opt-level = 3`
in every test build. Benchmark: `cargo build --timings` for `cargo check -p
jolt-atlas-verifier` and for `cargo nextest run --workspace --no-run`, cold, before and after,
on the same machine; the expectation is a strict decrease in total wall clock, magnitude
recorded, no numeric target. Memory, proof size and verifier time: unchanged, not re-measured.

## Design

### Architecture

```
before                                         after
  atlas-sumcheck ─┐                              atlas-sumcheck ─┐
  jolt-atlas-verifier ├─ jolt_claims ──┬─ jolt-riscv      jolt-atlas-verifier ├─ atlas_claims ── jolt-field
  jolt-atlas-core ─┘   (third_party)   ├─ jolt-lookup-tables ── jolt-riscv   jolt-atlas-core ─┘               jolt-poly
  jolt-akita ─────┘                    ├─ jolt-r1cs                           jolt-akita ─────┘               jolt-utils
                                       ├─ jolt-openings ── jolt-crypto        jolt-akita ── jolt-openings ── jolt-crypto
                                       └─ jolt-claims-derive                               (unchanged)
```

- **`atlas-claims`** (new, root-level workspace member, `atlas-claims/`): `lib.rs` re-exports
  what the vendored `lib.rs` re-exports today minus the derive macros: `ChallengeDrawError`,
  `InputClaims`, `MissingOpeningValue`, `NoChallenges`, `NoInputs`, `NoOutputs`,
  `OutputClaims`, `SumcheckChallenges`, `challenge`, `constant`, `derived`, `opening`, `Expr`,
  `Source`, `Term`, `SumcheckDomain`, `SymbolicSumcheck`. The `extern crate self as
  jolt_claims` alias (there for the derives' absolute paths) is dropped. `ops.rs` keeps the
  `Expr` operator impls; `formula_error.rs` the error type; `sumcheck.rs` the domain enum.
  Items in these files that only `protocols/` used are removed, each named in the PR.
  `Cargo.toml` is written in the workspace style (`[workspace.dependencies]` entry
  `atlas-claims = { path = "./atlas-claims" }`), with a header comment giving the upstream
  provenance (crate, rev, the six files).
- **Consumers**: `atlas-sumcheck` (relation core, shout, booleanity, hamming weight,
  ra-virtual, ps-shout, digit range, identity range check, opening reduction),
  `jolt-atlas-verifier` and `jolt-atlas-core` (every op relation, `plan.rs`): a one-token
  rename of the import path, no API change. `jolt-akita`: manifest dependency
  `jolt-claims` replaced; `MAX_ONE_HOT_TRACE_COLUMNS` defined in `trace_onehot/source.rs`;
  `schedules::one_hot_keys` takes the admitted widths from a constant. This is the first
  source delta `jolt-akita` carries; `PINS.md`'s `jolt-akita` section records it until the
  follow-up spec owns the crate.
- **Removed**: the five `third_party/` directories, their `exclude`/`members` entries, the
  `jolt-claims` `[workspace.dependencies]` row, their `Cargo.lock` entries (and transitively
  `strum 0.28`, if nothing else resolves it).
- **CI**: one new step in `verifier-isolation`.
- **Trust perimeter**: `docs/verifier.md` row for `third_party/` loses `jolt-claims`;
  `atlas-claims` is added to the atlas-crate rows as the symbolic-relation layer. The audit
  reading order gains the crate after `atlas-sumcheck`'s relation module.

### Alternatives Considered

- **Git dependencies on `a16z/jolt` instead of vendoring.** Rejected in discussion on
  2026-11-14: `jolt-claims` depends on `jolt-riscv` unconditionally in upstream's own
  manifest, so a git dependency pulls the same cone; and it is the opposite of owning the
  code, which is the program's purpose.
- **Ask upstream for a `zkvm` feature on `jolt-claims`, or a `jolt-claims-core` split, and
  wait.** Rejected: it keeps atlas's build on upstream's schedule and keeps the code
  upstream's. Upstream's split, if it happens, becomes a reference for our copy.
- **Prune `third_party/jolt-claims` in place** (delete `protocols/` and `r1cs.rs`, keep the
  directory). Considered (smaller diff); rejected by the user in favor of a first-class
  `atlas-claims` crate, so the ownership is visible in the crate map and `third_party/` holds
  only code we have not yet taken over.
- **Keep `lattice.rs` and `twist/`** (the balanced-digit algebra and Twist claim reductions,
  about 750 lines, no cone imports). Not copied: no atlas crate uses them; our clamp and
  remainder digit range checks (`atlas-sumcheck::digit_range`,
  `specs/clamp-dense-digit-range-check.md`) have their own encoding. They remain in upstream
  as a reference if a later spec wants them.
- **Keep `jolt-claims-derive`.** Not copied: the derives assume a static opening vocabulary and
  atlas opening ids are graph-indexed (`docs/upstream-inventory.md`); no atlas type derives
  them, and dropping the proc-macro crate removes `syn`/`prettyplease` from the build.
- **Replace `jolt-akita`'s admitted-width enumeration with `1..=64`.** Rejected: the keys must
  match the packaged schedule catalog `atlas-pcs` loads, and admitting keys the catalog lacks
  changes `one_hot_keys`' output; the explicit equal list with its equality test is exact.

## Documentation

- `third_party/PINS.md`: drop the five rows and the "`jolt-claims` and its dependency cone"
  section; the `jolt-akita` section gains its source delta; the intro says that vendoring is
  the transitional state for crates atlas has not yet taken over, and points at
  `docs/upstream-inventory.md` for owned crates' provenance.
- `docs/upstream-inventory.md`: the `opening_proof`/`jolt-claims` row says the relation layer
  is owned in `atlas-claims` from `f3265b5a`, listing the six files; the `lookup_tables/` row
  loses its "revisit if upstream splits the framework" note (moot).
- `docs/verifier.md`: trust-perimeter table and reading order as above.
- `CLAUDE.md`, `AGENTS.md`: crate map gains `atlas-claims` in the proof-system toolkit list
  (between `atlas-sumcheck`'s entry and `atlas-pcs`, or wherever dependency order puts it:
  it sits below `atlas-sumcheck`), and the `third_party/` sentence lists the remaining crates.
- `README.md`: no change (no user-facing behavior).
- Crate-level doc on `atlas-claims/src/lib.rs`: what the layer is, where it came from, the
  rule that changes are made here and upstream is consulted as a reference.

## Execution

Suggested commit sequence (each builds and passes its own gate):

1. **Add `atlas-claims`** with the six copied files, ported tests and manifest; workspace
   member; no consumer yet. Gate: `cargo nextest run -p atlas-claims`.
2. **Retarget consumers**: `jolt_claims::` → `atlas_claims::` across `atlas-sumcheck`,
   `jolt-atlas-verifier`, `jolt-atlas-core`; `jolt-akita` manifest, constant and admitted-width
   list with its equality test. Gate: `cargo check -p jolt-atlas-core --tests`, then
   `cargo nextest run -p jolt-atlas-core -E 'test(digest)'` and `-p jolt-akita`.
3. **Delete the cone**: the five directories, `Cargo.toml` members/exclude/workspace-deps,
   `Cargo.lock`. Gate: `cargo check --workspace --all-targets`; `cargo tree` shows none of the
   five.
4. **CI step and docs**: the graph step; `PINS.md`, `upstream-inventory.md`, `verifier.md`,
   `CLAUDE.md`, `AGENTS.md`, this spec's `Status`. Gate: CI green on `dev`.

Record the before/after `cargo build --timings` numbers in the PR description between steps 1
and 3 (before: current `dev`; after: step 3). Local runs stay narrow per `CLAUDE.md`; the
workspace run is CI's.

## References

- Upstream: `a16z/jolt` `crates/jolt-claims` at `f3265b5a9981f0c1359f2b63008824616f0c82fa`
  (`src/{claims,claim_data,symbolic,ops,formula_error,sumcheck}.rs`); `crates/jolt-akita`
  `src/schedules/mod.rs` and `src/trace_onehot/source.rs` at the same rev.
- `third_party/PINS.md` (the vendoring policy this spec starts retiring).
- `specs/adoption-program.md` stage 5 (adoption of the relation layer) and stage 8 (verifier
  cone closure); `docs/upstream-inventory.md`.
- `specs/verifier-audit-simplification.md` (every op's verification is a plan over relations;
  the reason the relation layer is the whole surface atlas needs).
- `docs/verifier.md` (trust perimeter and reading order).
