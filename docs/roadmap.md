# Roadmap: own the proof-system toolkit

Living document. Specs freeze at merge; this page tracks the program across them and is
updated as phases land. Last updated 2026-11-14.

## Principle

jolt-atlas owns the proof-system code it runs. Upstream `a16z/jolt` is a reference we read when
it ships something better (a field kernel, a sumcheck engine, a PCS adapter) and port into our
crates; it is not a dependency we track. `third_party/` is the transitional state for crates
we have not yet taken over, and the program ends with that directory gone. Owning a crate means:
it lives in the workspace under our name, we edit it, its tests are ours, and
`docs/upstream-inventory.md` records where it came from so a later port has a diff base.

Vendoring (`third_party/PINS.md`) got us upstream's relation layer, engine and lattice PCS
quickly, but it re-created the duplication joltworks had removed: today the workspace holds two
polynomial libraries, two sumcheck engines, two transcripts, two prefix-suffix lookup
frameworks and a field trait layered on another field trait. The program removes each pair.

External libraries stay external when they are generic and we do not change them: `rayon`,
`spongefish`, `blake2`, `prost`. LayerZero's `akita-*` crates (about 275,000 lines at the
pinned rev, MIT/Apache) are treated the same way for now; see "Open decisions".

## Current state

| Vendored crate | Lines | Used by atlas for | Atlas duplicate | Phase |
|---|---|---|---|---|
| `jolt-claims` + `-derive` | 20,264 | `SymbolicSumcheck`, `Expr`, claim traits (~1,100 lines of it) | none | 1 |
| `jolt-riscv`, `jolt-lookup-tables`, `jolt-r1cs` | 23,392 | nothing (pulled by `jolt-claims`) | `atlas-lookup-tables` is a different table set | 1 (deleted) |
| `jolt-akita` | 9,139 | the Akita adapter behind `atlas-pcs::akita` | none | 2 |
| `jolt-openings` | 1,261 | `CommitmentScheme`, prefix-packed layouts, `OpeningsError` | `atlas-pcs::commitment_scheme` | 2 |
| `jolt-crypto` | 5,459 | the `Commitment` trait (BN254 code off) | none | 2 |
| `jolt-sumcheck` | 4,913 | `prove_batch`, `verify_compressed`, `ProveRounds`, recorder | `atlas-sumcheck::upstream_sumcheck` facade, `sumcheck/` | 3 |
| `jolt-transcript` | 2,719 | spongefish prover/verifier states, codecs, grinding | `atlas-transcript` (thin layer) | 4 |
| `jolt-poly` | 6,365 | `UnivariatePoly`, `GruenSplitEqPolynomial`, `CompressedPoly`, binding order | `atlas-algebra::poly` (8,067 lines) | 5 |
| `jolt-utils` | 353 | one allocator helper | `atlas-algebra::utils` | 5 |
| `jolt-field` | 11,227 | Solinas Fp128 kernels (`Prime128OffsetA7F7`), `Ring`/`Field`, accumulators | `atlas-algebra::field` (`JoltField` extends upstream's; `Fp128` newtype) | 6 |
| arkworks fork pin (`ark-ff`, `ark-serialize`, `ark-std`) | external | `CanonicalSerialize` in 38 files, `BigInt`, signed ints, `test_rng` | — | 7 |

## Phases

Order is top-down: consumers first, so every phase is behavior-preserving (digest pins) and
the crates below it stay vendored until their turn. Each phase is one spec and one PR unless
the spec says otherwise.

1. **`atlas-claims`** — `specs/atlas-claims.md`. Own the relation layer; delete the RISC-V
   cone. Status: proposed.
2. **PCS adapter** — `specs/own-pcs-adapter.md`. `jolt-akita`, `jolt-openings` and
   `jolt-crypto` fold into `atlas-pcs`; `akita-*` becomes a direct dependency of `atlas-pcs`.
   Status: proposed.
3. **Sumcheck engine** — `specs/own-sumcheck-engine.md`. `jolt-sumcheck`'s engine becomes
   `atlas-sumcheck::engine`; the `upstream_sumcheck` facade goes. Status: proposed.
4. **Transcript** — `specs/own-transcript.md`. `jolt-transcript`'s split prover/verifier
   states, codecs and grinding become `atlas-transcript`. Status: proposed.
5. **Polynomials** — `specs/own-poly.md`. One polynomial library: `jolt-poly` and
   `atlas-algebra::poly` merge, `jolt-utils` folds in. Status: proposed.
6. **Field** — `specs/own-field.md`. One field crate: the Solinas kernels and one trait;
   the `Fp128` newtype and the two-layer `JoltField` go. Status: proposed.
7. **arkworks** — `specs/drop-arkworks.md`. Our own canonical serialization and big-integer
   helpers; the `a16z/arkworks-algebra` fork pin and `[patch.crates-io]` entries removed.
   Status: proposed.

Phases 5 and 6 keep the package names `jolt-poly` and `jolt-field` (the `akita-*` manifests
bind those names through the root `[patch."https://github.com/a16z/jolt"]`), owned and at the
repository root. Renaming them is possible only after the akita decision below.

## Gates every phase shares

- Digest pins unchanged (`cargo nextest run -p jolt-atlas-core -E 'test(digest)'`).
- CI Format, Clippy (`-D warnings`), Verifier isolation (including the no-BN254 graph check
  and, from phase 1, the no-RISC-V graph check), Test; the heavy job on `main`.
- The `prove` example on `models/gpt2-tiny` within the spec's stated tolerance; phases 5 and
  6 touch the sumcheck inner loops and gate on the real GPT-2 run.
- `docs/upstream-inventory.md` gains a provenance row; `third_party/PINS.md` loses the crate;
  `docs/verifier.md`'s perimeter table and the crate maps in `CLAUDE.md`/`AGENTS.md` follow.

## Open decisions

- **Fork `akita-*` or keep it external?** Forking brings 275,000 lines in-house for a PCS we
  do not currently modify; keeping it external means `jolt-field`/`jolt-poly` keep their
  package names and the API surface `akita-*` imports (`Prime`, `ExtField`, `Unreduced`,
  `Fold`, `solinas::parallel`, `UnivariatePoly`, `OmittedConstantPoly`, `MultilinearPoly`,
  `OneHotPolynomial`, `Point`). Recommendation: external until we need to change it; a thin
  compatibility surface in our owned crates is cheaper than owning a PCS. Revisit when the
  in-house lattice zk work starts (`lattice-only-end-state` in the team notes).
- **Where owned crates live.** Phase 1 puts `atlas-claims` at the root beside the other
  `atlas-*` crates; later phases follow. `third_party/` is deleted when phase 6 lands.

## Progress

| Phase | Spec | PR | Landed |
|---|---|---|---|
| 1 | `specs/atlas-claims.md` | — | — |
| 2 | `specs/own-pcs-adapter.md` | — | — |
| 3 | `specs/own-sumcheck-engine.md` | — | — |
| 4 | `specs/own-transcript.md` | — | — |
| 5 | `specs/own-poly.md` | — | — |
| 6 | `specs/own-field.md` | — | — |
| 7 | `specs/drop-arkworks.md` | — | — |
