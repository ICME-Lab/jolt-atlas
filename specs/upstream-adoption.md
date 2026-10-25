# Spec: Upstream Adoption — Replace joltworks with Vendored a16z/jolt Crates

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | #397                           |
| Depends on  | #396 (phase 1 lands before stage 1) |
| Supersedes  | the feature-gate/split plan recorded in #396's Alternatives Considered |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

`joltworks` is a snapshot of the monolithic `jolt-core`, maintained as a divergent fork while
upstream a16z/jolt decomposed into 31 crates and re-architected its claim management, transcripts
and sumcheck (see `docs/upstream-inventory.md`, the factual basis for this spec). Staying forked
means re-deriving upstream's soundness fixes by hand forever and building our own versions of
systems upstream already ships — including the lattice PCS (Akita) and the decomposed
prover/verifier layout we want. This spec replaces `joltworks` subsystem-by-subsystem with
vendored, rev-pinned upstream crates, re-porting atlas-specific deltas on top, so that the Akita
overhaul and the total-decomposition end state arrive as one adoption program instead of two
hand-built projects.

## Intent

### Goal

Retire `joltworks` as a fork: after this program, the proof-system toolkit consists of vendored
upstream crates (`jolt-field`, `jolt-transcript`, `jolt-sumcheck`, `jolt-poly`, `jolt-utils`,
the `jolt-lookup-tables` framework, `jolt-dory`, `jolt-hyperkzg`, `jolt-akita`, and —
contingent on the stage-4 decision — `jolt-openings`/`jolt-claims` as the claim system or as
internal dependencies of the PCS crates) at pinned revs under `third_party/`, plus atlas-only
crates holding what has no upstream counterpart (prefix–suffix shout, evaluation reduction, the
one-hot machinery, atlas lookup-table contents, and, on the stage-4 "no" branch, the atlas
claim accumulator). Every `joltworks` module named in `docs/upstream-inventory.md` is assigned
to a stage; none is left behind implicitly.

### Invariants

- **Vendoring, never copying**: each adopted crate is vendored whole at a recorded upstream rev
  under `third_party/`, wired by a root `Cargo.toml` patch (the `jolt-field` precedent on the
  Akita branch). No upstream file is ever copied into `joltworks`. Pin bumps are deliberate PRs
  whose body links the upstream diff.
- **One pin**: every vendored crate in one tree builds from the same upstream rev; mixed revs
  (today's split `arkworks dev/twist-shout` lockfile pins) are a defect this program removes.
- **Per-stage soundness gates**: every stage passes the soundness sweeps, the op-surface table,
  the full zoo, one real-model prove & verify, and — once `jolt-atlas-verifier` exists — its CI
  isolation build, so each swap is checked against the product verifier boundary. Stages are
  either **behavior-preserving** or **declared protocol-breaking** (proof format changes; gated
  by the full suite plus a serialization version bump and re-pinned goldens). No stage is
  silently protocol-breaking. Behavior preservation is checked against the *previous stage*,
  not within the build: the current zoo harness proves and verifies with the same code and
  cannot see a coordinated change, so each behavior-preserving stage pins, at the pre-stage
  commit, proof-bytes and transcript digests from deterministic zoo inputs, and asserts the
  post-stage build reproduces them byte for byte; protocol-breaking stages re-pin them.
- **Atlas deltas are re-ported, not lost**: before a subsystem is swapped, its atlas-specific
  changes are enumerated (perf work, the opening-accumulator semantics, zk hooks) and each is
  re-applied on the vendored crate, upstreamed, or retired with a stated reason.
- **The trace semantics contract is untouched**: `atlas-trace` fixed-point behavior, the
  fused-rescale split, lookup-table generation and padding are out of scope; only the proof
  toolkit underneath changes.

### Non-Goals

- Adopting upstream's zkVM product crates (`jolt-prover`, `jolt-verifier`, tracer, program,
  host, SDK): those verify RISC-V execution, not ONNX graphs. Our product layer
  (`jolt-atlas-core`, `jolt-atlas-verifier` per `specs/verifier-crate.md`) stays ours.
- Switching the default field or PCS in the early stages. Adopting `jolt-field` does not flip
  BN254 → Fp128; the Akita/Fp128 activation is its own stage with its own gate.
- The `zk` feature and `jolt-blindfold` adoption: deferred until zk is revived with tests.
- Upstreaming our deltas (desirable, separate effort per delta).

## Evaluation

### Acceptance Criteria

Per stage (the sequence is in Design):

- [ ] Stage 0 (its own PR, the program's first): the `Cargo.lock` arkworks pins are unified at
  one rev; the five flagged upstream fixes are *triaged* in our tree with a finding each (G1/G2
  trailing-byte decoding, verifier PCS floor, arkworks MSM, Dory table materialization, eq edge
  cases) — triage means an applicability verdict, recorded in `docs/upstream-inventory.md` in
  the same PR; *remediation* of any applicable finding is scheduled per finding (immediately if
  severe, else with the stage that adopts the affected subsystem, e.g. a Dory remediation rides
  stage 5). The J2 transcript bias analysis is reviewed against our Blake2b/Keccak challenge
  sampling, with the conclusion recorded in the stage-0 PR and re-examined by stage 2.
- [ ] Each adoption stage: `cargo nextest run --workspace` green; soundness sweeps green with no
  new tolerances; one real-model run (GPT-2 Dory minimum) green; the stage's PR states
  behavior-preserving or protocol-breaking, and protocol-breaking stages bump the proof
  serialization version and re-pin affected goldens.
- [ ] Each adoption stage deletes the replaced `joltworks` module in the same PR (no parallel
  implementations outlive a stage).
- [ ] After the final stage, `joltworks` is gone: every module from `docs/upstream-inventory.md`
  is either replaced by a vendored crate or lives in a named atlas-only crate (including the
  atlas claim accumulator if stage 4 concludes "no"), and `jolt-atlas-verifier`'s dependency
  cone satisfies the end state as revised in `specs/verifier-crate.md`: no `jolt-atlas-core`,
  no upstream prover *product* crates (`jolt-prover`, `jolt-spartan-prover`, `jolt-witness`),
  and the vendored toolkit's prover entry points (e.g. `jolt-sumcheck::prover`, which upstream
  exports unconditionally) barred by a `disallowed-methods`/`disallowed-types` clippy
  configuration in the verifier crate, run as part of the CI isolation job.
- [ ] Akita/Fp128 activation (within stage 5): before any default flips, the structural suites
  run as Fp128+Akita instances — the round-trip harness and sweeps are generic over
  `PCS`/field today only in signature (`PCS::Field = Fr` is hard-wired and the sweeps use
  `HyperKZG<Bn254>`), so the stage makes them generic and adds Akita-specific `mutate::Site`s;
  an honest real-model run alone does not cover malicious proofs against the new backend.
- [ ] A `third_party/PINS.md` records every vendored crate's upstream rev and last-triage date.

### Testing Strategy

The gates are the existing structural suites: the op-surface table, the three soundness sweeps,
the tiny zoo with transcript lockstep and serialization round-trips, the real-model heavy job,
and the importer golden hashes (unaffected; the import pipeline is out of scope). Stages that
change the proof format re-pin proof-shape-dependent expectations explicitly in the same PR.
The claim-lattice stage (4) additionally ports the opening-forgery sweep onto the adopted claim
system before the old accumulator is deleted — the sweep is the regression net for exactly the
code being replaced.

### Performance

Behavior-preserving stages must not regress the README benchmark table on
`models/gpt2 --pcs dory` (witness + commitment, IOP proving, opening proof, verify), checked
per stage with the `prove` example; atlas perf deltas being re-ported (fold routines, one-hot
commit paths, sparse RLC) are the mechanism, and a stage that cannot hold the line keeps the
delta and documents the divergence in `third_party/PINS.md`. Protocol-breaking stages may carry
costs **inherent to the adopted protocol** — stage 2's sponge proof-of-work and codec work is
the known case — which are not removable by any delta: the stage PR quantifies the regression
and accepts it explicitly on security grounds, rather than pretending a delta covers it. The
Akita activation is measured against the Dory baseline and justified on its own terms
(post-quantum PCS, upstream alignment), not required to win the benchmark.

## Design

### Architecture

Seven PRs — stage 0 plus stages 1–6 (large stages may split further; every PR references this
spec):

0. **Prerequisites** — lockfile unification, the five-fix triage, the J2 review. Lands first,
   independent of everything else including #396.
1. **`jolt-field`** — bump the Akita branch's vendored pin to current, land it on main
   independent of the Akita PCS. Behavior-preserving (BN254 remains the active field; Fp128
   becomes *available*).
2. **`jolt-transcript`** — adopt codecs, the bias budget, and the prover/verifier transcript
   split. Protocol-breaking (challenge derivation changes). The J2 bias review lands first so
   we know whether our current sampling has an exposure worth back-porting ahead of adoption.
3. **`jolt-sumcheck` + `jolt-poly` + `jolt-utils`** — the core engine swap, absorbing the
   homes of `msm`, `curve`, `config` and `utils`. The stage PR also records the disposition of
   every remaining shared-ancestry subprotocol per the inventory's buckets: the
   shout/booleanity/hamming family, `univariate_skip` (check whether upstream's round scheduler
   subsumes it), and the prefix–suffix lookup framework (`jolt-lookup-tables`; atlas table
   *contents* stay atlas-only). Target behavior-preserving modulo stage 2's format; the atlas
   fold-routine deltas are re-ported here.
4. **`jolt-openings` + `jolt-claims`** — the decision stage. Our opening accumulator is fully
   diverged, and upstream's claim lattice is architecturally the claim-ledger/ProofPlan design
   the verifier-simplification work specified. A time-boxed spike answers: can the lattice
   express atlas claim flows (node-indexed openings, multi-consumer evaluation reduction,
   deferred batches)? Yes → adopt and rebuild our flows in it; no → keep the atlas accumulator
   as an atlas-only crate implementing upstream's claim traits where possible, and record why.
   On the "no" branch, `jolt-openings`/`jolt-claims` are still vendored in stage 5 — upstream's
   PCS crates depend on them — but only as internal dependencies behind an atlas adapter at the
   product boundary.
5. **PCS: `jolt-dory`, `jolt-hyperkzg`, `jolt-akita`** — adopt the schemes, re-port the atlas
   Dory perf work, then activate Fp128+Akita behind a config gate with its own real-model
   sign-off before any default flips.
6. **Verifier-cone closure** — `jolt-atlas-verifier` (from `specs/verifier-crate.md`) depends
   only on vendored toolkit crates plus atlas-only verifier code, with upstream prover product
   crates excluded by the crate graph and the toolkit's prover entry points barred by the
   clippy disallow list in the CI isolation job (upstream toolkit crates export prover modules
   unconditionally, so "prover-free" is crate-level plus lint-level, not module-level — see the
   revised end state in `specs/verifier-crate.md`). This stage is that spec's declared end
   state; by then its
   own later phases (verification loop, preprocessing, per-op verifier moves, gated on the
   expression-engine migration — **a spec that does not exist yet and is the critical path for
   the full prover/verifier split**; see that spec's End state for the four-link completion
   chain) must have landed, since the cone can only close around code that lives in the crate.

Relationship to in-flight work: the `fp128-field` / `field-trait` / `akita-adapter` branches
(based on pre-rewrite main) are the raw material for stages 1 and 5; their content re-lands
through this program rather than rebasing wholesale.

### Alternatives Considered

- **Stay forked, cherry-pick fixes**: rejected — the inventory shows upstream re-architected;
  cherry-picking across different architectures decays into re-implementation, and we
  demonstrably miss fixes today (five open triage items, a split lockfile pin).
- **Re-fork wholesale from current upstream**: rejected — jolt-atlas-core consumes `joltworks`
  APIs in ~80 files; a big-bang re-fork is unreviewable and unbisectable. Staged adoption keeps
  every PR gated by the sweeps.
- **Git dependencies instead of vendoring**: rejected — upstream's own crates use path deps
  inside their workspace, and mixed git/path resolution breaks (the reason the Akita branch
  vendors `jolt-field`); vendoring also makes the audit surface and pin provenance explicit
  in-tree.
- **Build our own decomposition and claim ledger** (the pre-inventory plan): rejected — it
  duplicates systems upstream now ships; `specs/verifier-crate.md` records the superseded
  feature-gate/split ideas under Alternatives Considered.

## Documentation

`README.md` and `CLAUDE.md`/`AGENTS.md` crate maps per stage; `third_party/PINS.md` (new);
`docs/upstream-inventory.md` regenerated at stage 0 and referenced per stage;
`specs/verifier-crate.md` (PR #396, in review at the time of writing; lands before stage 1 per
Execution) is kept in sync — its End state section points here.

## Execution

This program depends on `specs/verifier-crate.md` (#396): its phase 1 (the proof-type boundary
and the CI isolation build) lands **before stage 1**, so every adoption stage is gated by the
isolation build from the start; and stage 6 is unreachable until that spec's later phases have
moved verification into the crate, since the cone can only close around code that lives there.
Only stage 0 is independent of #396 and can start immediately (lockfile unification, fix
triage, J2 review).

Stages 1–3 are sequential. Stage 4's spike can run concurrently with stage 3. Stage 5 depends
on stages 1–4: at `f3265b5a`, `jolt-dory` depends on `jolt-openings`, `jolt-poly`,
`jolt-transcript` and `jolt-akita`, so the PCS crates cannot vendor before their dependencies
have. Akita *activation* additionally requires the generic structural suites (see Acceptance).
Stage 6 lands whenever the cone's last prover dependency falls. Each stage's PR enumerates the
atlas deltas it re-ports (from the inventory's bucket notes) as a checklist in the PR body.

## References

- `docs/upstream-inventory.md` (divergence inventory, fix triage, drift measurements).
- `specs/verifier-crate.md` (#396): the product verifier boundary. Its phase 1 (the proof-type
  boundary) lands before this program; its later phases interleave with these stages; its end
  state is this program's stage 6.
- `specs/core-test-overhaul.md`: the sweeps and zoo that gate every stage.
- Upstream: a16z/jolt `f3265b5a` (2026-10-25), 31-crate workspace; the vendored `jolt-field`
  precedent at `third_party/jolt-field` on the Akita integration branch (pin `0e3faf07`).
