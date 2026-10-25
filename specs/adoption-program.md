# Spec: The Adoption Program — Upstream Toolkit, Expression Relations, Verifier Split

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | (filled when the PR opens, or at merge) |
| Depends on  | `specs/verifier-boundary.md` (stage 1) |
| Supersedes  | PRs #396 (program portions) and #397 (closed), and the unopened `spec/expression-engine` draft — one linear order replaces three interdependent documents |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Three efforts were specified separately and turned out to be one program in a forced order:
retiring the `joltworks` fork by adopting upstream a16z/jolt's decomposed crates (31 crates,
re-architected claims/transcripts/sumcheck — `docs/upstream-inventory.md` is the factual
basis), replacing hand-written per-op prover/verifier pairs with symbolic relations (upstream's
`jolt-claims` already ships the `Expr`/`SymbolicSumcheck` layer this requires), and completing
the prover/verifier crate split. The forcing is mutual: relations want `jolt-claims`, which
wants the upstream stack under it; the verifier split's remaining moves are blocked by per-op
verifier code that only the relation migration removes; and the adoption stages themselves are
ordered by upstream's own crate graph (`jolt-dory` depends on `jolt-openings`, `jolt-poly`,
`jolt-transcript`, `jolt-akita` at `f3265b5a`). This spec is that single order, stage by stage,
each stage one PR with stated gates.

## Intent

### Goal

Execute nine stages (0–8) that end with: the proof-system toolkit as vendored, rev-pinned
upstream crates plus named atlas-only crates; every algebraic-identity op defined by a symbolic
relation citing its `atlas_trace::semantics` function, with the verifier side derived from the
relation and never hand-written; and `jolt-atlas-verifier` holding all verification, its
dependency cone prover-free at crate level and lint level.

### Invariants

Each invariant names the mechanism that holds it.

- **Vendoring, never copying**: each adopted crate is vendored whole at a recorded upstream rev
  under `third_party/` with a root `Cargo.toml` patch (the Akita branch's `jolt-field`
  precedent); one rev for the whole tree; pin bumps are deliberate PRs linking the upstream
  diff. Mechanism: `third_party/PINS.md` + review; the split-pin defect (`b8f0f51`/`76bb3a4`
  arkworks locks) is removed at stage 0.
- **Per-stage gates**: every stage passes the soundness sweeps, the op-surface table, the zoo,
  one real-model prove & verify (GPT-2 Dory minimum), and — from stage 1 on — the
  `jolt-atlas-verifier` isolation build. Mechanism: CI plus the heavy job, named per stage PR.
- **No stage is silently protocol-breaking**: stages declare behavior-preserving or
  protocol-breaking. Behavior preservation is checked **across the stage, not within one
  build**: pinned pre-stage proof-bytes and transcript digests from deterministic zoo inputs,
  reproduced byte for byte (the in-build harness cannot see a coordinated prover+verifier
  change). Protocol-breaking stages bump the proof serialization version and re-pin. Mechanism:
  the pinned-digest checks.
- **For migrated ops the relation is the only verifier**: no hand-written
  `expected_output_claim` exists (the type is deleted — compiler); every relation carries a
  mandatory `atlas_trace::semantics` citation field (compiler) and its S ⊆ R note as doc
  (review, on the migration checklist). Prover fast paths are completeness-only and each has a
  named `relation_prover_equiv_*` differential test (review requirement).
- **Atlas deltas are re-ported, not lost**: enumerated per stage from the inventory's bucket
  notes, each re-applied, upstreamed, or retired with a stated reason. Mechanism: the checklist
  in each stage PR body.
- **The trace semantics contract is untouched**: fixed-point behavior, fused-rescale split,
  table generation, padding. Mechanism: importer goldens and the zoo, which re-execute it.
- **Every `joltworks` module named in `docs/upstream-inventory.md` is assigned to a stage**;
  none is left behind implicitly. Mechanism: the inventory's bucket table, updated per stage.

### Non-Goals

- Upstream's zkVM product crates (`jolt-prover`, `jolt-verifier`, tracer, program, host, SDK):
  they verify RISC-V execution, not ONNX graphs; our product layer stays ours.
- The lookup/one-hot family (Shout read-raf, booleanity, Hamming, gather, clamp, range checks)
  never becomes relations: a different protocol shape, already shared machinery.
- The `zk` feature and `jolt-blindfold`. The feature does not compile today (57 pre-existing
  errors) and no program stage repairs or chases it; **a dedicated zk-revival spec, written
  after this program completes**, owns restoring zero-knowledge on the end-state architecture
  (expected: adopt upstream `jolt-blindfold` rather than repair the fork). The zk-gated
  BlindFold `OutputClaimConstraint` machinery — a third encoding of the same identities (e.g.
  `ops/rsqrt.rs:228–289`) — is subsumed by relations in that spec. Recorded here so it is a
  decision, not an accident.
- Flipping the default field or PCS before stage 7's activation gates.
- Upstreaming our deltas (worthwhile, separate per-delta effort).
- A general constraint DSL: relations are sums of products over wires, degree ≤ 3 plus an
  engine-owned eq factor — what the existing op family uses, nothing more.

## Evaluation

### Acceptance Criteria

Per stage, in its own PR (gates from the Invariants apply to all; listed here is what is
specific):

- [ ] **Stage 0 — prerequisites** (independent of everything): arkworks lockfile pins unified;
  the five flagged upstream fixes triaged with an applicability verdict each, recorded in
  `docs/upstream-inventory.md` (remediation scheduled per finding — immediately if severe, else
  with the stage that adopts the affected subsystem); the J2 transcript bias analysis reviewed
  against our Blake2b/Keccak challenge sampling, conclusion recorded.
- [ ] **Stage 1 — verifier boundary**: `specs/verifier-boundary.md` implemented (its own spec
  and PR; listed here because later gates use its isolation build).
- [ ] **Stage 2 — `jolt-field`**: vendored pin current; **one field type for the tree** — the
  stage resolves how our `ark_bn254::Fr` (pinned arkworks fork, `joltworks::JoltField`) and
  upstream's `Ring`/bn254 wrapper unify (their backend wraps arkworks; the stage decides
  wrapper-at-the-boundary vs. a local `Ring` impl and records it). Behavior-preserving; BN254
  stays active, Fp128 becomes available.
- [ ] **Stage 3 — `jolt-transcript`**: codecs, bias budget, prover/verifier transcript split.
  Protocol-breaking. The PoW/codec cost is protocol-inherent: quantified in the PR and accepted
  on security grounds — no delta can remove it, and pretending one could would drop the
  hardening.
- [ ] **Stage 4 — `jolt-sumcheck` + `jolt-poly` + `jolt-utils`**: the engine-substrate swap,
  absorbing `msm`/`curve`/`config`/`utils` homes, with recorded dispositions for the
  shout/booleanity/Hamming family, `univariate_skip` (does upstream's round scheduler subsume
  it?), and the prefix–suffix lookup framework (atlas table *contents* stay atlas-only).
  Behavior-preserving modulo stage 3's format; atlas fold-routine deltas re-ported here.
- [ ] **Stage 5 — `jolt-openings`/`jolt-claims` + relations**: the time-boxed spike answers
  whether the claim lattice expresses atlas flows (node-indexed openings, multi-consumer
  evaluation reduction, deferred batches). Yes → adopt; no → the atlas accumulator stays as an
  atlas-only crate (upstream claim traits where possible) and `jolt-openings`/`jolt-claims`
  vendor later only as internal PCS dependencies behind an adapter. Then the **pilot
  relations**: `Mul` (`eq·(L·R − out·2^S − rem)`) and `Rsqrt`
  (`eq·(x̂·q + r_div − S³ + γ·(out² + r_sqrt − q))`) implemented on the adopted
  `Expr`/`SymbolicSumcheck` layer (or, on the "no" branch, a minimal same-`Term`-shape atlas
  IR, reason recorded), transcript-identical via pinned digests, hand-written
  `Params`/`Prover`/`Verifier` deleted in the same PR, S ⊆ R notes present. The opening-forgery
  sweep is ported onto the adopted claim system **before** the old accumulator is deleted.
- [ ] **Stage 6 — relation migrations**, one PR per family: fused-rescale (`Square`, `Cube`,
  einsum-dot rescale tie) → `Div`/`Iff`/Add–Sub operand ties → `Sum`/`MeanOfSquares` → softmax
  stages (post-#218, own decision point). Each deletes its pairs, pins digests, carries the
  migration checklist.
- [ ] **Stage 7 — PCS**: `jolt-dory`, `jolt-hyperkzg`, `jolt-akita` (dependencies satisfied by
  stages 2–5 per upstream's crate graph); atlas Dory perf deltas re-ported. **Akita/Fp128
  activation** additionally requires the structural suites as Fp128+Akita instances — the
  harness and sweeps are `Fr`/HyperKZG-hard-wired today and must become generic, with
  Akita-specific `mutate::Site`s — before any default flips; an honest real-model run alone
  does not cover malicious proofs on the new backend.
- [ ] **Stage 8 — verifier-cone closure**: with core emptied of per-op verifiers (stage 6), the
  `Verifier` state, `verify_iop`, deferred verification and verifier preprocessing (whose
  per-op registry dependency dissolved) move to `jolt-atlas-verifier`; `verify` re-attaches to
  `ONNXProof`; the relations and engine verifier side live in the crate. "Prover-free" holds at
  two enforceable levels: crate level (no `jolt-atlas-core`, no upstream prover product crates
  — cargo + isolation build) and lint level (a clippy `disallowed-methods`/`disallowed-types`
  bar on the vendored toolkit's prover entry points, e.g. `jolt-sumcheck::prover`, which
  upstream exports unconditionally — run in the isolation CI job).

Program definition of done (checkable): `joltworks` is gone — every inventory module replaced
or in a named atlas-only crate; `jolt-atlas-core` contains no verification logic;
`jolt-atlas-verifier` verifies a serialized proof from the model and public IO alone; isolation
build and prover-API lint green; `third_party/PINS.md` records every pin.

### Testing Strategy

The structural suites gate every stage (op-surface, three sweeps, zoo, importer goldens, heavy
job). New per this program: pinned cross-stage digests; `relation_prover_equiv_*` differential
tests for any specialized prover path; the forgery sweep ported to the adopted claim system in
stage 5; generic-PCS/field instantiations of the suites plus Akita `mutate::Site`s in stage 7.
No sweep tolerance may be added at any stage.

### Performance

Behavior-preserving stages: no regression on the README benchmark (`models/gpt2 --pcs dory`,
per-stage, measured with the `prove` example); atlas delta re-porting is the lever, and a stage
that cannot hold the line keeps the delta and documents the divergence in `PINS.md`.
Protocol-breaking stages may carry **protocol-inherent** costs (stage 3's PoW), quantified and
accepted explicitly. The relation engine's risk is the prover's sumcheck inner loop: mitigated
by compiling each static relation to a monomorphized fold at instance build, keeping the Gruen
split-eq factor engine-owned outside the relation, and the completeness-only fast-path valve —
upstream drives its Spartan/product sumchecks through this same symbolic layer at zkVM scale,
which is evidence, not proof, hence the per-stage gate. Akita activation is measured against
the Dory baseline and justified on its own terms, not required to win.

## Design

### Architecture

The order is forced, not chosen: stage 2 before 3–7 because everything bounds on the field
type; 3–4 before 5 because `jolt-claims` sits on `jolt-sumcheck`/`jolt-poly`/`jolt-transcript`;
5 before 6 because relations need the layer they're written in; 6 before 8 because the
verifier-crate moves are blocked by per-op verifier code until the migrations delete it; 2–5
before 7 because `jolt-dory`'s own `Cargo.toml` says so. Stage 0 is independent; stage 1 is
independent and early so its isolation build gates everything after it.

**A relation** is: named wires (output opening, operand openings, advice — each an
`OpeningId`), an expression (sum of terms, scalar or transcript-challenge coefficients times
products of wire references, degree ≤ 3), an engine-owned `eq(r,·)` factor, round/degree
geometry, and the mandatory semantics citation. The engine derives both directions and appends
exactly the declared wires on both sides (claim plumbing becomes declarative). Upstream's
`Expr<F, O, P, C>`/`Term`/`Source::{Opening,Derived,Challenge}` and `SymbolicSumcheck`
(`crates/jolt-claims`, at `f3265b5a`) are this design in production; stage 5 adopts it with
atlas `OpeningId`s as the opening source, falling back to a same-shape minimal IR only on
spike evidence.

Engine-driven and hand-written ops coexist between stages 5 and 8; the dispatch table says
which is which, and the op-surface table keeps one row per op throughout.

### Alternatives Considered

- **Three separate specs** (closed #396/#397 + the engine draft): rejected by experience — the
  documents encoded a circular discovery order and each review round surfaced another
  inter-spec contradiction; the order of operations is the design, so one document owns it.
- **Stay forked / cherry-pick / re-fork wholesale / git deps instead of vendoring / build our
  own claim ledger and IR**: rejected for the reasons recorded in closed #397 and the
  inventory — upstream re-architected, we demonstrably miss fixes, big-bang re-forks are
  unreviewable, mixed git/path resolution breaks, and upstream already ships the expression
  layer.
- **Relations before adoption** (the closed engine draft's implicit order): rejected — it
  front-runs the vendoring order and buys a fallback IR we expect to throw away.
- **Macro-generating prover/verifier pairs from one closure**: removes duplication but the
  identity stays code, not data — nothing to cite, audit, or interpret.

## Documentation

Per stage: `README.md` and `CLAUDE.md`/`AGENTS.md` crate maps; `third_party/PINS.md` (new,
stage 2); `docs/upstream-inventory.md` regenerated at stage 0 and updated per stage;
`CLAUDE.md` invariants gain "for migrated ops the relation is the source of truth" at stage 5;
the migration checklist (citation, S ⊆ R note, digests, pairs deleted) joins
`docs/pull-requests.md` at stage 5.

## Execution

Stage 0 can start today. Stage 1 is `specs/verifier-boundary.md`. Stages 2→5 are sequential;
the stage-5 spike may begin during stage 4. Stage 6's families are sequential PRs; stage 7 may
start after 5 and in parallel with 6; stage 8 lands last. Each stage PR enumerates its atlas
deltas from the inventory as a checklist.

## References

- `specs/verifier-boundary.md` (stage 1) and `specs/core-test-overhaul.md` (the gates).
- `docs/upstream-inventory.md`: divergence measurements, fix triage, bucket table.
- Upstream at `f3265b5a`: the 31-crate workspace; `crates/jolt-claims`
  (`Expr`/`SymbolicSumcheck`, `protocols/composed/relations.rs`); `crates/jolt-dory/Cargo.toml`
  (the stage-7 dependency facts); the vendored `jolt-field` precedent (pin `0e3faf07`).
- Closed PRs #396 and #397: design history and the review rounds whose findings this order
  absorbs.
