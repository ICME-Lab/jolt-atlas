# Spec: The Adoption Program — Upstream Toolkit, Expression Relations, Verifier Split

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Forpee                        |
| Created     | 2026-10-25                     |
| Status      | proposed                       |
| PR          | #400                           |
| Depends on  | —                              |
| Supersedes  | PR #399 (closed; its verifier-boundary spec is stage 1 here), PRs #396 (program portions) and #397 (closed), and the unopened `spec/expression-engine` draft — one linear order replaces four interdependent documents |

Status is `proposed` until the implementation merges, then `implemented`. A spec is frozen at
merge: it records what was decided and why at that point in time. The code is the source of
truth for current behavior, and nobody updates old specs when the code moves on.

## Summary

Four efforts were specified separately and turned out to be one program in a forced order:
retiring the `joltworks` fork by adopting upstream a16z/jolt's decomposed crates (31 crates,
re-architected claims/transcripts/sumcheck — `docs/upstream-inventory.md` is the factual
basis), replacing hand-written per-op prover/verifier pairs with symbolic relations (upstream's
`jolt-claims` already ships the `Expr`/`SymbolicSumcheck` layer this requires), completing the
prover/verifier crate split, and moving to the lattice stack (Fp128 + Akita) as the only field
and PCS. The forcing is mutual: relations want `jolt-claims`, which wants the upstream stack
under it; the verifier split's remaining moves are blocked by per-op verifier code that only the
relation migration removes; and the adoption stages themselves are ordered by upstream's own
crate graph (`jolt-akita` depends on `jolt-claims`, `jolt-crypto`, `jolt-field` at `f3265b5a`).
This spec is that single order, stage by stage, each stage one PR with stated gates. Its first
implementation PR is stage 1, the verifier boundary.

## Intent

### Goal

Execute nine stages (0–8) that end with: the proof-system toolkit as vendored, rev-pinned
upstream crates plus named atlas-only crates; every algebraic-identity op defined by a symbolic
relation bound to its `atlas_trace` op spec, with the verifier side derived from the relation
and never hand-written; `jolt-atlas-verifier` holding all verification, its dependency cone
prover-free at crate level and lint level; and Fp128 + Akita as the only field and PCS, with
BN254, Dory, HyperKZG and the `zk` feature gone from the tree.

### Invariants

Each invariant names the mechanism that holds it.

- **Vendoring, never copying**: each adopted crate is vendored whole at a recorded upstream rev
  under `third_party/` with a root `Cargo.toml` patch (the Akita branch's `jolt-field`
  precedent); one rev for the whole tree; pin bumps are deliberate PRs linking the upstream
  diff. Mechanism: `third_party/PINS.md` + review; the split-pin defect (`b8f0f51`/`76bb3a4`
  arkworks locks) is removed at stage 0.
- **Per-stage gates**: every stage passes the soundness sweeps, the op-surface table, the zoo,
  one real-model prove & verify (GPT-2 Dory minimum until stage 7a flips the default, GPT-2
  Akita after), and — from stage 1 on — the `jolt-atlas-verifier` isolation build. Mechanism:
  CI plus the heavy job, named per stage PR.
- **No stage is silently protocol-breaking**: stages declare behavior-preserving or
  protocol-breaking. Behavior preservation is checked **across the stage, not within one
  build** — the in-build zoo harness proves and verifies with the same code and cannot see a
  coordinated prover+verifier change. Mechanism: the cross-stage digest harness
  (`onnx_proof/testing/digests.rs`, created by stage 1), which pins, for each checked-in zoo
  model on its deterministic inputs, a hash of the serialized proof bytes and of the final
  transcript state. It pins the default backend at the time: HyperKZG until stage 7a lands
  (its setup is seeded, `hyperkzg/commitment_scheme.rs:38`), Akita from then on — including
  for any stage-6 family that lands after 7a. In every
  behavior-preserving stage the harness's pinned values are unchanged and the stage PR's first
  commit runs it green on the untouched tree, so the expected values provably predate the
  change. A protocol-breaking stage re-pins in a separate commit whose message names the cause.
  There is no proof-format version: the code is pre-production and old proofs need not verify.
  Classification per stage: 0, 1, 2, 4 (modulo stage 3's format), 5, 6, 7b and 8 are
  behavior-preserving; 3 and 7a (re-pins to Akita) are protocol-breaking. If stage 0 lands
  before stage 1's harness exists, its PR body records hand-run before/after hashes of the
  serialized proof for each checked-in zoo model on HyperKZG instead.
- **For migrated ops the relation is the only verifier**: no hand-written
  `expected_output_claim` exists (the type is deleted — compiler). Every relation is bound to
  the op it proves by `type Spec: atlas_trace::OpSpec` (e.g. `type Spec = atlas_trace::op::Mul`
  — compiler: the op must exist), and a test asserts each registered relation's `Spec` matches
  the `OpKind` it is dispatched under (`OpKind::of` is exhaustive). Each relation carries its
  S ⊆ R note as doc, naming the `atlas_trace::semantics` helper it relies on where one exists
  (e.g. `semantics::fixed::rebase` for fused rescale) — review, on the migration checklist. A
  relation bound to the right op with a wrong identity is caught by the soundness sweeps and
  the S ⊆ R review, not the compiler. Prover fast paths are completeness-only and each has a
  named `relation_prover_equiv_*` differential test (review requirement).
- **Migrations preserve the protocol**: a relation migration keeps the existing input claim,
  opening points and opening append order, so the digests reproduce. Changing an op's identity
  (e.g. to a zero-sum form) is a separate PR declared protocol-breaking. Mechanism: the digest
  harness.
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
- Zero-knowledge. The `zk` feature does not compile today (57 pre-existing errors under
  `cargo check -p jolt-atlas-core --features zk`) and stage 0 deletes it, with the BlindFold
  fork and the zk-gated `OutputClaimConstraint` machinery (e.g. `ops/rsqrt.rs:228–289`).
  Upstream's `jolt-blindfold` runs only on Dory/BN254 (`akita` and `zk` are mutually exclusive
  at `f3265b5a`), which this program removes. Lattice zero-knowledge is in-house research and a
  separate effort that builds on the end state.
- Backward compatibility of serialized proofs across stages (pre-production).
- Keeping BN254, Dory or HyperKZG as non-default backends after stage 7.
- GPU acceleration: a separate effort, and the expected source (with the lattice PCS) of
  Akita-era prover speedup.
- Upstreaming our deltas (worthwhile, separate per-delta effort).
- A general constraint DSL: relations are sums of products over wires, degree ≤ 3 plus an
  engine-owned eq factor — what the existing op family uses, nothing more.
- Minimizing verifier dependencies (no-std, an `atlas-trace` feature split): follow-ups.

## Evaluation

### Acceptance Criteria

The main list is **stage 1, the verifier boundary**, ticked by the stage 1 PR. It moves no
verification logic (see Design → Stage 1).

- [ ] `jolt-atlas-verifier` exists and owns `ONNXProof`, `Claims`, `ProofId`/`ProofType` and
  `ReducedOpeningProof` with their serialization (`proof_serialization.rs`). **Every field of
  `ONNXProof`, `ReducedOpeningProof` and `Claims` is private.** Each has a production
  `from_parts` taking every field (core's prover constructs `ReducedOpeningProof` including its
  `joint_opening_proof`) and production `&` read accessors for what core's free functions
  consume. Nothing about construction or reads is test-gated. `joltworks` types inside the
  proof (e.g. `SumcheckInstanceProof::compressed_polys`) keep their visibility; stages 3–4
  replace them.
- [ ] `cargo check -p jolt-atlas-verifier` builds in isolation with no `jolt-atlas-core`
  dependency; CI gains that build as a job.
- [ ] Core's inherent `ONNXProof` methods are free functions (`jolt_atlas_core::prove`,
  `prove_from_trace`, `verify` and their helpers) — both sides at once, because a type defined
  in another crate admits no inherent impls in core.
- [ ] A `test-util` feature adds `&mut` proof-field accessors; `testing/mutate.rs` uses only
  them. Hooks are inert unless invoked (examples build with dev-dependencies, so the benchmark
  binary carries the feature; it must measure the verifier downstream users build).
- [ ] `onnx_proof/testing/digests.rs` exists as specified in the Invariants, lands as the PR's
  first commit green on the pre-move tree, and passes unchanged at every later commit.
- [ ] `cargo nextest run --workspace` green with every sweep assertion and mutation/forgery
  site preserved in count and expectation; the real-model heavy job (GPT-2 Dory minimum)
  passes once on the branch.

### Later Stages

Each box is ticked by its own stage PR (gates from the Invariants apply to all; listed here is
what is specific).

- [ ] **Stage 0 — prerequisites** (stage 0 PR; independent of everything): arkworks lockfile
  pins unified; the five flagged upstream fixes triaged with an applicability verdict each,
  recorded in `docs/upstream-inventory.md` (remediation scheduled per finding — immediately if
  severe, else with the stage that adopts the affected subsystem); the J2 transcript bias
  analysis reviewed against our Blake2b/Keccak challenge sampling, conclusion recorded; the
  `zk` feature deleted from `jolt-atlas-core` and `joltworks` — `zk.rs`, the BlindFold fork
  code, the zk-gated `OutputClaimConstraint` machinery — and the zk paragraph removed from
  `CLAUDE.md`. If stage 1 lands first, it leaves `zk.rs`'s inherent-method call sites
  (`zk.rs:1601–1620`) as they are; the feature cannot regress. Behavior-preserving (digests,
  or hand-run proof hashes if the harness does not exist yet).
- [ ] **Stage 2 — `jolt-field`** (stage 2 PR): `jolt-field` vendored on main under
  `third_party/` at the Akita branch's pin (`0e3faf07`, bumped deliberately if needed);
  `joltworks::JoltField` implemented for upstream's `Fp128` (the `fp128-field` branch's shim is
  the starting point); the structural suites — op-surface, the three sweeps, the zoo, the
  digest harness — made generic over field and PCS, still instantiated for BN254 + HyperKZG/
  Dory. The BN254 scalar becomes upstream's `jolt_field::Fr` (the `bn254` feature): stages 3–4
  cannot run their BN254 gates otherwise, because upstream `jolt-poly` bounds on
  `jolt_field::JoltField` and `jolt-transcript`'s sealed `ChallengeCodec` is implemented for
  `jolt_field::Fr`, not `ark_bn254::Fr` (`jolt-transcript/src/challenge.rs` at `f3265b5a`) — a
  distinct Montgomery type that orphan rules and the seal keep atlas crates from bridging.
  HyperKZG and Dory expose `jolt_field::Fr` as `CommitmentScheme::Field` (so `PCS::Field = F`
  holds) and keep arkworks types internal, converting at the boundary. This work is temporary —
  stage 7b deletes it — and is the price of running stages 3–6 on a backend with a PCS.
  Behavior-preserving (digests): the conversion is a representation change, not an arithmetic
  one.
- [ ] **Stage 3 — `jolt-transcript`** (stage 3 PR): codecs, bias budget, prover/verifier
  transcript split. Protocol-breaking (re-pin). The PoW/codec cost is protocol-inherent:
  quantified in the PR and accepted on security grounds — no delta can remove it, and
  pretending one could would drop the hardening.
- [ ] **Stage 4a — `jolt-utils` and the recorded dispositions** (stage 4a PR): `jolt-utils`
  vendored and its allocator adopted; every module this stage names (`utils/*`, `msm/`,
  `config.rs`, `curve.rs`, the shout/booleanity/Hamming family, `univariate_skip`, the
  prefix–suffix lookup framework) has a recorded disposition in `docs/upstream-inventory.md` —
  the every-module-assigned invariant is the gate, not a promised destination. Behavior-preserving
  (digests).
- [ ] **Stage 4b — the engine swap onto `jolt-sumcheck` + `jolt-poly`** (one PR per op family,
  after a field-trait PR): split from stage 4 when the port met three facts the original entry
  did not anticipate. (1) `jolt-poly` bounds on upstream `jolt_field::JoltField`, which our
  generic `F: joltworks::field::JoltField` cannot carry — method names collide with upstream
  `Ring`/`Field`, and the `ark_bn254::Fr` kernel cannot implement a foreign trait — so the first
  PR unifies the field traits (our trait extends upstream's, the kernel stops being a
  `JoltField`, colliding methods resolve to upstream's). (2) `jolt-sumcheck` takes the facade
  `T: Transcript<Challenge = F>`, with no proof-of-work hook and full-field challenges, while
  stage 3's rounds grind on the split transcript and bind with the 125-bit `MontU128Challenge`.
  Both are met on the atlas side, so the vendored crate carries no delta: a scoped facade view of
  the split transcript grinds before each challenge it hands the engine, and a challenge's field
  image keeps its Montgomery limbs, so `JoltField::challenge_from_field` recovers the
  `MontU128Challenge` at no cost. The second PR swaps the batched-sumcheck driver onto upstream's
  `prove_batch`/`verify_compressed` with every existing instance running as an upstream batch
  member; it is protocol-breaking (upstream's absorption labels) and re-pins. (3) Adoption then
  ports every op protocol and the joltworks sumcheck subprotocols onto upstream's
  round-scheduler traits and `jolt-poly` types, so it lands op family by op family, each PR
  reproducing the digests and passing the performance gate. The operand polynomials stay ours
  where `jolt-poly`'s full-field binding measured slower than our 125-bit-challenge binding; the
  inventory records the measurement. Upstream's exceptional-challenge fix
  (`0337eda5`) and capped decoding of the sumcheck/polynomial proof types land with it. The
  split-eq delta is cheaper than the inventory feared — upstream `jolt-poly/split_eq.rs` already
  carries the Gruen machinery; remaining atlas fold-routine deltas are re-ported here.
- [ ] **Stage 5 — `jolt-openings`/`jolt-claims` + relations** (spike PR, then pilot PR): the
  spike answers whether the claim lattice expresses atlas flows, with a defined bar and box.
  **Pass** iff the `Mul` pilot relation, written as a `SymbolicSumcheck` over atlas
  `OpeningId`s, proves and verifies on a probe model (`testing/models.rs`) through the adopted
  claim system, **and** each stress flow is demonstrated as a working construct: a
  node-indexed opening fetched through the lattice, a multi-consumer evaluation reduction
  producing a `ReducedInstance`, and a deferred-batch registration verified after the node
  loop — "expressible on paper" is not a pass. The spike also confirms the layer carries a
  relation's input-claim expression separately from its summand (Design); if it carries only a
  summand, the input claim is layered on top. **Fail** evidence is the specific flow that
  cannot be expressed, recorded in the spike PR as the reason for the fallback. **Box: two
  weeks of focused effort**, after which the fallback (minimal same-`Term`-shape atlas IR) is
  taken without further debate; the fallback is deliberately cheap to exit later, so a wrong
  "no" is bounded, while an unboxed spike is the schedule hole this program exists to remove.
  Yes → adopt, vendoring `jolt-crypto` in the same PR (`jolt-openings` depends on it
  unconditionally at `f3265b5a`, default features off included); no → the atlas accumulator stays as an atlas-only crate (upstream claim traits
  where possible) and `jolt-openings`/`jolt-claims` vendor later only as internal PCS
  dependencies behind an adapter. Then the **pilot relations**, transcript-identical
  (digests): `Mul` in its fused form — input claim `q(r₀)·2^S + rem(r₀)`, where `q` is the
  **pre-clamp quotient opening** (`VirtualPoly::ClampAcc`-tied, as
  `fused_rebase::fused_input_claim` reads it today; the clamped node output would reject honest
  saturated executions, see `atlas-trace/src/op/fused.rs:33–39`), summand `eq(r₀,·)·L·R`
  (`ops/mul.rs:60–63, 227–243`), with the clamp lookup and the remainder range check remaining
  separate lookup-family protocols outside the relation; non-fused `Mul` (scale 0) takes the
  output opening as its input claim — and `Rsqrt`, its identity
  `x̂·q + r_div − S³ + γ·(out² + r_sqrt − q)` split into input claim and summand exactly as
  `ops/rsqrt.rs` performs it today. Both implemented on the adopted `Expr`/`SymbolicSumcheck`
  layer (or, on the "no" branch, the minimal atlas IR, reason recorded), hand-written
  `Params`/`Prover`/`Verifier` deleted in the same PR, S ⊆ R notes present. The
  opening-forgery sweep is ported onto the adopted claim system **before** the old accumulator
  is deleted.
- [ ] **Stage 6 — relation migrations** (one PR per family): fused-rescale (`Square`, `Cube`,
  einsum-dot rescale tie) → `Div`/`Iff`/Add–Sub operand ties → `Neg`, `ScalarConstDiv` and the
  remaining small algebraic verifiers → `Sum`/`MeanOfSquares` → softmax. **Softmax's
  disposition is decided by the closing family PR's author on a simple rule**: if the #218
  aux-vector rework has merged by then, the softmax stages migrate as the final relation
  family; if not, softmax joins the stage-8 move-as-code list (like the lookup family) and its
  relation migration is recorded as a named post-program follow-up — either way it has exactly
  one checkable disposition. The closing family PR asserts completeness against the op-surface
  table: every op is relation-migrated, on the recorded lookup-family list, or on the explicit
  deferred list with its stage-8 move assignment — stage 8 cannot start on an unaccounted
  remainder. Each PR deletes its pairs, reproduces digests, carries the migration checklist.
- [ ] **Stage 7a — Akita becomes the PCS** (stage 7a PR): `jolt-akita` vendored (its
  `jolt-crypto` dependency with it on the stage-5 fallback path; the adoption path already
  vendored it), starting from the `akita-adapter`
  branch's `AkitaScheme`; atlas Akita deltas recorded. Activation gates, all before the default
  flips: the structural suites instantiated for Fp128 + Akita (generic since stage 2) with
  Akita-specific `mutate::Site`s — an honest real-model run alone does not cover malicious
  proofs on the new backend — and a real-model prove & verify. Then the default field/PCS flips
  to Fp128 + Akita, the heavy job and the README benchmark move to Akita, the digest harness
  pins Akita, and the cost against the Dory baseline is measured and recorded in the PR —
  accepted regardless. Protocol-breaking (re-pin to Akita).
- [ ] **Stage 7b — BN254 removal** (stage 7b PR): Dory, HyperKZG, the BN254 `JoltField` impl
  and every pairing/curve path deleted from atlas code. The removal gate is the resolved
  production dependency graph, not source text: whole-crate vendoring keeps upstream's
  feature-gated BN254 backend in `third_party/` (`jolt-field`'s optional `ark-bn254` under its
  `bn254` feature), so the check is that `cargo tree` for every workspace member, with BN254
  features disabled, contains no `ark-bn254` or `ark-ec` — run in CI. The
  `a16z/arkworks-algebra` fork pin goes with them unless a resolved crate still requires
  `ark-ff` (recorded in `PINS.md`); the `prove` example's `--pcs` choices updated. Behavior-preserving
  (Akita digests).
- [ ] **Stage 8 — verifier-cone closure** (stage 8 PR). Stage 6 removes the
  *relation-family* verifiers but not everything: the lookup-family verifier-side handlers
  (ReLU/gather/clamp one-hot parameter builders, range-check verification) move as code, not
  relations, and `OperatorProofTrait` — which today mixes proving, verification and the
  committed-polynomial registry in one trait dispatched by `NodeCommittedPolynomials`
  (`ops/mod.rs:276–280`) — is split so the verifier-side methods and the model-only commitment
  metadata can cross without dragging prover code. Then the `Verifier` state, `verify_iop`,
  deferred verification and verifier preprocessing move to `jolt-atlas-verifier`; `verify`
  re-attaches to `ONNXProof`; the relations and engine verifier side live in the crate.
  "Prover-free" holds at two enforceable levels: crate level (no `jolt-atlas-core`, no
  upstream prover product crates — cargo + isolation build) and item level. Clippy's
  `disallowed-methods`/`disallowed-types` cannot ban a module path, so the bar is **an
  enumerated list of concrete public prover functions and types per vendored crate** (e.g. the
  items `jolt_sumcheck::prover` exports, which upstream ships unconditionally), maintained as a
  checklist item of the stage that vendors each crate, **plus** a module-path import check in
  the isolation CI job (reject `::prover` / `::witness` path segments appearing in
  `jolt-atlas-verifier` sources) as the backstop for items the enumeration misses.
  Behavior-preserving (digests).

Program definition of done (checkable): `joltworks` is gone — every inventory module replaced
or in a named atlas-only crate; `jolt-atlas-core` contains no verification logic;
`jolt-atlas-verifier` verifies a serialized proof from the model and public IO alone; Fp128 +
Akita is the only field and PCS, with no `zk` code in the tree and no `ark-bn254`, `ark-ec` or
pairing code in atlas sources or the resolved dependency graph (vendored upstream crates may
carry disabled BN254 features, per stage 7b);
isolation build, the enumerated prover-item disallow list and the import-path check green;
`third_party/PINS.md` records every pin.

### Testing Strategy

The structural suites gate every stage (op-surface, three sweeps, zoo, importer goldens, heavy
job). Stage 1 changes none of them except `testing/mutate.rs`'s accessor rewiring; no new
tolerance. New per this program: the cross-stage digest harness (stage 1);
`relation_prover_equiv_*` differential tests for any specialized prover path; a test binding
each relation's `Spec` to its dispatched `OpKind` (stage 5); the forgery sweep ported to the
adopted claim system in stage 5; field/PCS-generic suites (stage 2) instantiated for Fp128 +
Akita with Akita `mutate::Site`s (stage 7a). No sweep tolerance may be added at any stage.

### Performance

Behavior-preserving stages: no regression on the README benchmark, measured with the `prove`
example per stage (`models/gpt2 --pcs dory` until stage 7a lands, Akita from then on, whatever
stage-6 families remain). The baseline
is that command on `main` immediately before the stage; the gate is the median of 3 runs of
total prove time within +3% of the baseline median. The README table is HyperKZG today; stage 1
re-measures it on Dory so the published numbers match the gate. The `tracing` span names the
table cites (`ONNXProof::prove`, `ONNXProof::verify`, `ONNXProof::iop`, …) are kept as
instrument labels when stage 1 turns the methods into free functions. Atlas delta re-porting is
the lever, and a stage that cannot hold the line keeps the delta and documents the divergence
in `PINS.md`. Stage 1 is code motion only; the `test-util` inertness criterion keeps its
measurement honest. Protocol-breaking stages may carry **protocol-inherent** costs
(stage 3's PoW), quantified and accepted explicitly. The relation engine's risk is the prover's
sumcheck inner loop: mitigated by compiling each static relation to a monomorphized fold at
instance build, keeping the Gruen split-eq factor engine-owned outside the relation, and the
completeness-only fast-path valve — upstream drives its Spartan/product sumchecks through this
same symbolic layer at zkVM scale, which is evidence, not proof, hence the per-stage gate. The
Akita switch is measured against the Dory baseline and recorded, not required to win: the
expected speedup comes from the lattice PCS together with GPU work outside this program.

## Design

### Architecture

The order is forced, not chosen: stage 2 before 3–7 because everything bounds on the field
type; 3–4 before 5 because `jolt-claims` sits on `jolt-sumcheck`/`jolt-poly`/`jolt-transcript`;
5 before 6 because relations need the layer they're written in; 6 before 8 because the
verifier-crate moves are blocked by per-op verifier code until the migrations delete it; 5
before 7 because `jolt-akita`'s own `Cargo.toml` depends on `jolt-claims`; 7a before 7b because
the default must flip before the old backends can go. Stages 0 and 1 are independent; stage 1
is early so its isolation build and digest harness gate everything after it.

**A relation** is: named wires (output opening, operand openings, advice — each an
`OpeningId`), an **input-claim expression** over openings at the incoming point(s), a
**summand expression** (sum of terms, scalar or transcript-challenge coefficients times
products of wire references, degree ≤ 3), an engine-owned `eq(r,·)` factor, round/degree
geometry, and the op binding `type Spec: atlas_trace::OpSpec`. The engine absorbs the input
claim, runs the sumcheck on the summand, and derives both directions, appending exactly the
declared wires at the existing opening points in the existing order (claim plumbing becomes
declarative). Every fused-rescale op already has this shape: input claim `q·2^S + rem` via
`fused_rebase::fused_input_claim`, product summand. Upstream's
`Expr<F, O, P, C>`/`Term`/`Source::{Opening,Derived,Challenge}` and `SymbolicSumcheck`
(`crates/jolt-claims`, at `f3265b5a`) are this design in production; stage 5 adopts it with
atlas `OpeningId`s as the opening source, falling back to a same-shape minimal IR only on spike
evidence.

Engine-driven and hand-written ops coexist between stages 5 and 8; the dispatch table says
which is which, and the op-surface table keeps one row per op throughout.

### Stage 1 — Verifier Boundary

Soundness depends only on what the verifier checks, but verification is interleaved with the
prover across `jolt-atlas-core`, so an auditor asking "what exactly does the verifier accept?"
has no boundary to stand on. Stage 1 creates the boundary and enforces it from day one; later
stages tighten it monotonically.

```
common ─▶ joltworks ─▶ jolt-atlas-verifier ─▶ jolt-atlas-core (prover, witness, testing)
             ▲                 ▲
             └── atlas-trace ──┘        (atlas-onnx-import unchanged)
```

What moves is exactly the no-core-dependency set, closed under field types: the proof's field
types (`EvalReductionProof`, `Openings`) are in `joltworks` and `ModelExecutionIO` in
`atlas-trace`, so nothing pulls core in. Everything else fails a dependency check today:

- *Verification logic* calls core's per-op `OperatorVerifier` — moving it is a cargo cycle. It
  stays in core as free functions over the moved type (forced by coherence anyway, and done for
  prove and verify symmetrically in one commit). The `#[cfg(test)]` lockstep comparison against
  `ProverDebugInfo` stays in core's free `verify`, untouched; `ProverDebugInfo` stays core-owned.
- *Verifier preprocessing* (`AtlasVerifierPreprocessing`) embeds `AtlasSharedPreprocessing`,
  whose `get_models_committed_polynomials` calls the per-op registry
  (`NodeCommittedPolynomials` in `ops/`) — the same cycle.

Those cycles dissolve at stage 8. Until then two reachable-but-unused exceptions to
"prover-free" are review-held, not compiler-held: `joltworks` prover APIs (the crate is
depended on whole) and `atlas-trace` execution (`Model::trace` shares a crate with the graph
and `semantics` the verifier legitimately needs).

**Field-access design.** All fields are private. `from_parts` takes every field (prover
construction is production code); reads used by core's free functions are public `&`
accessors (production code); `test-util` gates only `&mut` access, so the production proof API
is immutable after construction and the soundness harness reaches fields only through the
gated hooks.

### Alternatives Considered

- **Separate specs** (closed #396/#397/#399 + the engine draft): rejected by experience — the
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
- **Zero-sum relations** (`eq·(L·R − q·2^S − rem)` for Mul): rejected for the migration — it
  opens `q` and `rem` at the new sumcheck point, changing the initial claim, round messages and
  openings, so a pilot could not show it changed nothing; an identity change is its own
  protocol-breaking PR.
- **Citing an `atlas_trace::semantics` function path**: rejected — `semantics` holds helpers
  (`rebase`, `rsqrt`, tables), not op definitions (Add's lives in `op/linear.rs`), and a path
  string is not compiler-checked. The op-type binding is.
- **Keeping `ark_bn254::Fr` as the BN254 scalar through stage 7b** (skipping the stage-2
  adaptation because 7b deletes it): rejected — upstream `jolt-poly` and `jolt-transcript`
  accept only `jolt_field` types, so stages 3–4 would have no BN254 backend to gate on, and
  Fp128 has no PCS before Akita. The throwaway adaptation is cheaper than a vendored
  compatibility delta in two upstream crates.
- **Keeping Dory/BN254 for zero-knowledge via `jolt-blindfold`**: rejected — the zk we want is
  lattice zk; BlindFold is Dory-only and ours does not compile.
- **A `verifier-only` feature flag instead of a crate** (stage 1): rejected — features are
  additive and give no auditable dependency edge.
- **Moving `verify` at stage 1 behind a dependency-inversion trait** (core registers per-op
  verifiers at runtime): rejected — a dynamic seam inside the audit surface that the end state
  doesn't want, bought only to claim an entry point early.

## Documentation

Per stage: `README.md` and `CLAUDE.md`/`AGENTS.md` crate maps (stage 1 adds
`jolt-atlas-verifier` and proof-type ownership); `CONTRIBUTING.md` CI table (stage 1's
isolation job); `third_party/PINS.md` (new, stage 2); `docs/upstream-inventory.md` regenerated
at stage 0 and updated per stage; `CLAUDE.md` loses the `zk` paragraph at stage 0 and gains
"for migrated ops the relation is the source of truth" at stage 5; the migration checklist
(op binding, S ⊆ R note, digests, pairs deleted) joins `docs/pull-requests.md` at stage 5; the
README benchmark and the heavy-job commands move to Akita at stage 7a.

## Execution

Stage 1's commits: (1) the digest harness, green on the untouched tree; (2) convert core's
inherent `ONNXProof` methods to free functions in place (mechanical, reviewable alone);
(3) create the crate, move the types + serialization, make fields private, add `from_parts` +
`&` accessors, update call sites; (4) `test-util` `&mut` accessors and `testing/mutate.rs`
rewiring; (5) the CI isolation job.

Program order: stages 0 and 1 can start today. Stages 2→5 are sequential; the stage-5 spike may
begin during stage 4. Stage 6's families are sequential PRs. Stage 7a may be developed and
land after 5, in parallel with 6; 7b lands after 7a. Gates follow the cutover: any stage-6
family landing after 7a reproduces Akita digests, runs the GPT-2 Akita heavy job, and measures
against the Akita baseline. Stage 8 lands last. Each stage PR enumerates its atlas
deltas from the inventory as a checklist.

## References

- `specs/core-test-overhaul.md`: the sweeps and zoo gating every stage.
- `docs/upstream-inventory.md`: divergence measurements, fix triage, bucket table.
- Upstream at `f3265b5a`: the 31-crate workspace; `crates/jolt-claims`
  (`Expr`/`SymbolicSumcheck`, `protocols/composed/relations.rs`); `crates/jolt-akita/Cargo.toml`
  (the stage-7 dependency facts); `crates/jolt-prover/Cargo.toml` (`akita` and `zk` mutually
  exclusive); the vendored `jolt-field` precedent (pin `0e3faf07`).
- Closed PRs #396, #397 and #399: design history, including the review rounds that produced
  the coherence, field-access and cycle analyses and the orders this spec absorbs.
