# Native BlindFold integration for ONNX models

## Status and review base

This is an implementation plan. The ONNX integration is not implemented by
this document, and the experimental ONNX ZK tests still fail.

The branch starts at native verifier PR [#375](https://github.com/ICME-Lab/jolt-atlas/pull/375),
commit `f26127dc1978acdc02a89b3318b0e8779603bf5d`. It targets `main`, with
#375 as its merge prerequisite. The dependency order is #369, #371, #372,
#373, #374, #375, then this integration. Work can proceed before those PRs
merge. The ordinary verifier PR #370 is a separate path.

Review the new work against the pinned prerequisite, rather than attributing
its native relations and optimizations to this integration. The initial
commit adds only this plan. No additional performance result is claimed.

## Problem and intended result

The experimental dispatcher in
[`jolt-atlas-core/src/onnx_proof/zk.rs`](../jolt-atlas-core/src/onnx_proof/zk.rs)
has missing operator relations and an incomplete migration of verification.
Its HyperKZG bundle also reveals a joint evaluation claim. Filling in the
operator stubs alone would not establish that the bundle hides all private
values.

The native APIs already provide Dory hiding openings, BlindFold constraints,
registered graphs and tensor boundaries. The new frontend should translate
an Atlas `Model` into those relations and verify execution against an
authenticated model registration. Here, translation means replacing each
model operation with one or more native proof operations while preserving
its integer computation and tensor layout.

The first supported privacy contract has public model structure, shapes,
scales and weights. Execution inputs and outputs can remain hidden through
their commitments. Any values explicitly declared public must be bound to
those same commitments. The registration and its public parameters must come
from the verifier's expected model. A registration supplied only by the
prover does not identify that model.

## Reuse map and remaining work

All paths below are relative to `joltworks/src/poly/commitment/dory/`.
Having a native relation does not establish that an ONNX operation can be
translated without checking its supported domains.

| Frontend responsibility | Existing native implementation | Integration still required |
|---|---|---|
| Shared hidden tensors and model registration | `native_graph.rs`, `native_registration.rs` | Deterministic model translation, node mapping and authenticated public constants |
| Add, subtract, multiplication and contractions | `native_add.rs`, `native_mul.rs`, `native_einsum.rs` | Exact shifts, clamping, broadcasting and supported accumulator bounds |
| Sum and mean of squares | `native_reduce.rs` | Logical reduction counts, padded shapes and rescaling |
| Softmax | `NativeGraph::softmax_with_checked_centering` | Scale and shape checks, exact row centering and integer-kernel parity |
| Sine and cosine | `NativeGraph::trig` | Periodic reduction, supported scales and exact table parity |
| Activations | `native_clamped_lookup.rs` | Register each exact activation table and validate its domain |
| Division and reciprocal square root | `native_division.rs`, `native_reciprocal.rs`, `native_rsqrt.rs` | Audit each Atlas division variant, signed behavior and exceptional inputs |
| Gather, layouts and logical operations | `native_hidden_lookup.rs`, `native_layout.rs`, `native_concat.rs`, `native_logic.rs` | Original-table identity, index rules, axis mapping, padding and masks |
| Public or external tensor boundaries | `native_boundary.rs` | Bind declared values, tensor positions and the caller's context to the original commitments |

## Implementation sequence

### 1. Translate a model without a private witness

- Accept the tracer's `Model`, rather than an application-specific export.
- Validate input ordering, node references, shapes, scales, arity and outputs.
  Report unsupported operations or parameter domains as errors.
- Derive native tensor identities and auxiliary operations deterministically
  from the model and public declarations. Do not choose the registered program
  from an observed private execution.
- Register constants, the graph, the declared public input policy and the
  original-to-native input/output mapping. Reuse public preprocessing while
  generating fresh blinds for private values.
- Compare every translated operation with the Atlas integer kernel, including
  logical versus padded dimensions. Do not silently substitute mathematical
  real-number operations for Atlas's integer computation.

### 2. Connect proof construction and verification

- Use `NativeGraphPreprocessing` and `NativeGraphProof` to construct the model
  proof. Use `NativeRegisteredGraph::verify` against the expected registration.
- Preserve the original commitments when operations share a tensor. Require
  boundary proofs where values cross into another proof or become public.
- Define the public API and serialized proof format explicitly. The old
  HyperKZG bundle is not wire-compatible with a native Dory proof.
- Audit the existing `prove_zk` and `verify_zk` call sites before choosing a
  compatibility wrapper or an explicit API migration. Do not silently change
  which inputs and outputs the verifier checks.
- Do not include the old cleartext joint evaluation claim or private tensor
  values in the new proof format. Record the complete public statement.

### 3. Preserve and extend the rejection tests

- Exercise honest construction, serialization and fresh verification for each
  supported operator and mixed graph.
- Preserve the security intent of
  `test_zk_constant_binding_rejects_cross_model_attack` and
  `test_zk_rejects_rebound_input_for_square_model`. Establish that the honest
  proof verifies first, then reject a changed model or verifier-facing input.
  A panic or unsupported path does not pass an attack regression.
- Reject changed outputs, registrations, graph mappings, shapes, tensor
  commitments and external contexts. Include independently valid proofs that
  use different hidden producers.
- Exercise false quotients, invalid remainders, saturation, scalar tensors,
  zero coordinates, padding and invalid gather indices.
- If bundle-specific test code needs migration, record the replacement test
  and the invariant it checks. Do not delete or ignore a failing control to
  produce a green result.

### 4. Validate models and restore blocking ONNX CI

- Add a small model fixture that runs the actual frontend and native proof
  APIs. It must not bypass translation by constructing a native graph in the
  test itself.
- Reproduce a pinned transformer export with a recorded operator inventory,
  model parameters, input policy and accepted serialized proof. Report setup,
  proving, verification, proof size and peak memory separately. Numerical
  execution alone is not proof completion.
- Keep the native ZK checks. Restore the ONNX ZK checks as blocking only after
  the complete applicable suite passes on this branch. Retain the current
  failure record and account for existing ignored tests explicitly.

## Completion gates

- [ ] Deterministic frontend translation and explicit supported domains.
- [ ] Registered native prove/verify API with a documented privacy boundary.
- [ ] Exact operator and mixed-model parity fixtures.
- [ ] Honest proofs and all required binding/rejection controls pass.
- [ ] Serialized proofs verify in a fresh process against the expected key.
- [ ] Strict native and core Clippy pass with ZK enabled.
- [ ] Native and ONNX ZK suites pass without newly ignored failing tests.
- [ ] ONNX CI is blocking and passes at the final review head.
- [ ] Complete model evidence records source, workload and measurement scope.

## Measurement policy

This is primarily a new frontend capability. A failing dispatcher is not a
valid timing baseline. Until an equivalent complete proof is measured on both
sources, proving time, verification time, proof size and memory improvements
are unmeasured. Keep the prerequisite PRs' individual measurements separate.
This plan does not complete an outer proof or private-model support.
