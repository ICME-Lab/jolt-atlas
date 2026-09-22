# Native ONNX frontend validation

Implementation commit `5868306460e3a145dc4d00673e3adcc507f4e7e4` adds a supported model frontend on the native
verifier prerequisite `f26127dc1978acdc02a89b3318b0e8779603bf5d`.
The next evidence commit adds only this record.

## Linux checks

| Check | Result |
|---|---|
| Core Clippy, all targets, ZK enabled | Passed with warnings denied |
| Native frontend tests | 10 passed, 0 failed, 0 ignored |
| Complete native proofs constructed and verified by the new tests | Normalization and softmax |
| Source files matching the tested Linux tree | 476 |
| Authenticated collected validation files | 14 |

Rust 1.95.0 on the retained Linux Xeon E7-8890 v4. The optimized repository
test profile retains debug assertions. Rayon uses eight workers and Cargo
tests use one thread. Workspace crates were cleaned before the source switch;
the shared dependency cache was held under an exclusive validation lock.
Commands and full logs are in this directory. Test/build durations in the
logs are validation costs, not model performance measurements.

The numerical controls compare every original model intermediate with
`Model::execute_graph`, including saturation, negative floor division,
logical mean counts, scalar layouts and rebased trigonometric/softmax blocks.
The normalization proof is serialized and decoded with the statement and
registration before verification. The same proof rejects different expected
constants, changed context, changed graph outputs, a substituted output
commitment and a freshly blinded statement. Repeated commitments keep the
public constant fixed and refresh the private input. Additional tests cover
input ordering, malformed topology and shapes, unsupported domains and
softmax centering overflow.

These are native frontend checks. The old public-IO attack tests are not
ported or called passing here. The proof's hidden IO must be linked to any
external expected values through the existing tensor boundary API. The
experimental HyperKZG dispatcher remains incomplete and its full suite has
not been repaired by this port.

## Source and attempt records

`port-sources.json` pins the original application files.
`tested-source-files.json` hashes the complete extracted Linux tree.
`source-audit.json` records comparison with the local implementation; the
subsequently expanded integration document is excluded from that source
comparison. No Rust source differs. The original collected evidence archive
authenticates as `0b550ecf028e686ac2a679b45cc42d694cf88b46bc18650441df4bbac28b43fe`.

The first strict Clippy attempt rejected missing public API documentation.
Its log is preserved. The correction adds documentation and no lint allowance.
A slow proxy transfer and an early launcher with no uploaded script produced
no build result; the accepted run used a complete, authenticated direct transfer
with the already pinned host key. No runtime test is removed or newly ignored.

The frontend is a new capability. An equivalent complete proof comparison is
unmeasured, so this record claims no improvement in proving time, verification
time, proof size or peak memory. See [the supported domains and remaining
integration work](../native-onnx-integration.md).
