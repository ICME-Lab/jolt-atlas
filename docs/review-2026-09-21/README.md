# Atlas review consolidation

This index preserves the scope, source identities and validation for the September 21 consolidation. There are seven review units. The first five are opened together; the final two remain published branches until earlier reviews merge.

## Review queue

| Unit | Topic | Review | Description |
|---|---|---|---|
| 1 | Tensor correctness and commitments | [#369](https://github.com/ICME-Lab/jolt-atlas/pull/369) | [Review 1](reviews/01.md) |
| 2 | Ordinary verification and encoding | [#370](https://github.com/ICME-Lab/jolt-atlas/pull/370) | [Review 2](reviews/02.md) |
| 3 | Native foundations and boundaries | [#371](https://github.com/ICME-Lab/jolt-atlas/pull/371) | [Review 3](reviews/03.md) |
| 4 | Native tensor operators | [#372](https://github.com/ICME-Lab/jolt-atlas/pull/372) | [Review 4](reviews/04.md) |
| 5 | Table access and generation | [#373](https://github.com/ICME-Lab/jolt-atlas/pull/373) | [Review 5](reviews/05.md) |
| 6 | Native proving time and memory | [Published branch, held](https://github.com/ICME-Lab/jolt-atlas/tree/review/native-prover) | [Review 6](reviews/06.md) |
| 7 | Native verification and encoding | [Published branch, held](https://github.com/ICME-Lab/jolt-atlas/tree/review/native-verifier) | [Review 7](reviews/07.md) |

The first five units are the active review queue. Units 6 and 7 remain available for review as branches and will become PRs when earlier reviews merge. [Commit review map](commits.md) lists the component changes in order.

## Dependencies

```text
perf/round-2
  tensor-correctness
    ordinary-verifier
    native-foundations
      native-operators
        native-generation
          native-prover (held)
            native-verifier (held)
```

Each code branch has a separate review description with its actual base, classified changes, concrete correctness examples and original measurements. Recorded performance comparisons have different controls. No cumulative performance improvement is claimed for the assembled branches.

## Preservation

The original source branches remain available. Closed PRs retain their discussions and evidence. [sources.json](sources.json) maps all 67 originals (including #368, opened after the first inventory) to their replacements, with immutable reviewed heads. Antoine's #366 is included with authorship preserved, and its PR and original base branch remain untouched. Independent #287 and #288 are unchanged.

## Validation method

Checks use Rust 1.95.0 on Linux with `RUSTFLAGS=-D warnings`, eight Rayon workers and the repository's optimized test profile, including debug assertions. Every tracked source file is authenticated against its Git snapshot. Each source switch clears all local workspace crates, and builds run under one exclusive lock. Summaries identify the tested commit, commands, exit codes and timing. Build times are validation costs, not performance measurements.

An initial run shared Cargo artifacts across source trees and was discarded when a stale dependency surfaced. The published acceptance records use the subsequent clean, serialized runs. An earlier release-profile run also encountered two tests that require debug assertions; no assertion or proof predicate was weakened.

The ZK-enabled ONNX dispatcher is checked separately from native BlindFold. An exact unchanged-base comparison records the existing dispatcher errors and the activation-test diagnostics removed by the correctness branch. Passing native-library checks do not establish a complete private ONNX dispatcher or a complete outer proof.

## Selected Linux results

| Unit | Ordinary native library | ZK native library | Other checks |
|---|---:|---:|---|
| Correctness | Included in 356 workspace passes | Separate dispatcher fails to compile | Default workspace lint and formatting pass |
| Ordinary verifier | 174 parallel, 176 serial | 206 serial | Final affine suite 176 passes; three complete proofs match exactly; fixed tables match all reference entries |
| Native foundations | 162 | 213 | Workspace lint passes with and without ZK |
| Native operators | 163 | 259 | Workspace lint passes with and without ZK |
| Native generation | 163 | 295 | Native lint passes with ZK |
| Native prover | 188 | 331 | Workspace lint passes with and without ZK |
| Native verifier | 203 | 353 | Workspace lint passes with and without ZK, including the independent storage oracle |

Counts refer to their individual configurations, so shared tests occur in multiple rows. Formatting passes at all seven final heads. [branches.json](branches.json) records exact commands, heads, ignored cases and failures. The ordinary library matrix uses `d190e141`; final head `b70711b0` has a documented two-file backend/iterator cleanup, validated separately by backend tests, strict lint, complete proof comparisons and full fixed-table references. Its exact intervening diff is included.

The [post-validation source audit](after-validation-audit.json) confirms that all tracked files in twelve selected source snapshots, including their lockfiles, remained unchanged. Recorded dispatcher failures are described below and retained in the evidence. No green whole-workspace ZK test result is claimed.

## Soundness context

The already merged [#244](https://github.com/ICME-Lab/jolt-atlas/pull/244) is a concrete malicious-prover repair. A proof made with another model constant could carry public reduced claims replaced with the verifier's expected values; the old cleartext check did not bind those claims to the hidden openings. The fix added that binding and introduced `test_zk_constant_binding_rejects_cross_model_attack` to reject the patched bundle. This remains merged work and is not counted as a new consolidation result. Already merged [#247](https://github.com/ICME-Lab/jolt-atlas/pull/247) binds public input tensors into the transcript before challenges.

New native relations have their own rejection examples: different registered weights, a consumer for a different hidden producer, a false quotient with a consistent field equation, or a later tied maximum. These demonstrate constraints enforced by the new API. They are not presented as attacks on an earlier API that lacked that capability.

## Measurement scope

| Review unit | Concrete result to review | Scope |
|---|---|---|
| Ordinary verification | 23,032,033,118 to 8,109,826,681 cycles | Activation cache on the same saved proof; not outer proving time |
| Ordinary encoding | 5,188,295 to 2,941,543 bytes | Canonical proof recovered exactly |
| Native graph composition | 7.770 to 2.042 seconds | Four-table component, one observation per variant, both controls include required equality arguments |
| Native table access | 102.1 to 66.5 KiB | Eight-index gather, complete proof and statement |
| Native proving | 31.08 to 22.04 minutes | One complete policy observation per variant; whole-process peak did not fall |
| Native memory | 140.26 to 125.08 GiB | One complete action observation per variant; commitments plus proving rose from 1,710.55 to 1,721.48 seconds |
| Native verifier memory | 1.4848 to 0.9876 GiB | Same complete saved receipt, two observations per variant; separate host-memory experiment |
| Native verification | 101.053578 to 59.088359 seconds | Same saved proof, complete validation with rejection controls, two observations per variant |

These comparisons are retained evidence from the original PRs. Each review description gives its own control and limitations. Values in different rows do not share one baseline, and the ratios must not be multiplied. The fully assembled branches have not received a new complete-model timing comparison.

## Repeating a check

Use a fresh clone and check out the immutable head in `branches.json`. Install Rust 1.95.0, set `RUSTFLAGS=-D warnings`, `RAYON_NUM_THREADS=8`, `RUST_TEST_THREADS=4`, `CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0`, and run the commands in the linked summary. The test profile keeps optimization and debug assertions. Use a separate Cargo target directory for each source tree, or clean all local workspace packages before switching sources.

The retained runner scripts describe the producing host's archive layout. The Git heads and summary commands are sufficient for a fresh checkout; they do not require that host or its old build cache. The ordinary example `verifier_feature_parity` writes a canonical proof, compact proof and public IO into a fresh directory. Run it in parallel, serial and optional-feature builds, then compare all three files. Each invocation verifies both encodings and rejects a changed public output.

## Integration choices

- Preserve the serial tensor helpers already in round two, then add the missing round-two call sites and strict equal-length zip behavior.
- Keep both polynomial reference tests from #301; its final test relocation does not need a duplicate patch.
- Place vector registration and boundaries in foundations, scalar/tensor coverage in operators, and the uncommitted-witness control with its constructor in generation.
- Split #343 into shared range commitments in the prover branch and checked transport in the verifier branch.
- Move #344's activation-test constant repair to the common correctness prerequisite.
- Include the current #337 sparse-row cache and #363 fully populated index representation, including their latest evidence and regressions.
- Keep #356–#358 as separate commits for allocation/copy review. Their earlier combined complete-model observation did not improve overall proving.
- Repair #357's missing evaluation-trait import. No proof predicate or workflow check is relaxed.

## Experimental ONNX test result

After the compiler and lint repairs, the foundation head `0a7de79f` runs the separate core ZK suite: **127 passed, 26 failed, 29 ignored**. Failures include unimplemented operator paths, missing verifier relations and unsuccessful division/gather verification. The historical constant-binding and rebound-input regressions cannot complete through that dispatcher. They are historical explanations, not fresh passing soundness controls in this consolidation. The native registration and boundary rejection tests have their own passing records.

The exact unchanged base fails to compile with ZK, so these runtime failures are not labeled inherited by an exact-base runtime comparison. No failing test is disabled. Native API validation and compilation success do not establish a complete private ONNX dispatcher, a green whole-workspace ZK test job, or an outer proof.
