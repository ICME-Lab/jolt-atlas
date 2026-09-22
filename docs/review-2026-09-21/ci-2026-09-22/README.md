# Native CI coverage and experimental ONNX status

The main workflow previously tested only `jolt-atlas-core --features zk`. That exercises the experimental ONNX dispatcher and misses the native BlindFold tests in `joltworks`. The current dispatcher references removed APIs on main and the ordinary branches. On the native branches it compiles, but fails tests for missing or incomplete relations.

The main `Clippy & Test (zk)` matrix entry now checks and tests `joltworks --features zk`. The original ONNX Clippy and test commands remain in a separately named experimental job. Both commands are attempted, tests use `--no-fail-fast`, and the job summary reports both outcomes. A failed command leaves that experimental job visibly failed, with job-level `continue-on-error` so it does not fail the main workflow. This is an explicit CI scope change, not a completion of the experimental proof system. No tests are deleted or marked ignored and no proof relation is relaxed.

The ordinary workspace, model proofs, formatting and WebAssembly jobs remain. Checkout is updated from the obsolete v3 action to v7.0.1, using Node 24. Actionlint v1.7.12 accepts the final workflow. The documentation branch also adopts the two existing `values` iterator lint repairs from the correctness branch, without changing iteration order or mutations.

## Fresh Linux validation

Rust 1.95.0, strict warnings, eight Rayon workers and four test workers on the retained Xeon host. Workspace crates are cleaned between the two source archives under an exclusive build lock. Sources and patches are checked by their SHA-256 identities before use. The test profile retains optimization and debug assertions.

| Source | Native all-target Clippy | Native ZK library tests | Documentation tests |
|---|---|---:|---|
| Common correctness branch | Pass | 191 passed | 1 passed, 1 existing ignored |
| Documentation branch plus iterator fixes | Pass | 188 passed | 1 passed, 1 existing ignored |

The common native suite was run before the final workflow split; its native commands and Rust sources are identical. The documentation source includes the final workflow and iterator fixes. The source JSON and patches pin both tested trees. The difference in library counts reflects the common correctness changes already present in #369.

The seven implementation branches change only `.github/workflows/rust.yml` in this follow-up. Their earlier native validation retains its exact Rust source scope. The documentation PR also contains the two iterator changes and this updated archive.

`validation/` contains the complete logs and command summaries. `collected.json` authenticates all eight files against hashes read from the producing host. CI reruns on GitHub establish current per-PR workflow status; these local results do not claim completion of the experimental dispatcher or a new performance gain.
