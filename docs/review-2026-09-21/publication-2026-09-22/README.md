# Remaining Atlas reviews opened, September 22

At the user’s request, all remaining branches now have PRs targeting main.

| PR | Branch | Review prerequisite |
|---|---|---|
| [#374](https://github.com/ICME-Lab/jolt-atlas/pull/374) | `review/native-prover` | #373 |
| [#375](https://github.com/ICME-Lab/jolt-atlas/pull/375) | `review/native-verifier` | #374 |
| [#376](https://github.com/ICME-Lab/jolt-atlas/pull/376) | `review/consolidation-records` | None for the documentation |

The code heads are unchanged from the main-target alignment. Their existing Linux validation retains its recorded scope. The documentation branch records main ancestry through a merge with an unchanged tree, then updates the index and descriptions. Its entire comparison with main is restricted to `docs/review-2026-09-21/`.

`code-prs.json` and `created-records.json` preserve creation readback. The documentation PR head in its creation receipt predates this final publication record. `records-history.json` preserves the documentation merge and tree comparison. `docs-validation.json` records checksum and scope checks immediately before opening the documentation PR. Original log and patch bytes retain their exact whitespace. Authored text passes the whitespace check. The top-level manifest covers the final archive.

No new runtime tests or performance measurements were run, and no compute instance was started. The combined implementation branches remain unmeasured as a single benchmark. Existing experimental ONNX limitations and dated CI results retain their original scope.
