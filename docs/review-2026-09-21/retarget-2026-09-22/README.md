# Retarget Atlas reviews to main

PR #288 merged on 2026-09-22 at 08:11:51 UTC as `e6cadf160dc8b6fa8b7a183ff24ce71b6ef84518`. Its tree equals the earlier round-two base exactly. All five open authored PRs now target `main`; their prerequisite order remains documented in their bodies.

| Branch | Previously tested head | Current head | Source tree changed |
|---|---|---|---|
| `review/tensor-correctness` | `0b28a6baff0ee3bf499726c0d3e63d703f3b007b` | `d95b380893393b5a22f220cd993e23038f0cf27c` | No |
| `review/ordinary-verifier` | `b70711b0499dec29badc4d70152c2521b86ed91f` | `3bc92a8328a7f15bb7ced362ad50b4f19b65742d` | No |
| `review/native-foundations` | `0a7de79f358c86ddad43c21cc2431086e82296f3` | `0c1639adb04721b6195e14ecfb1ccd8c76608375` | No |
| `review/native-operators` | `fb1b116c00b97e0a6855bb8517e4f93b33561bb4` | `5a2e52eaf1751385aa589c508176492ae995fef2` | No |
| `review/native-generation` | `02163ea6302999fc84fb49468c4a064ad3f4466d` | `1bde949f18edcb7b1f59d52b457c105e90eea55d` | No |
| `review/native-prover` | `6d3e3a24daf3a120baa05711e8ebac28d801daa9` | `16e21ae947d50edf8412304db6a184f1ba74ae73` | No |
| `review/native-verifier` | `2e415e2a8bfd71678b7b62766a149b259dc166ce` | `739e2255701eab85b240f9ce33e75e865ea7d585` | No |

The correctness branch records the squash merge with the `ours` merge strategy only after verifying that main's tree equals its already-contained round-two baseline. Ordinary prerequisite merges propagate that history to the other branches. No source file changed, no history was force-pushed, and the existing source validation retains its original scope.

The saved before/after GitHub snapshots retain full descriptions, labels, heads and targets. The source-tree assertions and merge ancestry are in [history-alignment.json](history-alignment.json). To repeat either source check, run `git diff OLD_HEAD NEW_HEAD --exit-code` and `git merge-base --is-ancestor e6cadf160dc8b6fa8b7a183ff24ce71b6ef84518 NEW_HEAD` in an Atlas clone.
