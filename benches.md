# GPT-2 proof benchmarks

Measured on 2026-11-23 on an Apple M3 Pro with 18 GB RAM and macOS 15.7.4.
Both CPU backends used the same build (Cargo `test` profile, Rust 1.99.1)
and five Rayon threads.

The input was one sequence of 16 synthetic tokens from the `prove` example.
Both backends used the full-size GPT-2 export with `logits` as the sole graph
output, so they proved the same statement. Each backend ran three times,
alternating which backend ran first in each pair. All proofs verified.

| Backend | Setup (s) | Prove (s) | Verify (s) | Input tokens/s | Proof (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BitZ CPU | 0.177 | 1.91 | 0.088 | 8.38 | 1.33 |
| Akita CPU | 4.64 | 58.72 | 3.35 | 0.27 | 4.69 |
| GPU | — | — | — | — | — |

Times are medians of the corresponding `prove` output lines. Input tokens
per second is 16 divided by the median proof generation time. The Akita
`prove` time includes the lazy setup of its commitment groups. GPU timings
are pending a GPU backend.

## Reproduce

Use the Python environment from `scripts/requirements-models.txt` and
run `python scripts/download_gpt2.py` to export the model.

```sh
model_dir=$(mktemp -d)
python - "$model_dir" <<'PY'
import sys
from pathlib import Path
from shutil import copyfile

sys.path.insert(0, "scripts")
from onnx_export_utils import keep_single_output

target = Path(sys.argv[1]) / "network.onnx"
copyfile("models/gpt2/network.onnx", target)
keep_single_output(target, "logits")
PY

cargo build --locked -p jolt-atlas-core --features akita --example prove --profile test
for pcs in bitz akita akita bitz bitz akita; do
  RAYON_NUM_THREADS=5 target/debug/examples/prove --pcs "$pcs" --model "$model_dir" --seq-len 16
done
```
