#!/usr/bin/env python3
"""Download the GPT-2 ONNX model using Hugging Face Optimum.

The script:
  1. Installs required pip packages (optimum, onnxruntime, transformers) if missing.
  2. Exports GPT-2 to ONNX via `optimum-cli`.
  3. Renames model.onnx → network.onnx (convention used by other models).
  4. Saves GPT-2's tokenizer (tokenizer.json + friends) alongside the model — the
     `optimum.exporters.onnx` export step above doesn't write it, and benches that need
     real text (e.g. perplexity on WikiText) can't tokenize without it.

Output directory: atlas-onnx-tracer/models/gpt2/
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "atlas-onnx-tracer" / "models" / "gpt2"


def ensure_packages():
    """Install optimum[exporters], onnxruntime, and transformers if not already present."""
    pkgs = ["optimum[exporters]", "optimum[onnxruntime]", "transformers"]
    print("Ensuring required Python packages are installed …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
    )


def export_model():
    """Export GPT-2 to ONNX using optimum-cli."""
    model_onnx = MODEL_DIR / "model.onnx"
    if model_onnx.exists():
        print(f"model.onnx already exists at {MODEL_DIR}, skipping export.")
        return

    print(f"Exporting GPT-2 to ONNX → {MODEL_DIR} …")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            "gpt2",
            str(MODEL_DIR),
        ],
    )

    if not model_onnx.exists():
        sys.exit(f"ERROR: Export finished but {model_onnx} not found.")

    print("Export complete.")


def rename_network():
    """Rename model.onnx → network.onnx for compatibility."""
    src = MODEL_DIR / "model.onnx"
    dst = MODEL_DIR / "network.onnx"
    if dst.exists():
        print("network.onnx already exists, skipping rename.")
        return
    if not src.exists():
        sys.exit(f"ERROR: {src} not found, cannot rename.")
    print("Renaming model.onnx → network.onnx …")
    src.rename(dst)
    print("Done.")


def save_tokenizer():
    """Save GPT-2's tokenizer (tokenizer.json + friends) into MODEL_DIR."""
    tokenizer_json = MODEL_DIR / "tokenizer.json"
    if tokenizer_json.exists():
        print("tokenizer.json already exists, skipping.")
        return
    from transformers import AutoTokenizer

    print(f"Saving GPT-2 tokenizer → {MODEL_DIR} …")
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(MODEL_DIR)
    if not tokenizer_json.exists():
        sys.exit(f"ERROR: tokenizer save finished but {tokenizer_json} not found.")
    print("Tokenizer saved.")


def main():
    ensure_packages()
    export_model()
    rename_network()
    save_tokenizer()
    print(f"\n✅  GPT-2 ONNX model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
