#!/usr/bin/env python3
"""Download the GPT-2 ONNX model using Hugging Face Optimum.

The script:
  1. Installs required pip packages (optimum, onnxruntime) if missing.
  2. Exports GPT-2 to ONNX via `optimum-cli`.
  3. Copies model.onnx → network.onnx (convention used by other models).

Output directory: atlas-onnx-tracer/models/gpt2/
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "atlas-onnx-tracer" / "models" / "gpt2"


def ensure_packages():
    """Install optimum[exporters] and onnxruntime if not already present."""
    pkgs = ["optimum[exporters]", "optimum[onnxruntime]"]
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


def copy_network():
    """Copy model.onnx → network.onnx for compatibility."""
    src = MODEL_DIR / "model.onnx"
    dst = MODEL_DIR / "network.onnx"
    if dst.exists():
        print("network.onnx already exists, skipping copy.")
        return
    print("Copying model.onnx → network.onnx …")
    shutil.copy2(src, dst)
    print("Done.")


def main():
    ensure_packages()
    export_model()
    copy_network()
    print(f"\n✅  GPT-2 ONNX model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
