#!/usr/bin/env python3
"""Download the BAAI/bge-small-en-v1.5 ONNX model using Hugging Face Optimum.

The script:
  1. Installs required pip packages (optimum, onnxruntime) if missing.
  2. Exports BAAI/bge-small-en-v1.5 to ONNX via `optimum-cli`.
  3. Renames model.onnx -> network.onnx (convention used by other models).

Output directory: onnx-tracer/models/bge-small-en-v1.5/
"""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "onnx-tracer" / "models" / "bge-small-en-v1.5"


def ensure_packages():
    """Install optimum[exporters] and onnxruntime if not already present."""
    pkgs = ["optimum[exporters]", "optimum[onnxruntime]"]
    print("Ensuring required Python packages are installed ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
    )


def export_model():
    """Export BAAI/bge-small-en-v1.5 to ONNX using optimum-cli."""
    model_onnx = MODEL_DIR / "model.onnx"
    if model_onnx.exists():
        print(f"model.onnx already exists at {MODEL_DIR}, skipping export.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} to ONNX -> {MODEL_DIR} ...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            MODEL_ID,
            str(MODEL_DIR),
        ],
    )

    if not model_onnx.exists():
        sys.exit(f"ERROR: Export finished but {model_onnx} not found.")

    print("Export complete.")


def rename_network():
    """Rename model.onnx -> network.onnx for compatibility."""
    src = MODEL_DIR / "model.onnx"
    dst = MODEL_DIR / "network.onnx"
    if dst.exists():
        print("network.onnx already exists, skipping rename.")
        return
    if not src.exists():
        sys.exit(f"ERROR: {src} not found, cannot rename.")
    print("Renaming model.onnx -> network.onnx ...")
    src.rename(dst)
    print("Done.")


def main():
    ensure_packages()
    export_model()
    rename_network()
    print(f"\nOK: {MODEL_ID} ONNX model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
