#!/usr/bin/env python3
"""Download the Qwen2-0.5B ONNX model using Hugging Face Optimum.

The script:
  1. Installs required pip packages (optimum, onnxruntime) if missing.
  2. Exports Qwen2-0.5B to ONNX via `optimum-cli` (without KV-cache).
  3. Renames model.onnx → network.onnx (convention used by other models).

Output directory: atlas-onnx-tracer/models/qwen/
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "atlas-onnx-tracer" / "models" / "qwen"


def ensure_packages():
    """Install optimum[exporters] and onnxruntime if not already present."""
    pkgs = ["optimum[exporters]", "optimum[onnxruntime]"]
    print("Ensuring required Python packages are installed …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
    )


def export_model():
    """Export Qwen2-0.5B to ONNX using optimum-cli (without KV-cache).

    Uses --task text-generation (not text-generation-with-past) so the
    exported model has simple dynamic dims (batch_size, sequence_length)
    without Min(...) shape expressions that Tract cannot parse.
    """
    model_onnx = MODEL_DIR / "model.onnx"
    if model_onnx.exists():
        print(f"model.onnx already exists at {MODEL_DIR}, skipping export.")
        return

    print(f"Exporting Qwen2-0.5B to ONNX → {MODEL_DIR} …")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            "Qwen/Qwen2-0.5B",
            "--task",
            "text-generation",
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


def main():
    ensure_packages()
    export_model()
    rename_network()
    print(f"\n✅  Qwen2-0.5B ONNX model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
