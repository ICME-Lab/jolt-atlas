#!/usr/bin/env python3
"""Download the GPT-2 ONNX model using Hugging Face Optimum.

Exports with the default (with-past) task at opset 18; the importer elides
the empty KV-cache inputs. Output directory: models/gpt2/
"""

from onnx_export_utils import REPO_ROOT, ensure_packages, export_onnx, rename_network

MODEL_DIR = REPO_ROOT / "models" / "gpt2"


def main():
    ensure_packages()
    export_onnx("gpt2", MODEL_DIR, "--task", "text-generation-with-past", "--opset", "18")
    rename_network(MODEL_DIR)
    print(f"\nOK: GPT-2 ONNX model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
