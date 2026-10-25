#!/usr/bin/env python3
"""Download the Qwen2-0.5B ONNX model using Hugging Face Optimum.

Exports with ``--task text-generation`` (no KV cache) so the model has plain
``batch_size`` / ``sequence_length`` dynamic dims, at opset 18. The weights
land in ``model.onnx_data`` (2.5 GB, fp32). Output directory: models/qwen/
"""

from onnx_export_utils import REPO_ROOT, ensure_packages, export_onnx, rename_network

MODEL_DIR = REPO_ROOT / "models" / "qwen"


def main():
    ensure_packages()
    export_onnx("Qwen/Qwen2-0.5B", MODEL_DIR, "--task", "text-generation", "--opset", "18")
    rename_network(MODEL_DIR)
    print(f"\nOK: Qwen2-0.5B ONNX model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
