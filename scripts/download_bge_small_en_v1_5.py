#!/usr/bin/env python3
"""Download the BAAI/bge-small-en-v1.5 ONNX model using Hugging Face Optimum.

Exports at opset 14 and keeps a single graph output (see
``onnx_export_utils.keep_single_output``). Output directory:
models/bge-small-en-v1.5/
"""

from onnx_export_utils import (
    REPO_ROOT,
    ensure_packages,
    export_onnx,
    keep_single_output,
    rename_network,
)

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_DIR = REPO_ROOT / "models" / "bge-small-en-v1.5"


def main():
    ensure_packages()
    export_onnx(MODEL_ID, MODEL_DIR, "--task", "feature-extraction", "--opset", "14")
    if (MODEL_DIR / "model.onnx").exists():
        keep_single_output(MODEL_DIR / "model.onnx", "token_embeddings")
    rename_network(MODEL_DIR)
    print(f"\nOK: {MODEL_ID} ONNX model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
