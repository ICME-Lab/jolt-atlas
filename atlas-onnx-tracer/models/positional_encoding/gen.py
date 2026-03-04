#!/usr/bin/env python3
"""
Generate a positional-encoding-style ONNX model for testing both Sin and Cos.

Input is an "angles" tensor. The model computes:
    sin_part = sin(angles)
    cos_part = cos(angles)
    output = sin_part + cos_part

This keeps a single model output while still exercising both trigonometric ops.
"""

import os

import onnx
import torch
import torch.onnx


class PositionalEncodingTrig(torch.nn.Module):
    def __init__(self):
        super(PositionalEncodingTrig, self).__init__()

    def forward(self, angles):
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        return sin_part + cos_part


def main():
    model = PositionalEncodingTrig()
    model.eval()

    # Shape: [batch=1, sequence_length=8, half_dim=4]
    # Values represent precomputed angles (e.g. position * inverse_frequency).
    dummy_input = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.1, 0.01, 0.001],
                [2.0, 0.2, 0.02, 0.002],
                [3.0, 0.3, 0.03, 0.003],
                [4.0, 0.4, 0.04, 0.004],
                [5.0, 0.5, 0.05, 0.005],
                [6.0, 0.6, 0.06, 0.006],
                [7.0, 0.7, 0.07, 0.007],
            ]
        ],
        dtype=torch.float32,
    )

    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["angles"],
        output_names=["output"],
        dynamic_axes=None,
    )

    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    external_data_file = output_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)
        print(f"Removed external data file: {external_data_file}")

    print(f"ONNX model saved to {output_path}")
    print("Input shape: [1, 8, 4]")
    print("Output shape: [1, 8, 4]")
    print("\nModel operations:")
    print("  Sin: sin(angles)")
    print("  Cos: cos(angles)")
    print("  Add: sin(angles) + cos(angles)")

    test_input = dummy_input
    with torch.no_grad():
        output = model(test_input)
    print("\nTest verification:")
    print(f"  Input[0][0]: {test_input[0, 0].tolist()}")
    print(f"  Output[0][0] (sin + cos): {output[0, 0].tolist()}")


if __name__ == "__main__":
    main()
