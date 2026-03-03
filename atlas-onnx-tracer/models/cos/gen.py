#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing the Cos operation.
The model computes cos(x).
"""

import os

import onnx
import torch
import torch.onnx


class SimpleCos(torch.nn.Module):
    def __init__(self):
        super(SimpleCos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


def main():
    model = SimpleCos()
    model.eval()

    dummy_input = torch.tensor([-2.0, -1.0, 0.0, 1.0])

    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["input"],
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
    print("Input shape: [4]")
    print("Output shape: [4]")
    print("\nModel operations:")
    print("  Cos: cos(input)")

    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    with torch.no_grad():
        test_output = model(test_input)
    print("\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output: {test_output.tolist()}")


if __name__ == "__main__":
    main()
