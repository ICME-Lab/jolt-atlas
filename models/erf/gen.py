#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing the Erf (error function) operation.
The model computes erf(x) which maps input values to the range (-1, 1).
"""

import os

import onnx
import torch
import torch.onnx


class SimpleErf(torch.nn.Module):
    def __init__(self):
        super(SimpleErf, self).__init__()

    def forward(self, x):
        # Error function
        return torch.erf(x)


def main():
    # Create model instance
    model = SimpleErf()
    model.eval()

    # Create dummy input with simple shape [4]
    # Using inputs that show erf's characteristic behavior
    dummy_input = torch.tensor([-2.0, -1.0, 0.0, 1.0])

    # Export to ONNX
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

    # Load and re-save to ensure all data is embedded (not external)
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    # Clean up any external data files (safety measure)
    external_data_file = output_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)
        print(f"Removed external data file: {external_data_file}")

    print(f"ONNX model saved to {output_path}")
    print("Input shape: [4]")
    print("Output shape: [4]")
    print("\nModel operations:")
    print("  Erf: erf(input)")
    print("  Maps values to range (-1, 1)")

    # Test with sample input
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    with torch.no_grad():
        test_output = model(test_input)
    print("\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output: {test_output.tolist()}")
    print("  Expected: approximately [-0.995, -0.843, 0.0, 0.843]")


if __name__ == "__main__":
    main()
