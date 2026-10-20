#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing the Rsqrt (reciprocal square root) operation.
The model computes 1 / sqrt(x) which should be optimized to a single Rsqrt node by the tracer.
"""

import torch
import torch.onnx
import onnx
import os

class SimpleRsqrt(torch.nn.Module):
    def __init__(self):
        super(SimpleRsqrt, self).__init__()
    
    def forward(self, x):
        # Reciprocal square root: 1 / sqrt(x)
        # This pattern should be recognized and converted to Rsqrt node
        sqrt_x = torch.sqrt(x)
        rsqrt_x = 1.0 / sqrt_x
        return rsqrt_x

def main():
    # Create model instance
    model = SimpleRsqrt()
    model.eval()
    
    # Create dummy input with simple shape [4]
    dummy_input = torch.tensor([1.0, 4.0, 9.0, 16.0])
    
    # Export to ONNX
    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
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
    print(f"Input shape: [4]")
    print(f"Output shape: [4]")
    print(f"\nModel operations:")
    print(f"  1. Sqrt: sqrt(input)")
    print(f"  2. Div: 1.0 / sqrt_result")
    print(f"  Expected to be optimized to Rsqrt node")
    
    # Test with sample input
    test_input = torch.tensor([1.0, 4.0, 9.0, 16.0])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output: {test_output.tolist()}")
    print(f"  Expected: [1.0, 0.5, 0.333..., 0.25] (1/sqrt of each input)")

if __name__ == "__main__":
    main()