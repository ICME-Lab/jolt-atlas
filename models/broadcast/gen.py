#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing Add, Sub, and Mul operations with broadcasting.
The model takes an input tensor and applies operations with constants that require broadcasting.
Broadcasting can happen on up to 3 dimensions.
All dimensions are powers of two.

Examples of broadcasting (dimensions align from right):
- Input [4] + constant [1, 4] -> broadcast input to [1, 4]
- Input [2, 4] + constant [8, 2, 4] -> broadcast input to [8, 2, 4]
- Input [1] + constant [4, 8, 16] -> broadcast input to [4, 8, 16]
"""

import torch
import torch.onnx
import numpy as np

class SimpleAddSubMulBroadcast(torch.nn.Module):
    def __init__(self):
        super(SimpleAddSubMulBroadcast, self).__init__()
        # Define constants with different shapes to trigger broadcasting
        # Input will be [4], constants will have power-of-2 dimensions
        self.const1 = torch.nn.Parameter(torch.full((1, 4), 100.0), requires_grad=False)
        self.const2 = torch.nn.Parameter(torch.full((2, 1, 4), 50.0), requires_grad=False)
        self.const3 = torch.nn.Parameter(torch.full((2, 8, 4), 2.0), requires_grad=False)
    
    def forward(self, x):
        # x shape: [4]
        # Operations with broadcasting - each operation broadcasts to match output dims
        add_result = x + self.const1  # [4] + [1, 4] -> [1, 4]
        sub_result = add_result - self.const2  # [1, 4] - [2, 1, 4] -> [2, 1, 4]
        mul_result = sub_result * self.const3  # [2, 1, 4] * [2, 8, 4] -> [2, 8, 4]
        return mul_result

def main():
    # Create model instance
    model = SimpleAddSubMulBroadcast()
    model.eval()
    
    # Create dummy input with shape [4]
    # Use small input values to ensure no overflow
    dummy_input = torch.randn(4) * 10.0  # Values roughly in range [-30, 30]
    
    # Export to ONNX
    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Use version 14 to avoid conversion issues
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    # Load and re-save to ensure all data is embedded (not external)
    import onnx
    from onnx import numpy_helper
    
    onnx_model = onnx.load(output_path)
    
    # Save with location=None to embed all data
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    
    # Clean up any external data files
    import os
    external_data_file = output_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)
        print(f"Removed external data file: {external_data_file}")
    
    print(f"ONNX model saved to {output_path}")
    print(f"Input shape: [4]")
    print(f"Output shape: [2, 8, 4]")
    print(f"\nModel operations (with broadcasting):")
    print(f"  1. Add: input[4] + const1[1, 4] -> [1, 4]")
    print(f"  2. Sub: result[1, 4] - const2[2, 1, 4] -> [2, 1, 4]")
    print(f"  3. Mul: result[2, 1, 4] * const3[2, 8, 4] -> [2, 8, 4]")
    print(f"\nExpected output for input x: (x + 100.0 - 50.0) * 2.0 = (x + 50.0) * 2.0")
    print(f"Output will be broadcasted across all [2, 8, 4] dimensions")
    
    # Test with sample input
    test_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output shape: {list(test_output.shape)}")
    print(f"  Output[0, 0, :]: {test_output[0, 0, :].tolist()}")
    print(f"  Expected: {[(x + 50.0) * 2.0 for x in test_input.tolist()]}")
    print(f"  (Same values broadcasted across all [2, 8] positions)")

if __name__ == "__main__":
    main()