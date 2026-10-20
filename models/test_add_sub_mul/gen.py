#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing Add, Sub, and Mul operations.
The model takes an input vector of size [1<<16] = 65536 and applies:
1. Add: input + const1
2. Sub: (input + const1) - const2
3. Mul: ((input + const1) - const2) * const3 (output)

Values are chosen to avoid overflow with 64-bit signed integers.
"""

import torch
import torch.onnx
import numpy as np

class SimpleAddSubMul(torch.nn.Module):
    def __init__(self):
        super(SimpleAddSubMul, self).__init__()
        # Use small constants to avoid overflow
        # For 64-bit signed integers, max is ~9 * 10^18
        # With vector size 65536, we use very small values
        self.const1 = torch.nn.Parameter(torch.full((65536,), 100.0), requires_grad=False)
        self.const2 = torch.nn.Parameter(torch.full((65536,), 50.0), requires_grad=False)
        self.const3 = torch.nn.Parameter(torch.full((65536,), 2.0), requires_grad=False)
    
    def forward(self, x):
        # x shape: [65536]
        # Add operation
        add_result = x + self.const1
        # Sub operation
        sub_result = add_result - self.const2
        # Mul operation (output)
        mul_result = sub_result * self.const3
        return mul_result

def main():
    # Create model instance
    model = SimpleAddSubMul()
    model.eval()
    
    # Create dummy input with shape [65536]
    # Use small input values to ensure no overflow
    dummy_input = torch.randn(65536) * 10.0  # Values roughly in range [-30, 30]
    
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
    print(f"Input shape: [65536] = [1 << 16]")
    print(f"Output shape: [65536]")
    print(f"\nModel operations:")
    print(f"  1. Add: input + 100.0")
    print(f"  2. Sub: (input + 100.0) - 50.0")
    print(f"  3. Mul: ((input + 100.0) - 50.0) * 2.0")
    print(f"\nExpected output for input x: (x + 100.0 - 50.0) * 2.0 = (x + 50.0) * 2.0")
    
    # Test with sample input
    test_input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 65531)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification (first 5 elements):")
    print(f"  Input: {test_input[:5].tolist()}")
    print(f"  Output: {test_output[:5].tolist()}")
    print(f"  Expected: {[(x + 50.0) * 2.0 for x in test_input[:5].tolist()]}")

if __name__ == "__main__":
    main()
