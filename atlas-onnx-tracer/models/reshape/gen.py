#!/usr/bin/env python3
"""
Generate a simple ONNX model that triggers Reshape operations.
When tensors with incompatible shapes are used in operations,
ONNX automatically inserts Reshape nodes. This is similar to the
broadcast model but focuses on reshape rather than broadcasting.

Example: Input [4] used with constant [1, 4] will trigger a reshape from [4] to [1, 4]
"""

import torch
import torch.onnx
import numpy as np

class SimpleReshapeTest(torch.nn.Module):
    def __init__(self):
        super(SimpleReshapeTest, self).__init__()
        # Constants with different shapes to trigger reshaping
        # Const with shape [1, 4] - when added to [4], input will be reshaped
        self.const1 = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), requires_grad=False)
        # Const with shape [2, 1, 4] - will trigger further reshaping
        self.const2 = torch.nn.Parameter(torch.ones(2, 1, 4) * 5.0, requires_grad=False)
    
    def forward(self, x):
        # x shape: [4]
        # Add with const1 [1, 4] - this should trigger reshape of x from [4] to [1, 4]
        result1 = x + self.const1  # [4] + [1, 4] -> reshape x to [1, 4], then add -> [1, 4]
        
        # Add with const2 [2, 1, 4] - this should trigger reshape from [1, 4] to [2, 1, 4]
        result2 = result1 + self.const2  # [1, 4] + [2, 1, 4] -> reshape to [2, 1, 4], then add -> [2, 1, 4]
        
        return result2

def main():
    # Create model instance
    model = SimpleReshapeTest()
    model.eval()
    
    # Create dummy input with shape [4]
    dummy_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
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
    print(f"Output shape: [2, 1, 4]")
    print(f"\nModel operations (with implicit reshapes):")
    print(f"  1. Reshape: input[4] -> [1, 4] (implicit)")
    print(f"  2. Add: [1, 4] + const1[1, 4] -> [1, 4]")
    print(f"  3. Reshape: [1, 4] -> [2, 1, 4] (implicit via broadcasting)")
    print(f"  4. Add: [2, 1, 4] + const2[2, 1, 4] -> [2, 1, 4]")
    print(f"\nExpected: ONNX should insert Reshape nodes where dimension changes occur")
    
    # Test with sample input
    test_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output shape: {list(test_output.shape)}")
    print(f"  Output[0, 0, :]: {test_output[0, 0, :].tolist()}")
    print(f"  Output[1, 0, :]: {test_output[1, 0, :].tolist()}")
    expected = [(x + c + 5.0) for x, c in zip(test_input.tolist(), [1.0, 2.0, 3.0, 4.0])]
    print(f"  Expected: {expected}")

if __name__ == "__main__":
    main()
