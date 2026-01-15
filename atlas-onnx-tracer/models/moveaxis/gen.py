#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing MoveAxis (Transpose) operation.
The model takes an input tensor and swaps two dimensions using the Transpose operation.
All dimensions are powers of two.

Example:
- Input [2, 4, 8] -> MoveAxis dims 1 -> 2 -> Output [2, 8, 4]
"""

import torch
import torch.onnx
import numpy as np

class SimpleMoveAxis(torch.nn.Module):
    def __init__(self):
        super(SimpleMoveAxis, self).__init__()
    
    def forward(self, x):
        # x shape: [2, 4, 8]
        # Swap dimensions 1 and 2, keeping dimension 0 in place
        # This is equivalent to permute(0, 2, 1) which gives [2, 8, 4]
        result = x.permute(0, 2, 1)
        return result

def main():
    # Create model instance
    model = SimpleMoveAxis()
    model.eval()
    
    # Create dummy input with shape [2, 4, 8]
    # Use small input values for clarity
    dummy_input = torch.randn(2, 4, 8) * 10.0  # Values roughly in range [-30, 30]
    
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
    print(f"Input shape: [2, 4, 8]")
    print(f"Output shape: [2, 8, 4]")
    print(f"\nModel operations:")
    print(f"  MoveAxis: input[2, 4, 8] -> permute(0, 2, 1) -> output[2, 8, 4]")
    print(f"  (Swapping dimensions 1 and 2, keeping dimension 0 unchanged)")
    
    # Test with sample input
    test_input = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification:")
    print(f"  Input shape: {list(test_input.shape)}")
    print(f"  Output shape: {list(test_output.shape)}")
    print(f"  Input[0, 0, 0]: {test_input[0, 0, 0].item()}")
    print(f"  Output[0, 0, 0]: {test_output[0, 0, 0].item()}")
    print(f"  (permute(0,2,1) maps input[i,j,k] to output[i,k,j])")

if __name__ == "__main__":
    main()
