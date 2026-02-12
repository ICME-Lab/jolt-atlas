#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing the Tanh (hyperbolic tangent) operation.
The model computes tanh(x) which maps input values to the range (-1, 1).
"""

import torch
import torch.onnx
import onnx
import os

class SimpleTanh(torch.nn.Module):
    def __init__(self):
        super(SimpleTanh, self).__init__()
    
    def forward(self, x):
        # Hyperbolic tangent: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        return torch.tanh(x)

def main():
    # Create model instance
    model = SimpleTanh()
    model.eval()
    
    # Create dummy input with simple shape [4]
    # Using inputs that show tanh's characteristic behavior
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
    print(f"  Tanh: tanh(input)")
    print(f"  Maps values to range (-1, 1)")
    
    # Test with sample input
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest verification:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output: {test_output.tolist()}")
    print(f"  Expected: approximately [-0.964, -0.762, 0.0, 0.762]")

if __name__ == "__main__":
    main()
