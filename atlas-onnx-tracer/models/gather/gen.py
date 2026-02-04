#!/usr/bin/env python3
"""
Generate a simple ONNX model for testing the Gather operation.
The model uses a Gather node to select specific elements from an input tensor.
"""

import torch
import torch.onnx
import onnx
import os

class SimpleGather(torch.nn.Module):
    def __init__(self):
        super(SimpleGather, self).__init__()
        # Register data as a constant buffer
        self.register_buffer('data', torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]))
    
    def forward(self, indices):
        # Gather operation: select elements from constant data using indices
        # This should be converted to a Gather node
        gathered = torch.index_select(self.data, dim=0, index=indices)
        return gathered

def main():
    # Create model instance
    model = SimpleGather()
    model.eval()
    
    # Create dummy input (indices)
    # Indices tensor with shape [4] (power of 2)
    dummy_input = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    
    # Export to ONNX
    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=['indices'],
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
    print(f"Data (constant): [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]")
    print(f"Input (indices) shape: [4]")
    print(f"Output shape: [4]")
    print(f"\nModel operations:")
    print(f"  1. Gather: select elements from constant data using input indices")
    print(f"  Expected to be converted to Gather node")
    
    # Test with sample input
    test_indices = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    with torch.no_grad():
        test_output = model(test_indices)
    print(f"\nTest verification:")
    print(f"  Indices: {test_indices.tolist()}")
    print(f"  Output: {test_output.tolist()}")
    print(f"  Expected: [10.0, 30.0, 50.0, 70.0] (elements at indices 0, 2, 4, 6)")

if __name__ == "__main__":
    main()