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
        # Embedding table - 2D tensor [65, 64]
        self.embedding = torch.nn.Embedding(65, 64)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(65 * 64, dtype=torch.float32).reshape(65, 64)
            )
    
    def forward(self, indices):
        # Gather operation: embedding lookup
        # indices shape: [1, 64], embedding shape: [65, 64]
        # Output shape: [1, 64, 64]
        return self.embedding(indices)

def main():
    # Create model instance
    model = SimpleGather()
    model.eval()
    
    # Create dummy input (indices)
    # Indices tensor with shape [1, 64]
    dummy_input = torch.tensor([
        [0, 3, 7, 2, 5, 1, 4, 6,
         15, 14, 13, 12, 11, 10, 9, 8,
         5, 5, 5, 5, 5, 5, 5, 5,
         0, 1, 2, 3, 4, 5, 6, 7,
         64, 63, 62, 61, 60, 59, 58, 57,
         56, 55, 54, 53, 52, 51, 50, 49,
         48, 47, 46, 45, 44, 43, 42, 41,
         40, 39, 38, 37, 36, 35, 34, 33]
    ], dtype=torch.long)
    
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
    print("Input shape: [1, 64] -> Output shape: [1, 64, 64]")
    
    # Test with sample input
    test_indices = torch.tensor([
        [0, 3, 7, 2, 5, 1, 4, 6,
         15, 14, 13, 12, 11, 10, 9, 8,
         5, 5, 5, 5, 5, 5, 5, 5,
         0, 1, 2, 3, 4, 5, 6, 7,
         64, 63, 62, 61, 60, 59, 58, 57,
         56, 55, 54, 53, 52, 51, 50, 49,
         48, 47, 46, 45, 44, 43, 42, 41,
         40, 39, 38, 37, 36, 35, 34, 33]
    ], dtype=torch.long)
    with torch.no_grad():
        test_output = model(test_indices)
    print(f"Test output shape: {test_output.shape}")

if __name__ == "__main__":
    main()