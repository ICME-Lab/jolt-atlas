#!/usr/bin/env python3
"""
Generate a simple MLP ONNX model with 4 layers using MatMul + Square operations.
The model architecture:
  Input: [batch_size, 4]
  Layer 1: MatMul [4, 16] + Square
  Layer 2: MatMul [16, 8] + Square
  Layer 3: MatMul [8, 8] + Square
  Layer 4: MatMul [8, 4] + Square
  Output: [batch_size, 4]

All dimensions are powers of 2. Weights are initialized with realistic MLP values
using Xavier/Glorot initialization scaled down to avoid overflow in fixed-point arithmetic.
"""

import torch
import torch.nn as nn
import numpy as np


class SquareMLP4Layer(nn.Module):
    def __init__(self):
        super(SquareMLP4Layer, self).__init__()
        
        # Define 4 linear layers with power-of-two dimensions
        # Using realistic weight initialization (Xavier/Glorot) but scaled down
        # to avoid overflow with fixed-point arithmetic (scale=128)
        
        # Layer dimensions: 4 -> 16 -> 8 -> 8 -> 4
        self.fc1 = nn.Linear(4, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.fc3 = nn.Linear(8, 8, bias=False)
        self.fc4 = nn.Linear(8, 4, bias=False)
        
        # Initialize weights with Xavier/Glorot initialization
        # Scale factor of 0.5 balances avoiding underflow and overflow
        # with 4 layers of squaring in fixed-point arithmetic
        scale_factor = 0.5
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight)
            self.fc1.weight.data *= scale_factor
            
            nn.init.xavier_uniform_(self.fc2.weight)
            self.fc2.weight.data *= scale_factor
            
            nn.init.xavier_uniform_(self.fc3.weight)
            self.fc3.weight.data *= scale_factor
            
            nn.init.xavier_uniform_(self.fc4.weight)
            self.fc4.weight.data *= scale_factor
    
    def forward(self, x):
        # Layer 1: MatMul + Square
        x = self.fc1(x)
        x = torch.pow(x, 2.0)
        
        # Layer 2: MatMul + Square
        x = self.fc2(x)
        x = torch.pow(x, 2.0)
        
        # Layer 3: MatMul + Square
        x = self.fc3(x)
        x = torch.pow(x, 2.0)
        
        # Layer 4: MatMul + Square
        x = self.fc4(x)
        x = torch.pow(x, 2.0)
        
        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model instance
    model = SquareMLP4Layer()
    model.eval()
    
    # Create dummy input with shape [batch_size=1, 4]
    dummy_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    
    print("Model Architecture:")
    print("=" * 60)
    print("Input shape: [batch_size, 4]")
    print("Layer 1: MatMul [4, 16] + Square -> [batch_size, 16]")
    print("Layer 2: MatMul [16, 8] + Square -> [batch_size, 8]")
    print("Layer 3: MatMul [8, 8] + Square -> [batch_size, 8]")
    print("Layer 4: MatMul [8, 4] + Square -> [batch_size, 4]")
    print("Output shape: [batch_size, 4]")
    print("=" * 60)
    
    # Print sample weights to show they're realistic
    print("\nSample weights (first layer, first 4x4 subset):")
    print(model.fc1.weight.data[:4, :4].numpy())
    
    # Test with sample input
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"\nTest input: {dummy_input.numpy()}")
    print(f"Test output: {test_output.numpy()}")
    
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
    
    onnx_model = onnx.load(output_path)
    
    # Save with location=None to embed all data
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    
    # Clean up any external data files
    import os
    external_data_file = output_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)
        print(f"\nRemoved external data file: {external_data_file}")
    
    print(f"\n✓ ONNX model saved to {output_path}")
    print(f"✓ All dimensions are powers of 2")
    print(f"✓ Weights initialized with Xavier/Glorot (scaled for fixed-point)")
    print(f"✓ 4 layers: MatMul + Square activation")


if __name__ == "__main__":
    main()
