#!/usr/bin/env python3
"""
Simple MeanOfSquares test model.
Computes mean of squares along axis 1: E[x^2] = (1/n) * sum(x^2)
"""
import torch
import torch.nn as nn

class MeanOfSquaresModel(nn.Module):
    """Simple model that computes mean of squares along the last dimension."""
    
    def forward(self, x):
        # Mean of squares: (1/n) * sum(x^2)
        # This should be converted to MeanOfSquares operation
        return torch.mean(x * x, dim=1, keepdim=True)

# Create and export model
model = MeanOfSquaresModel()
model.eval()

# Input: [2, 4] - batch of 2, sequence length of 4
x = torch.randn(2, 4)

# Export to ONNX
torch.onnx.export(
    model,
    (x,),
    "network.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=False,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to network.onnx")
print(f"Input shape: {x.shape}")
print(f"Output shape: {model(x).shape}")
