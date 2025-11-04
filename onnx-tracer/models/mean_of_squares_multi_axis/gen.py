#!/usr/bin/env python3
"""
MeanOfSquares test model with multiple axis reduction.
Computes mean of squares along axes [1, 2].
"""
import torch
import torch.nn as nn

class MeanOfSquaresMultiAxisModel(nn.Module):
    """Model that computes mean of squares along multiple dimensions."""
    
    def forward(self, x):
        # Mean of squares along axes 1 and 2
        # Input: [batch, height, width] -> Output: [batch, 1, 1]
        return torch.mean(x * x, dim=[1, 2], keepdim=True)

# Create and export model
model = MeanOfSquaresMultiAxisModel()
model.eval()

# Input: [2, 3, 4] - batch of 2, height 3, width 4
x = torch.randn(2, 3, 4)

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
print(f"Expected reduction: axes [1, 2] -> product = {x.shape[1] * x.shape[2]}")
