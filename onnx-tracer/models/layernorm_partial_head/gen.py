"""
Partial LayerNorm Head Model
This model implements only the FIRST 3 steps of LayerNorm computation:
1. Calculate mean (SUM + DIV)
2. Center the data (SUB)
3. Calculate variance (MEANOFSQUARES)
"""

import torch
import json
from torch import nn
import math
from dataclasses import dataclass
from torch.nn import functional as F


class LayerNormPartialHead(nn.Module):
    """
    Implements only the first 3 steps of LayerNorm:
    Step 1: Calculate mean
    Step 2: Center the data
    Step 3: Calculate variance
    
    Output: The variance (before adding epsilon and taking RSQRT)
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
    
    def forward(self, x):
        # Input: [16, 16]
        # Add batch dimension temporarily for computation
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 16, 16]
        
        # Step 1: Calculate mean (SUM + DIV)
        mean = x.mean(dim=-1, keepdim=True)  # [1, 16, 1]
        
        # Step 2: Center the data (SUB)
        centered = x - mean  # [1, 16, 16]
        
        # Step 3: Calculate variance (MEANOFSQUARES)
        variance = (centered ** 2).mean(dim=-1, keepdim=True)  # [1, 16, 1]
        
        # STOP HERE - Return variance
        # Remove batch dimension for output: [16, 1]
        if variance.dim() == 3:
            variance = variance.squeeze(0)
        
        return variance


# Create Partial LayerNorm head model with fixed batch size
shape = [16, 16]
x = torch.ones(16, 16)
model = LayerNormPartialHead(normalized_shape=16)

# Forward pass
torch_out = model(x)

print("=" * 60)
print("Partial LayerNorm Head Model - Steps 1, 2, 3 Only")
print("=" * 60)

# Export ONNX
torch.onnx.export(model, x, "network.onnx",
       export_params=True,
       opset_version=14,
       do_constant_folding=True,
       input_names=['input'],
       output_names=['output'])

# Save test data
d = x.detach().numpy().reshape([-1]).tolist()
output_shape = list(torch_out.shape)
data = dict(
    input_shapes=[shape],
    input_data=[d],
    output_data=[torch_out.detach().numpy().reshape([-1]).tolist()]
)

json.dump(data, open("input.json", 'w'))




