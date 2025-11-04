"""
LayerNorm Head Model
This model implements only the operations that happen before the first matmul
in the self-attention model. Based on the test output, this includes:
1. Input [16, 16]
2. Reshape to [1, 16, 16] 
3. LayerNorm operations (SUM, DIV, SUB, MEANOFSQUARES, etc.)

This is essentially a LayerNorm preprocessing head.
"""

import torch
import json
from torch import nn
import math
from dataclasses import dataclass
from torch.nn import functional as F


class LayerNormHead(nn.Module):
    """
    Implements the exact operations before the first matmul:
    - Input reshape
    - Layer normalization computation
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # LayerNorm doesn't have learnable parameters in this head version
        # We'll implement the raw computation
    
    def forward(self, x):
        # Input: [16, 16]
        # Reshape to add batch dimension: [1, 16, 16]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 16, 16]
        
        # LayerNorm computation - matches the operations in the test log:
        # Operations 2-12 from the test output
        
        # 1. Calculate mean (SUM + DIV)
        mean = x.mean(dim=-1, keepdim=True)  # [1, 16, 1]
        
        # 2. Center the data (SUB)
        centered = x - mean  # [1, 16, 16]
        
        # 3. Calculate variance (MEANOFSQUARES)
        variance = (centered ** 2).mean(dim=-1, keepdim=True)  # [1, 16, 1]
        
        # 4. Add epsilon and compute reciprocal square root
        rstd = torch.rsqrt(variance + self.eps)  # [1, 16, 1]
        
        # 5. Apply normalization
        normalized = centered * rstd  # [1, 16, 16]
        
        return normalized


class LayerNormHeadFixed(nn.Module):
    """
    Fixed batch size version that outputs [16, 16] instead of [1, 16, 16]
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
        
        # LayerNorm computation
        mean = x.mean(dim=-1, keepdim=True)  # [1, 16, 1]
        centered = x - mean  # [1, 16, 16]
        variance = (centered ** 2).mean(dim=-1, keepdim=True)  # [1, 16, 1]
        rstd = torch.rsqrt(variance + self.eps)  # [1, 16, 1]
        normalized = centered * rstd  # [1, 16, 16]
        
        # Remove batch dimension for output: [16, 16]
        if normalized.dim() == 3:
            normalized = normalized.squeeze(0)
        
        return normalized


# Create LayerNorm head model with fixed batch size
shape = [16, 16]
x = torch.ones(16, 16)
model = LayerNormHeadFixed(normalized_shape=16)

# Forward pass
torch_out = model(x)

# Export ONNX
torch.onnx.export(model, x, "network.onnx",
       export_params=True,
       opset_version=14,
       do_constant_folding=True,
       input_names=['input'],
       output_names=['output'])

# Save test data
d = x.detach().numpy().reshape([-1]).tolist()
data = dict(
    input_shapes=[shape],
    input_data=[d],
    output_data=[torch_out.detach().numpy().reshape([-1]).tolist()]
)

json.dump(data, open("input.json", 'w'))

print("LayerNorm Head model successfully exported to ONNX format")
print(f"Input shape: {shape}")
print(f"Output shape: {list(torch_out.shape)}")
print("Operations: Input reshape + LayerNorm computation")
print("This implements operations 0-12 from the self-attention test (before first matmul)")

# Also create a version that shows what the operations do
print("\nDemonstrating the operations:")
print("1. Input: shape", x.shape)
x_batch = x.unsqueeze(0)
print("2. After reshape: shape", x_batch.shape)
mean = x_batch.mean(dim=-1, keepdim=True)
print("3. Mean calculation: shape", mean.shape)
centered = x_batch - mean
print("4. After centering: shape", centered.shape)
variance = (centered ** 2).mean(dim=-1, keepdim=True)
print("5. Variance calculation: shape", variance.shape)
eps = 1e-5
rstd = torch.rsqrt(variance + eps)
print("6. Reciprocal std: shape", rstd.shape)
normalized = centered * rstd
print("7. After normalization: shape", normalized.shape)
final_output = normalized.squeeze(0)
print("8. Final output: shape", final_output.shape)
