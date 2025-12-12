#!/usr/bin/env python3
"""
FFN model with GELU activation: input -> matmul -> gelu -> matmul -> output
Uses non-power-of-two dimensions (30x31) for more general testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFN_GELU(nn.Module):
    """
    Feed-forward network with GELU activation:
    - Input: 30 dimensions
    - Hidden: 31 dimensions
    - Output: 30 dimensions
    - Operations: Linear (matmul + bias) -> GELU -> Linear (matmul + bias)
    """
    def __init__(self, input_dim=30, hidden_dim=31, output_dim=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # First matmul + bias
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second matmul + bias
        
        # Initialize weights with small random values
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # First layer: matmul + bias + gelu
        x = self.fc1(x)        # Linear transformation (matmul + bias)
        x = F.gelu(x)          # GELU activation
        
        # Output layer: matmul + bias (no activation)
        x = self.fc2(x)        # Final linear transformation
        return x

# Create model
input_dim = 30
hidden_dim = 31
output_dim = 30

model = FFN_GELU(input_dim, hidden_dim, output_dim)
model.eval()

# Create some dummy training data to make the model learn something meaningful
num_samples = 64
X_train = torch.randn(num_samples, input_dim)
# Create a simple pattern: output is a learned transformation of input
y_train = torch.matmul(X_train, torch.randn(input_dim, output_dim)) + torch.randn(output_dim) * 0.1

# Quick training to ensure the model has meaningful weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training FFN with GELU...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    output = model(X_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Set model to evaluation mode
model.eval()

# Create dummy input for ONNX export (batch size 1)
dummy_input = torch.randn(1, input_dim)

# Test the model with dummy input
with torch.no_grad():
    test_output = model(dummy_input)
    print(f"\nModel test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Input: {dummy_input.numpy().flatten()[:10]}...")  # Show first 10 values
    print(f"Output: {test_output.numpy().flatten()[:10]}...")

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=15,
    do_constant_folding=True,
)

print(f"\n✅ Exported ONNX model to network.onnx")
print(f"Model architecture:")
print(f"  Input dimension: {input_dim}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Output dimension: {output_dim}")
print(f"  Operations: Input -> Linear -> GELU -> Linear -> Output")
print(f"  Matrix multiplication shapes:")
print(f"    First matmul: ({input_dim},) × ({input_dim}, {hidden_dim}) = ({hidden_dim},)")
print(f"    Second matmul: ({hidden_dim},) × ({hidden_dim}, {output_dim}) = ({output_dim},)")
