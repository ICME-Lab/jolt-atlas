import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Single linear layer: [batch_size, 4] x [4, 4] -> [batch_size, 4]
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: [batch_size, 4]
        # Linear: [batch_size, 4] x [4, 4] -> [batch_size, 4]
        x = self.linear(x)
        # ReLU activation
        x = self.relu(x)
        # Output: [batch_size, 4]
        return x

# Create model
model = SimpleMLP()

# Generate some training data
# Input: [batch_size, 4] where batch_size can vary
batch_size = 32
X = torch.randn(batch_size, 4)
# Create target data (simple transformation for training)
y = torch.abs(X) + 0.1 * torch.randn(batch_size, 4)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training simple MLP...")

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the model
model.eval()
with torch.no_grad():
    # Test with batch_size=1 for export
    test_input = torch.randn(1, 4)
    test_output = model(test_input)
    
    print("\nModel test:")
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)
    print("Input:", test_input.numpy().flatten())
    print("Output:", test_output.numpy().flatten())

# Export to ONNX
# Use batch_size=1 for the export (can be dynamic)
dummy_input = torch.randn(1, 4)

torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},    # Variable batch size
        'output': {0: 'batch_size'}
    }
)

print("✅ Exported ONNX model to network.onnx")

# Print model architecture details
print("\nModel architecture:")
print("Input dimension: 4 (power of 2)")
print("Output dimension: 4 (power of 2)")
print("Operations: Input -> Linear(4,4) -> ReLU -> Output")
print("Matrix multiplication shape:")
print("  MatMul: [batch_size, 4] × [4, 4] -> [batch_size, 4]")
