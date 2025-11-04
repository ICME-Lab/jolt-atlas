import torch
import torch.nn as nn
import numpy as np
import onnx

# Load the original perceptron_2.onnx to extract exact parameters
original_model = onnx.load('./tests/og.onnx')

# Extract weight and bias tensors
weights = {}
for init in original_model.graph.initializer:
    weights[init.name] = onnx.numpy_helper.to_array(init)

class Perceptron2(nn.Module):
    """
    Multi-layer perceptron matching the original perceptron_2.onnx architecture.
    Architecture: Input [1, 4] -> Linear(4, 30) -> ReLU -> Linear(30, 30) -> ReLU -> Linear(30, 30) -> ReLU -> Linear(30, 3) -> ReLU -> Output [1, 3]
    """
    def __init__(self):
        super(Perceptron2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 30),    # layers.0
            nn.ReLU(),           # layers.1
            nn.Linear(30, 30),   # layers.2
            nn.ReLU(),           # layers.3
            nn.Linear(30, 30),   # layers.4
            nn.ReLU(),           # layers.5
            nn.Linear(30, 3),    # layers.6
            nn.ReLU(),           # Final ReLU (layers.7)
        )
        
        # Load exact weights from original model
        # Note: PyTorch uses transposed weights compared to ONNX
        with torch.no_grad():
            self.layers[0].weight.copy_(torch.from_numpy(weights['layers.0.weight']))
            self.layers[0].bias.copy_(torch.from_numpy(weights['layers.0.bias']))
            self.layers[2].weight.copy_(torch.from_numpy(weights['layers.2.weight']))
            self.layers[2].bias.copy_(torch.from_numpy(weights['layers.2.bias']))
            self.layers[4].weight.copy_(torch.from_numpy(weights['layers.4.weight']))
            self.layers[4].bias.copy_(torch.from_numpy(weights['layers.4.bias']))
            self.layers[6].weight.copy_(torch.from_numpy(weights['layers.6.weight']))
            self.layers[6].bias.copy_(torch.from_numpy(weights['layers.6.bias']))
    
    def forward(self, x):
        return self.layers(x)

# Create model with exact weights from original
print("Loading weights from original perceptron_2.onnx...")
model = Perceptron2()
print("✓ Model created with original weights")

# Test the model with the expected test input
model.eval()
with torch.no_grad():
    # Test with the exact input from the test: [1, 2, 3, 4]
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    test_output = model(test_input)
    

# Export to ONNX
# Use batch_size=1 for the export
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


