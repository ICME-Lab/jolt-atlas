import torch
import torch.nn as nn
import math

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Continuous base-2 sigmoid
        y = torch.sigmoid(x)
        return y

# Create model
model = Sigmoid()
model.eval()

# Example input
x = torch.randn(1, 64)

# Export to ONNX
torch.onnx.export(
    model,
    (x,),
    "network.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=11,
    do_constant_folding=True,
)

print("Exported sigmoid model to network.onnx")
