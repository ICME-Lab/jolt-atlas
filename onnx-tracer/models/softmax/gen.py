import torch
import torch.nn as nn
import math

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Continuous base-2 sigmoid
        softmax = nn.Softmax(dim=0)
        y = softmax(x)
        return y

# Create model
model = Softmax()
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

print("Exported softmax model to network.onnx")
