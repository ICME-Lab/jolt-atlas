import torch
import torch.nn as nn
import math

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln2 = math.log(2.0)
        self.q = 256

    def forward(self, x):
        return self.q * torch.sigmoid(x * math.log(2.0))

# Create model
model = Sigmoid()
model.eval()

# Example input
x = torch.randn(1, 10)

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
