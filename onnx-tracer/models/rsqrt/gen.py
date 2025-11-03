import torch
import torch.nn as nn

class RsqrtModel(nn.Module):
    def forward(self, x):
        # PyTorch doesn't have a direct rsqrt op, but x.pow(-0.5) exports as Rsqrt in ONNX
        return torch.rsqrt(x)

# Instantiate and prepare model
model = RsqrtModel()
model.eval()

# Example input
x = torch.tensor([4.0], dtype=torch.float32)

# Export to ONNX
torch.onnx.export(
    model,
    (x,),
    "network.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=11,  # Rsqrt is supported from opset 11 onward
    do_constant_folding=True,
)