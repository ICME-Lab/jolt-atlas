# Minimal Pow(x, 3) fixture: `y = x ** 3` exports as ONNX `Pow(x, 3)`, which the
# tracer lowers to the unary `Cube` op. Regression fixture for the Pow-lowering
# operand wiring: the broadcast exponent constant must not reach the prover as a
# second operand (see test_pow_cube in jolt-atlas-core/src/onnx_proof/e2e_tests.rs).
# Note: `x * x * x` or `x ** 2` would NOT cover this path (no Pow(3) node).
import json

import torch


class Pow3(torch.nn.Module):
    def forward(self, x):
        return x ** 3


torch.manual_seed(42)
model = Pow3().eval()
x = 0.1 * torch.rand(1, 16)

kwargs = dict(export_params=True,
              opset_version=10,
              do_constant_folding=True,
              input_names=['input'],
              output_names=['output'],
              dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}})
try:
    # torch >= 2.9 defaults to the dynamo exporter, which cannot target opset 10
    torch.onnx.export(model, x, "network.onnx", dynamo=False, **kwargs)
except TypeError:  # older torch without the `dynamo` kwarg
    torch.onnx.export(model, x, "network.onnx", **kwargs)

data = dict(input_shapes=[[1, 16]],
            input_data=[x.detach().numpy().reshape([-1]).tolist()])
json.dump(data, open("input.json", 'w'))
