import json

import torch
import torch.nn as nn


class ConcatModel(nn.Module):
    def forward(self, head_0, head_1):
        # Concat along the last dimension.
        return torch.cat([head_0, head_1], dim=-1)


def main() -> None:
    torch.manual_seed(7)

    # Fully static shapes to avoid symbolic dimension names (e.g. s26).
    batch = 1
    seq = 4
    head_dim = 16

    model = ConcatModel().eval()

    head_0 = torch.randn(batch, seq, head_dim)
    head_1 = torch.randn(batch, seq, head_dim)

    with torch.no_grad():
        output = model(head_0, head_1)

    torch.onnx.export(
        model,
        (head_0, head_1),
        "network.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["head_0", "head_1"],
        output_names=["output"],
        # Intentionally no dynamic_axes to keep dims as concrete dim_value.
    )

    # Keep model weights in a single ONNX file when possible.
    try:
        import onnx

        m = onnx.load("network.onnx", load_external_data=True)
        onnx.save_model(m, "network.onnx", save_as_external_data=False)
    except Exception:
        pass

    payload = {
        "input_shapes": [[batch, seq, head_dim], [batch, seq, head_dim]],
        "input_data": [head_0.reshape(-1).tolist(), head_1.reshape(-1).tolist()],
        "output_data": [output.reshape(-1).tolist()],
    }

    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
