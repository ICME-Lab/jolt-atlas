import json

import torch
import torch.nn as nn


class SliceModel(nn.Module):
    def forward(self, x):
        # Variant 1: basic trailing-axis range slice.
        out_basic = x[:, :, :4]

        # Variant 2: explicit middle-axis range slice.
        out_axis = x[:, 1:4, :]

        # Variant 3: two-axis bounded slice (no stepping).
        out_box = x[:, 0:3, 2:6]

        # VM currently supports a single graph output, so merge all slice
        # variants into one tensor while preserving each slice op in the graph.
        merged = torch.cat(
            [
                out_basic.reshape(x.shape[0], -1),
                out_axis.reshape(x.shape[0], -1),
                out_box.reshape(x.shape[0], -1),
            ],
            dim=1,
        )

        return merged


def main() -> None:
    torch.manual_seed(11)

    # Keep static power-of-two input dimensions for downstream tooling.
    shape = [2, 4, 8]
    model = SliceModel().eval()
    x = torch.randn(*shape)

    with torch.no_grad():
        output = model(x)

    torch.onnx.export(
        model,
        x,
        "network.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # Keep model weights in one ONNX file when possible.
    try:
        import onnx

        m = onnx.load("network.onnx", load_external_data=True)
        onnx.save_model(m, "network.onnx", save_as_external_data=False)
    except Exception:
        pass

    payload = {
        "input_shapes": [shape],
        "input_data": [x.reshape(-1).tolist()],
        "output_data": [output.reshape(-1).tolist()],
    }

    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
