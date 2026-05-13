#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
EPS = 1e-12


def per_input_channel_max(weight: torch.Tensor) -> list[float]:
    # HF stores Linear weights as [out, in]. SmoothQuant scales the input axis.
    return weight.float().abs().amax(dim=0).tolist()


def merge_max(*xs: list[float]) -> list[float]:
    out = list(xs[0])
    for x in xs[1:]:
        if len(out) != len(x):
            raise ValueError(f"length mismatch: {len(out)} vs {len(x)}")
        out = [max(a, b) for a, b in zip(out, x)]
    return out


def make_scale(act_max: list[float], weight_max: list[float], alpha: float) -> list[float]:
    if len(act_max) != len(weight_max):
        raise ValueError(f"length mismatch: {len(act_max)} vs {len(weight_max)}")
    scales = []
    for a, w in zip(act_max, weight_max):
        a = max(float(a), EPS)
        w = max(float(w), EPS)
        scales.append((a**alpha) / (w ** (1.0 - alpha)))
    return scales


def summary(xs: list[float]) -> dict:
    t = torch.tensor(xs, dtype=torch.float32)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "p50": float(t.quantile(0.50)),
        "p90": float(t.quantile(0.90)),
        "p99": float(t.quantile(0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation-stats",
        default="qwen2-awy/calibration/activation_stats_128.json",
    )
    parser.add_argument(
        "--model",
        default="atlas-onnx-tracer/models/qwen/model.safetensors",
    )
    parser.add_argument(
        "--output",
        default="qwen2-awy/calibration/smoothquant_scales_alpha_0_5.json",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    stats = json.loads(Path(args.activation_stats).read_text())
    out_layers = []

    with safe_open(args.model, framework="pt", device="cpu") as f:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}"
            layer_stats = stats["layers"][layer]

            wq = per_input_channel_max(f.get_tensor(f"{prefix}.self_attn.q_proj.weight"))
            wk = per_input_channel_max(f.get_tensor(f"{prefix}.self_attn.k_proj.weight"))
            wv = per_input_channel_max(f.get_tensor(f"{prefix}.self_attn.v_proj.weight"))
            attn_w = merge_max(wq, wk, wv)

            wg = per_input_channel_max(f.get_tensor(f"{prefix}.mlp.gate_proj.weight"))
            wu = per_input_channel_max(f.get_tensor(f"{prefix}.mlp.up_proj.weight"))
            mlp_w = merge_max(wg, wu)

            down_w = per_input_channel_max(f.get_tensor(f"{prefix}.mlp.down_proj.weight"))

            attn_scale = make_scale(
                layer_stats["attn_in"]["max_abs"], attn_w, args.alpha
            )
            mlp_scale = make_scale(layer_stats["mlp_in"]["max_abs"], mlp_w, args.alpha)
            down_scale = make_scale(
                layer_stats["down_in"]["max_abs"], down_w, args.alpha
            )

            out_layers.append(
                {
                    "layer": layer,
                    "attn_in": {
                        "scale": attn_scale,
                        "activation_max_abs_summary": summary(
                            layer_stats["attn_in"]["max_abs"]
                        ),
                        "weight_max_abs_summary": summary(attn_w),
                        "scale_summary": summary(attn_scale),
                    },
                    "mlp_in": {
                        "scale": mlp_scale,
                        "activation_max_abs_summary": summary(
                            layer_stats["mlp_in"]["max_abs"]
                        ),
                        "weight_max_abs_summary": summary(mlp_w),
                        "scale_summary": summary(mlp_scale),
                    },
                    "down_in": {
                        "scale": down_scale,
                        "activation_max_abs_summary": summary(
                            layer_stats["down_in"]["max_abs"]
                        ),
                        "weight_max_abs_summary": summary(down_w),
                        "scale_summary": summary(down_scale),
                    },
                }
            )

    output = {
        "alpha": args.alpha,
        "definition": "X_smooth = X / scale, W_smooth = W * scale along the input channel axis",
        "activation_stats": args.activation_stats,
        "model": args.model,
        "samples": stats["samples"],
        "tokens": stats["tokens"],
        "layers": out_layers,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"wrote {out_path}")
    print(f"alpha={args.alpha} samples={stats['samples']} tokens={stats['tokens']}")
    for name in ["attn_in", "mlp_in", "down_in"]:
        best = sorted(
            (
                (layer["layer"], layer[name]["scale_summary"]["max"])
                for layer in out_layers
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        print(name, best)


if __name__ == "__main__":
    main()
