#!/usr/bin/env python3
"""
Generate a small transformer-block-style ONNX model whose feed-forward path uses
the Sigmoid activation.
"""

import json
import math
import os

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class SigmoidSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)

        return self.out_proj(attended)


class SigmoidTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, num_heads: int = 4, ffn_dim: int = 32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = SigmoidSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffn_up = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        ff = self.ffn_up(self.ln_2(x))
        ff = torch.sigmoid(ff)
        ff = self.ffn_down(ff)
        return x + ff


def main():
    torch.manual_seed(7)

    batch = 1
    seq_len = 4
    embed_dim = 16
    input_shape = [batch, seq_len, embed_dim]

    model = SigmoidTransformerBlock(embed_dim=embed_dim, num_heads=4, ffn_dim=32)
    model.eval()

    dummy_input = torch.randn(batch, seq_len, embed_dim)

    with torch.no_grad():
        output = model(dummy_input)

    output_path = "network.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    external_data_file = output_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)
        print(f"Removed external data file: {external_data_file}")

    payload = {
        "input_shapes": [input_shape],
        "input_data": [dummy_input.reshape(-1).tolist()],
        "output_data": [output.reshape(-1).tolist()],
    }
    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"ONNX model saved to {output_path}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {list(output.shape)}")
    print("\nModel operations:")
    print("  LayerNorm -> SelfAttention -> residual")
    print("  LayerNorm -> Linear -> Sigmoid -> Linear -> residual")
    print("  Sigmoid is used in the feed-forward path of the transformer block")


if __name__ == "__main__":
    main()
