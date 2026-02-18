"""nanoGPT-1M: ~1M parameter nanoGPT variant. Reference: https://github.com/karpathy/nanoGPT"""

import json, math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.n_head = c.n_head
        self.n_embd = c.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float(-10))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.attn = LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln_2, self.mlp = LayerNorm(c.n_embd), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 128


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=LayerNorm(c.n_embd),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    model.eval()

    x = torch.randint(cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        out = model(x)
    torch.onnx.export(
        model, x, "network.onnx",
        export_params=True, opset_version=10, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    data = dict(
        input_shapes=[[1, cfg.block_size]],
        input_data=x.reshape(-1).tolist(),
        output_data=[o.detach().numpy().reshape(-1).tolist() for o in out],
    )
    json.dump(data, open("input.json", "w"))
