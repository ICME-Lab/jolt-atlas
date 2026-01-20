"""
Causal Self-Attention only model (no LayerNorm, no FFN)
"""

import torch
import json
from torch import nn
import math
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention (non-causal - no masking)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #q shape:(B, nh, T, hs), k transpose shape (B, nh, hs, T) -> (B, nh, T, T)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


# Simple configuration for testing
shape = [1, 64, 64]
x = torch.ones(1, 64, 64)
config = GPTConfig(block_size=64, vocab_size=65, n_layer=1, n_head=4, n_embd=64, dropout=0.0, bias=False)
model = CausalSelfAttention(config)
torch_out = model(x)

torch.onnx.export(model, x, "network.onnx",
       export_params=True,        # store the trained parameter weights inside the model file
        opset_version=18,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output']) # the model's output names

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_shapes = [shape],
            input_data = [d],
            output_data = [((torch_out).detach().numpy()).reshape([-1]).tolist()])

# Serialize data into file:
json.dump( data, open( "input.json", 'w' ) )
