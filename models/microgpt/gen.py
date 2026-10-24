"""
Generate a minimal GPT ONNX model (MicroGPT) based on Karpathy's microGPT script.
Follows the exact same patterns as the nanoGPT model in this repo but with the
microGPT script's architectural choices:

Architecture:
  - Token + Position Embeddings
  - 1 Transformer Block:
      - RMSNorm (manual, no learned params)
      - Multi-Head Causal Self-Attention (4 heads)
      - RMSNorm
      - MLP with ReLU
  - Language Model Head

Key differences from standard GPT-2 (matching the microGPT script):
  - RMSNorm instead of LayerNorm
  - ReLU instead of GELU
  - No biases in linear layers
  - Separate Q/K/V projections instead of fused c_attn
"""

import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMSNorm as in the microGPT script: normalize by root mean square, no learned parameters."""
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, following the microGPT script structure."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Separate Q, K, V projections (matching the microGPT script)
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        # Output projection
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

        # Causal mask — use -10 instead of -inf for fixed-point compatibility
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float(-10))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)
        return y


class MLP(nn.Module):
    """Feed-forward MLP with ReLU activation, matching the microGPT script."""

    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Transformer block: RMSNorm + Attention + RMSNorm + MLP, with residual connections."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    """
    Minimal GPT model following the microGPT script architecture.

    Differences from standard GPT-2:
      - RMSNorm instead of LayerNorm (as in the script)
      - ReLU instead of GELU (as in the script)
      - No biases in linear layers (as in the script)
      - Separate Q/K/V projections (as in the script)
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        # Initial RMSNorm (as in the script: applied right after embedding)
        self.norm0 = RMSNorm(n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        ])

        # Language model head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights with small std for fixed-point safety
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Token + position embeddings
        tok_emb = self.wte(idx)      # (B, T, n_embd)
        pos_emb = self.wpe(pos)      # (1, T, n_embd)
        x = tok_emb + pos_emb

        # Initial RMSNorm
        x = self.norm0(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output logits
        logits = self.lm_head(x)     # (B, T, vocab_size)
        return logits


def export_to_onnx():
    # Model hyperparameters (matching the microGPT Python script)
    vocab_size = 32   # power of 2, close to the script's 28 (27 chars + BOS)
    n_embd = 16       # embedding dimension (script uses 16)
    n_head = 4        # attention heads (script uses 4)
    n_layer = 1       # transformer layers (script uses 1)
    block_size = 16   # sequence length (script uses 16)
    head_dim = n_embd // n_head  # 4

    model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size)
    model.eval()

    # Token ID input
    shape = [1, block_size]
    x = torch.randint(vocab_size, (1, block_size))

    with torch.no_grad():
        torch_out = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MicroGPT Architecture:")
    print(f"  vocab_size:  {vocab_size}")
    print(f"  n_embd:      {n_embd}")
    print(f"  n_head:      {n_head}")
    print(f"  head_dim:    {head_dim}")
    print(f"  n_layer:     {n_layer}")
    print(f"  block_size:  {block_size}")
    print(f"  num params:  {n_params}")
    print(f"\n  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(torch_out.shape)}")

    # Export using opset 10 (matches working nanoGPT model)
    torch.onnx.export(
        model, x, "network.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("\nExported to network.onnx")

    # Save test data for verification
    d = x.detach().numpy().reshape([-1]).tolist()
    data = dict(
        input_shapes=[shape],
        input_data=[d],
        output_data=[torch_out.detach().numpy().reshape([-1]).tolist()]
    )
    json.dump(data, open("input.json", 'w'))
    print("Saved test data to input.json")


if __name__ == "__main__":
    export_to_onnx()
