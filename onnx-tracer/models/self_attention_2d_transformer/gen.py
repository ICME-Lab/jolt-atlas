"""
Self-Attention model with maximum 2 dimensions
This model operates entirely in 2D space without any 3D tensors
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


class LayerNorm2D(nn.Module):
    """2D LayerNorm that operates on the last dimension"""
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps
    
    def forward(self, x):
        # x shape: [seq_len, embd_dim]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class SelfAttention2D(nn.Module):
    """Self-attention that works entirely in 2D"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # QKV projection as separate layers for 2D operation
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        seq_len, d_model = x.shape  # [seq_len, d_model]
        
        # Compute Q, K, V
        q = self.q_proj(x)  # [seq_len, d_model]
        k = self.k_proj(x)  # [seq_len, d_model]
        v = self.v_proj(x)  # [seq_len, d_model]
        
        # Reshape for multi-head attention (still 2D)
        # We'll flatten the computation to avoid 3D tensors
        
        # Simple attention without multi-head separation to keep 2D
        # Scale by sqrt(d_model) instead of head_size for simplicity
        scale = 1.0 / math.sqrt(d_model)
        
        # Attention scores: [seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [seq_len, d_model]
        out = torch.matmul(attn_weights, v)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class MLP2D(nn.Module):
    """MLP that operates in 2D"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block2D(nn.Module):
    """Transformer block that operates entirely in 2D"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm2D(config.n_embd)
        self.attn = SelfAttention2D(config)
        self.ln2 = LayerNorm2D(config.n_embd)
        self.mlp = MLP2D(config)
    
    def forward(self, x):
        # x shape: [seq_len, d_model]
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# Create 2D-only model
shape = [16, 16]  # Input shape: [seq_len, d_model]
x = torch.ones(16, 16)
config = GPTConfig(block_size=16, vocab_size=17, n_layer=1, n_head=4, n_embd=16, dropout=0.0, bias=False)
model = Block2D(config)

# Forward pass
torch_out = model(x)

# Export ONNX
torch.onnx.export(model, x, "network.onnx",
       export_params=True,
       opset_version=14,
       do_constant_folding=True,
       input_names=['input'],
       output_names=['output'])

# Save test data
d = x.detach().numpy().reshape([-1]).tolist()
data = dict(
    input_shapes=[shape],
    input_data=[d],
    output_data=[torch_out.detach().numpy().reshape([-1]).tolist()]
)

json.dump(data, open("input.json", 'w'))

print("Model successfully exported to ONNX format (2D only)")
print(f"Input shape: {shape}")
print(f"Output shape: {list(torch_out.shape)}")
print("Maximum tensor dimensionality: 2")
print("All GELU activations replaced with ReLU")
