"""
Reference: https://github.com/karpathy/nanoGPT :)
Modified version with ReLU activation, smaller input size, and fixed batch size of 1
Input shape: [16, 16] (no batch dimension in ONNX export)
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
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


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

        # manual implementation of attention with simplified masking for ONNX compatibility
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #q shape:(B, nh, T, hs), k transpose shape (B, nh, hs, T) -> (B, nh, T, T)
        
        # For ONNX compatibility, we'll create a simple upper triangular mask
        # Create indices for masking
        i_indices = torch.arange(T, device=x.device, dtype=torch.float32).view(T, 1)
        j_indices = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T)
        mask = (i_indices >= j_indices).float()
        mask = mask.view(1, 1, T, T)
        
        # Apply causal mask
        att = att * mask + (1.0 - mask) * (-1e9)
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y
    

  
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x)  # Replace GELU with ReLU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FixedBatchWrapper(nn.Module):
    """Wrapper to handle fixed batch size for ONNX export"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Add batch dimension for the internal model
        if x.dim() == 2:  # If input is [16, 16], add batch dim
            x = x.unsqueeze(0)  # Make it [1, 16, 16]
        
        # Run the model
        output = self.model(x)
        
        # Remove batch dimension for output
        if output.dim() == 3:  # If output is [1, 16, 16], remove batch dim
            output = output.squeeze(0)  # Make it [16, 16]
        
        return output


# Create smaller model with FIXED batch size of 1, input shape [16, 16]
shape = [16, 16]  # No batch dimension
x = torch.ones(1, 16, 16)  # Still need batch dim for PyTorch training/inference
config = GPTConfig(block_size=16, vocab_size=17, n_layer=2, n_head=4, n_embd=16, dropout=0.0, bias=False)
model = Block(config)

# Create wrapper for fixed batch size export
wrapped_model = FixedBatchWrapper(model)

# Test the wrapped model with [16, 16] input
x_no_batch = torch.ones(16, 16)
torch_out = wrapped_model(x_no_batch)

# Export ONNX with fixed batch size (no dynamic batch dimension)
torch.onnx.export(wrapped_model, x_no_batch, "network.onnx",
       export_params=True,        # store the trained parameter weights inside the model file
        opset_version=14,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        # No dynamic_axes - this forces the batch size to be fixed at 1
        )

# For the input data, we'll flatten the [16, 16] shape (without batch dimension)
d = ((x_no_batch).detach().numpy()).reshape([-1]).tolist()  # Remove batch dimension

data = dict(input_shapes = [shape],  # Shape without batch dimension
            input_data = [d],
            output_data = [((torch_out).detach().numpy()).reshape([-1]).tolist()])  # Remove batch dimension

# Serialize data into file:
json.dump( data, open( "input.json", 'w' ) )

