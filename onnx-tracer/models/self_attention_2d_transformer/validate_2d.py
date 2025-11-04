"""
Validation script to verify tensor dimensions in the 2D-only model
"""

import torch
import numpy as np
from gen import Block2D, GPTConfig

def check_tensor_dimensions():
    """Hook into PyTorch operations to track tensor dimensions"""
    max_dims = 0
    operations = []
    
    def hook_fn(module, input, output):
        nonlocal max_dims, operations
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    dims = inp.dim()
                    max_dims = max(max_dims, dims)
                    operations.append(f"{module.__class__.__name__} input[{i}]: {dims}D {list(inp.shape)}")
        
        if isinstance(output, torch.Tensor):
            dims = output.dim()
            max_dims = max(max_dims, dims)
            operations.append(f"{module.__class__.__name__} output: {dims}D {list(output.shape)}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    dims = out.dim()
                    max_dims = max(max_dims, dims)
                    operations.append(f"{module.__class__.__name__} output[{i}]: {dims}D {list(out.shape)}")
    
    # Create model and add hooks
    config = GPTConfig(block_size=16, vocab_size=17, n_layer=1, n_head=4, n_embd=16, dropout=0.0, bias=False)
    model = Block2D(config)
    
    # Register hooks on all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Run forward pass
    x = torch.randn(16, 16)
    with torch.no_grad():
        output = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return max_dims, operations


if __name__ == "__main__":
    print("🔍 Analyzing tensor dimensions in 2D-only model...")
    print("=" * 60)
    
    max_dims, operations = check_tensor_dimensions()
    
    print(f"\n📊 Results:")
    print(f"Maximum tensor dimensions: {max_dims}D")
    
    if max_dims <= 2:
        print("✅ SUCCESS: Model operates with maximum 2 dimensions!")
    else:
        print(f"❌ FAILURE: Model uses {max_dims}D tensors")
    
    print(f"\n📝 Detailed operations:")
    for op in operations[:20]:  # Show first 20 operations
        print(f"  {op}")
    
    if len(operations) > 20:
        print(f"  ... and {len(operations) - 20} more operations")
    
    print("\n" + "=" * 60)
    
    # Additional verification
    print("🧪 Additional verification:")
    config = GPTConfig(block_size=16, vocab_size=17, n_layer=1, n_head=4, n_embd=16, dropout=0.0, bias=False)
    model = Block2D(config)
    
    test_inputs = [
        torch.randn(16, 16),
        torch.randn(8, 16), 
        torch.randn(32, 16)
    ]
    
    for i, x in enumerate(test_inputs):
        try:
            output = model(x)
            print(f"  Test {i+1}: {list(x.shape)} -> {list(output.shape)} ✅")
        except Exception as e:
            print(f"  Test {i+1}: {list(x.shape)} -> Error: {e} ❌")
