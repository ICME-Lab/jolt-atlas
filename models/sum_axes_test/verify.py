#!/usr/bin/env python3
"""
Verify the generated ONNX model structure and sum operations.
"""

import onnx
import numpy as np


def main():
    # Load the ONNX model
    model = onnx.load("network.onnx")
    
    print("=" * 80)
    print("ONNX Model Structure")
    print("=" * 80)
    
    # Print input information
    print("\nInputs:")
    for input_tensor in model.graph.input:
        print(f"  - {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    
    # Print output information
    print("\nOutputs:")
    for output_tensor in model.graph.output:
        print(f"  - {output_tensor.name}: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
    
    # Print nodes (operations)
    print(f"\nNodes ({len(model.graph.node)} total):")
    sum_operations = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == "ReduceSum":
            # Extract axes attribute
            axes = None
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = list(attr.ints) if attr.ints else attr.i
            sum_operations.append((i, node.name, node.input, node.output, axes))
            print(f"  [{i}] {node.op_type} (axes={axes})")
            print(f"      Input: {node.input}")
            print(f"      Output: {node.output}")
        else:
            print(f"  [{i}] {node.op_type}: {node.input} -> {node.output}")
    
    print(f"\n{'=' * 80}")
    print(f"Summary: Found {len(sum_operations)} Sum/ReduceSum operations")
    print("=" * 80)
    
    # Detailed analysis of each sum operation
    for idx, (node_idx, name, inputs, outputs, axes) in enumerate(sum_operations, 1):
        print(f"\nSum Operation {idx} (Node {node_idx}):")
        print(f"  Axes: {axes}")
        print(f"  Input tensors: {inputs}")
        print(f"  Output tensor: {outputs}")


if __name__ == "__main__":
    main()
