#!/usr/bin/env python3
"""
Test Sum(axis=0) on 1D vector.
Input: [8]
Output: [1] (keepdims=1 to return tensor not scalar)
"""

import onnx
from onnx import helper, TensorProto


def create_model():
    # Input: [8]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [8])
    
    # Axes constant
    axes_0 = helper.make_tensor('axes_0', TensorProto.INT64, [1], [0])
    
    # Sum(axis=0): [8] -> [1]
    sum_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'axes_0'],
        outputs=['output'],
        keepdims=1,  # Keep dims to return [1] instead of scalar
        name='sum_1d_axis0'
    )
    
    # Output: [1]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])
    
    # Create graph
    graph_def = helper.make_graph(
        nodes=[sum_node],
        name='Sum1DAxis0',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[axes_0]
    )
    
    model_def = helper.make_model(graph_def, producer_name='sum_1d_axis0')
    model_def.opset_import[0].version = 13
    
    return model_def


def main():
    print("Creating Sum(axis=0) on 1D vector model...")
    model = create_model()
    
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    onnx.save(model, "network.onnx")
    print("✓ Saved to network.onnx")
    print("\n  Input: [8]")
    print("  Sum(axis=0) -> Output: [1]")


if __name__ == "__main__":
    main()
