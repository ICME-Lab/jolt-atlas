#!/usr/bin/env python3
"""
Test Sum(axis=0) on 2D matrix.
Input: [4, 8]
Output: [8]
"""

import onnx
from onnx import helper, TensorProto


def create_model():
    # Input: [4, 8]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 8])
    
    # Axes constant
    axes_0 = helper.make_tensor('axes_0', TensorProto.INT64, [1], [0])
    
    # Sum(axis=0): [4, 8] -> [8]
    sum_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'axes_0'],
        outputs=['output'],
        keepdims=0,
        name='sum_2d_axis0'
    )
    
    # Output: [8]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [8])
    
    # Create graph
    graph_def = helper.make_graph(
        nodes=[sum_node],
        name='Sum2DAxis0',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[axes_0]
    )
    
    model_def = helper.make_model(graph_def, producer_name='sum_2d_axis0')
    model_def.opset_import[0].version = 13
    
    return model_def


def main():
    print("Creating Sum(axis=0) on 2D matrix model...")
    model = create_model()
    
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    onnx.save(model, "network.onnx")
    print("✓ Saved to network.onnx")
    print("\n  Input: [4, 8]")
    print("  Sum(axis=0) -> Output: [8]")


if __name__ == "__main__":
    main()
