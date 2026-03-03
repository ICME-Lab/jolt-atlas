#!/usr/bin/env python3
"""
Test Sum(axis=2) on 3D tensor with batch_size=1.
Input: [1, 4, 8]
Output: [1, 4]
"""

import onnx
from onnx import helper, TensorProto


def create_model():
    # Input: [1, 4, 8]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 8])
    
    # Axes constant
    axes_2 = helper.make_tensor('axes_2', TensorProto.INT64, [1], [2])
    
    # Sum(axis=2): [1, 4, 8] -> [1, 4]
    sum_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'axes_2'],
        outputs=['output'],
        keepdims=0,
        name='sum_3d_axis2'
    )
    
    # Output: [1, 4]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    
    # Create graph
    graph_def = helper.make_graph(
        nodes=[sum_node],
        name='Sum3DAxis2',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[axes_2]
    )
    
    model_def = helper.make_model(graph_def, producer_name='sum_3d_axis2')
    model_def.opset_import[0].version = 13
    
    return model_def


def main():
    print("Creating Sum(axis=2) on 3D tensor model...")
    model = create_model()
    
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    onnx.save(model, "network.onnx")
    print("✓ Saved to network.onnx")
    print("\n  Input: [1, 4, 8]")
    print("  Sum(axis=2) -> Output: [1, 4]")


if __name__ == "__main__":
    main()
