#!/usr/bin/env python3
"""
Generate an ONNX model for testing Sum operations with different axes and dimensions.

This model tests 4 different sum scenarios in a chain:
1. Sum(axis=2) on 3D tensor [1, 4, 8] -> [1, 4]
2. Sum(axis=1) on 2D matrix [4, 8] -> [4]
3. Sum(axis=0) on 2D matrix [4, 8] -> [8]
4. Sum(axis=0) on 1D vector [8] -> scalar

The batch dimension (first dim) is always 1, which gets normalized away during processing.
Uses only operations compatible with the atlas-onnx-tracer runtime.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, TensorProto


def create_sum_axes_test_model():
    """Create ONNX model manually using ONNX helper to have full control.
    
    Strategy: Use Add operations to incorporate intermediate sums into the final result,
    ensuring all 4 sum operations are part of the computational graph leading to the output.
    
    Chain of operations:
    1. Input [1, 4, 8] 
    2. Sum1 axis=2: [1, 4, 8] -> [1, 4] (test 3D with batch=1, axis=2)
    3. Reshape input: [1, 4, 8] -> [4, 8]
    4. Sum2 axis=1: [4, 8] -> [4] (test 2D, axis=1)
    5. Sum3 axis=0: [4, 8] -> [8] (test 2D, axis=0) 
    6. Reshape Sum1: [1, 4] -> [4] and Add to Sum2
    7. Reshape Sum2+Sum1: [4] -> [4, 1] and broadcast to [4, 8]
    8. Add to create [4, 8], then Sum axis=0 -> [8]
    9. Add Sum3 to this [8]
    10. Sum4 axis=0: [8] -> scalar (test 1D, axis=0)
    """
    
    # Input tensor: [1, 4, 8]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 8])
    
    # Create axes as constants
    axes_2 = helper.make_tensor('axes_2', TensorProto.INT64, [1], [2])
    axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
    axes_0 = helper.make_tensor('axes_0', TensorProto.INT64, [1], [0])
    
    # TEST 1: Sum(axis=2) on [1, 4, 8] -> [1, 4]
    sum1_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'axes_2'],
        outputs=['sum1_output'],  # [1, 4]
        keepdims=0,
        name='sum1_3d_axis2'
    )
    
    # Reshape input [1, 4, 8] -> [4, 8]
    reshape_input_shape = helper.make_tensor(
        name='reshape_input_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 8]
    )
    reshape_input_node = helper.make_node(
        'Reshape',
        inputs=['input', 'reshape_input_shape'],
        outputs=['reshaped_input_4x8'],  # [4, 8]
        name='reshape_input'
    )
    
    # TEST 2: Sum(axis=1) on [4, 8] -> [4]
    sum2_node = helper.make_node(
        'ReduceSum',
        inputs=['reshaped_input_4x8', 'axes_1'],
        outputs=['sum2_output'],  # [4]
        keepdims=0,
        name='sum2_2d_axis1'
    )
    
    # TEST 3: Sum(axis=0) on [4, 8] -> [8]
    sum3_node = helper.make_node(
        'ReduceSum',
        inputs=['reshaped_input_4x8', 'axes_0'],
        outputs=['sum3_output'],  # [8]
        keepdims=0,
        name='sum3_2d_axis0'
    )
    
    # Reshape sum1 [1, 4] -> [4] to add to sum2
    reshape_sum1_shape = helper.make_tensor(
        name='reshape_sum1_shape',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[4]
    )
    reshape_sum1_node = helper.make_node(
        'Reshape',
        inputs=['sum1_output', 'reshape_sum1_shape'],
        outputs=['sum1_reshaped'],  # [4]
        name='reshape_sum1'
    )
    
    # Add sum1 and sum2: [4] + [4] -> [4]
    add1_node = helper.make_node(
        'Add',
        inputs=['sum1_reshaped', 'sum2_output'],
        outputs=['combined_sum1_sum2'],  # [4]
        name='add1'
    )
    
    # Reshape [4] -> [1, 4] 
    reshape_combined_shape = helper.make_tensor(
        name='reshape_combined_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, 4]
    )
    reshape_combined_node = helper.make_node(
        'Reshape',
        inputs=['combined_sum1_sum2', 'reshape_combined_shape'],
        outputs=['combined_reshaped'],  # [1, 4]
        name='reshape_combined'
    )
    
    # Tile [1, 4] -> [2, 4] (repeat twice on axis 0)
    tile_shape = helper.make_tensor(
        name='tile_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[2, 1]
    )
    tile_node = helper.make_node(
        'Tile',
        inputs=['combined_reshaped', 'tile_shape'],
        outputs=['tiled'],  # [2, 4]
        name='tile'
    )
    
    # Reshape [2, 4] -> [8]
    reshape_tile_shape = helper.make_tensor(
        name='reshape_tile_shape',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[8]
    )
    reshape_tile_node = helper.make_node(
        'Reshape',
        inputs=['tiled', 'reshape_tile_shape'],
        outputs=['tiled_reshaped'],  # [8]
        name='reshape_tiled'
    )
    
    # Add sum3 and tiled: [8] + [8] -> [8]
    add2_node = helper.make_node(
        'Add',
        inputs=['sum3_output', 'tiled_reshaped'],
        outputs=['final_combined'],  # [8]
        name='add2'
    )
    
    # TEST 4: Sum(axis=0) on [8] -> [1]
    sum4_node = helper.make_node(
        'ReduceSum',
        inputs=['final_combined', 'axes_0'],
        outputs=['sum4_output'],  # [1]
        keepdims=1,
        name='sum4_1d_axis0'
    )
    
    # Final output tensor (shape [1])
    output_tensor = helper.make_tensor_value_info('sum4_output', TensorProto.FLOAT, [1])
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes=[
            sum1_node, reshape_input_node, sum2_node, sum3_node,
            reshape_sum1_node, add1_node, reshape_combined_node,
            tile_node, reshape_tile_node, add2_node, sum4_node
        ],
        name='SumAxesTest',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[
            axes_2, axes_1, axes_0,
            reshape_input_shape, reshape_sum1_shape, reshape_combined_shape,
            tile_shape, reshape_tile_shape
        ]
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='sum_axes_test')
    model_def.opset_import[0].version = 13
    
    return model_def


def main():
    # Create model
    print("Creating ONNX model...")
    model = create_sum_axes_test_model()
    
    # Check and fix the model
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    # Save the model
    output_path = "network.onnx"
    onnx.save(model, output_path)
    print(f"✓ ONNX model saved to {output_path}")
    
    print("\nModel architecture:")
    print("  Input: [1, 4, 8]")
    print("  Sum1: axis=2 on [1, 4, 8] -> [1, 4]")
    print("  Reshape: [1, 4, 8] -> [4, 8]")
    print("  Sum2: axis=1 on [4, 8] -> [4]")
    print("  Sum3: axis=0 on [4, 8] -> [8]")
    print("  Sum4: axis=0 on [8] -> [1] (final output)")
    print("\n✓ All 4 sum operations included in the model")


if __name__ == "__main__":
    main()
