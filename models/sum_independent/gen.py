#!/usr/bin/env python3
"""
Generate an ONNX model for testing Sum operations with different axes and dimensions.

This model tests 4 different sum scenarios using INDEPENDENT data sources:
1. Sum(axis=2) on 3D tensor [1, 4, 8] -> [1, 4]
2. Sum(axis=1) on 2D matrix [4, 8] -> [4]
3. Sum(axis=0) on 2D matrix [4, 8] -> [8]
4. Sum(axis=0) on 1D vector [8] -> [1]

Each sum operation works on completely independent data - no node output reuse.
All intermediate results are combined at the end using Add operations.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def create_sum_independent_model():
    """Create ONNX model with 4 independent sum operations.
    
    Strategy: Create separate constant tensors for each sum operation,
    ensuring no data reuse between operations. All sums contribute to final output.
    """
    
    # Input tensor: [1, 4, 8] (will be multiplied to create variations)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 8])
    
    # Create axes as constants
    axes_2 = helper.make_tensor('axes_2', TensorProto.INT64, [1], [2])
    axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
    axes_0 = helper.make_tensor('axes_0', TensorProto.INT64, [1], [0])
    
    # Create constant multipliers to generate independent data from input
    const1 = numpy_helper.from_array(np.ones((1, 4, 8), dtype=np.float32) * 1.0, name='const1')
    const2 = numpy_helper.from_array(np.ones((1, 4, 8), dtype=np.float32) * 2.0, name='const2')
    const3 = numpy_helper.from_array(np.ones((1, 4, 8), dtype=np.float32) * 3.0, name='const3')
    const4 = numpy_helper.from_array(np.ones((1, 4, 8), dtype=np.float32) * 4.0, name='const4')
    
    # ============================================================================
    # INDEPENDENT DATA SOURCE 1: For Sum1 (axis=2 on 3D)
    # ============================================================================
    mul1_node = helper.make_node(
        'Mul',
        inputs=['input', 'const1'],
        outputs=['data1'],  # [1, 4, 8]
        name='mul1'
    )
    
    # TEST 1: Sum(axis=2) on [1, 4, 8] -> [1, 4]
    sum1_node = helper.make_node(
        'ReduceSum',
        inputs=['data1', 'axes_2'],
        outputs=['sum1_output'],  # [1, 4]
        keepdims=0,
        name='sum1_3d_axis2'
    )
    
    # ============================================================================
    # INDEPENDENT DATA SOURCE 2: For Sum2 (axis=1 on 2D)
    # ============================================================================
    mul2_node = helper.make_node(
        'Mul',
        inputs=['input', 'const2'],
        outputs=['data2_3d'],  # [1, 4, 8]
        name='mul2'
    )
    
    # Reshape to 2D
    reshape2_shape = helper.make_tensor(
        name='reshape2_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 8]
    )
    reshape2_node = helper.make_node(
        'Reshape',
        inputs=['data2_3d', 'reshape2_shape'],
        outputs=['data2'],  # [4, 8]
        name='reshape2'
    )
    
    # TEST 2: Sum(axis=1) on [4, 8] -> [4]
    sum2_node = helper.make_node(
        'ReduceSum',
        inputs=['data2', 'axes_1'],
        outputs=['sum2_output'],  # [4]
        keepdims=0,
        name='sum2_2d_axis1'
    )
    
    # ============================================================================
    # INDEPENDENT DATA SOURCE 3: For Sum3 (axis=0 on 2D)
    # ============================================================================
    mul3_node = helper.make_node(
        'Mul',
        inputs=['input', 'const3'],
        outputs=['data3_3d'],  # [1, 4, 8]
        name='mul3'
    )
    
    # Reshape to 2D
    reshape3_shape = helper.make_tensor(
        name='reshape3_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 8]
    )
    reshape3_node = helper.make_node(
        'Reshape',
        inputs=['data3_3d', 'reshape3_shape'],
        outputs=['data3'],  # [4, 8]
        name='reshape3'
    )
    
    # TEST 3: Sum(axis=0) on [4, 8] -> [8]
    sum3_node = helper.make_node(
        'ReduceSum',
        inputs=['data3', 'axes_0'],
        outputs=['sum3_output'],  # [8]
        keepdims=0,
        name='sum3_2d_axis0'
    )
    
    # ============================================================================
    # INDEPENDENT DATA SOURCE 4: For Sum4 (axis=0 on 1D)
    # ============================================================================
    mul4_node = helper.make_node(
        'Mul',
        inputs=['input', 'const4'],
        outputs=['data4_3d'],  # [1, 4, 8]
        name='mul4'
    )
    
    # Reshape to 2D [4, 8]
    reshape4_2d_shape = helper.make_tensor(
        name='reshape4_2d_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 8]
    )
    reshape4_2d_node = helper.make_node(
        'Reshape',
        inputs=['data4_3d', 'reshape4_2d_shape'],
        outputs=['data4_2d'],  # [4, 8]
        name='reshape4_2d'
    )
    
    # Sum to [8]
    sum_to_8_node = helper.make_node(
        'ReduceSum',
        inputs=['data4_2d', 'axes_0'],
        outputs=['data4'],  # [8]
        keepdims=0,
        name='sum_to_8'
    )
    
    # TEST 4: Sum(axis=0) on [8] -> [1]
    sum4_node = helper.make_node(
        'ReduceSum',
        inputs=['data4', 'axes_0'],
        outputs=['sum4_output'],  # [1]
        keepdims=1,
        name='sum4_1d_axis0'
    )
    
    # ============================================================================
    # COMBINE ALL RESULTS
    # ============================================================================
    # Now we need to combine all the sum results into a single output
    # sum1: [1, 4], sum2: [4], sum3: [8], sum4: [1]
    
    # Reshape sum1 [1, 4] -> [4]
    reshape_sum1_shape = helper.make_tensor(
        name='reshape_sum1_shape',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[4]
    )
    reshape_sum1_node = helper.make_node(
        'Reshape',
        inputs=['sum1_output', 'reshape_sum1_shape'],
        outputs=['sum1_flat'],  # [4]
        name='reshape_sum1'
    )
    
    # Add sum1 and sum2: [4] + [4] -> [4]
    add1_node = helper.make_node(
        'Add',
        inputs=['sum1_flat', 'sum2_output'],
        outputs=['combined_12'],  # [4]
        name='add1'
    )
    
    # Reshape [4] -> [4, 1]
    reshape_combined_shape = helper.make_tensor(
        name='reshape_combined_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 1]
    )
    reshape_combined_node = helper.make_node(
        'Reshape',
        inputs=['combined_12', 'reshape_combined_shape'],
        outputs=['combined_12_2d'],  # [4, 1]
        name='reshape_combined'
    )
    
    # Tile [4, 1] -> [4, 2]
    tile_shape = helper.make_tensor(
        name='tile_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, 2]
    )
    tile_node = helper.make_node(
        'Tile',
        inputs=['combined_12_2d', 'tile_shape'],
        outputs=['tiled'],  # [4, 2]
        name='tile'
    )
    
    # Reshape [4, 2] -> [8]
    reshape_tile_shape = helper.make_tensor(
        name='reshape_tile_shape',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[8]
    )
    reshape_tile_node = helper.make_node(
        'Reshape',
        inputs=['tiled', 'reshape_tile_shape'],
        outputs=['combined_12_flat'],  # [8]
        name='reshape_tile'
    )
    
    # Add sum3: [8] + [8] -> [8]
    add2_node = helper.make_node(
        'Add',
        inputs=['combined_12_flat', 'sum3_output'],
        outputs=['combined_123'],  # [8]
        name='add2'
    )
    
    # Reshape [8] -> [1, 8]
    reshape_123_shape = helper.make_tensor(
        name='reshape_123_shape',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, 8]
    )
    reshape_123_node = helper.make_node(
        'Reshape',
        inputs=['combined_123', 'reshape_123_shape'],
        outputs=['combined_123_2d'],  # [1, 8]
        name='reshape_123'
    )
    
    # Sum to [1]
    sum_combined_node = helper.make_node(
        'ReduceSum',
        inputs=['combined_123_2d', 'axes_1'],
        outputs=['combined_123_sum'],  # [1]
        keepdims=1,
        name='sum_combined'
    )
    
    # Add sum4: [1] + [1] -> [1]
    add3_node = helper.make_node(
        'Add',
        inputs=['combined_123_sum', 'sum4_output'],
        outputs=['final_output'],  # [1]
        name='add3'
    )
    
    # Final output tensor (shape [1])
    output_tensor = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [1])
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes=[
            # Data source 1
            mul1_node, sum1_node,
            # Data source 2
            mul2_node, reshape2_node, sum2_node,
            # Data source 3
            mul3_node, reshape3_node, sum3_node,
            # Data source 4
            mul4_node, reshape4_2d_node, sum_to_8_node, sum4_node,
            # Combine results
            reshape_sum1_node, add1_node, reshape_combined_node, tile_node,
            reshape_tile_node, add2_node, reshape_123_node, sum_combined_node, add3_node
        ],
        name='SumIndependent',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[
            const1, const2, const3, const4,
            axes_2, axes_1, axes_0,
            reshape2_shape, reshape3_shape, reshape4_2d_shape,
            reshape_sum1_shape, reshape_combined_shape, tile_shape,
            reshape_tile_shape, reshape_123_shape
        ]
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='sum_independent')
    model_def.opset_import[0].version = 13
    
    return model_def


def main():
    # Create model
    print("Creating ONNX model with independent sum operations...")
    model = create_sum_independent_model()
    
    # Check and fix the model
    onnx.checker.check_model(model)
    print("✓ Model validation passed")
    
    # Save the model
    output_path = "network.onnx"
    onnx.save(model, output_path)
    print(f"✓ ONNX model saved to {output_path}")
    
    print("\nModel architecture:")
    print("  Input: [1, 4, 8]")
    print("  ")
    print("  Independent data sources (using Mul with different constants):")
    print("    Data1 = input * 1.0  -> Sum1: axis=2 on [1, 4, 8] -> [1, 4]")
    print("    Data2 = input * 2.0  -> Sum2: axis=1 on [4, 8] -> [4]")
    print("    Data3 = input * 3.0  -> Sum3: axis=0 on [4, 8] -> [8]")
    print("    Data4 = input * 4.0  -> Sum4: axis=0 on [8] -> [1]")
    print("  ")
    print("  All sum results are combined via Add operations -> [1] (final output)")
    print("\n✓ All 4 sum operations included with NO node output reuse")


if __name__ == "__main__":
    main()
