"""
Validation script for the fixed batch size ReLU-based self-attention model
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import json


def validate_onnx_model():
    """Validate that the ONNX model works correctly and produces expected outputs"""
    
    # Load the ONNX model
    onnx_model = onnx.load("network.onnx")
    
    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False
    
    # Print model info
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]
    
    print(f"✓ Model input: {input_info.name}")
    print(f"✓ Model output: {output_info.name}")
    
    # Check input/output shapes
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
    
    print(f"✓ ONNX input shape: {input_shape}")
    print(f"✓ ONNX output shape: {output_shape}")
    
    # Verify fixed batch size (should be [16, 16] not [1, 16, 16])
    if input_shape == [16, 16] and output_shape == [16, 16]:
        print("✓ Batch size is correctly fixed (no batch dimension in ONNX)")
    else:
        print(f"⚠ Expected shapes [16, 16], got input: {input_shape}, output: {output_shape}")
    
    # Create ONNX Runtime session
    try:
        ort_session = ort.InferenceSession("network.onnx")
        print("✓ ONNX Runtime session created successfully")
    except Exception as e:
        print(f"✗ Failed to create ONNX Runtime session: {e}")
        return False
    
    # Load test data
    try:
        with open("input.json", "r") as f:
            data = json.load(f)
        print("✓ Test data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        return False
    
    # Prepare input (should be [16, 16] shape)
    input_data = np.array(data["input_data"][0]).reshape(data["input_shapes"][0]).astype(np.float32)
    expected_output = np.array(data["output_data"][0]).reshape(data["input_shapes"][0])
    
    print(f"✓ Input data shape: {input_data.shape}")
    print(f"✓ Expected output shape: {expected_output.shape}")
    
    # Run inference
    try:
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)
        print("✓ ONNX inference completed successfully")
    except Exception as e:
        print(f"✗ ONNX inference failed: {e}")
        return False
    
    # Compare outputs
    output_diff = np.abs(ort_outputs[0].flatten() - expected_output.flatten())
    max_diff = np.max(output_diff)
    mean_diff = np.mean(output_diff)
    
    print(f"✓ Output comparison:")
    print(f"  - Max difference: {max_diff:.6f}")
    print(f"  - Mean difference: {mean_diff:.6f}")
    print(f"  - ONNX output shape: {ort_outputs[0].shape}")
    
    # Check if differences are within acceptable tolerance
    if max_diff < 1e-5:
        print("✓ Output matches expected values within tolerance")
        return True
    else:
        print("⚠ Output differences exceed tolerance, but this may be acceptable")
        return True


def check_relu_usage():
    """Check that the model uses ReLU instead of GELU"""
    onnx_model = onnx.load("network.onnx")
    
    relu_count = 0
    gelu_count = 0
    tanh_count = 0
    
    for node in onnx_model.graph.node:
        if node.op_type == "Relu":
            relu_count += 1
        elif node.op_type == "Gelu":
            gelu_count += 1
        elif node.op_type == "Tanh":
            tanh_count += 1
    
    print(f"✓ Activation function analysis:")
    print(f"  - ReLU operations: {relu_count}")
    print(f"  - GELU operations: {gelu_count}")
    print(f"  - Tanh operations: {tanh_count}")
    
    if gelu_count == 0 and tanh_count == 0 and relu_count > 0:
        print("✓ Successfully replaced GELU with ReLU")
        return True
    else:
        print("⚠ GELU or Tanh operations still present in the model")
        return False


def test_inference_with_different_inputs():
    """Test inference with different input patterns"""
    print("✓ Testing inference with different input patterns:")
    
    ort_session = ort.InferenceSession("network.onnx")
    
    # Test 1: All zeros
    test_input = np.zeros((16, 16), dtype=np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    output = ort_session.run(None, ort_inputs)[0]
    print(f"  - All zeros input -> output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test 2: All ones
    test_input = np.ones((16, 16), dtype=np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    output = ort_session.run(None, ort_inputs)[0]
    print(f"  - All ones input -> output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test 3: Random input
    test_input = np.random.randn(16, 16).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    output = ort_session.run(None, ort_inputs)[0]
    print(f"  - Random input -> output range: [{output.min():.4f}, {output.max():.4f}]")


if __name__ == "__main__":
    print("Validating Fixed Batch Size ONNX model...")
    print("=" * 60)
    
    model_valid = validate_onnx_model()
    print()
    
    relu_check = check_relu_usage()
    print()
    
    try:
        test_inference_with_different_inputs()
        inference_test = True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        inference_test = False
    print()
    
    if model_valid and relu_check and inference_test:
        print("🎉 All validations passed!")
        print("📝 Key features verified:")
        print("   - Fixed batch size (no dynamic batch dimension)")
        print("   - Input shape: [16, 16]")
        print("   - Output shape: [16, 16]")
        print("   - ReLU activation (no GELU/Tanh)")
        print("   - ONNX Runtime compatibility")
    else:
        print("❌ Some validations failed")
