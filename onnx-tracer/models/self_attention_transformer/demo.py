#!/usr/bin/env python3
"""
Demonstration script for the fixed batch size ReLU-based self-attention model
Shows how to use the model with [16, 16] input (no batch dimension)
"""

import onnxruntime as ort
import numpy as np
import json


def load_model():
    """Load the ONNX model"""
    try:
        session = ort.InferenceSession("network.onnx")
        print("✓ Model loaded successfully")
        
        # Print model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"  - Input name: {input_info.name}")
        print(f"  - Input shape: {input_info.shape}")
        print(f"  - Output name: {output_info.name}")
        print(f"  - Output shape: {output_info.shape}")
        
        return session
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def run_inference(session, input_data, description=""):
    """Run inference on the model"""
    try:
        # Ensure input is the correct shape and type
        if input_data.shape != (16, 16):
            raise ValueError(f"Input must be shape (16, 16), got {input_data.shape}")
        
        input_data = input_data.astype(np.float32)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})
        
        output = result[0]
        print(f"✓ {description}")
        print(f"  - Input shape: {input_data.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return output
    except Exception as e:
        print(f"✗ Inference failed for {description}: {e}")
        return None


def demonstrate_single_sample_inference():
    """Demonstrate how to use the model for single sample inference"""
    print("\n" + "="*60)
    print("Single Sample Inference Demonstration")
    print("="*60)
    
    session = load_model()
    if session is None:
        return
    
    print("\n1. Testing with different input patterns:")
    
    # Test 1: Identity-like pattern
    input1 = np.eye(16, dtype=np.float32)
    run_inference(session, input1, "Identity matrix input")
    
    # Test 2: Constant values
    input2 = np.ones((16, 16), dtype=np.float32) * 0.5
    run_inference(session, input2, "Constant 0.5 input")
    
    # Test 3: Random pattern
    np.random.seed(42)  # For reproducibility
    input3 = np.random.randn(16, 16).astype(np.float32)
    run_inference(session, input3, "Random normal input")
    
    # Test 4: Sequential pattern
    input4 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        input4[i, :i+1] = 1.0  # Causal pattern
    run_inference(session, input4, "Causal pattern input")
    
    print("\n2. Comparing with test data:")
    # Load the test data from input.json
    try:
        with open("input.json", "r") as f:
            test_data = json.load(f)
        
        test_input = np.array(test_data["input_data"][0]).reshape(test_data["input_shapes"][0]).astype(np.float32)
        expected_output = np.array(test_data["output_data"][0]).reshape(test_data["input_shapes"][0])
        
        actual_output = run_inference(session, test_input, "Test data input")
        
        if actual_output is not None:
            diff = np.abs(actual_output - expected_output)
            print(f"  - Max difference from expected: {diff.max():.8f}")
            print(f"  - Mean difference from expected: {diff.mean():.8f}")
        
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")


def demonstrate_production_usage():
    """Show how this model would be used in a production setting"""
    print("\n" + "="*60)
    print("Production Usage Example")
    print("="*60)
    
    session = load_model()
    if session is None:
        return
    
    print("\n3. Production-style inference pipeline:")
    
    # Simulate a production scenario
    def process_sample(data):
        """Process a single sample through the model"""
        # Preprocessing
        if data.shape != (16, 16):
            print(f"  ⚠ Reshaping input from {data.shape} to (16, 16)")
            data = np.resize(data, (16, 16))
        
        # Normalization (example)
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        # Inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: data.astype(np.float32)})
        
        # Postprocessing
        output = result[0]
        
        return output
    
    # Process multiple samples
    samples = [
        np.random.randn(16, 16),
        np.random.randn(16, 16),
        np.random.randn(16, 16)
    ]
    
    for i, sample in enumerate(samples):
        output = process_sample(sample)
        print(f"  Sample {i+1}: input range [{sample.min():.3f}, {sample.max():.3f}] "
              f"-> output range [{output.min():.3f}, {output.max():.3f}]")


def compare_batch_vs_fixed():
    """Compare this fixed batch model with dynamic batch approach"""
    print("\n" + "="*60)
    print("Fixed vs Dynamic Batch Comparison")
    print("="*60)
    
    session = load_model()
    if session is None:
        return
    
    print("\n4. Advantages of fixed batch size:")
    print("   ✓ No batch dimension in ONNX (input: [16, 16] not [1, 16, 16])")
    print("   ✓ Simpler deployment (no need to handle variable batch sizes)")
    print("   ✓ Potentially faster inference (optimized for single sample)")
    print("   ✓ Lower memory footprint (fixed memory allocation)")
    print("   ✓ Better for real-time applications")
    
    print("\n   Considerations:")
    print("   - Can only process one sample at a time")
    print("   - Need to call inference multiple times for multiple samples")
    print("   - May be less efficient for batch processing scenarios")
    
    # Demonstrate timing (simple example)
    import time
    
    test_input = np.random.randn(16, 16).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: test_input})
    
    # Timing
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: test_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"\n   Performance: ~{avg_time:.2f}ms per inference (averaged over {num_runs} runs)")


if __name__ == "__main__":
    print("🚀 Fixed Batch Size Self-Attention Model Demo")
    print("This model takes input shape [16, 16] with NO batch dimension")
    
    demonstrate_single_sample_inference()
    demonstrate_production_usage()
    compare_batch_vs_fixed()
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("Key takeaway: This model is optimized for single-sample inference")
    print("with a fixed input shape of [16, 16] (no batch dimension)")
    print("="*60)
