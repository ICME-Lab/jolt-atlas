#!/usr/bin/env python3
"""
Test script for authorization model inference.
Tests the trained model on various authorization scenarios.
"""

import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from gen import MLPAuthorizationClassifier, create_feature_mapping, create_one_hot_vector, generate_synthetic_transaction

def load_model_artifacts():
    """Load the trained model and metadata"""
    
    # Load metadata
    with open('vocab.json', 'r') as f:
        vocab_data = json.load(f)
        vocab_mapping = vocab_data['vocab_mapping']
        feature_mapping = vocab_data['feature_mapping']
    
    with open('meta.json', 'r') as f:
        meta = json.load(f)
    
    # Load PyTorch model
    pytorch_model = MLPAuthorizationClassifier()
    pytorch_model.load_state_dict(torch.load('authorization_model.pth'))
    pytorch_model.eval()
    
    # Load ONNX model
    onnx_session = ort.InferenceSession('network.onnx')
    
    return pytorch_model, onnx_session, vocab_mapping, feature_mapping, meta

def test_pytorch_vs_onnx(pytorch_model, onnx_session, vocab_mapping):
    """Test that PyTorch and ONNX models produce identical outputs"""
    
    print("🔍 Testing PyTorch vs ONNX model consistency")
    print("=" * 50)
    
    # Generate test transactions
    test_cases = []
    for _ in range(10):
        transaction = generate_synthetic_transaction()
        one_hot = create_one_hot_vector(transaction, vocab_mapping)
        test_cases.append((transaction, one_hot))
    
    max_diff = 0.0
    all_close = True
    
    for i, (transaction, one_hot) in enumerate(test_cases):
        
        # PyTorch prediction
        with torch.no_grad():
            torch_input = torch.FloatTensor(one_hot).unsqueeze(0)
            torch_output = pytorch_model(torch_input)
            torch_probs = torch.softmax(torch_output, dim=1).numpy()
        
        # ONNX prediction
        onnx_input = one_hot.reshape(1, -1).astype(np.float32)
        onnx_output = onnx_session.run(None, {'input': onnx_input})[0]
        onnx_probs = torch.softmax(torch.FloatTensor(onnx_output), dim=1).numpy()
        
        # Compare outputs
        diff = np.max(np.abs(torch_probs - onnx_probs))
        max_diff = max(max_diff, diff)
        
        if diff > 1e-6:
            all_close = False
            print(f"❌ Test {i+1}: Large difference detected: {diff:.2e}")
        else:
            print(f"✅ Test {i+1}: Outputs match (diff: {diff:.2e})")
    
    print(f"\nMaximum difference: {max_diff:.2e}")
    
    if all_close:
        print("✅ All tests passed! PyTorch and ONNX models are consistent.")
    else:
        print("❌ Some tests failed! Models may not be equivalent.")
    
    return all_close

def test_authorization_scenarios(onnx_session, vocab_mapping):
    """Test specific authorization scenarios"""
    
    print("\n🧪 Testing Authorization Scenarios")
    print("=" * 50)
    
    test_scenarios = [
        {
            'name': 'High trust, sufficient budget',
            'transaction': {
                'budget': 15, 'trust': 7, 'amount': 8, 'category': 0, 
                'velocity': 2, 'day': 1, 'time': 1, 'risk': 0
            },
            'expected': 'AUTHORIZED'
        },
        {
            'name': 'Amount exceeds budget',
            'transaction': {
                'budget': 5, 'trust': 4, 'amount': 12, 'category': 0,
                'velocity': 2, 'day': 1, 'time': 1, 'risk': 0
            },
            'expected': 'DENIED'
        },
        {
            'name': 'Low trust, high amount',
            'transaction': {
                'budget': 15, 'trust': 1, 'amount': 12, 'category': 0,
                'velocity': 2, 'day': 1, 'time': 1, 'risk': 0
            },
            'expected': 'DENIED'
        },
        {
            'name': 'High velocity transactions',
            'transaction': {
                'budget': 15, 'trust': 7, 'amount': 8, 'category': 0,
                'velocity': 7, 'day': 1, 'time': 1, 'risk': 0
            },
            'expected': 'DENIED'
        },
        {
            'name': 'Restricted category for untrusted merchant',
            'transaction': {
                'budget': 15, 'trust': 0, 'amount': 5, 'category': 2,
                'velocity': 2, 'day': 1, 'time': 1, 'risk': 0
            },
            'expected': 'DENIED'
        },
        {
            'name': 'Late night high-value transaction',
            'transaction': {
                'budget': 15, 'trust': 7, 'amount': 14, 'category': 0,
                'velocity': 2, 'day': 1, 'time': 3, 'risk': 0
            },
            'expected': 'DENIED'
        }
    ]
    
    correct_predictions = 0
    
    for scenario in test_scenarios:
        
        # Create one-hot vector
        one_hot = create_one_hot_vector(scenario['transaction'], vocab_mapping)
        
        # ONNX prediction
        onnx_input = one_hot.reshape(1, -1).astype(np.float32)
        onnx_output = onnx_session.run(None, {'input': onnx_input})[0]
        probs = torch.softmax(torch.FloatTensor(onnx_output), dim=1).numpy()[0]
        
        # Get prediction
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class] * 100
        
        if predicted_class == 0:
            prediction = 'AUTHORIZED'
            auth_prob = probs[0] * 100
            deny_prob = probs[1] * 100
        else:
            prediction = 'DENIED'
            auth_prob = probs[0] * 100
            deny_prob = probs[1] * 100
        
        # Check if correct
        is_correct = prediction == scenario['expected']
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {scenario['name']}")
        print(f"   Expected: {scenario['expected']}, Got: {prediction}")
        print(f"   Confidence: {confidence:.1f}% (Auth: {auth_prob:.1f}%, Deny: {deny_prob:.1f}%)")
        print()
    
    accuracy = correct_predictions / len(test_scenarios) * 100
    print(f"Scenario Test Accuracy: {correct_predictions}/{len(test_scenarios)} ({accuracy:.1f}%)")
    
    return accuracy

def test_edge_cases(onnx_session, vocab_mapping):
    """Test edge cases and boundary conditions"""
    
    print("\n🔬 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        {
            'name': 'All minimum values',
            'transaction': {
                'budget': 0, 'trust': 0, 'amount': 0, 'category': 0,
                'velocity': 0, 'day': 0, 'time': 0, 'risk': 0
            }
        },
        {
            'name': 'All maximum values',
            'transaction': {
                'budget': 15, 'trust': 7, 'amount': 15, 'category': 3,
                'velocity': 7, 'day': 6, 'time': 3, 'risk': 0
            }
        },
        {
            'name': 'Perfect authorization scenario',
            'transaction': {
                'budget': 15, 'trust': 7, 'amount': 5, 'category': 0,
                'velocity': 1, 'day': 2, 'time': 1, 'risk': 0
            }
        },
        {
            'name': 'Perfect denial scenario',
            'transaction': {
                'budget': 0, 'trust': 0, 'amount': 15, 'category': 3,
                'velocity': 7, 'day': 6, 'time': 3, 'risk': 0
            }
        }
    ]
    
    for case in edge_cases:
        
        # Create one-hot vector
        one_hot = create_one_hot_vector(case['transaction'], vocab_mapping)
        
        # ONNX prediction
        onnx_input = one_hot.reshape(1, -1).astype(np.float32)
        onnx_output = onnx_session.run(None, {'input': onnx_input})[0]
        probs = torch.softmax(torch.FloatTensor(onnx_output), dim=1).numpy()[0]
        
        # Get prediction
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class] * 100
        
        if predicted_class == 0:
            prediction = 'AUTHORIZED'
        else:
            prediction = 'DENIED'
        
        print(f"📋 {case['name']}")
        print(f"   Decision: {prediction}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Auth: {probs[0]*100:.1f}%, Deny: {probs[1]*100:.1f}%")
        print()

def test_model_robustness(onnx_session, vocab_mapping, num_tests=1000):
    """Test model robustness with random inputs"""
    
    print(f"🔄 Testing Model Robustness ({num_tests} random tests)")
    print("=" * 50)
    
    predictions = []
    confidences = []
    
    for _ in range(num_tests):
        
        # Generate random transaction
        transaction = generate_synthetic_transaction()
        one_hot = create_one_hot_vector(transaction, vocab_mapping)
        
        # ONNX prediction
        onnx_input = one_hot.reshape(1, -1).astype(np.float32)
        onnx_output = onnx_session.run(None, {'input': onnx_input})[0]
        probs = torch.softmax(torch.FloatTensor(onnx_output), dim=1).numpy()[0]
        
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        predictions.append(predicted_class)
        confidences.append(confidence)
    
    # Calculate statistics
    auth_rate = np.mean([p == 0 for p in predictions]) * 100
    deny_rate = np.mean([p == 1 for p in predictions]) * 100
    avg_confidence = np.mean(confidences) * 100
    min_confidence = np.min(confidences) * 100
    max_confidence = np.max(confidences) * 100
    
    print(f"Authorization rate: {auth_rate:.1f}%")
    print(f"Denial rate: {deny_rate:.1f}%")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print(f"Confidence range: {min_confidence:.1f}% - {max_confidence:.1f}%")
    
    # Check for any NaN or invalid outputs
    has_nan = np.any(np.isnan(confidences))
    has_inf = np.any(np.isinf(confidences))
    
    if has_nan or has_inf:
        print("❌ Model produced NaN or infinite values!")
    else:
        print("✅ Model is robust - no NaN or infinite values detected.")

def main():
    """Run all tests"""
    
    print("🎯 Authorization Model Testing Suite")
    print("=" * 60)
    
    try:
        # Load model artifacts
        pytorch_model, onnx_session, vocab_mapping, feature_mapping, meta = load_model_artifacts()
        
        print("✅ Model artifacts loaded successfully")
        print(f"📊 Model info: {meta['input_size']} → {meta['hidden_size']} → {meta['output_size']}")
        print(f"📈 Training accuracy: {meta['training_accuracy']:.1f}%")
        print()
        
        # Test PyTorch vs ONNX consistency
        consistency_passed = test_pytorch_vs_onnx(pytorch_model, onnx_session, vocab_mapping)
        
        # Test authorization scenarios
        scenario_accuracy = test_authorization_scenarios(onnx_session, vocab_mapping)
        
        # Test edge cases
        test_edge_cases(onnx_session, vocab_mapping)
        
        # Test robustness
        test_model_robustness(onnx_session, vocab_mapping)
        
        # Summary
        print("\n📋 Test Summary")
        print("=" * 50)
        print(f"✅ Model consistency: {'PASS' if consistency_passed else 'FAIL'}")
        print(f"📊 Scenario accuracy: {scenario_accuracy:.1f}%")
        print(f"🎯 Overall assessment: {'GOOD' if consistency_passed and scenario_accuracy >= 80 else 'NEEDS IMPROVEMENT'}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise

if __name__ == '__main__':
    main()
