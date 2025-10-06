#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import json
import re

# Load vocab and class mappings
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

with open('labels.json', 'r') as f:
    class_mapping = json.load(f)

def preprocess_text(text, vocab_dict, max_features=512):
    """Convert text to TF-IDF feature vector"""
    # Simple tokenization (should match training preprocessing)
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Create feature vector
    features = np.zeros(max_features, dtype=np.float32)
    
    for word in words:
        if word in vocab_dict:
            idx = vocab_dict[word]['index']
            if idx < max_features:
                features[idx] += 1.0  # Simple term frequency
    
    return features

print("Testing ONNX Small MLP model:")
print("-" * 30)

# Load ONNX model
session = ort.InferenceSession("network.onnx")

# Test with single input
test_text = "The government plans new trade policies."
features = preprocess_text(test_text, vocab)

print("Testing with batch_size=1:")
input_data = features.reshape(1, -1)
print(f"Input shape: {input_data.shape}")

# Run inference
output = session.run(None, {"input": input_data})[0]
print(f"Output shape: {output.shape}")
print(f"Output (first 5): {output[0][:5]}")

predicted_class_idx = np.argmax(output[0][:5])  # Only consider first 5 real classes
real_classes = ["business", "entertainment", "politics", "sport", "tech"]
predicted_class = real_classes[predicted_class_idx]
confidence = output[0][predicted_class_idx]

print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")

print("\nTesting with batch_size=2:")
input_data_batch = np.vstack([features, features])
print(f"Input shape: {input_data_batch.shape}")

output_batch = session.run(None, {"input": input_data_batch})[0]
print(f"Output shape: {output_batch.shape}")
print(f"Output row 0 (first 5): {output_batch[0][:5]}")
print(f"Output row 1 (first 5): {output_batch[1][:5]}")

pred0 = real_classes[np.argmax(output_batch[0][:5])]
pred1 = real_classes[np.argmax(output_batch[1][:5])]
print(f"Predicted class (row 0): {pred0}")
print(f"Predicted class (row 1): {pred1}")
