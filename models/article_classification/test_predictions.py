#!/usr/bin/env python3

import torch
import torch.nn as nn
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the model class (needed for loading)
class MLPClassifierSmall(nn.Module):
    """
    Smaller MLP classifier with power-of-two dimensions for SNARK compatibility.
    Architecture: Input -> Linear(512->256) -> ReLU -> Linear(256->128) -> ReLU -> Linear(128->8)
    Max dimension: 512
    """
    def __init__(self, vocab_size=512, hidden_size=256, num_classes=8):
        super(MLPClassifierSmall, self).__init__()
        
        # All dimensions are powers of 2 for SNARK efficiency, max 512
        self.fc1 = nn.Linear(vocab_size, hidden_size)       # 512 -> 256
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2) # 256 -> 128  
        self.fc3 = nn.Linear(hidden_size // 2, num_classes) # 128 -> 8
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input: [batch_size, 512]
        x = self.fc1(x)           # [batch_size, 256]
        x = self.relu(x)          # [batch_size, 256]
        x = self.fc2(x)           # [batch_size, 128]
        x = self.relu(x)          # [batch_size, 128]
        x = self.fc3(x)           # [batch_size, 8]
        return x

# Load the model and mappings
model = torch.load('model.pt', weights_only=False)
model.eval()

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
    features = np.zeros(max_features)
    
    for word in words:
        if word in vocab_dict:
            idx = vocab_dict[word]['index']
            if idx < max_features:
                features[idx] += 1.0  # Simple term frequency
    
    return features

# Test texts
test_texts = [
    "The government plans new trade policies.",
    "The latest computer model has impressive features.",
    "The football match ended in a thrilling draw.",
    "The new movie has received rave reviews from critics.",
    "The stock market saw a significant drop today."
]

expected_classes = ["business", "tech", "sport", "entertainment", "business"]

print("Testing Small MLP model predictions:")
print("-" * 50)

for i, text in enumerate(test_texts):
    # Preprocess text
    features = preprocess_text(text, vocab)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        confidence = output[0][predicted_class_idx].item()
    
    # Map to class name (only use first 5 classes, ignore dummy classes)
    if predicted_class_idx < 5:
        predicted_class = class_mapping[str(predicted_class_idx)]
    else:
        predicted_class = "unknown"
    
    expected = expected_classes[i]
    match = "✅" if predicted_class == expected else "❌"
    
    print(f"Text {i+1}: '{text}'")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted_class} (confidence: {confidence:.4f})")
    print(f"Match: {match}")
    
    # Show all predictions for first 5 classes
    print("All predictions:")
    real_classes = ["business", "entertainment", "politics", "sport", "tech"]
    for j, class_name in enumerate(real_classes):
        print(f"  {class_name}: {output[0][j].item():.4f}")
    print()
