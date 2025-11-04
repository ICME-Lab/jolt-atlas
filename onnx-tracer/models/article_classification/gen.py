#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import re
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Configuration (Smaller dimensions)
# --------------------------
VOCAB_SIZE = 512   # Power of 2, reduced from 1024
HIDDEN_SIZE = 256  # Power of 2, reduced from 512
NUM_CLASSES = 8    # Pad to power of 2 (original 5 classes)
BATCH_SIZE = 32
EPOCHS = 25        # Slightly more epochs for smaller model
LEARNING_RATE = 0.001

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

# --------------------------
# 1. Load and preprocess data
# --------------------------
print("Loading BBC dataset...")
df = pd.read_csv('bbc_data.csv')

print(f"Dataset loaded: {len(df)} articles")
print("Label distribution:")
print(df['labels'].value_counts())

# Get unique labels
unique_labels = sorted(df['labels'].unique())
print(f"Loaded {len(df)} texts with {len(unique_labels)} unique labels: {set(unique_labels)}")

# --------------------------
# 2. Create TF-IDF features with smaller power-of-2 vocab size
# --------------------------
# Use top 512 features (power of 2) for SNARK efficiency
vectorizer = TfidfVectorizer(
    max_features=VOCAB_SIZE,
    stop_words='english',
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['data'])
X_dense = X.toarray()

# Save vocabulary mapping for inference
vocab = vectorizer.get_feature_names_out()
vocab_mapping = {}
for idx, word in enumerate(vocab):
    # For compatibility with existing code, store index and a dummy IDF
    vocab_mapping[word] = {
        "index": idx,
        "idf": 1.0  # Simplified for MLP - actual TF-IDF is handled in preprocessing
    }

with open('vocab.json', 'w') as f:
    json.dump(vocab_mapping, f, indent=2)

print(f"✅ Vocabulary saved with {len(vocab_mapping)} features")

# --------------------------
# 3. Encode labels and pad to power of 2
# --------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['labels'])

# Create mapping for the 5 real classes to 8 padded classes
real_classes = label_encoder.classes_
class_mapping = {i: real_classes[i] for i in range(len(real_classes))}
# Pad with dummy classes for power of 2
for i in range(len(real_classes), NUM_CLASSES):
    class_mapping[i] = f"dummy_class_{i}"

# Save class mapping
with open('labels.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)

print(f"✅ Class mapping saved: {class_mapping}")

# --------------------------
# 4. Split data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_dense, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --------------------------
# 5. Train model
# --------------------------
model = MLPClassifierSmall(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Only use the first 5 outputs for the real classes
        outputs_real = outputs[:, :len(real_classes)]
        loss = criterion(outputs_real, batch_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# --------------------------
# 6. Evaluate model
# --------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # Only use the first 5 outputs for prediction
    test_outputs_real = test_outputs[:, :len(real_classes)]
    test_predictions = torch.argmax(test_outputs_real, dim=1)
    
accuracy = accuracy_score(y_test, test_predictions.numpy())
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_predictions.numpy(), target_names=real_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions.numpy()))

# --------------------------
# 7. Export ONNX model
# --------------------------
model.eval()
# Use batch_size=1 for inference
dummy_input = torch.randn(1, VOCAB_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("✅ ONNX model exported successfully")

# --------------------------
# 8. Save PyTorch model for testing
# --------------------------
torch.save(model, 'model.pt')
print("✅ PyTorch model saved")

print("✅ Training complete, ONNX and mappings saved.")
