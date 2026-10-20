#!/usr/bin/env python3

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# Try importing ONNX
try:
    import onnx
    import torch.onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Model export will be disabled.")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------
# Configuration (Power-of-2 dimensions for SNARK compatibility)
# --------------------------
INPUT_SIZE = 64     # Power of 2 for one-hot encoded features
HIDDEN_SIZE = 32    # Power of 2, smaller than article classification
NUM_CLASSES = 4     # Power of 2 (2 real classes: authorized/denied + 2 padding)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Feature dimensions (all power of 2)
BUDGET_BUCKETS = 16      # Budget ranges
TRUST_BUCKETS = 8        # Merchant trust levels
AMOUNT_BUCKETS = 16      # Transaction amount ranges  
CATEGORY_BUCKETS = 4     # Transaction categories
VELOCITY_BUCKETS = 8     # Transaction velocity
DAY_BUCKETS = 8          # Day of week (padded to power of 2)
TIME_BUCKETS = 4         # Time of day
RISK_BUCKETS = 0         # Risk score buckets (removed to fit 64 dimensions)

# Verify total input size
TOTAL_FEATURES = (BUDGET_BUCKETS + TRUST_BUCKETS + AMOUNT_BUCKETS + 
                 CATEGORY_BUCKETS + VELOCITY_BUCKETS + DAY_BUCKETS + 
                 TIME_BUCKETS + RISK_BUCKETS)
assert TOTAL_FEATURES == INPUT_SIZE, f"Feature dimensions don't match INPUT_SIZE: {TOTAL_FEATURES} != {INPUT_SIZE}"


class MLPAuthorizationClassifier(nn.Module):
    """
    MLP classifier for authorization decisions with power-of-two dimensions.
    Architecture follows article classification pattern:
    Input -> Linear(64->32) -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->4)
    No sigmoid - outputs raw logits for classification.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super(MLPAuthorizationClassifier, self).__init__()
        
        # All dimensions are powers of 2 for SNARK efficiency
        self.fc1 = nn.Linear(input_size, hidden_size)           # 64 -> 32
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)     # 32 -> 16  
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)     # 16 -> 4
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input: [batch_size, 64] - one-hot encoded features
        x = self.fc1(x)           # [batch_size, 32]
        x = self.relu(x)          # [batch_size, 32]
        x = self.fc2(x)           # [batch_size, 16]
        x = self.relu(x)          # [batch_size, 16]
        x = self.fc3(x)           # [batch_size, 4] - raw logits
        return x


def create_feature_mapping():
    """Create feature encoding mappings for one-hot representation"""
    
    feature_ranges = {
        'budget': BUDGET_BUCKETS,
        'trust': TRUST_BUCKETS, 
        'amount': AMOUNT_BUCKETS,
        'category': CATEGORY_BUCKETS,
        'velocity': VELOCITY_BUCKETS,
        'day': DAY_BUCKETS,
        'time': TIME_BUCKETS,
        'risk': RISK_BUCKETS
    }
    
    # Create vocab mapping
    vocab_mapping = {}
    feature_mapping = {}
    current_idx = 0
    
    for feature_type, bucket_count in feature_ranges.items():
        feature_values = [f'{feature_type}_{i}' for i in range(bucket_count)]
        feature_mapping[feature_type] = feature_values
        
        for value in feature_values:
            vocab_mapping[value] = {
                "index": current_idx,
                "feature_type": feature_type
            }
            current_idx += 1
    
    return vocab_mapping, feature_mapping


def generate_synthetic_transaction():
    """Generate a single synthetic transaction with realistic authorization patterns"""
    
    # Generate base features
    budget_level = random.randint(0, BUDGET_BUCKETS - 1)  # 0 = low budget, 15 = high budget
    trust_level = random.randint(0, TRUST_BUCKETS - 1)    # 0 = untrusted, 7 = highly trusted
    amount_level = random.randint(0, AMOUNT_BUCKETS - 1)  # 0 = small amount, 15 = large amount
    category = random.randint(0, CATEGORY_BUCKETS - 1)    # 0-3: different merchant categories
    velocity = random.randint(0, VELOCITY_BUCKETS - 1)    # 0 = low velocity, 7 = high velocity
    day_of_week = random.randint(0, 6)                    # 0-6 for actual days, 7 is padding
    time_of_day = random.randint(0, TIME_BUCKETS - 1)     # 0-3: time periods
    risk_score = 0  # Not used since RISK_BUCKETS = 0
    
    # Authorization logic with realistic business rules
    authorized = True
    
    # Rule 1: Insufficient budget
    if amount_level > budget_level + 2:  # Amount significantly exceeds budget
        authorized = False
    
    # Rule 2: Low trust merchant with high amount
    if trust_level <= 2 and amount_level >= 10:
        authorized = False
    
    # Rule 3: High velocity (too many transactions)
    if velocity >= 6:
        authorized = False
        
    # Rule 4: Restricted categories for untrusted merchants
    if trust_level <= 1 and category in [2, 3]:  # Categories 2,3 are restricted
        authorized = False
    
    # Rule 5: Late night high-value transactions (suspicious)
    if time_of_day == 3 and amount_level >= 12:  # Late night + high amount
        authorized = False
    
    # Add some noise - occasionally approve borderline cases
    if not authorized and random.random() < 0.1:  # 10% chance to approve denied cases
        authorized = True
    
    # Add some noise - occasionally deny good cases  
    if authorized and random.random() < 0.05:  # 5% chance to deny approved cases
        authorized = False
    
    return {
        'budget': budget_level,
        'trust': trust_level,
        'amount': amount_level,
        'category': category,
        'velocity': velocity,
        'day': day_of_week,
        'time': time_of_day,
        'risk': risk_score,
        'authorized': authorized
    }


def create_one_hot_vector(transaction, vocab_mapping):
    """Convert transaction to one-hot encoded vector"""
    
    vector = np.zeros(INPUT_SIZE)
    
    # Map each feature to its one-hot position
    features = ['budget', 'trust', 'amount', 'category', 'velocity', 'day', 'time', 'risk']
    
    for feature in features:
        value = transaction[feature]
        feature_key = f'{feature}_{value}'
        
        if feature_key in vocab_mapping:
            idx = vocab_mapping[feature_key]['index']
            vector[idx] = 1.0
    
    return vector


def generate_dataset(num_samples=10000):
    """Generate synthetic authorization dataset"""
    
    print(f"Generating {num_samples} synthetic transactions...")
    
    vocab_mapping, feature_mapping = create_feature_mapping()
    
    X = []
    y = []
    
    for i in range(num_samples):
        transaction = generate_synthetic_transaction()
        
        # Create one-hot input vector
        input_vector = create_one_hot_vector(transaction, vocab_mapping)
        X.append(input_vector)
        
        # Create label (one-hot for 4 classes: [authorized, denied, padding, padding])
        if transaction['authorized']:
            label = [1, 0, 0, 0]  # Authorized
        else:
            label = [0, 1, 0, 0]  # Denied
        
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Print dataset statistics
    authorized_count = np.sum(y[:, 0])
    denied_count = np.sum(y[:, 1])
    print(f"Authorized: {authorized_count} ({authorized_count/num_samples*100:.1f}%)")
    print(f"Denied: {denied_count} ({denied_count/num_samples*100:.1f}%)")
    
    return X, y, vocab_mapping, feature_mapping

def train_model(X, y, vocab_mapping):
    """Train the authorization model"""
    
    print("\n🧠 Training Authorization Model")
    print("=" * 50)
    
    # Split into train/validation sets
    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    
    indices = np.random.permutation(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = MLPAuthorizationClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Convert one-hot labels to class indices
            label_indices = torch.argmax(labels, dim=1)
            loss = criterion(outputs, label_indices)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                actual = torch.argmax(labels, dim=1)
                
                val_correct += (predicted == actual).sum().item()
                val_total += labels.size(0)
        
        train_loss /= len(train_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'authorization_model.pth')
    
    print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('authorization_model.pth'))
    
    # Calculate final training accuracy
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predicted = torch.argmax(train_outputs, dim=1)
        train_labels = torch.argmax(y_train_tensor, dim=1)
        final_train_acc = (train_predicted == train_labels).float().mean().item() * 100
    
    return model, final_train_acc, best_val_acc


def test_model_performance(model, X, y):
    """Test model performance on the dataset"""
    
    print("\n📊 Model Performance Analysis")
    print("=" * 50)
    
    model.eval()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted = torch.argmax(outputs, dim=1)
        actual = torch.argmax(y_tensor, dim=1)
        
        # Calculate accuracy
        accuracy = (predicted == actual).float().mean().item() * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Calculate per-class metrics
        authorized_mask = (actual == 0)
        denied_mask = (actual == 1)
        
        if authorized_mask.sum() > 0:
            auth_acc = (predicted[authorized_mask] == actual[authorized_mask]).float().mean().item() * 100
            print(f"Authorized Class Accuracy: {auth_acc:.2f}%")
        
        if denied_mask.sum() > 0:
            deny_acc = (predicted[denied_mask] == actual[denied_mask]).float().mean().item() * 100
            print(f"Denied Class Accuracy: {deny_acc:.2f}%")
        
        # Print confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual.numpy(), predicted.numpy())
        print(f"\nConfusion Matrix:")
        print(f"             Predicted")
        print(f"           Auth  Deny")
        print(f"Actual Auth {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"       Deny {cm[1,0]:4d} {cm[1,1]:4d}")


def test_sample_scenarios(model, vocab_mapping):
    """Test the model with specific authorization scenarios"""
    
    print("\n🧪 Testing Sample Authorization Scenarios")
    print("=" * 50)
    
    model.eval()
    
    test_scenarios = [
        {
            'name': 'High trust merchant, good budget (should authorize)',
            'transaction': {
                'budget': 12, 'trust': 7, 'amount': 8, 'category': 0,
                'velocity': 2, 'day': 1, 'time': 1, 'risk': 1
            }
        },
        {
            'name': 'Amount exceeds budget (should deny)', 
            'transaction': {
                'budget': 3, 'trust': 5, 'amount': 12, 'category': 1,
                'velocity': 3, 'day': 2, 'time': 2, 'risk': 2
            }
        },
        {
            'name': 'Untrusted merchant, high amount (should deny)',
            'transaction': {
                'budget': 10, 'trust': 1, 'amount': 11, 'category': 2,
                'velocity': 4, 'day': 3, 'time': 1, 'risk': 3
            }
        },
        {
            'name': 'High velocity transactions (should deny)',
            'transaction': {
                'budget': 8, 'trust': 6, 'amount': 5, 'category': 0,
                'velocity': 7, 'day': 4, 'time': 2, 'risk': 2
            }
        },
        {
            'name': 'High risk score (should deny)',
            'transaction': {
                'budget': 9, 'trust': 4, 'amount': 6, 'category': 1,
                'velocity': 3, 'day': 1, 'time': 1, 'risk': 7
            }
        },
        {
            'name': 'Late night high-value transaction (should deny)',
            'transaction': {
                'budget': 15, 'trust': 5, 'amount': 13, 'category': 0,
                'velocity': 2, 'day': 5, 'time': 3, 'risk': 3
            }
        }
    ]
    
    with torch.no_grad():
        for scenario in test_scenarios:
            # Create one-hot vector
            input_vector = create_one_hot_vector(scenario['transaction'], vocab_mapping)
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            
            # Get prediction
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            
            auth_prob = probabilities[0][0].item()
            deny_prob = probabilities[0][1].item()
            
            decision = "AUTHORIZED" if predicted_class == 0 else "DENIED"
            confidence = max(auth_prob, deny_prob)
            
            print(f"\n{scenario['name']}")
            print(f"  Decision: {decision}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Auth Prob: {auth_prob:.1%}, Deny Prob: {deny_prob:.1%}")


def export_to_onnx(model, output_path='network.onnx'):
    """Export trained model to ONNX format"""
    
    if not ONNX_AVAILABLE:
        print("❌ Cannot export to ONNX: torch.onnx not available")
        return False
    
    print(f"\n📦 Exporting model to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input with correct size (64 features)
    dummy_input = torch.randn(1, INPUT_SIZE)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Model exported successfully to {output_path}")
        
        # Verify export
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verified")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False


def save_metadata(vocab_mapping, feature_mapping, model_stats):
    """Save model metadata for testing"""
    
    # Save vocab with the correct structure for test.py
    vocab_data = {
        'vocab_mapping': vocab_mapping,
        'feature_mapping': feature_mapping
    }
    
    with open('vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    # Save model metadata
    metadata = {
        "model_type": "authorization_classifier",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": NUM_CLASSES,
        "training_accuracy": model_stats['training_accuracy'],
        "validation_accuracy": model_stats['validation_accuracy'],
        "architecture": {
            "type": "MLP",
            "layers": [
                {"type": "Linear", "in_features": INPUT_SIZE, "out_features": HIDDEN_SIZE},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": HIDDEN_SIZE, "out_features": HIDDEN_SIZE // 2},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": HIDDEN_SIZE // 2, "out_features": NUM_CLASSES}
            ]
        },
        "feature_buckets": {
            "budget": BUDGET_BUCKETS,
            "trust": TRUST_BUCKETS,
            "amount": AMOUNT_BUCKETS,
            "category": CATEGORY_BUCKETS,
            "velocity": VELOCITY_BUCKETS,
            "day": DAY_BUCKETS,
            "time": TIME_BUCKETS,
            "risk": RISK_BUCKETS
        }
    }
    
    with open('meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Metadata saved to vocab.json and meta.json")


if __name__ == '__main__':
    print("🚀 Starting Authorization Model Training")
    print("=" * 60)
    
    # Generate dataset
    X, y, vocab_mapping, feature_mapping = generate_dataset(num_samples=10000)
    
    # Train model
    model, train_acc, val_acc = train_model(X, y, vocab_mapping)
    
    # Test model performance
    test_model_performance(model, X, y)
    
    # Test sample scenarios
    test_sample_scenarios(model, vocab_mapping)
    
    # Export to ONNX
    export_success = export_to_onnx(model, 'network.onnx')
    
    # Save metadata
    model_stats = {
        "dataset_size": len(X),
        "train_size": int(0.8 * len(X)),
        "val_size": int(0.2 * len(X)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_accuracy": train_acc,
        "validation_accuracy": val_acc
    }
    save_metadata(vocab_mapping, feature_mapping, model_stats)
    
    print(f"\n✨ Authorization model training complete!")
    if export_success:
        print(f"📁 Model saved as: network.onnx")
        print(f"📁 Metadata saved as: vocab.json, meta.json")
    
    print("\n📋 Model Summary:")
    print(f"  • Input size: {INPUT_SIZE} (one-hot encoded features)")
    print(f"  • Architecture: {INPUT_SIZE} → {HIDDEN_SIZE} → {HIDDEN_SIZE//2} → {NUM_CLASSES}")
    print(f"  • Classes: Authorized (0), Denied (1), Padding (2,3)")
    print(f"  • Features: Budget, Trust, Amount, Category, Velocity, Day, Time, Risk")