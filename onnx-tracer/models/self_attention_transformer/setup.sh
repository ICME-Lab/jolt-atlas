#!/bin/bash

# Setup script for creating the fixed batch size ReLU-based self-attention model

set -e

echo "Setting up virtual environment for self_attention_fixed_batch model..."

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "Generating ONNX model with fixed batch size..."

# Generate the ONNX model
python gen.py

echo "Validating ONNX model..."

# Validate the model
python validate.py

echo "Setup and validation complete!"
echo "Files generated:"
echo "  - network.onnx (ONNX model with fixed batch size)"
echo "  - input.json (test data)"
echo ""
echo "Model characteristics:"
echo "  - Input shape: [16, 16] (no batch dimension)"
echo "  - Output shape: [16, 16] (no batch dimension)"
echo "  - Fixed batch size of 1"
echo ""
echo "To activate the virtual environment manually:"
echo "  source .venv/bin/activate"
