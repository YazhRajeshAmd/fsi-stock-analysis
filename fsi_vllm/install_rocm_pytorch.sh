#!/bin/bash

# Install ROCm version of PyTorch for AMD MI300X
echo "Installing ROCm PyTorch for AMD MI300X..."

# Uninstall CUDA version of PyTorch
pip uninstall torch torchvision torchaudio -y

# Install ROCm version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

echo "ROCm PyTorch installation completed!"
