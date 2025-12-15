#!/bin/bash
# ============================================================================
# AWS EC2 Instance Setup Script
# Cloud LLM Inference Benchmark Project
# MSML 650 - Cloud Computing
# ============================================================================
# This script sets up an AWS EC2 GPU instance with all necessary dependencies
# for running vLLM and SGLang inference benchmarks.
# 
# Usage: Run this script after SSHing into your EC2 instance
#   chmod +x setup_instance.sh
#   ./setup_instance.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Cloud LLM Benchmark - Instance Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================
# 1. System Update
# ============================================
echo ""
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

# ============================================
# 2. Install Basic Dependencies
# ============================================
echo ""
echo "Step 2: Installing basic dependencies..."
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    python3 \
    python3-pip \
    python3-venv \
    unzip

print_status "Basic dependencies installed"

# ============================================
# 3. Install Docker
# ============================================
echo ""
echo "Step 3: Installing Docker..."

# Remove old versions if exist
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install Docker
sudo apt install -y docker.io

# Start Docker and enable on boot
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
sudo usermod -aG docker $USER

print_status "Docker installed"

# ============================================
# 4. Install NVIDIA Container Toolkit
# ============================================
echo ""
echo "Step 4: Installing NVIDIA Container Toolkit..."

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "NVIDIA driver not found. Using Deep Learning AMI is recommended."
    print_warning "If using a custom AMI, install NVIDIA drivers first."
else
    print_status "NVIDIA driver found"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

print_status "NVIDIA Container Toolkit installed"

# ============================================
# 5. Setup Python Environment
# ============================================
echo ""
echo "Step 5: Setting up Python environment..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies for benchmarking
pip3 install --user \
    requests \
    aiohttp \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    rich \
    python-dotenv \
    openai

print_status "Python environment setup complete"

# ============================================
# 6. Create Working Directory
# ============================================
echo ""
echo "Step 6: Setting up working directory..."

# Create benchmark directory if not exists
mkdir -p ~/cloud-llm-inference-benchmark
mkdir -p ~/cloud-llm-inference-benchmark/results/vllm
mkdir -p ~/cloud-llm-inference-benchmark/results/sglang

print_status "Working directory created"

# ============================================
# 7. Verify Installation
# ============================================
echo ""
echo "=============================================="
echo "  Verifying Installation"
echo "=============================================="

# Check Docker
if docker --version &> /dev/null; then
    print_status "Docker: $(docker --version)"
else
    print_error "Docker installation failed"
fi

# Check NVIDIA Docker
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_status "NVIDIA Docker runtime: Working"
else
    print_warning "NVIDIA Docker runtime: May need to logout and login again"
fi

# Check Python
print_status "Python: $(python3 --version)"

# ============================================
# Final Instructions
# ============================================
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: Log out and log back in for Docker group changes to take effect."
echo ""
echo "Next steps:"
echo "  1. Log out and log back in: exit"
echo "  2. Upload your project files"
echo "  3. Set your HuggingFace token: export HUGGINGFACE_TOKEN='your_token'"
echo "  4. Build and run Docker containers"
echo ""
echo "Quick test after re-login:"
echo "  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo ""

