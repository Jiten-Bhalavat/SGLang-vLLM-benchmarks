# Local GPU Setup Guide - RTX 4070 (8GB VRAM)

## 🎯 Benchmark Matrix: 3 Models × 2 Frameworks

| Model | Parameters | VRAM | vLLM | SGLang |
|-------|------------|------|------|--------|
| `facebook/opt-125m` | 125M | ~0.5 GB | ✓ | ✓ |
| `Qwen/Qwen2-1.5B` | 1.5B | ~3.5 GB | ✓ | ✓ |
| `microsoft/phi-2` | 2.7B | ~6 GB | ✓ | ✓ |

**Total: 6 benchmarks**

---

## 📋 Prerequisites Checklist

- [x] NVIDIA RTX 4070 (8GB VRAM)
- [x] CUDA drivers installed
- [ ] Docker Desktop for Windows
- [ ] WSL2 (Windows Subsystem for Linux)
- [ ] NVIDIA Container Toolkit

---

## 🚀 STEP 1: Install Docker Desktop

1. **Download**: https://www.docker.com/products/docker-desktop/
2. **Install** and follow the prompts
3. **Restart your computer**
4. **Open Docker Desktop** and complete initial setup
5. Accept the terms and skip the tutorial

---

## 🐧 STEP 2: Install WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

**Restart your computer** after installation.

After restart, Ubuntu will open automatically to complete setup:
- Create a username (e.g., `benchmark`)
- Create a password (remember this!)

---

## 🔧 STEP 3: Configure Docker for GPU

### 3.1 Enable WSL Integration in Docker Desktop

1. Open **Docker Desktop**
2. Go to **Settings** (gear icon)
3. **Resources** → **WSL Integration**
4. Toggle ON for **Ubuntu**
5. Click **Apply & Restart**

### 3.2 Install NVIDIA Container Toolkit in WSL

Open **Ubuntu** (from Start menu) and run:

```bash
# Update packages
sudo apt-get update

# Install prerequisites
sudo apt-get install -y curl

# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
```

### 3.3 Restart Docker Desktop

1. Right-click Docker icon in system tray
2. Click **Restart**

### 3.4 Test GPU Access

In Ubuntu terminal:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your RTX 4070 GPU info!

---

## 📁 STEP 4: Copy Project to WSL

In Ubuntu terminal:

```bash
# Create projects directory
mkdir -p ~/projects
cd ~/projects

# Copy from Windows (adjust the path if your Windows username is different)
# The /mnt/d/ corresponds to D: drive
cp -r "/mnt/d/International/Study Materials/Semester-3/MSML 650 - Cloud/Project/cloud-llm-inference-benchmark" ./

# Go to project
cd cloud-llm-inference-benchmark

# Verify files
ls -la
```

---

## 🐳 STEP 5: Build Docker Images

This step takes about **20-25 minutes** total.

```bash
# Build vLLM image (~10-12 minutes)
echo "Building vLLM Docker image..."
docker build -f docker/Dockerfile.vllm -t vllm-server .

# Build SGLang image (~10-12 minutes)
echo "Building SGLang Docker image..."
docker build -f docker/Dockerfile.sglang -t sglang-server .

echo "Done! Both images built successfully."
```

---

## 🏃 STEP 6: Run Full Benchmark Suite

### Option A: Automated Script (Recommended)

```bash
# Set HuggingFace token
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Make script executable
chmod +x scripts/run_local_benchmark.sh

# Run all 6 benchmarks
./scripts/run_local_benchmark.sh
```

This will automatically run all 6 benchmarks (3 models × 2 frameworks).

**Estimated time: 1.5-2 hours**

### Option B: Manual Step-by-Step

If you prefer to run benchmarks one at a time:

```bash
# Set token
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Install Python dependencies
pip install -r requirements.txt
```

#### Benchmark 1: vLLM + OPT-125M
```bash
docker run -d --gpus all --name vllm-server -p 8000:8000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="facebook/opt-125m" vllm-server

# Wait ~2 min, then check: curl http://localhost:8000/health

python3 src/benchmark_runner.py --framework vllm \
    --config configs/workload_medium.json --model "facebook/opt-125m"

docker stop vllm-server && docker rm vllm-server
```

#### Benchmark 2: SGLang + OPT-125M
```bash
docker run -d --gpus all --name sglang-server -p 30000:30000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="facebook/opt-125m" sglang-server

# Wait ~2 min, then check: curl http://localhost:30000/health

python3 src/benchmark_runner.py --framework sglang \
    --config configs/workload_medium.json --model "facebook/opt-125m"

docker stop sglang-server && docker rm sglang-server
```

#### Benchmark 3: vLLM + Qwen2-1.5B
```bash
docker run -d --gpus all --name vllm-server -p 8000:8000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="Qwen/Qwen2-1.5B" vllm-server

# Wait ~5 min for larger model

python3 src/benchmark_runner.py --framework vllm \
    --config configs/workload_medium.json --model "Qwen/Qwen2-1.5B"

docker stop vllm-server && docker rm vllm-server
```

#### Benchmark 4: SGLang + Qwen2-1.5B
```bash
docker run -d --gpus all --name sglang-server -p 30000:30000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="Qwen/Qwen2-1.5B" sglang-server

python3 src/benchmark_runner.py --framework sglang \
    --config configs/workload_medium.json --model "Qwen/Qwen2-1.5B"

docker stop sglang-server && docker rm sglang-server
```

#### Benchmark 5: vLLM + Phi-2
```bash
docker run -d --gpus all --name vllm-server -p 8000:8000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="microsoft/phi-2" vllm-server

# Wait ~5 min

python3 src/benchmark_runner.py --framework vllm \
    --config configs/workload_medium.json --model "microsoft/phi-2"

docker stop vllm-server && docker rm vllm-server
```

#### Benchmark 6: SGLang + Phi-2
```bash
docker run -d --gpus all --name sglang-server -p 30000:30000 \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -e MODEL_NAME="microsoft/phi-2" sglang-server

python3 src/benchmark_runner.py --framework sglang \
    --config configs/workload_medium.json --model "microsoft/phi-2"

docker stop sglang-server && docker rm sglang-server
```

---

## 📊 STEP 7: Generate Analysis Report

```bash
python3 src/parse_results.py --results-dir results/ --export-csv --generate-report
```

---

## 📈 STEP 8: Visualize Results

### Option A: Copy results to Windows and open Jupyter

```bash
# Copy results back to Windows
cp -r results "/mnt/d/International/Study Materials/Semester-3/MSML 650 - Cloud/Project/cloud-llm-inference-benchmark/"
```

Then on Windows, open PowerShell:
```powershell
cd "D:\International\Study Materials\Semester-3\MSML 650 - Cloud\Project\cloud-llm-inference-benchmark"
pip install jupyter pandas matplotlib seaborn
jupyter notebook notebooks/benchmark_visualization.ipynb
```

### Option B: Run Jupyter in WSL

```bash
pip install jupyter
jupyter notebook --no-browser
```

Copy the URL and open in Windows browser.

---

## ⏱️ Time Estimate

| Step | Time |
|------|------|
| Install Docker Desktop | 10 min |
| Install WSL2 | 10 min |
| Configure GPU access | 10 min |
| Build Docker images | 25 min |
| Run 6 benchmarks | 60-90 min |
| Generate report | 5 min |
| **Total** | **~2-2.5 hours** |

---

## 🔧 Troubleshooting

### "nvidia-smi" not found in WSL
The NVIDIA driver is shared from Windows. Make sure:
- You have the latest NVIDIA driver on Windows
- Docker Desktop is running

### Docker GPU test fails
```bash
# Restart Docker Desktop
# Then in WSL:
sudo systemctl restart docker
```

### Out of memory error
- Close other GPU applications (games, etc.)
- Use `workload_small.json` instead of medium

### Container crashes immediately
```bash
# Check logs
docker logs vllm-server
docker logs sglang-server
```

### Server takes too long to start
Larger models (Phi-2) can take 5-10 minutes to load. Be patient!

---

## 📝 For Your Report

When writing your report, include:

**Hardware Setup:**
- GPU: NVIDIA RTX 4070 (8GB VRAM)
- OS: Windows 11 with WSL2
- Docker: Docker Desktop with NVIDIA Container Toolkit

**Models Tested:**
- OPT-125M (125M parameters) - Baseline small model
- Qwen2-1.5B (1.5B parameters) - Medium model
- Phi-2 (2.7B parameters) - Larger model

**Frameworks Compared:**
- vLLM (PagedAttention, continuous batching)
- SGLang (RadixAttention, structured generation)

This gives you a comprehensive comparison across model sizes!
