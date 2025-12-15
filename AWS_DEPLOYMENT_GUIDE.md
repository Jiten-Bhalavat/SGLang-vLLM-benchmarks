# AWS Deployment Guide - Complete Step-by-Step

## 🎯 Benchmark Matrix: 3 Models × 2 Frameworks

| Model | Parameters | vLLM | SGLang |
|-------|------------|------|--------|
| `facebook/opt-125m` | 125M | ✓ | ✓ |
| `Qwen/Qwen2-1.5B` | 1.5B | ✓ | ✓ |
| `mistralai/Mistral-7B-v0.1` | 7B | ✓ | ✓ |

**Total: 6 benchmarks**

---

## 📋 Prerequisites

1. **AWS Account** with EC2 access
2. **HuggingFace Account** with access token
   - Go to: https://huggingface.co/settings/tokens
   - Create a token with "Read" access
   - **Accept model licenses** for:
     - https://huggingface.co/mistralai/Mistral-7B-v0.1
     - https://huggingface.co/Qwen/Qwen2-1.5B

---

## 💰 Cost Estimate

| Instance | GPU | Cost/Hour | Est. Total Time | Est. Total Cost |
|----------|-----|-----------|-----------------|-----------------|
| g5.xlarge | A10G (24GB) | $1.01 | ~3 hours | **~$3.00** |
| g4dn.xlarge | T4 (16GB) | $0.53 | ~4 hours | **~$2.00** |

**Recommendation:** Use `g5.xlarge` for faster benchmarks with all models.

---

## 🚀 STEP 1: Launch EC2 Instance (AWS Console)

### 1.1 Go to EC2 Dashboard
- Login to AWS Console → Search "EC2" → Click "Launch Instance"

### 1.2 Configure Instance

| Setting | Value |
|---------|-------|
| **Name** | `llm-benchmark-server` |
| **AMI** | Search "Deep Learning AMI GPU PyTorch" → Select Ubuntu version |
| **Instance Type** | `g5.xlarge` (recommended) |
| **Key Pair** | Create new → Name: `llm-key` → Download `.pem` file |
| **Security Group** | Create new with these rules: |

**Security Group Inbound Rules:**
| Type | Port | Source |
|------|------|--------|
| SSH | 22 | My IP |
| Custom TCP | 8000 | My IP |
| Custom TCP | 30000 | My IP |

| Setting | Value |
|---------|-------|
| **Storage** | 100 GB, gp3 |

### 1.3 Launch and Wait
- Click "Launch Instance"
- Wait 2-3 minutes until status shows "Running"
- Copy the **Public IPv4 address**

---

## 🔌 STEP 2: Connect to Instance

### On Windows (PowerShell):
```powershell
# Move key to .ssh folder
mkdir ~/.ssh -ErrorAction SilentlyContinue
Move-Item ~/Downloads/llm-key.pem ~/.ssh/

# Connect (replace YOUR_IP with your EC2 public IP)
ssh -i ~/.ssh/llm-key.pem ubuntu@YOUR_IP
```

### On Mac/Linux:
```bash
chmod 400 ~/Downloads/llm-key.pem
ssh -i ~/Downloads/llm-key.pem ubuntu@YOUR_IP
```

---

## ⚙️ STEP 3: Setup Instance (Run on EC2)

Copy and paste this entire block:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Setup complete! Please logout and login again."
```

**IMPORTANT: Logout and login again:**
```bash
exit
```

Then reconnect:
```bash
ssh -i ~/.ssh/llm-key.pem ubuntu@YOUR_IP
```

---

## 📤 STEP 4: Upload Project Files

### From your LOCAL machine (new terminal):

```powershell
# Navigate to project folder
cd "D:\International\Study Materials\Semester-3\MSML 650 - Cloud\Project"

# Upload to EC2 (replace YOUR_IP)
scp -i ~/.ssh/llm-key.pem -r cloud-llm-inference-benchmark ubuntu@YOUR_IP:~/
```

---

## 🐳 STEP 5: Build Docker Images (Run on EC2)

```bash
# Go to project directory
cd ~/cloud-llm-inference-benchmark

# Build vLLM image (takes ~10-15 minutes)
echo "Building vLLM Docker image..."
docker build -f docker/Dockerfile.vllm -t vllm-server .

# Build SGLang image (takes ~10-15 minutes)
echo "Building SGLang Docker image..."
docker build -f docker/Dockerfile.sglang -t sglang-server .

echo "Docker images built successfully!"
```

---

## 🏃 STEP 6: Run Full Benchmark Suite

```bash
# Set your HuggingFace token (REQUIRED!)
export HUGGINGFACE_TOKEN="hf_your_token_here"

# Make script executable
chmod +x scripts/run_full_benchmark.sh

# Run all 6 benchmarks (3 models × 2 frameworks)
./scripts/run_full_benchmark.sh
```

**This will take approximately 2-3 hours.** You can:
- Leave the terminal running
- Use `screen` or `tmux` to run in background:
  ```bash
  screen -S benchmark
  ./scripts/run_full_benchmark.sh
  # Press Ctrl+A, then D to detach
  # Reconnect later with: screen -r benchmark
  ```

---

## 📥 STEP 7: Download Results

### From your LOCAL machine:

```powershell
# Create local results folder
mkdir benchmark_results

# Download all results (replace YOUR_IP)
scp -i ~/.ssh/llm-key.pem -r ubuntu@YOUR_IP:~/cloud-llm-inference-benchmark/results/* ./benchmark_results/
```

---

## 📊 STEP 8: Visualize Results (Local)

```bash
cd cloud-llm-inference-benchmark
jupyter notebook notebooks/benchmark_visualization.ipynb
```

---

## ⚠️ STEP 9: TERMINATE INSTANCE (CRITICAL!)

**GPU instances cost ~$1/hour even when idle!**

### Option A: AWS Console
1. Go to EC2 → Instances
2. Select your instance
3. Actions → Instance State → **Terminate**

### Option B: AWS CLI
```bash
aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID
```

---

## 🔧 Troubleshooting

### "Permission denied" SSH error
```bash
chmod 400 ~/.ssh/llm-key.pem
```

### Docker GPU not working
```bash
# Verify GPU is visible
nvidia-smi

# Test Docker GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Model download fails
- Check HuggingFace token is correct
- Accept model license on HuggingFace website
- For Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.1

### Out of memory error
- Use smaller models first (OPT-125M, Qwen-1.5B)
- Or upgrade to `g5.2xlarge` for more GPU memory

### Server won't start
```bash
# Check container logs
docker logs vllm-server
docker logs sglang-server
```

---

## 📁 Expected Output Files

After running benchmarks, you'll have:

```
results/
├── vllm/
│   ├── benchmark_facebook_opt-125m_TIMESTAMP.json
│   ├── benchmark_Qwen_Qwen2-1.5B_TIMESTAMP.json
│   └── benchmark_mistralai_Mistral-7B_TIMESTAMP.json
├── sglang/
│   ├── benchmark_facebook_opt-125m_TIMESTAMP.json
│   ├── benchmark_Qwen_Qwen2-1.5B_TIMESTAMP.json
│   └── benchmark_mistralai_Mistral-7B_TIMESTAMP.json
├── benchmark_summary.csv
└── analysis_report.json
```

---

## ⏱️ Time Breakdown (Approximate)

| Step | Time |
|------|------|
| Launch EC2 & Setup | 15 min |
| Upload project | 2 min |
| Build Docker images | 25 min |
| Benchmark OPT-125M (×2) | 20 min |
| Benchmark Qwen-1.5B (×2) | 40 min |
| Benchmark Mistral-7B (×2) | 90 min |
| Download results | 2 min |
| **Total** | **~3 hours** |

---

Good luck with your benchmarks! 🚀
