# Cloud Deployment of Model Serving Platforms: Benchmarking vLLM and SGLang

**Course:** MSML 650 - Cloud Computing  
**Project:** Benchmarking and Deployment of LLM Inference Frameworks in Cloud Environments

## 📋 Project Overview

This project benchmarks two leading LLM inference frameworks—**vLLM** and **SGLang**—across three different model sizes deployed on AWS EC2 GPU instances.

### Benchmark Matrix: 3 Models × 2 Frameworks

| Model | Parameters | vLLM | SGLang |
|-------|------------|------|--------|
| `facebook/opt-125m` | 125M | ✓ | ✓ |
| `Qwen/Qwen2-1.5B` | 1.5B | ✓ | ✓ |
| `microsoft/phi-2` | 2.7B | ✓ | ✓ |

### Metrics Evaluated
- **Throughput**: Tokens per second (TPS), Requests per second (RPS)
- **Latency**: Average, P50, P90, P95, P99
- **Scalability**: Performance across model sizes
- **Cost Efficiency**: Cost per million tokens

## 📁 Project Structure

```
cloud-llm-inference-benchmark/
├── docker/
│   ├── Dockerfile.vllm          # vLLM server container
│   └── Dockerfile.sglang        # SGLang server container
├── scripts/
│   ├── run_full_benchmark.sh    # Main benchmark script (3×2 matrix)
│   ├── setup_instance.sh        # EC2 instance setup
│   └── deploy_aws.sh            # AWS deployment helper
├── configs/
│   ├── workload_small.json      # 10 requests, 64 tokens
│   ├── workload_medium.json     # 50 requests, 128 tokens
│   └── workload_large.json      # 100 requests, 256 tokens
├── src/
│   ├── load_test.py             # Async load testing
│   ├── benchmark_runner.py      # Benchmark orchestrator
│   └── parse_results.py         # Results analyzer
├── notebooks/
│   └── benchmark_visualization.ipynb
├── results/
│   ├── vllm/                    # vLLM benchmark results
│   └── sglang/                  # SGLang benchmark results
├── report/
│   └── report_template.md       # Report template
├── AWS_DEPLOYMENT_GUIDE.md      # Step-by-step AWS instructions
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU (RTX 4070 8GB or similar)
- Docker Desktop with WSL2
- HuggingFace account with access token

### Deployment Options

| Option | GPU | Time | Cost |
|--------|-----|------|------|
| **Local (RTX 4070)** | 8GB VRAM | ~2 hours | Free |
| AWS EC2 (g5.xlarge) | A10G 24GB | ~3 hours | ~$3.00 |

### Local GPU Setup (Recommended)
1. Install Docker Desktop + WSL2
2. Configure NVIDIA Container Toolkit
3. Build Docker images
4. Run benchmark suite

**See [LOCAL_GPU_GUIDE.md](LOCAL_GPU_GUIDE.md) for detailed instructions.**

### AWS Cloud Setup (Alternative)
**See [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) for cloud deployment.**

## 🐳 Docker Images

### vLLM Server
```bash
docker build -f docker/Dockerfile.vllm -t vllm-server .
docker run --gpus all -p 8000:8000 -e HUGGINGFACE_TOKEN=$HF_TOKEN -e MODEL_NAME="facebook/opt-125m" vllm-server
```

### SGLang Server
```bash
docker build -f docker/Dockerfile.sglang -t sglang-server .
docker run --gpus all -p 30000:30000 -e HUGGINGFACE_TOKEN=$HF_TOKEN -e MODEL_NAME="facebook/opt-125m" sglang-server
```

## 📊 Running Benchmarks

### Full Benchmark Suite (Recommended)
```bash
export HUGGINGFACE_TOKEN="your_token"
./scripts/run_full_benchmark.sh
```

This runs all 6 benchmarks automatically.

### Individual Benchmark
```bash
python src/benchmark_runner.py \
    --framework vllm \
    --config configs/workload_medium.json \
    --model "facebook/opt-125m"
```

## 📈 Analysis & Visualization

### Generate Report
```bash
python src/parse_results.py --results-dir results/ --export-csv --generate-report
```

### Visualization Notebook
Open `notebooks/benchmark_visualization.ipynb` to generate:
- Throughput comparison charts
- Latency distribution graphs
- Cost efficiency analysis
- Model scaling comparisons

## 🔬 Framework Comparison

### vLLM
- **PagedAttention**: Efficient KV-cache management
- **Continuous Batching**: Dynamic request batching
- **High Throughput**: Optimized for maximum tokens/second

### SGLang
- **RadixAttention**: Tree-based caching
- **Structured Generation**: Optimized for constrained outputs
- **Prefix Sharing**: Efficient prompt reuse

## 📚 References

1. [vLLM GitHub](https://github.com/vllm-project/vllm)
2. [SGLang GitHub](https://github.com/sgl-project/sglang)
3. [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/g5/)

## ⚠️ Important Notes

1. **GPU Required**: Real benchmarks require NVIDIA GPU (AWS EC2)
2. **HuggingFace Token**: Required for model downloads
3. **Model Licenses**: Accept licenses on HuggingFace for Mistral and Qwen
4. **Cost Awareness**: Always terminate EC2 instances when done!

---

**MSML 650 - Cloud Computing Project**
