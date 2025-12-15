# Cloud Deployment of Model Serving Platforms: Benchmarking vLLM and SGLang on Cloud GPUs

**Course:** MSML 650 - Cloud Computing  
**Date:** [Insert Date]  
**Author:** [Your Name]  

---

## Abstract

This report presents a comprehensive benchmark study comparing two leading Large Language Model (LLM) inference frameworks—vLLM and SGLang—deployed on AWS EC2 GPU instances. We evaluate performance metrics including throughput (tokens per second), latency distributions, and cost efficiency. Our findings provide insights for selecting optimal inference frameworks for production LLM deployments.

---

## 1. Introduction

### 1.1 Background

Large Language Models (LLMs) have revolutionized natural language processing, but their deployment at scale presents significant infrastructure challenges. The inference phase—generating text from a trained model—requires substantial computational resources and careful optimization to achieve acceptable latency and throughput.

### 1.2 Motivation

Cloud-based GPU instances offer a scalable solution for LLM inference, but the choice of serving framework significantly impacts performance and cost. This project evaluates two prominent frameworks:

- **vLLM**: Developed by UC Berkeley, featuring PagedAttention for efficient memory management
- **SGLang**: A structured generation language with optimized batching and caching

### 1.3 Objectives

1. Deploy vLLM and SGLang on AWS EC2 GPU instances
2. Benchmark performance across multiple workload configurations
3. Analyze throughput, latency, and cost efficiency
4. Provide recommendations for framework selection

---

## 2. Framework Overview

### 2.1 vLLM Architecture

vLLM introduces several key innovations:

- **PagedAttention**: Manages key-value cache in non-contiguous memory blocks, similar to virtual memory paging
- **Continuous Batching**: Dynamically batches requests to maximize GPU utilization
- **Optimized CUDA Kernels**: Custom kernels for attention computation

```
┌─────────────────────────────────────────┐
│              vLLM Server                │
├─────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────────┐  │
│  │  Scheduler  │   │ PagedAttention  │  │
│  │  (Batching) │   │   (KV Cache)    │  │
│  └─────────────┘   └─────────────────┘  │
│  ┌─────────────────────────────────────┐│
│  │       Model Executor (CUDA)         ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### 2.2 SGLang Architecture

SGLang provides:

- **RadixAttention**: Tree-based caching for structured generation
- **Constrained Decoding**: Efficient generation with grammar constraints
- **Prefix Sharing**: Reuses computation for common prompt prefixes

```
┌─────────────────────────────────────────┐
│             SGLang Server               │
├─────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────────┐  │
│  │   Router    │   │ RadixAttention  │  │
│  │             │   │    (Cache)      │  │
│  └─────────────┘   └─────────────────┘  │
│  ┌─────────────────────────────────────┐│
│  │       Runtime (FlashInfer)          ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

---

## 3. Cloud Deployment Setup

### 3.1 AWS EC2 Configuration

| Parameter | Value |
|-----------|-------|
| Instance Type | g5.xlarge |
| GPU | NVIDIA A10G (24GB) |
| vCPUs | 4 |
| Memory | 16 GB |
| Storage | 100 GB gp3 |
| Region | us-east-1 |
| AMI | Deep Learning AMI (Ubuntu 20.04) |

### 3.2 Cost Analysis

| Instance Type | GPU | Hourly Cost (USD) |
|--------------|-----|-------------------|
| g4dn.xlarge | T4 (16GB) | $0.526 |
| g5.xlarge | A10G (24GB) | $1.006 |
| g5.2xlarge | A10G (24GB) | $1.212 |
| p3.2xlarge | V100 (16GB) | $3.06 |

### 3.3 Docker Configuration

Both frameworks were containerized using Docker for reproducible deployment:

- Base Image: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- Python Version: 3.10
- PyTorch Version: 2.1.2 (CUDA 12.1)

### 3.4 Model Selection

| Model | Parameters | Context Length |
|-------|------------|----------------|
| facebook/opt-125m | 125M | 2048 |
| [Alternative model] | [Size] | [Context] |

---

## 4. Benchmark Methodology

### 4.1 Workload Configurations

| Workload | Requests | Max Tokens | Concurrency |
|----------|----------|------------|-------------|
| Small | 10 | 64 | 2 |
| Medium | 50 | 128 | 5 |
| Large | 100 | 256 | 10 |

### 4.2 Metrics Collected

1. **Throughput**
   - Tokens per second (TPS)
   - Requests per second (RPS)

2. **Latency**
   - Average latency
   - Percentiles: P50, P90, P95, P99
   - Time to first token (TTFT)

3. **Resource Utilization**
   - GPU utilization (%)
   - GPU memory usage (GB)

4. **Reliability**
   - Success rate (%)
   - Error rate

### 4.3 Testing Procedure

1. Start inference server
2. Execute warmup requests (2-10)
3. Run benchmark workload
4. Collect and store metrics
5. Repeat for each framework and workload combination

---

## 5. Results

### 5.1 Throughput Comparison

[INSERT THROUGHPUT CHART: results/throughput_comparison.png]

| Framework | Workload | Throughput (TPS) | Throughput (RPS) |
|-----------|----------|------------------|------------------|
| vLLM | Small | [VALUE] | [VALUE] |
| vLLM | Medium | [VALUE] | [VALUE] |
| vLLM | Large | [VALUE] | [VALUE] |
| SGLang | Small | [VALUE] | [VALUE] |
| SGLang | Medium | [VALUE] | [VALUE] |
| SGLang | Large | [VALUE] | [VALUE] |

### 5.2 Latency Distribution

[INSERT LATENCY CHART: results/latency_distribution.png]

| Framework | Avg Latency (s) | P50 (s) | P95 (s) | P99 (s) |
|-----------|-----------------|---------|---------|---------|
| vLLM | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| SGLang | [VALUE] | [VALUE] | [VALUE] | [VALUE] |

### 5.3 Cost Efficiency

[INSERT COST CHART: results/cost_comparison.png]

| Framework | Avg TPS | Cost/Hour | Cost per 1M Tokens |
|-----------|---------|-----------|-------------------|
| vLLM | [VALUE] | $1.006 | $[VALUE] |
| SGLang | [VALUE] | $1.006 | $[VALUE] |

### 5.4 GPU Utilization

[INSERT HEATMAP: results/gpu_utilization_heatmap.png]

| Framework | GPU Utilization (%) | Memory Usage (GB) |
|-----------|---------------------|-------------------|
| vLLM | [VALUE] | [VALUE] |
| SGLang | [VALUE] | [VALUE] |

---

## 6. Discussion

### 6.1 Performance Analysis

**Throughput:**
- [Analysis of which framework achieved higher throughput and why]
- [Discussion of PagedAttention vs RadixAttention impact]

**Latency:**
- [Analysis of latency differences]
- [Discussion of tail latency (P95/P99) implications]

### 6.2 Cost-Performance Trade-offs

- [Analysis of cost per token for each framework]
- [Recommendations based on budget constraints]

### 6.3 Scalability Considerations

- [How each framework scales with increased concurrency]
- [Memory efficiency under load]

### 6.4 Best GPU-Model Combinations

Based on our analysis:

| Use Case | Recommended Framework | GPU |
|----------|----------------------|-----|
| High throughput | vLLM | A10G/A100 |
| Low latency | [Framework] | [GPU] |
| Cost sensitive | [Framework] | T4 |
| Structured generation | SGLang | [GPU] |

### 6.5 Limitations

- [Discuss any limitations of the study]
- [Model size constraints]
- [Single GPU vs multi-GPU testing]

---

## 7. Conclusion

### 7.1 Key Findings

1. **Throughput**: [Summary of throughput findings]
2. **Latency**: [Summary of latency findings]
3. **Cost Efficiency**: [Summary of cost findings]

### 7.2 Recommendations

For production deployments:
- **High-throughput scenarios**: Recommend [framework]
- **Latency-sensitive applications**: Recommend [framework]
- **Cost-optimized deployments**: Recommend [framework]

### 7.3 Future Work

- Benchmark with larger models (7B, 13B parameters)
- Multi-GPU deployment comparison
- Evaluate newer framework versions
- Test with different GPU architectures (H100, L40S)

---

## 8. References

1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
   - GitHub: https://github.com/vllm-project/vllm

2. Zheng, L., et al. (2023). "SGLang: Efficient Execution of Structured Language Model Programs."
   - GitHub: https://github.com/sgl-project/sglang

3. AWS EC2 GPU Instances Documentation
   - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html

4. NVIDIA CUDA Documentation
   - https://docs.nvidia.com/cuda/

5. HuggingFace Transformers Documentation
   - https://huggingface.co/docs/transformers/

---

## Appendix A: Code Repository

Project code is available at: [GitHub repository URL or submission location]

### Directory Structure
```
cloud-llm-inference-benchmark/
├── docker/           # Dockerfiles for vLLM and SGLang
├── scripts/          # Deployment and benchmark scripts
├── configs/          # Workload configurations
├── src/              # Python benchmarking code
├── notebooks/        # Visualization notebook
├── results/          # Benchmark results
└── report/           # This report
```

---

## Appendix B: Raw Data

[Include or reference the raw benchmark data files]

- `results/vllm/benchmark_*.json`
- `results/sglang/benchmark_*.json`
- `results/benchmark_results_combined.csv`

---

## Appendix C: Reproducibility

### Prerequisites
1. AWS account with EC2 access
2. HuggingFace account and access token
3. Python 3.10+, Docker

### Steps to Reproduce
1. Launch EC2 GPU instance (g5.xlarge)
2. Clone project repository
3. Run `./scripts/setup_instance.sh`
4. Build Docker images
5. Run benchmarks: `./scripts/run_benchmarks.sh`
6. Analyze results in Jupyter notebook

---

*Report generated: [Date]*

