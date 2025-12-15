#!/bin/bash
# ============================================================================
# Full Benchmark Suite - 3 Models × 2 Frameworks
# Cloud LLM Inference Benchmark Project
# MSML 650 - Cloud Computing
# ============================================================================
# 
# This script runs the complete benchmark matrix:
#   - 3 Models: OPT-125M, Qwen2-1.5B, Mistral-7B
#   - 2 Frameworks: vLLM, SGLang
#   - Total: 6 benchmark runs
#
# Prerequisites:
#   - AWS EC2 GPU instance (g5.xlarge recommended)
#   - Docker with NVIDIA runtime installed
#   - HuggingFace token exported
#
# Usage:
#   export HUGGINGFACE_TOKEN="your_token"
#   ./run_full_benchmark.sh
#
# Estimated Time: 2-3 hours total
# Estimated Cost: ~$3-4 (g5.xlarge @ $1/hr)
# ============================================================================

set -e

# ============= CONFIGURATION =============
# Models to benchmark (small → large)
MODELS=(
    "facebook/opt-125m"
    "Qwen/Qwen2-1.5B"
    "mistralai/Mistral-7B-v0.1"
)

# Frameworks
FRAMEWORKS=("vllm" "sglang")

# Ports
VLLM_PORT=8000
SGLANG_PORT=30000

# Workload to use
WORKLOAD="configs/workload_medium.json"

# Results directory
RESULTS_DIR="results"

# ============= COLORS =============
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============= FUNCTIONS =============

print_header() {
    echo ""
    echo -e "${CYAN}=============================================="
    echo "  $1"
    echo -e "==============================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check HuggingFace token
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        print_error "HUGGINGFACE_TOKEN not set!"
        echo "Run: export HUGGINGFACE_TOKEN='your_token'"
        exit 1
    fi
    print_status "HuggingFace token found"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed!"
        exit 1
    fi
    print_status "Docker installed"
    
    # Check NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime not working!"
        exit 1
    fi
    print_status "NVIDIA Docker runtime working"
    
    # Check GPU
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

stop_all_containers() {
    echo "Stopping any existing containers..."
    docker stop vllm-server sglang-server 2>/dev/null || true
    docker rm vllm-server sglang-server 2>/dev/null || true
}

run_vllm_benchmark() {
    local model=$1
    local model_safe_name=$(echo $model | tr '/' '_')
    
    print_header "vLLM Benchmark: $model"
    
    # Stop any existing container
    docker stop vllm-server 2>/dev/null || true
    docker rm vllm-server 2>/dev/null || true
    
    # Start vLLM server
    echo "Starting vLLM server with $model..."
    docker run -d --gpus all \
        --name vllm-server \
        -p $VLLM_PORT:8000 \
        -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
        -e MODEL_NAME="$model" \
        vllm-server
    
    # Wait for server to be ready
    echo "Waiting for vLLM server to load model (this may take a few minutes)..."
    local max_wait=600  # 10 minutes max
    local waited=0
    while ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        echo "  Waited ${waited}s..."
        
        # Check if container is still running
        if ! docker ps | grep -q vllm-server; then
            print_error "vLLM container crashed! Check logs:"
            docker logs vllm-server
            return 1
        fi
        
        if [ $waited -ge $max_wait ]; then
            print_error "Timeout waiting for vLLM server"
            docker logs vllm-server
            return 1
        fi
    done
    
    print_status "vLLM server is ready!"
    
    # Run benchmark
    echo "Running benchmark..."
    python3 src/benchmark_runner.py \
        --framework vllm \
        --config $WORKLOAD \
        --model "$model" \
        --url "http://localhost:$VLLM_PORT" \
        --results-dir "$RESULTS_DIR"
    
    # Stop server
    docker stop vllm-server
    docker rm vllm-server
    
    print_status "vLLM benchmark complete for $model"
}

run_sglang_benchmark() {
    local model=$1
    local model_safe_name=$(echo $model | tr '/' '_')
    
    print_header "SGLang Benchmark: $model"
    
    # Stop any existing container
    docker stop sglang-server 2>/dev/null || true
    docker rm sglang-server 2>/dev/null || true
    
    # Start SGLang server
    echo "Starting SGLang server with $model..."
    docker run -d --gpus all \
        --name sglang-server \
        -p $SGLANG_PORT:30000 \
        -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
        -e MODEL_NAME="$model" \
        sglang-server
    
    # Wait for server to be ready
    echo "Waiting for SGLang server to load model (this may take a few minutes)..."
    local max_wait=600  # 10 minutes max
    local waited=0
    while ! curl -s http://localhost:$SGLANG_PORT/health > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        echo "  Waited ${waited}s..."
        
        # Check if container is still running
        if ! docker ps | grep -q sglang-server; then
            print_error "SGLang container crashed! Check logs:"
            docker logs sglang-server
            return 1
        fi
        
        if [ $waited -ge $max_wait ]; then
            print_error "Timeout waiting for SGLang server"
            docker logs sglang-server
            return 1
        fi
    done
    
    print_status "SGLang server is ready!"
    
    # Run benchmark
    echo "Running benchmark..."
    python3 src/benchmark_runner.py \
        --framework sglang \
        --config $WORKLOAD \
        --model "$model" \
        --url "http://localhost:$SGLANG_PORT" \
        --results-dir "$RESULTS_DIR"
    
    # Stop server
    docker stop sglang-server
    docker rm sglang-server
    
    print_status "SGLang benchmark complete for $model"
}

generate_report() {
    print_header "Generating Analysis Report"
    
    python3 src/parse_results.py \
        --results-dir "$RESULTS_DIR" \
        --export-csv \
        --generate-report
    
    print_status "Report generated!"
}

# ============= MAIN SCRIPT =============

print_header "Full Benchmark Suite: 3 Models × 2 Frameworks"

echo ""
echo "Benchmark Matrix:"
echo "┌─────────────────────────────────┬────────┬─────────┐"
echo "│ Model                           │ vLLM   │ SGLang  │"
echo "├─────────────────────────────────┼────────┼─────────┤"
echo "│ facebook/opt-125m (125M)        │   ✓    │    ✓    │"
echo "│ Qwen/Qwen2-1.5B (1.5B)          │   ✓    │    ✓    │"
echo "│ mistralai/Mistral-7B-v0.1 (7B)  │   ✓    │    ✓    │"
echo "└─────────────────────────────────┴────────┴─────────┘"
echo ""
echo "Estimated time: 2-3 hours"
echo ""

# Check prerequisites
check_prerequisites

# Stop any running containers
stop_all_containers

# Create results directories
mkdir -p $RESULTS_DIR/vllm $RESULTS_DIR/sglang

# Install Python dependencies
print_header "Installing Python Dependencies"
pip install -q -r requirements.txt
print_status "Dependencies installed"

# Track timing
START_TIME=$(date +%s)

# Run all benchmarks
BENCHMARK_COUNT=0
TOTAL_BENCHMARKS=$((${#MODELS[@]} * ${#FRAMEWORKS[@]}))

for model in "${MODELS[@]}"; do
    for framework in "${FRAMEWORKS[@]}"; do
        BENCHMARK_COUNT=$((BENCHMARK_COUNT + 1))
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}  Benchmark $BENCHMARK_COUNT of $TOTAL_BENCHMARKS${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        if [ "$framework" == "vllm" ]; then
            run_vllm_benchmark "$model" || print_warning "vLLM benchmark failed for $model"
        else
            run_sglang_benchmark "$model" || print_warning "SGLang benchmark failed for $model"
        fi
    done
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

# Generate final report
generate_report

# Final summary
print_header "Benchmark Complete!"

echo ""
echo "Summary:"
echo "  Total benchmarks: $TOTAL_BENCHMARKS"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in:"
echo "  - $RESULTS_DIR/vllm/*.json"
echo "  - $RESULTS_DIR/sglang/*.json"
echo "  - $RESULTS_DIR/benchmark_summary.csv"
echo "  - $RESULTS_DIR/analysis_report.json"
echo ""
echo "Next steps:"
echo "  1. Download results to local machine"
echo "  2. Run visualization notebook"
echo "  3. Complete your report"
echo ""
echo -e "${GREEN}Don't forget to TERMINATE your EC2 instance to avoid charges!${NC}"

