#!/bin/bash
# ============================================================================
# Local GPU Benchmark Script - RTX 4070 (8GB VRAM)
# Cloud LLM Inference Benchmark Project
# MSML 650 - Cloud Computing
# ============================================================================
#
# Benchmark Matrix: 3 Models × 2 Frameworks = 6 Benchmarks
#   - facebook/opt-125m      (125M params)  - Small baseline
#   - Qwen/Qwen2-1.5B        (1.5B params)  - Medium
#   - microsoft/phi-2        (2.7B params)  - Large
#
# Usage:
#   export HUGGINGFACE_TOKEN="your_token"
#   ./scripts/run_local_benchmark.sh
#
# Estimated Time: ~1.5-2 hours
# ============================================================================

set -e

# ============= CONFIGURATION =============
MODELS=(
    "facebook/opt-125m"
    "Qwen/Qwen2-1.5B"
    "microsoft/phi-2"
)

MODEL_NAMES=(
    "OPT-125M"
    "Qwen2-1.5B"
    "Phi-2"
)

WORKLOAD="configs/workload_medium.json"
RESULTS_DIR="results"

# Colors
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
    
    # Check Docker GPU
    echo "Testing Docker GPU access..."
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "Docker GPU not working!"
        echo "Make sure NVIDIA Container Toolkit is installed"
        exit 1
    fi
    print_status "Docker GPU access working"
    
    # Show GPU info
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}

stop_all_containers() {
    docker stop vllm-server sglang-server 2>/dev/null || true
    docker rm vllm-server sglang-server 2>/dev/null || true
}

run_vllm_benchmark() {
    local model=$1
    local model_name=$2
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  vLLM + $model_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Stop existing container
    docker stop vllm-server 2>/dev/null || true
    docker rm vllm-server 2>/dev/null || true
    
    # Start vLLM server
    echo "Starting vLLM server with $model..."
    docker run -d --gpus all \
        --name vllm-server \
        -p 8000:8000 \
        -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
        -e MODEL_NAME="$model" \
        vllm-server
    
    # Wait for server to be ready
    echo "Waiting for model to load (this may take a few minutes)..."
    local max_wait=600
    local waited=0
    
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        
        # Show progress
        if [ $((waited % 30)) -eq 0 ]; then
            echo "  Still loading... ${waited}s elapsed"
        fi
        
        # Check if container crashed
        if ! docker ps | grep -q vllm-server; then
            print_error "vLLM container crashed!"
            echo "Logs:"
            docker logs vllm-server 2>&1 | tail -20
            return 1
        fi
        
        if [ $waited -ge $max_wait ]; then
            print_error "Timeout waiting for vLLM server"
            return 1
        fi
    done
    
    print_status "vLLM server ready!"
    
    # Run benchmark
    echo "Running benchmark..."
    python3 src/benchmark_runner.py \
        --framework vllm \
        --config $WORKLOAD \
        --model "$model" \
        --url "http://localhost:8000" \
        --results-dir "$RESULTS_DIR"
    
    # Stop server
    docker stop vllm-server
    docker rm vllm-server
    
    print_status "vLLM + $model_name complete!"
}

run_sglang_benchmark() {
    local model=$1
    local model_name=$2
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  SGLang + $model_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Stop existing container
    docker stop sglang-server 2>/dev/null || true
    docker rm sglang-server 2>/dev/null || true
    
    # Start SGLang server
    echo "Starting SGLang server with $model..."
    docker run -d --gpus all \
        --name sglang-server \
        -p 30000:30000 \
        -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
        -e MODEL_NAME="$model" \
        sglang-server
    
    # Wait for server to be ready
    echo "Waiting for model to load (this may take a few minutes)..."
    local max_wait=600
    local waited=0
    
    while ! curl -s http://localhost:30000/health > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        
        if [ $((waited % 30)) -eq 0 ]; then
            echo "  Still loading... ${waited}s elapsed"
        fi
        
        if ! docker ps | grep -q sglang-server; then
            print_error "SGLang container crashed!"
            echo "Logs:"
            docker logs sglang-server 2>&1 | tail -20
            return 1
        fi
        
        if [ $waited -ge $max_wait ]; then
            print_error "Timeout waiting for SGLang server"
            return 1
        fi
    done
    
    print_status "SGLang server ready!"
    
    # Run benchmark
    echo "Running benchmark..."
    python3 src/benchmark_runner.py \
        --framework sglang \
        --config $WORKLOAD \
        --model "$model" \
        --url "http://localhost:30000" \
        --results-dir "$RESULTS_DIR"
    
    # Stop server
    docker stop sglang-server
    docker rm sglang-server
    
    print_status "SGLang + $model_name complete!"
}

# ============= MAIN SCRIPT =============

clear
print_header "LLM Inference Benchmark Suite"
echo ""
echo "  GPU: NVIDIA RTX 4070 (8GB VRAM)"
echo ""
echo "  Benchmark Matrix: 3 Models × 2 Frameworks"
echo ""
echo "  ┌─────────────────────────┬────────┬─────────┐"
echo "  │ Model                   │ vLLM   │ SGLang  │"
echo "  ├─────────────────────────┼────────┼─────────┤"
echo "  │ OPT-125M (125M params)  │   ✓    │    ✓    │"
echo "  │ Qwen2-1.5B (1.5B)       │   ✓    │    ✓    │"
echo "  │ Phi-2 (2.7B params)     │   ✓    │    ✓    │"
echo "  └─────────────────────────┴────────┴─────────┘"
echo ""
echo "  Estimated time: 1.5-2 hours"
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

# Track time
START_TIME=$(date +%s)
BENCHMARK_COUNT=0
TOTAL_BENCHMARKS=6

# Run all benchmarks
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    
    # vLLM benchmark
    BENCHMARK_COUNT=$((BENCHMARK_COUNT + 1))
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Benchmark $BENCHMARK_COUNT of $TOTAL_BENCHMARKS${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    
    run_vllm_benchmark "$model" "$model_name" || print_warning "vLLM benchmark failed for $model_name"
    
    # SGLang benchmark
    BENCHMARK_COUNT=$((BENCHMARK_COUNT + 1))
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Benchmark $BENCHMARK_COUNT of $TOTAL_BENCHMARKS${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    
    run_sglang_benchmark "$model" "$model_name" || print_warning "SGLang benchmark failed for $model_name"
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

# Generate final report
print_header "Generating Analysis Report"
python3 src/parse_results.py \
    --results-dir "$RESULTS_DIR" \
    --export-csv \
    --generate-report

# Final summary
print_header "🎉 Benchmark Complete!"
echo ""
echo "  Total benchmarks: 6"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "  Results saved in:"
echo "    - $RESULTS_DIR/vllm/*.json"
echo "    - $RESULTS_DIR/sglang/*.json"
echo "    - $RESULTS_DIR/benchmark_summary.csv"
echo "    - $RESULTS_DIR/analysis_report.json"
echo ""
echo "  Next steps:"
echo "    1. Open notebooks/benchmark_visualization.ipynb"
echo "    2. Generate charts for your report"
echo "    3. Complete report using report/report_template.md"
echo ""
