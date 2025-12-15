#!/bin/bash
# ============================================================================
# Benchmark Execution Script
# Cloud LLM Inference Benchmark Project
# MSML 650 - Cloud Computing
# ============================================================================
# This script runs benchmarks on vLLM and SGLang servers
#
# Usage:
#   ./run_benchmarks.sh [OPTIONS]
#
# Options:
#   --framework [vllm|sglang|both]  Framework to benchmark (default: both)
#   --config [small|medium|large]   Workload size (default: medium)
#   --model MODEL_NAME              Model name (default: facebook/opt-125m)
# ============================================================================

set -e

# Default values
FRAMEWORK="both"
CONFIG_SIZE="medium"
MODEL_NAME="facebook/opt-125m"
VLLM_URL="http://localhost:8000"
SGLANG_URL="http://localhost:30000"
RESULTS_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --config)
            CONFIG_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --vllm-url)
            VLLM_URL="$2"
            shift 2
            ;;
        --sglang-url)
            SGLANG_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=============================================="
echo "  Cloud LLM Inference Benchmark Runner"
echo -e "==============================================${NC}"
echo ""
echo "Configuration:"
echo "  Framework: $FRAMEWORK"
echo "  Workload:  $CONFIG_SIZE"
echo "  Model:     $MODEL_NAME"
echo ""

# Check if Python script exists
if [ ! -f "src/benchmark_runner.py" ]; then
    echo "Error: benchmark_runner.py not found in src/"
    echo "Make sure you're in the project root directory"
    exit 1
fi

# Function to check server health
check_server() {
    local url=$1
    local name=$2
    
    echo -n "Checking $name server at $url... "
    
    if curl -s --connect-timeout 5 "$url/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}Not responding${NC}"
        return 1
    fi
}

# Function to run benchmark for a framework
run_benchmark() {
    local framework=$1
    local url=$2
    
    echo ""
    echo -e "${BLUE}Running $framework benchmark...${NC}"
    echo "================================================"
    
    python3 src/benchmark_runner.py \
        --framework "$framework" \
        --config "configs/workload_${CONFIG_SIZE}.json" \
        --url "$url" \
        --model "$MODEL_NAME" \
        --results-dir "$RESULTS_DIR"
}

# Main execution
echo ""
echo "Checking server availability..."

# Run benchmarks based on framework selection
case $FRAMEWORK in
    "vllm")
        if check_server "$VLLM_URL" "vLLM"; then
            run_benchmark "vllm" "$VLLM_URL"
        else
            echo "Error: vLLM server not available"
            exit 1
        fi
        ;;
    "sglang")
        if check_server "$SGLANG_URL" "SGLang"; then
            run_benchmark "sglang" "$SGLANG_URL"
        else
            echo "Error: SGLang server not available"
            exit 1
        fi
        ;;
    "both")
        VLLM_OK=false
        SGLANG_OK=false
        
        if check_server "$VLLM_URL" "vLLM"; then
            VLLM_OK=true
        fi
        
        if check_server "$SGLANG_URL" "SGLang"; then
            SGLANG_OK=true
        fi
        
        if [ "$VLLM_OK" = true ]; then
            run_benchmark "vllm" "$VLLM_URL"
        fi
        
        if [ "$SGLANG_OK" = true ]; then
            run_benchmark "sglang" "$SGLANG_URL"
        fi
        
        if [ "$VLLM_OK" = false ] && [ "$SGLANG_OK" = false ]; then
            echo "Error: No servers available"
            exit 1
        fi
        ;;
    *)
        echo "Invalid framework: $FRAMEWORK"
        echo "Use: vllm, sglang, or both"
        exit 1
        ;;
esac

# Parse and display results
echo ""
echo -e "${BLUE}=============================================="
echo "  Benchmark Complete!"
echo -e "==============================================${NC}"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""

# Run results parser if available
if [ -f "src/parse_results.py" ]; then
    echo "Generating analysis report..."
    python3 src/parse_results.py --results-dir "$RESULTS_DIR" --export-csv --generate-report
fi

echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. Run visualization notebook: notebooks/benchmark_visualization.ipynb"
echo "  3. Download results for report: scp -r results/ local-machine:path/"

