#!/bin/bash
# Login to HuggingFace if token provided
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token $HUGGINGFACE_TOKEN
fi

# Set default model if not provided
MODEL=${MODEL_NAME:-"facebook/opt-125m"}

# Start vLLM server with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
