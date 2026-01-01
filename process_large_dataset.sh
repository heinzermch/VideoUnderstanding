#!/bin/bash
# Wrapper script to process large image datasets in chunks
# This prevents OOM by running each chunk in a fresh process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
IMAGE_DIR="/home/michael/Videos/model_input/frames/"
CHUNK_SIZE=2000
OUTPUT_DIR="output_files/frame_level"
MODEL_ID="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--image-dir DIR] [--chunk-size N] [--output-dir DIR] [--model-id MODEL]"
            exit 1
            ;;
    esac
done

echo "Processing large dataset in chunks..."
echo "  Image directory: $IMAGE_DIR"
echo "  Chunk size: $CHUNK_SIZE images"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_ID"
echo ""

python3 scripts/process_in_chunks.py \
    --image-dir "$IMAGE_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --model-id "$MODEL_ID" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ All chunks processed!"

