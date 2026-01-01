#!/bin/bash
# Quick script to reprocess chunk 8 and merge with existing CSV

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
IMAGE_DIR="/home/michael/Videos/model_input/frames/"
CHUNK_SIZE=2000
OUTPUT_DIR="output_files/frame_level"
MODEL_ID="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
CHUNK_NUM=8

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
        --chunk-num)
            CHUNK_NUM="$2"
            shift 2
            ;;
        --merged-csv)
            MERGED_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--image-dir DIR] [--chunk-size N] [--output-dir DIR] [--model-id MODEL] [--chunk-num N] [--merged-csv FILE]"
            exit 1
            ;;
    esac
done

echo "Reprocessing chunk $CHUNK_NUM..."
echo "  Image directory: $IMAGE_DIR"
echo "  Chunk size: $CHUNK_SIZE images"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_ID"
echo ""

CMD="python3 scripts/reprocess_chunk.py \
    --chunk-num $CHUNK_NUM \
    --image-dir \"$IMAGE_DIR\" \
    --chunk-size $CHUNK_SIZE \
    --model-id \"$MODEL_ID\" \
    --output-dir \"$OUTPUT_DIR\""

if [ -n "$MERGED_CSV" ]; then
    CMD="$CMD --merged-csv \"$MERGED_CSV\""
fi

eval $CMD

echo ""
echo "✅ Chunk $CHUNK_NUM reprocessed and merged!"

