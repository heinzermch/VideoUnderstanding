#!/bin/bash

# Define the base output directory
INPUT_DIR="/home/michael/Videos/model_input_ocr"
OUTPUT_DIR="/home/michael/Videos/model_input/osd_frames/"


# Create the base directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the input directory
cd "$INPUT_DIR" || exit 1

# Enable case-insensitive globbing to match both .mp4 and .MP4
shopt -s nocaseglob

# Number of parallel jobs (default to 24, or set manually via MAX_JOBS environment variable)
MAX_JOBS=${MAX_JOBS:-24}

# Function to process a single video
process_video() {
    local video="$1"
    local video_name="${video%.*}"
    local output_subdir="$OUTPUT_DIR/$video_name"
    
    mkdir -p "$output_subdir"
    
    echo "Processing: $video -> $output_subdir"
    
    # -vf "fps=10": 10 frame every 1 second
    # -vf "scale=1280:720": Rescale to 720p
    # -frame_pts 1: Name file based on video timestamp
    # %05d: Pad with 5 zeros (e.g., 00010.png)
    ffmpeg -i "$video" \
           -vf "fps=5,scale=1280:720" \
           -frame_pts 1 \
           "$output_subdir/%05d.png" -hide_banner -loglevel error
    
    echo "Completed: $video"
}

# Export function and variables for parallel execution
export -f process_video
export OUTPUT_DIR

# Process all videos in parallel with job limit
for video in *.MOV; do
    # Check if any mp4 files exist to avoid errors in empty folders
    [ -e "$video" ] || continue
    
    # Wait if we've reached the max number of parallel jobs
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 0.1
    done
    
    # Process video in background
    process_video "$video" &
done

# Wait for all background jobs to complete
wait

echo "Done! All videos processed."