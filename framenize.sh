#!/bin/bash

# Define the base output directory
INPUT_DIR="/home/michael/Videos/model_input"
OUTPUT_DIR="/home/michael/Videos/model_input/frames/"

# Create the base directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the input directory
cd "$INPUT_DIR" || exit 1

# Loop through all mp4 files in the input directory
for video in *.mp4; do
    # Check if any mp4 files exist to avoid errors in empty folders
    [ -e "$video" ] || continue

    # Get the filename without the extension
    video_name="${video%.*}"

    # Define and create the specific subfolder for this video
    output_subdir="$OUTPUT_DIR/$video_name"
    mkdir -p "$output_subdir"

    echo "Processing: $video -> $output_subdir"

    # -vf "fps=1": 1 frame every 1 second
    # -vf "scale=1280:720": Rescale to 720p
    # -frame_pts 1: Name file based on video timestamp
    # %05d: Pad with 5 zeros (e.g., 00010.png)
    ffmpeg -i "$video" \
           -vf "fps=1,scale=1280:720" \
           -frame_pts 1 \
           "$output_subdir/%05d.png" -hide_banner -loglevel error
done

echo "Done! All videos processed."