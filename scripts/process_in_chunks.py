#!/usr/bin/env python3
"""
Split large image datasets into chunks and process each chunk separately.
This prevents memory accumulation by running each chunk in a fresh process.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def split_image_list(image_files, chunk_size):
    """Split image list into chunks."""
    chunks = []
    for i in range(0, len(image_files), chunk_size):
        chunks.append(image_files[i:i + chunk_size])
    return chunks

def save_chunk_list(chunk_files, output_file):
    """Save chunk file list to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(chunk_files, f, indent=2)

def load_chunk_list(input_file):
    """Load chunk file list from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def process_chunk(chunk_idx, chunk_files, script_path, model_id, output_dir, base_timestamp):
    """Process a single chunk by running the main script with a file list."""
    print(f"\n{'='*60}")
    print(f"Processing chunk {chunk_idx + 1} ({len(chunk_files)} images)")
    print(f"{'='*60}")
    
    # Create temporary file list
    chunk_list_file = os.path.join(output_dir, f"chunk_{chunk_idx + 1}_filelist.json")
    save_chunk_list(chunk_files, chunk_list_file)
    
    # Run the main script with this chunk
    cmd = [
        sys.executable,
        script_path,
        "--model-id", model_id,
        "--output-dir", output_dir,
        "--max-images", str(len(chunk_files)),
        "--chunk-file", chunk_list_file,
        "--chunk-idx", str(chunk_idx + 1),
        "--base-timestamp", base_timestamp
    ]
    
    result = subprocess.run(cmd, check=False)
    
    # Clean up temporary file list
    if os.path.exists(chunk_list_file):
        os.remove(chunk_list_file)
    
    return result.returncode == 0

def merge_csv_files(output_dir, base_timestamp, pattern="qwen_batch_vllm_results_*_chunk_*.csv"):
    """Merge all chunk CSV files into a single file."""
    import pandas as pd
    import glob
    
    # Find all chunk CSV files
    chunk_files = sorted(glob.glob(os.path.join(output_dir, pattern)))
    
    if not chunk_files:
        print(f"No chunk CSV files found matching pattern: {pattern}")
        return None
    
    print(f"\nMerging {len(chunk_files)} chunk files...")
    
    # Read and concatenate all chunks
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_csv(chunk_file)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {os.path.basename(chunk_file)}")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    merged_file = os.path.join(output_dir, f"qwen_batch_vllm_results_{base_timestamp}_merged_rows_{len(merged_df)}.csv")
    merged_df.to_csv(merged_file, index=False)
    
    print(f"\n✅ Merged {len(merged_df)} total rows into: {merged_file}")
    
    return merged_file

def main():
    parser = argparse.ArgumentParser(
        description="Split large image datasets into chunks and process separately"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Number of images per chunk (default: 2000)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default='cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit',
        help="Model ID to use for inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_files/frame_level",
        help="Directory to save output CSV files"
    )
    # Determine script path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_script = os.path.join(project_root, "qwen_batch_vllm.py")
    
    parser.add_argument(
        "--script-path",
        type=str,
        default=default_script,
        help="Path to the main processing script"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging CSV files at the end"
    )
    
    args = parser.parse_args()
    
    # Add project root to Python path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import here to avoid circular dependencies
    from data_processing.image_loading import load_images
    
    # Load all image files
    print("Loading image file list...")
    image_files = load_images(args.image_dir, keep_every_nth_image=2)
    print(f"Found {len(image_files)} images")
    
    # Split into chunks
    chunks = split_image_list(image_files, args.chunk_size)
    print(f"Split into {len(chunks)} chunks of up to {args.chunk_size} images each")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate base timestamp for all chunks
    import time
    base_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Process each chunk
    success_count = 0
    for chunk_idx, chunk_files in enumerate(chunks):
        success = process_chunk(
            chunk_idx, chunk_files, args.script_path, 
            args.model_id, args.output_dir, base_timestamp
        )
        if success:
            success_count += 1
        else:
            print(f"❌ Chunk {chunk_idx + 1} failed!")
            # Continue with other chunks
    
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/{len(chunks)} chunks successfully")
    print(f"{'='*60}")
    
    # Merge CSV files if requested
    if not args.skip_merge and success_count > 0:
        merge_csv_files(args.output_dir, base_timestamp)

if __name__ == "__main__":
    main()

