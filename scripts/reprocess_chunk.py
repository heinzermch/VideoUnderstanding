#!/usr/bin/env python3
"""
Reprocess a specific chunk and merge it with the existing merged CSV file.
"""

import os
import sys
import subprocess
import argparse
import json
import glob
import pandas as pd
from pathlib import Path

# Add project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_processing.image_loading import load_images

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

def find_merged_csv(output_dir):
    """Find the most recent merged CSV file."""
    pattern = os.path.join(output_dir, "qwen_batch_vllm_results_*_merged_rows_*.csv")
    merged_files = sorted(glob.glob(pattern))
    
    if not merged_files:
        return None
    
    # Return the most recent one (last in sorted list)
    return merged_files[-1]

def merge_chunk_with_csv(chunk_file, merged_csv_file, output_dir, chunk_num):
    """Replace chunk data in merged CSV with new chunk data."""
    print(f"\n{'='*60}")
    print(f"Merging chunk {chunk_num} with existing merged CSV")
    print(f"{'='*60}")
    
    # Load the new chunk data
    print(f"Loading new chunk data from: {os.path.basename(chunk_file)}")
    chunk_df = pd.read_csv(chunk_file)
    print(f"  Loaded {len(chunk_df)} rows from chunk {chunk_num}")
    
    # Load the merged CSV
    print(f"Loading merged CSV: {os.path.basename(merged_csv_file)}")
    merged_df = pd.read_csv(merged_csv_file)
    print(f"  Loaded {len(merged_df)} rows from merged CSV")
    
    # Find rows that belong to chunk 8 in the merged CSV
    # We'll identify them by checking if the file paths match the chunk files
    # Since we don't have a chunk column, we'll need to match by file paths
    chunk_file_paths = set(chunk_df['file_name'].values)
    
    # Remove old chunk 8 rows (rows where file_name matches chunk 8 files)
    before_count = len(merged_df)
    merged_df = merged_df[~merged_df['file_name'].isin(chunk_file_paths)]
    removed_count = before_count - len(merged_df)
    print(f"  Removed {removed_count} old rows that match chunk {chunk_num} files")
    
    # Add the new chunk data
    merged_df = pd.concat([merged_df, chunk_df], ignore_index=True)
    print(f"  Added {len(chunk_df)} new rows from chunk {chunk_num}")
    
    # Sort by file_name to maintain consistency
    merged_df = merged_df.sort_values('file_name').reset_index(drop=True)
    
    # Save the updated merged CSV
    # Use the same filename as the original merged CSV
    merged_df.to_csv(merged_csv_file, index=False)
    
    print(f"\n✅ Updated merged CSV: {os.path.basename(merged_csv_file)}")
    print(f"   Total rows: {len(merged_df)}")
    
    return merged_csv_file

def main():
    parser = argparse.ArgumentParser(
        description="Reprocess a specific chunk and merge with existing CSV"
    )
    parser.add_argument(
        "--chunk-num",
        type=int,
        required=True,
        help="Chunk number to reprocess (1-indexed, e.g., 8 for chunk 8)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/michael/Videos/model_input/frames/",
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
    parser.add_argument(
        "--merged-csv",
        type=str,
        default=None,
        help="Path to merged CSV file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--base-timestamp",
        type=str,
        default=None,
        help="Base timestamp for output files (auto-generated if not provided)"
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
    
    args = parser.parse_args()
    
    # Validate chunk number
    if args.chunk_num < 1:
        print("Error: chunk-num must be >= 1")
        sys.exit(1)
    
    chunk_idx = args.chunk_num - 1  # Convert to 0-indexed
    
    # Load all image files
    print("Loading image file list...")
    image_files = load_images(args.image_dir, keep_every_nth_image=2)
    print(f"Found {len(image_files)} images")
    
    # Split into chunks
    chunks = split_image_list(image_files, args.chunk_size)
    print(f"Split into {len(chunks)} chunks of up to {args.chunk_size} images each")
    
    # Validate chunk number
    if chunk_idx >= len(chunks):
        print(f"Error: chunk {args.chunk_num} does not exist. Only {len(chunks)} chunks available.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate or use provided base timestamp
    import time
    if args.base_timestamp:
        base_timestamp = args.base_timestamp
    else:
        base_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Process the chunk
    chunk_files = chunks[chunk_idx]
    print(f"\n{'='*60}")
    print(f"Reprocessing chunk {args.chunk_num} ({len(chunk_files)} images)")
    print(f"{'='*60}")
    
    success = process_chunk(
        chunk_idx, chunk_files, args.script_path, 
        args.model_id, args.output_dir, base_timestamp
    )
    
    if not success:
        print(f"❌ Chunk {args.chunk_num} processing failed!")
        sys.exit(1)
    
    # Find the chunk output file
    chunk_pattern = os.path.join(args.output_dir, f"qwen_batch_vllm_results_*_chunk_{args.chunk_num}.csv")
    chunk_files_found = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files_found:
        print(f"❌ Could not find output file for chunk {args.chunk_num}")
        print(f"   Searched for pattern: {chunk_pattern}")
        sys.exit(1)
    
    chunk_file = chunk_files_found[-1]  # Use most recent if multiple
    print(f"\n✅ Chunk {args.chunk_num} processed successfully")
    print(f"   Output file: {os.path.basename(chunk_file)}")
    
    # Find or use provided merged CSV
    if args.merged_csv:
        merged_csv_file = args.merged_csv
        if not os.path.exists(merged_csv_file):
            print(f"❌ Merged CSV file not found: {merged_csv_file}")
            sys.exit(1)
    else:
        merged_csv_file = find_merged_csv(args.output_dir)
        if not merged_csv_file:
            print("❌ Could not find merged CSV file")
            print("   Please provide --merged-csv or ensure a merged CSV exists in output directory")
            sys.exit(1)
    
    print(f"\nFound merged CSV: {os.path.basename(merged_csv_file)}")
    
    # Merge the new chunk with the existing merged CSV
    merge_chunk_with_csv(chunk_file, merged_csv_file, args.output_dir, args.chunk_num)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully reprocessed chunk {args.chunk_num} and merged with CSV")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

