#!/usr/bin/env python3
"""
Merge all qwen_batch_vllm_results CSV files into a single CSV file.
Stores the merged file in the merged folder.
"""

import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

def merge_qwen_batch_vllm_results():
    # Define paths
    frame_level_dir = Path("output_files/frame_level")
    merged_dir = frame_level_dir / "merged"
    
    # Create merged directory if it doesn't exist
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all qwen_batch_vllm_results CSV files
    pattern = str(frame_level_dir / "qwen_batch_vllm_results_*_chunk_*.csv")
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print("No qwen_batch_vllm_results CSV files found.")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # Read and concatenate all CSV files
    dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            total_rows += len(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue
    
    if not dataframes:
        print("No valid CSV files to merge.")
        return
    
    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Remove duplicates based on file_name (keep first occurrence)
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['file_name'], keep='first')
    duplicates_removed = initial_count - len(merged_df)
    
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed} duplicate entries (based on file_name).")
    
    # Sort by file_name
    merged_df = merged_df.sort_values(by='file_name').reset_index(drop=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"qwen_batch_vllm_results_merged_{timestamp}_rows_{len(merged_df)}.csv"
    output_path = merged_dir / output_filename
    
    # Write merged CSV
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nMerged {len(merged_df)} unique rows from {len(csv_files)} files.")
    print(f"Output saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    merge_qwen_batch_vllm_results()
