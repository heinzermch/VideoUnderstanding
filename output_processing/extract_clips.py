#!/usr/bin/env python3
"""
Extract video clips from clip-level CSV file.

This script reads a clip-level CSV file (with columns: video, start, end)
and extracts the corresponding video segments from the original video files.
"""

import pandas as pd
import os
import subprocess
import argparse
from pathlib import Path


def mmss_to_seconds(time_str):
    """Convert mm:ss format to seconds.
    
    Args:
        time_str: Time string in mm:ss format (e.g., "05:44")
    
    Returns:
        Total seconds (e.g., 344)
    """
    parts = time_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}. Expected mm:ss")
    minutes, seconds = int(parts[0]), int(parts[1])
    return minutes * 60 + seconds


def find_video_file(video_name, input_dir):
    """Find the video file in the input directory.
    
    Tries common video extensions: .mp4, .MP4, .mov, .MOV, .avi, .AVI, .mkv, .MKV
    
    Args:
        video_name: Name of the video (without extension)
        input_dir: Directory to search for the video file
    
    Returns:
        Path to the video file if found, None otherwise
    """
    common_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV']
    
    for ext in common_extensions:
        video_path = os.path.join(input_dir, f"{video_name}{ext}")
        if os.path.exists(video_path):
            return video_path
    
    return None


def extract_clip(input_video, start_time, end_time, output_path):
    """Extract a video clip using ffmpeg.
    
    Args:
        input_video: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path where the output clip should be saved
    
    Returns:
        True if successful, False otherwise
    """
    duration = end_time - start_time
    
    # Build ffmpeg command
    # -ss: start time
    # -t: duration
    # -c copy: copy codecs (faster, but may not work if start time is not a keyframe)
    # -avoid_negative_ts make_zero: handle timestamp issues
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # Copy codecs for speed (no re-encoding)
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        # Run ffmpeg with minimal output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error extracting clip: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("  Error: ffmpeg not found. Please install ffmpeg.")
        return False


def sanitize_filename(name):
    """Sanitize a string to be used as a filename.
    
    Args:
        name: String to sanitize
    
    Returns:
        Sanitized string safe for use as filename
    """
    # Replace problematic characters
    import re
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing spaces and dots
    name = name.strip(' .')
    return name


def extract_clips_from_csv(csv_path, input_dir, output_dir, max_duration=None):
    """Extract video clips from a clip-level CSV file.
    
    Args:
        csv_path: Path to the clip-level CSV file
        input_dir: Directory containing the original video files
        output_dir: Directory where extracted clips should be saved
        max_duration: Maximum duration in seconds. Clips longer than this will be skipped (default: None, no limit)
    """
    # Read the CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_columns = ['video', 'start', 'end']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Process each row
    total_clips = len(df)
    successful = 0
    failed = 0
    skipped = 0
    missing_videos = []
    
    if max_duration is not None:
        print(f"Maximum duration filter: {max_duration} seconds")
    
    print(f"\nProcessing {total_clips} clips...")
    
    for idx, row in df.iterrows():
        video_name = row['video']
        start_time_str = row['start']
        end_time_str = row['end']
        
        print(f"\n[{idx + 1}/{total_clips}] Processing: {video_name} ({start_time_str} - {end_time_str})")
        
        # Find the video file
        video_path = find_video_file(video_name, input_dir)
        if video_path is None:
            print(f"  Warning: Video file not found for '{video_name}'")
            missing_videos.append(video_name)
            failed += 1
            continue
        
        # Convert time strings to seconds
        try:
            start_seconds = mmss_to_seconds(start_time_str)
            end_seconds = mmss_to_seconds(end_time_str)
        except ValueError as e:
            print(f"  Error: {e}")
            failed += 1
            continue
        
        if start_seconds >= end_seconds:
            print(f"  Error: Start time ({start_time_str}) must be before end time ({end_time_str})")
            failed += 1
            continue
        
        # Check maximum duration
        duration = end_seconds - start_seconds
        if max_duration is not None and duration > max_duration:
            print(f"  Skipped: Duration ({duration}s) exceeds maximum ({max_duration}s)")
            skipped += 1
            continue
        
        # Create output filename
        # Format: video_name_start_end.mp4
        safe_video_name = sanitize_filename(video_name)
        output_filename = f"{safe_video_name}_{start_time_str.replace(':', '-')}_{end_time_str.replace(':', '-')}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Extract the clip
        if extract_clip(video_path, start_seconds, end_seconds, output_path):
            print(f"  Success: {output_filename}")
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total clips: {total_clips}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if skipped > 0:
        print(f"  Skipped (exceeded max duration): {skipped}")
    if missing_videos:
        print(f"\n  Missing video files ({len(missing_videos)}):")
        for video in set(missing_videos):
            print(f"    - {video}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract video clips from clip-level CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_clips.py \\
    -i output_files/clip_level/video_level_results_is_there_at_least_one_person_visible_.csv \\
    --input-dir /home/michael/Videos/model_input \\
    --output-dir extracted_clips
        """
    )
    parser.add_argument(
        '-i', '--input-csv',
        required=True,
        help='Path to the clip-level CSV file (with columns: video, start, end)'
    )
    parser.add_argument(
        '--input-dir', default='/home/michael/Videos/model_input',
        help='Directory containing the original video files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='/home/michael/Videos/cuts/water',
        help='Directory where extracted clips should be saved'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=None,
        help='Maximum duration in seconds. Clips longer than this will be skipped (default: no limit)'
    )
    
    args = parser.parse_args()
    
    extract_clips_from_csv(args.input_csv, args.input_dir, args.output_dir, args.max_duration)


if __name__ == "__main__":
    main()

