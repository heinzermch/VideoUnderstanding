# This script loads the processed csv and extracts frame level information to combine
# it to video level. So if we detect an object in continuous frames it should be synthezied in the form
# of mm:ss-mm:ss
# As an example, assume we have a detection in the file:
#
#file_name,Is there at least one person visible?,"Is there a lake, river, or large body of water?"
#/home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00028.png,No,No
#/home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00030.png,Yes,No
#/home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00032.png,Yes,No
#/home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00034.png,No,No
#
# The questions should be separated and split into two files, one per category:
# File name video_level_results_person_visible
# This should be cominbed to video,start,end
# 5seenwanderung_letztersee_hd,00:30,00:34
# 
# One should be careful to change the frame numbers in the file names to proper mm:ss format.

import pandas as pd
import os
import re
import argparse


def extract_video_name(file_path):
    """Extract video name from file path.
    
    Example: /home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00030.png
    -> 5seenwanderung_letztersee_hd
    """
    # Get the directory name that contains the frame file
    dir_name = os.path.basename(os.path.dirname(file_path))
    return dir_name


def extract_frame_number(file_path):
    """Extract frame number from filename.
    
    Example: /home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00030.png
    -> 30
    """
    filename = os.path.basename(file_path)
    # Extract the number from filename (e.g., 00030.png -> 30)
    match = re.match(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return None


def seconds_to_mmss(seconds):
    """Convert seconds to mm:ss format.
    
    Example: 30 -> 00:30, 125 -> 02:05
    """
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def find_consecutive_ranges(frame_data, question_column):
    """Find consecutive frames with 'Yes' answers for a given question.
    
    Args:
        frame_data: DataFrame with columns including 'video', 'frame_number', and question_column
        question_column: Name of the question column to process
    
    Returns:
        List of tuples (video, start_frame, end_frame) for consecutive 'Yes' ranges
    """
    ranges = []
    
    # Group by video
    for video_name, video_group in frame_data.groupby('video'):
        # Sort by frame number
        video_group = video_group.sort_values('frame_number')
        
        # Find consecutive 'Yes' frames
        current_start = None
        current_end = None
        
        for _, row in video_group.iterrows():
            if row[question_column] == 'Yes':
                if current_start is None:
                    # Start a new range
                    current_start = row['frame_number']
                    current_end = row['frame_number']
                else:
                    # Continue the current range
                    current_end = row['frame_number']
            else:
                # End the current range if it exists
                if current_start is not None:
                    ranges.append((video_name, current_start, current_end))
                    current_start = None
                    current_end = None
        
        # Don't forget the last range if it ends at the last frame
        if current_start is not None:
            ranges.append((video_name, current_start, current_end))
    
    return ranges


def sanitize_filename(text):
    """Convert question text to a valid filename."""
    # Replace problematic characters with underscores
    filename = re.sub(r'[<>:"/\\|?*,]', '_', text)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    return filename.lower()


def process_frame_to_video_level(input_csv, output_dir=None, min_duration=3):
    """Process frame-level CSV and create clip-level CSVs for each question.
    
    Args:
        input_csv: Path to input frame-level CSV file
        output_dir: Directory to save output files (default: output_files/clip_level)
        min_duration: Minimum duration in seconds for a clip to be saved (default: 3)
    """
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Extract video name and frame number from file_name column
    df['video'] = df['file_name'].apply(extract_video_name)
    df['frame_number'] = df['file_name'].apply(extract_frame_number)
    
    # Get all question columns (all columns except 'file_name', 'video', 'frame_number')
    question_columns = [col for col in df.columns 
                       if col not in ['file_name', 'video', 'frame_number']]
    
    # Set output directory
    if output_dir is None:
        output_dir = "output_files/clip_level"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each question column
    for question in question_columns:
        print(f"\nProcessing question: {question}")
        
        # Find consecutive ranges
        ranges = find_consecutive_ranges(df, question)
        
        if not ranges:
            print(f"  No 'Yes' answers found for: {question}")
            continue
        
        # Filter out clips that are shorter than min_duration
        # Since frame numbers represent seconds, duration = end_frame - start_frame
        filtered_ranges = [
            (video, start_frame, end_frame) 
            for video, start_frame, end_frame in ranges 
            if (end_frame - start_frame) > min_duration
        ]
        
        if not filtered_ranges:
            print(f"  No clips longer than {min_duration} seconds found for: {question}")
            continue
        
        print(f"  Filtered {len(ranges) - len(filtered_ranges)} clips of {min_duration} seconds or less")
        
        # Convert to DataFrame with video, start, end columns
        clip_data = []
        for video, start_frame, end_frame in filtered_ranges:
            clip_data.append({
                'video': video,
                'start': seconds_to_mmss(start_frame),
                'end': seconds_to_mmss(end_frame)
            })
        
        clip_df = pd.DataFrame(clip_data)
        
        # Create output filename
        safe_question_name = sanitize_filename(question)
        output_filename = f"video_level_results_{safe_question_name}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to CSV
        clip_df.to_csv(output_path, index=False)
        print(f"  Saved {len(clip_df)} ranges to {output_path}")
        print(f"  Sample output:")
        print(clip_df.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Convert frame-level CSV to clip-level CSVs'
    )
    parser.add_argument(
        '-i', '--input_csv',
        default='output_files/frame_level/qwen_batch_vllm_results_20251226_185030_rows_1711.csv',
        help='Path to input frame-level CSV file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='output_files/clip_level',
        help='Output directory for clip-level CSVs (default: output_files/clip_level)'
    )
    parser.add_argument(
        '-m', '--min-duration',
        type=int,
        default=5,
        help='Minimum duration in seconds for a clip to be saved (default: 5)'
    )
    
    args = parser.parse_args()
    
    process_frame_to_video_level(args.input_csv, args.output_dir, args.min_duration)


if __name__ == "__main__":
    main()
