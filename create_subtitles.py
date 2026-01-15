#!/usr/bin/env python3
"""
Script to create SRT subtitle files from CSV data aggregated by half seconds.
Creates subtitles for: altitude, speed_kmh, current_ampere, battery_voltage, milliamps
"""

import csv
import os
from pathlib import Path


def seconds_to_srt_timestamp(seconds):
    """Convert seconds (float) to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_value(value, unit):
    """Format value with unit, handling NaN and None values"""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    # Format to reasonable precision
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.3f} {unit}"
        elif abs(value) < 1:
            return f"{value:.2f} {unit}"
        elif abs(value) < 100:
            return f"{value:.1f} {unit}"
        else:
            return f"{int(value)} {unit}"
    return f"{value} {unit}"


def create_srt_file(data, column_name, unit, output_path):
    """Create an SRT subtitle file from data"""
    subtitle_index = 1
    srt_content = []
    
    # Use all entries for half-second precision
    for i, row in enumerate(data):
        half_seconds = float(row['half_seconds'])
        value = row.get(column_name)
        
        # Start time is the current half_second mark
        start_time = seconds_to_srt_timestamp(half_seconds)
        # End time is 0.5 seconds later for half-second intervals
        end_time = seconds_to_srt_timestamp(half_seconds + 0.5)
        
        # Format the value with unit
        formatted_value = format_value(value, unit)
        
        # Add subtitle entry
        srt_content.append(f"{subtitle_index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(formatted_value)
        srt_content.append("")  # Empty line between entries
        
        subtitle_index += 1
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))
    
    print(f"Created subtitle file: {output_path}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_subtitles.py <input_csv_file>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_dir = Path("/home/michael/Projects/VideoUnderstanding/output_files/frame_level/subs")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV data
    data = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for key in ['half_seconds', 'altitude', 'speed_kmh', 'current_ampere', 
                       'battery_voltage', 'milliamps']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            data.append(row)
    
    # Define categories and their units
    categories = {
        'altitude': 'm',
        'speed_kmh': 'km/h',
        'current_ampere': 'A',
        'battery_voltage': 'V',
        'milliamps': 'MAh'
    }
    
    # Get base filename without extension
    base_name = Path(input_csv).stem
    
    # Create subtitle file for each category
    for column_name, unit in categories.items():
        output_filename = f"{base_name}_{column_name}.srt"
        output_path = output_dir / output_filename
        create_srt_file(data, column_name, unit, output_path)
    
    print(f"\nAll subtitle files created in: {output_dir}")


if __name__ == "__main__":
    main()

