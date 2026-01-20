#!/usr/bin/env python3
# merge_subtitles_into_one.py

# Take all subtitles, combine them into one file. The file should combine, speed, altitutde, current and battery voltage.
# The subtitle should be of the format:
# 1
# 00:00:00,000 --> 00:00:00,500
# Speed     Altitude    Current   Battery Voltage
# 10 km/h   100 m        1 A      3.6 V
# 2
# 00:00:00,500 --> 00:00:01,000
# Speed     Altitude    Current   Battery Voltage
# 15 km/h   100 m       11 A      3.7 V
# ...
# It should ensure that the values are consitently formatted. Each section gets 20 characters.
# The goal is that with a mono spaced font, the subtitles are readable and consistent.

import os
from pathlib import Path
from typing import Dict, List, Tuple


def parse_srt_file(filepath: str) -> Dict[str, str]:
    """
    Parse an SRT subtitle file and return a dictionary mapping timestamps to values.
    
    Args:
        filepath: Path to the SRT file
        
    Returns:
        Dictionary mapping timestamp strings (e.g., "00:00:00,000 --> 00:00:00,500") to subtitle values
    """
    entries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this is a sequence number (numeric)
        if line.isdigit():
            # Next line should be the timestamp
            if i + 1 < len(lines):
                timestamp = lines[i + 1].strip()
                # Next line should be the value
                if i + 2 < len(lines):
                    value = lines[i + 2].strip()
                    entries[timestamp] = value
                    i += 4  # Skip sequence, timestamp, value, and blank line
                    continue
        
        i += 1
    
    return entries


def format_value(value: str, width: int = 20) -> str:
    """
    Format a value to fit within the specified width, left-aligned.
    
    Args:
        value: The value string to format
        width: The width of the field (default 20)
        
    Returns:
        Formatted string with the value left-aligned in the specified width
    """
    return value.ljust(width)


def merge_subtitles(input_dir: str, output_filename: str = None):
    """
    Merge multiple subtitle files into one combined subtitle file.
    
    Args:
        input_dir: Directory containing the subtitle files
        output_filename: Optional output filename. If not provided, generates one based on input files.
    """
    input_path = Path(input_dir)
    
    # Find the subtitle files
    speed_file = None
    altitude_file = None
    current_file = None
    battery_file = None
    
    for file in input_path.glob("*.srt"):
        filename = file.name.lower()
        if "speed" in filename and "kmh" in filename:
            speed_file = file
        elif "altitude" in filename:
            altitude_file = file
        elif "current" in filename and "ampere" in filename:
            current_file = file
        elif "battery" in filename and "voltage" in filename:
            battery_file = file
    
    # Check that all required files are found
    missing = []
    if not speed_file:
        missing.append("speed_kmh")
    if not altitude_file:
        missing.append("altitude")
    if not current_file:
        missing.append("current_ampere")
    if not battery_file:
        missing.append("battery_voltage")
    
    if missing:
        raise FileNotFoundError(f"Missing subtitle files: {', '.join(missing)}")
    
    print(f"Found files:")
    print(f"  Speed: {speed_file.name}")
    print(f"  Altitude: {altitude_file.name}")
    print(f"  Current: {current_file.name}")
    print(f"  Battery: {battery_file.name}")
    
    # Parse all subtitle files
    print("Parsing subtitle files...")
    speed_entries = parse_srt_file(str(speed_file))
    altitude_entries = parse_srt_file(str(altitude_file))
    current_entries = parse_srt_file(str(current_file))
    battery_entries = parse_srt_file(str(battery_file))
    
    # Get all unique timestamps (they should all be the same, but we'll use union to be safe)
    all_timestamps = set(speed_entries.keys()) | set(altitude_entries.keys()) | \
                     set(current_entries.keys()) | set(battery_entries.keys())
    
    # Sort timestamps
    sorted_timestamps = sorted(all_timestamps)
    
    print(f"Found {len(sorted_timestamps)} subtitle entries")
    
    # Generate output filename if not provided
    if output_filename is None:
        # Use the base name from one of the input files
        base_name = speed_file.stem.replace("_speed_kmh", "")
        output_filename = f"{base_name}_merged.srt"
    
    output_path = input_path / output_filename
    
    # Write merged subtitle file
    print(f"Writing merged subtitle file to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, timestamp in enumerate(sorted_timestamps, start=1):
            # Get values (with fallback to empty string if missing)
            speed = speed_entries.get(timestamp, "")
            altitude = altitude_entries.get(timestamp, "")
            current = current_entries.get(timestamp, "")
            battery = battery_entries.get(timestamp, "")
            
            # Write sequence number
            f.write(f"{idx}\n")
            
            # Write timestamp
            f.write(f"{timestamp}\n")
            
            # Write header line
            header = format_value("Speed") + format_value("Altitude") + format_value("Current") + format_value("Battery Voltage")
            f.write(f"{header}\n")
            
            # Write values line
            values = format_value(speed) + format_value(altitude) + format_value(current) + format_value(battery)
            f.write(f"{values}\n")
            
            # Write blank line
            f.write("\n")
    
    print(f"Successfully created merged subtitle file: {output_path}")


if __name__ == "__main__":
    # Default input directory
    input_directory = "/home/michael/Projects/VideoUnderstanding/output_files/frame_level/subs"
    
    merge_subtitles(input_directory)
