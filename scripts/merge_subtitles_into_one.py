#!/usr/bin/env python3
# merge_subtitles_into_one.py

# Take all subtitles, combine them into one file. The file should combine speed, altitude, current, battery voltage, and MAh used.
# The subtitle should be of the format:
# 1
# 00:00:00,000 --> 00:00:00,500
# Speed          Altitude       Current        Battery Voltage MAh Used
# 10 km/h        100 m          1 A           4.10 V          9
# 2
# 00:00:00,500 --> 00:00:01,000
# Speed          Altitude       Current        Battery Voltage MAh Used
# 15 km/h        100 m          11 A          3.70 V          10
# ...
# It should ensure that the values are consistently formatted. Each section gets 15 characters.
# Battery voltage is formatted to 2 decimals (x.xx format).
# MAh used is displayed as an integer.
# The goal is that with a mono spaced font, the subtitles are readable and consistent.
# Uses NBSP (non-breaking space) characters for padding.

import os
from pathlib import Path
from typing import Dict, List, Tuple

NBSP = '\u00A0'


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


def format_value(value: str, width: int = 15) -> str:
    """
    Format a value to fit within the specified width, left-aligned using NBSP for padding.
    
    Args:
        value: The value string to format
        width: The width of the field (default 15)
        
    Returns:
        Formatted string with the value left-aligned in the specified width, padded with NBSP
    """
    return value + (NBSP * (width - len(value)))


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
    milliamps_file = None
    
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
        elif "milliamps" in filename:
            milliamps_file = file
    
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
    if not milliamps_file:
        missing.append("milliamps")
    
    if missing:
        raise FileNotFoundError(f"Missing subtitle files: {', '.join(missing)}")
    
    print(f"Found files:")
    print(f"  Speed: {speed_file.name}")
    print(f"  Altitude: {altitude_file.name}")
    print(f"  Current: {current_file.name}")
    print(f"  Battery: {battery_file.name}")
    print(f"  Milliamps: {milliamps_file.name}")
    
    # Parse all subtitle files
    print("Parsing subtitle files...")
    speed_entries = parse_srt_file(str(speed_file))
    altitude_entries = parse_srt_file(str(altitude_file))
    current_entries = parse_srt_file(str(current_file))
    battery_entries = parse_srt_file(str(battery_file))
    milliamps_entries = parse_srt_file(str(milliamps_file))
    
    # Get all unique timestamps (they should all be the same, but we'll use union to be safe)
    all_timestamps = set(speed_entries.keys()) | set(altitude_entries.keys()) | \
                     set(current_entries.keys()) | set(battery_entries.keys()) | \
                     set(milliamps_entries.keys())
    
    # Sort timestamps
    sorted_timestamps = sorted(all_timestamps)
    
    print(f"Found {len(sorted_timestamps)} subtitle entries")
    
    # Generate output filename if not provided
    if output_filename is None:
        # Use the base name from one of the input files
        base_name = speed_file.stem.replace("_speed_kmh", "")
        output_filename = f"{base_name}_merged.srt"
    
    output_path = input_path / output_filename
    
    def format_battery_voltage(battery_str: str) -> str:
        """
        Format battery voltage to x.xx format (2 decimals).
        
        Args:
            battery_str: Battery voltage string (e.g., "4.1 V" or "3.9 V")
            
        Returns:
            Formatted battery voltage string (e.g., "4.10 V")
        """
        if not battery_str:
            return ""
        
        # Extract the number part
        try:
            # Remove "V" and any whitespace, then convert to float
            voltage_str = battery_str.replace("V", "").strip()
            voltage = float(voltage_str)
            # Format to 2 decimals
            return f"{voltage:.2f} V"
        except (ValueError, AttributeError):
            # If parsing fails, return original
            return battery_str
    
    def extract_mah_integer(mah_str: str) -> str:
        """
        Extract integer value from mah string (e.g., "9 MAh" -> "9").
        
        Args:
            mah_str: Milliampere-hours string (e.g., "9 MAh" or "10 MAh")
            
        Returns:
            Integer value as string (e.g., "9" or "10")
        """
        if not mah_str:
            return ""
        
        try:
            # Extract the number part (before "MAh")
            parts = mah_str.split()
            if parts:
                # Try to parse as integer
                mah_value = int(float(parts[0]))  # Handle cases like "9.0 MAh"
                return str(mah_value)
            return ""
        except (ValueError, AttributeError, IndexError):
            return ""
    
    # Write merged subtitle file
    print(f"Writing merged subtitle file to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, timestamp in enumerate(sorted_timestamps, start=1):
            # Get values (with fallback to empty string if missing)
            speed = speed_entries.get(timestamp, "")
            altitude = altitude_entries.get(timestamp, "")
            current = current_entries.get(timestamp, "")
            battery_raw = battery_entries.get(timestamp, "")
            milliamps_raw = milliamps_entries.get(timestamp, "")
            
            # Format battery voltage to 2 decimals
            battery = format_battery_voltage(battery_raw)
            
            # Extract mahs used as integer
            mahs_used = extract_mah_integer(milliamps_raw)
            
            # Write sequence number
            f.write(f"{idx}\n")
            
            # Write timestamp
            f.write(f"{timestamp}\n")
            
            # Write header line
            header = format_value("Speed") + format_value("Altitude") + format_value("Current") + format_value("Battery Voltage") + format_value("MAh Used")
            f.write(f"{header}\n")
            
            # Write values line
            values = format_value(speed) + format_value(altitude) + format_value(current) + format_value(battery) + format_value(mahs_used)
            f.write(f"{values}\n")
            
            # Write blank line
            f.write("\n")
    
    print(f"Successfully created merged subtitle file: {output_path}")


if __name__ == "__main__":
    # Default input directory
    input_directory = "/home/michael/Projects/VideoUnderstanding/output_files/frame_level/subs"
    
    merge_subtitles(input_directory)
