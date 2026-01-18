#!/usr/bin/env python3
"""
Process DVR aggregated CSV file:
- Sets index to half_seconds * 2
- Keeps half_seconds as time_code
- Rounds altitude, speed_kmh, and current_ampere to integers
- Keeps battery_voltage and milliamps as is
"""

import pandas as pd
import sys
import os


def process_dvr_aggregated(input_file, output_file=None):
    """
    Process the DVR aggregated CSV file according to specifications.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, defaults to input_file with _processed suffix)
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Create index from half_seconds * 2
    df['index'] = (df['half_seconds'] * 2).astype(int)
    df.set_index('index', inplace=True)
    
    # Rename half_seconds to time_code
    df.rename(columns={'half_seconds': 'time_code'}, inplace=True)
    
    # Round altitude, speed_kmh, and current_ampere to integers
    df['altitude'] = df['altitude'].round().astype(int)
    df['speed_kmh'] = df['speed_kmh'].round().astype(int)
    df['current_ampere'] = df['current_ampere'].round().astype(int)
    
    # Keep only the specified columns
    columns_to_keep = ['time_code', 'altitude', 'speed_kmh', 'current_ampere', 
                       'battery_voltage', 'milliamps']
    df = df[columns_to_keep]
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=True)
    print(f"Processed file saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_dvr_aggregated.py <input_csv_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    process_dvr_aggregated(input_file, output_file)
