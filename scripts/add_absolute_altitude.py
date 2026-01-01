#!/usr/bin/env python3
"""
Script to add a constant value to the altitude column in a CSV file.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


def add_constant_to_altitude(input_file: str, constant: float, output_file: str = None):
    """
    Read a CSV file, add a constant to the altitude column, and save the result.
    
    Args:
        input_file: Path to the input CSV file
        constant: Constant value to add to altitude
        output_file: Optional path to output file. If None, overwrites input file.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if altitude column exists
    if 'altitude' not in df.columns:
        print(f"Error: 'altitude' column not found in CSV file.", file=sys.stderr)
        print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    # Add constant to altitude
    df['altitude'] = df['altitude'] + constant
    
    # Determine output file path
    if output_file is None:
        output_file = input_file
    
    # Write the modified CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully added {constant} to altitude values.")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Add a constant value to the altitude column in a CSV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s aggregated_by_seconds.csv 100.5
  %(prog)s input.csv 50.0 -o output.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default='output_files/frame_level/dvr/dvr_ocr_results_20260101_144013_rows_4077_aggregated_by_seconds.csv',
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        'constant',
        type=float,
        nargs='?',
        default=815.5,
        help='Constant value to add to altitude'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_files/frame_level/dvr/dvr_ocr_results_20260101_144013_rows_4077_aggregated_by_seconds_absolute_altitude.csv',
        help='Output file path (default: overwrites input file)'
    )
    
    args = parser.parse_args()
    
    add_constant_to_altitude(args.input_file, args.constant, args.output)


if __name__ == '__main__':
    main()

