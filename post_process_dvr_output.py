#!/usr/bin/env python3
"""
Post-process DVR OCR output CSV file.
- Separates individual columns properly
- Removes units from numeric values
- Cleans multiline content
"""

import pandas as pd
import re
import sys
import os


def extract_lat_lon(value):
    """Extract latitude and longitude from text like 'LAT 46.9738599, LON 8.5870643'
    Validates that both have at least 4 decimal places, otherwise returns None"""
    if pd.isna(value) or value == "":
        return None, None
    
    # Remove multiline content, keep only first line
    value = str(value).split('\n')[0].strip()
    
    # Extract LAT and LON values
    lat_match = re.search(r'LAT\s+([+-]?\d+\.?\d*)', value, re.IGNORECASE)
    lon_match = re.search(r'LON\s+([+-]?\d+\.?\d*)', value, re.IGNORECASE)
    
    lat = lat_match.group(1) if lat_match else None
    lon = lon_match.group(1) if lon_match else None
    
    # Validate that both have at least 4 decimal places
    if lat is not None:
        if '.' in lat:
            decimal_part = lat.split('.')[1]
            if len(decimal_part) < 4:
                lat = None
        else:
            # No decimal point, invalid
            lat = None
    
    if lon is not None:
        if '.' in lon:
            decimal_part = lon.split('.')[1]
            if len(decimal_part) < 4:
                lon = None
        else:
            # No decimal point, invalid
            lon = None
    
    return lat, lon


def clean_time(value):
    """Clean time value, keep format like '00:00'"""
    if pd.isna(value) or value == "":
        return None
    
    # Remove multiline content, keep only first line
    value = str(value).split('\n')[0].strip()
    
    # Extract time pattern (HH:MM or MM:SS)
    time_match = re.search(r'(\d{1,2}:\d{2})', value)
    if time_match:
        return time_match.group(1)
    
    return value.strip()


def clean_altitude(value):
    """Remove units from altitude (M, ALT = prefix)"""
    if pd.isna(value) or value == "":
        return None
    
    # Remove multiline content, keep only first line
    value = str(value).split('\n')[0].strip()
    
    # Remove "ALT = " prefix if present
    value = re.sub(r'ALT\s*=\s*', '', value, flags=re.IGNORECASE)
    
    # Extract numeric value and remove "M" unit
    # Look for pattern like "1.8 M" or "ALT = 1.9 M"
    match = re.search(r'(\d+\.?\d*)\s*M\b', value, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Fallback: just extract number
    match = re.search(r'(\d+\.?\d*)', value)
    if match:
        return match.group(1)
    
    return None


def clean_speed(value):
    """Remove 'km/h' unit from speed"""
    if pd.isna(value) or value == "":
        return None
    
    # Remove multiline content, keep only first line
    value = str(value).split('\n')[0].strip()
    
    # Extract numeric value and remove "km/h" unit
    match = re.search(r'([+-]?\d+\.?\d*)', value)
    if match:
        return match.group(1)
    
    return None


def clean_milliamps(value):
    """Remove 'MA', 'MAh' units from milliamps and validate it's an integer"""
    if pd.isna(value) or value == "":
        return None
    
    # Remove multiline content, keep only first line
    value = str(value).split('\n')[0].strip()
    
    # Skip if it looks like a time (e.g., "63:2") or voltage or temperature
    if re.match(r'\d+:\d+', value) or 'V' in value.upper() or '°C' in value or 'C' in value.upper():
        return None
    
    # Extract numeric value and remove "MA", "MAh" units
    # Look for pattern like "11 MA", "12 MAh"
    match = re.search(r'(\d+\.?\d*)\s*(?:MA|MAh)', value, re.IGNORECASE)
    if match:
        num_str = match.group(1)
    else:
        # Fallback: just extract number if no unit found
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            num_str = match.group(1)
        else:
            return None
    
    # Validate it's an integer (no decimal part)
    try:
        num_float = float(num_str)
        if num_float.is_integer():
            return str(int(num_float))
        else:
            # Not an integer, return None
            return None
    except (ValueError, TypeError):
        return None


def clean_current_ampere(value):
    """Remove 'A' unit from current ampere and validate it's a float of form x.xx (two decimal places)"""
    if pd.isna(value) or value == "":
        return None
    
    # Handle multiline content - look for the actual current value
    # The multiline content often has format like "6. 3.52 A" on a line
    lines = str(value).split('\n')
    
    # First, try to find a line with "A" unit (current ampere)
    for line in lines:
        line = line.strip()
        # Look for pattern like "3.52 A" or "6. 3.52 A"
        match = re.search(r'(\d+\.?\d*)\s*A\b', line, re.IGNORECASE)
        if match:
            num_str = match.group(1)
            # Validate it's a float of form x.xx (two decimal places)
            try:
                num_float = float(num_str)
                # Check if it has exactly two decimal places
                if '.' in num_str:
                    decimal_part = num_str.split('.')[1]
                    if len(decimal_part) == 2:
                        return num_str
                # If it's a whole number, format it as x.00
                if num_float.is_integer():
                    return f"{int(num_float)}.00"
                return None
            except (ValueError, TypeError):
                return None
    
    # If not found in multiline, check first line
    value = lines[0].strip()
    
    # Skip if it looks like coordinates or other data
    if 'LAT' in value.upper() or 'LON' in value.upper():
        return None
    
    # Extract numeric value with "A" unit
    match = re.search(r'(\d+\.?\d*)\s*A\b', value, re.IGNORECASE)
    if match:
        num_str = match.group(1)
        # Validate it's a float of form x.xx (two decimal places)
        try:
            num_float = float(num_str)
            # Check if it has exactly two decimal places
            if '.' in num_str:
                decimal_part = num_str.split('.')[1]
                if len(decimal_part) == 2:
                    return num_str
            # If it's a whole number, format it as x.00
            if num_float.is_integer():
                return f"{int(num_float)}.00"
            return None
        except (ValueError, TypeError):
            return None
    
    return None


def clean_battery_voltage(value):
    """Remove 'V' unit from battery voltage and validate it's between 0 and 5"""
    if pd.isna(value) or value == "":
        return None
    
    # Handle multiline content - look for voltage values
    # The multiline content often has format like "6. 4.06 V" on a line
    lines = str(value).split('\n')
    
    # First, try to find a line with "V" unit (voltage)
    # Prefer values between 0-5 volts (battery voltage range)
    voltage_values = []
    for line in lines:
        line = line.strip()
        # Look for pattern like "4.06 V" or "24.4 V" or "6. 4.06 V"
        match = re.search(r'(\d+\.?\d*)\s*V\b', line, re.IGNORECASE)
        if match:
            try:
                voltage = float(match.group(1))
                voltage_values.append((voltage, line))
            except (ValueError, TypeError):
                continue
    
    # If we found voltage values, prefer the one in the 0-5V range (battery voltage)
    if voltage_values:
        # Sort by preference: 0-5V range first, then others
        voltage_values.sort(key=lambda x: (not (0 <= x[0] <= 5), x[0]))
        voltage = voltage_values[0][0]
        # Validate it's between 0 and 5
        if 0 <= voltage <= 5:
            return str(voltage)
        else:
            return None
    
    # If not found in multiline, check first line
    value = lines[0].strip()
    
    # Extract numeric value with "V" unit
    match = re.search(r'(\d+\.?\d*)\s*V\b', value, re.IGNORECASE)
    if match:
        try:
            voltage = float(match.group(1))
            # Validate it's between 0 and 5
            if 0 <= voltage <= 5:
                return str(voltage)
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    return None


def validate_milliamps_sequence(milliamps_series):
    """
    Validate milliamps sequence:
    - Must be monotonically increasing
    - Cannot jump by more than 100 between consecutive values
    - Uses the last valid value for comparison even if there are None values in between
    - Returns a new series with invalid values set to None
    """
    validated = milliamps_series.copy()
    last_valid_value = None
    
    for idx, current_val in enumerate(milliamps_series):
        if pd.isna(current_val) or current_val is None:
            # Keep None values as None, but don't reset last_valid_value
            continue
        
        try:
            current_int = int(current_val)
            
            if last_valid_value is not None:
                # Check if it's decreasing
                if current_int < last_valid_value:
                    validated.iloc[idx] = None
                    continue
                
                # Check if jump is more than 100
                jump = current_int - last_valid_value
                if jump > 100:
                    validated.iloc[idx] = None
                    continue
            
            # Valid value, update last_valid_value
            last_valid_value = current_int
            
        except (ValueError, TypeError):
            # Invalid value, set to None
            validated.iloc[idx] = None
            # Don't reset last_valid_value - keep the last valid value
    
    return validated


def post_process_dvr_output(input_file, output_file=None):
    """
    Post-process DVR OCR output CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_file with '_processed' suffix)
    """
    print(f"Reading input file: {input_file}")
    
    # Read CSV with proper handling of multiline values
    # Use engine='python' for better multiline handling
    try:
        df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True, engine='python')
    except Exception as e:
        print(f"Warning: Error reading CSV: {e}")
        print("Trying with default engine...")
        df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Create new dataframe with cleaned columns
    processed_df = pd.DataFrame()
    processed_df['file_name'] = df['file_name']
    
    # Process latitude and longitude
    lat_lon_col = 'What is the latitude and longitude shown in the top right of the frame?'
    if lat_lon_col in df.columns:
        lat_lon_data = df[lat_lon_col].apply(extract_lat_lon)
        processed_df['latitude'] = [x[0] for x in lat_lon_data]
        processed_df['longitude'] = [x[1] for x in lat_lon_data]
    
    # Process time
    time_col = 'What is the time of recording shown in the middle right of the frame?'
    if time_col in df.columns:
        processed_df['time'] = df[time_col].apply(clean_time)
    
    # Process altitude
    altitude_col = 'What is the altitude of the drone shown in the middle right of the frame?'
    if altitude_col in df.columns:
        processed_df['altitude'] = df[altitude_col].apply(clean_altitude)
    
    # Process speed
    speed_col = 'What is the speed of the drone shown in the middle right in km/h?'
    if speed_col in df.columns:
        processed_df['speed_kmh'] = df[speed_col].apply(clean_speed)
    
    # Process milliamps
    milliamps_col = 'What is the milliamps used shown on the middle left?'
    if milliamps_col in df.columns:
        processed_df['milliamps'] = df[milliamps_col].apply(clean_milliamps)
        # Validate milliamps sequence (must be increasing, no jumps > 100)
        processed_df['milliamps'] = validate_milliamps_sequence(processed_df['milliamps'])
    
    # Process current ampere
    current_col = 'What is the current ampere usage shown on the middle left?'
    if current_col in df.columns:
        processed_df['current_ampere'] = df[current_col].apply(clean_current_ampere)
    
    # Process battery voltage
    voltage_col = 'What is the battery voltage shown on the bottom middle (between 2 and 5 volts)?'
    if voltage_col in df.columns:
        processed_df['battery_voltage'] = df[voltage_col].apply(clean_battery_voltage)
    
    # Determine output file path
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.csv"
    
    print(f"\nSaving processed data to: {output_file}")
    processed_df.to_csv(output_file, index=False)
    
    print(f"Processed {len(processed_df)} rows")
    print(f"Output columns: {list(processed_df.columns)}")
    
    # Show sample of processed data
    print("\nSample of processed data:")
    print(processed_df.head(200).to_string())
    print('--------------------------------')
    print(processed_df.tail(200).to_string())
    
    # Debug output: Count None values for each column (indicates data issues)
    print("\n" + "="*60)
    print("DEBUG: None value counts (data quality issues):")
    print("="*60)
    total_rows = len(processed_df)
    for col in processed_df.columns:
        if col == 'file_name':
            continue  # Skip file_name column
        # Count None/NaN values (including string 'None')
        none_count = sum(1 for x in processed_df[col] if pd.isna(x) or x is None or str(x).strip() == 'None')
        percentage = (none_count / total_rows * 100) if total_rows > 0 else 0
        print(f"  {col:25s}: {none_count:5d} None values ({percentage:5.1f}%)")
    print("="*60)
    
    return processed_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python post_process_dvr_output.py <input_csv_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    post_process_dvr_output(input_file, output_file)

