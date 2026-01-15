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

# Optional matplotlib import for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
    """Remove 'A' unit from current ampere and validate it's a float of form x.xx (two decimal places)
    
    Two cases:
    1. If there's a single "." in the value, it can be considered the valid ampere measurement (remove the 'A')
    2. If there's a "6." in there, that starts the ampere measurement (question number prefix)
    """
    if pd.isna(value) or value == "":
        return None
    
    value_str = str(value)
    
    # Skip if it looks like coordinates or other data
    if 'LAT' in value_str.upper() or 'LON' in value_str.upper():
        return None
    
    # Case 1: Look for pattern like "6. X.XX" or "6. X.XX A" where "6." is the question number prefix
    # Search all lines for "6." pattern
    lines = value_str.split('\n')
    for line in lines:
        line = line.strip()
        # Pattern: "6." followed by optional space, then a number with decimal (may or may not have 'A')
        # Must have a decimal point in the number (e.g., "6. 15.75" or "6. 3.52 A")
        match = re.search(r'6\.\s*(\d+\.\d+)(?:\s*A\b)?', line, re.IGNORECASE)
        if match:
            num_str = match.group(1)
            # Validate and format to two decimal places
            try:
                num_float = float(num_str)
                # Format to exactly two decimal places
                return f"{num_float:.2f}"
            except (ValueError, TypeError):
                pass
    
    # Case 2: Look for pattern with single "." (like "1.52" or "1.40 A")
    # Check all lines, but prefer the one with 'A' if present
    # First, try to find one with 'A' unit
    for line in lines:
        line = line.strip()
        dot_count = line.count('.')
        if dot_count == 1:
            # Single dot case - extract the number (with or without 'A' unit)
            match = re.search(r'(\d+\.\d+)(?:\s*A\b)?', line, re.IGNORECASE)
            if match:
                num_str = match.group(1)
                try:
                    num_float = float(num_str)
                    # Format to exactly two decimal places
                    return f"{num_float:.2f}"
                except (ValueError, TypeError):
                    pass
    
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


def validate_speed_sequence(speed_series):
    """
    Validate speed sequence:
    - For speeds below 10 km/h: if it increases or decreases more than 50% 
      compared to the last valid value, set to None
    - Uses the last valid value for comparison even if there are None values in between
    - Returns a new series with invalid values set to None
    """
    validated = speed_series.copy()
    last_valid_value = None
    
    for idx, current_val in enumerate(speed_series):
        if pd.isna(current_val) or current_val is None:
            # Keep None values as None, but don't reset last_valid_value
            continue
        
        try:
            current_float = float(current_val)
            
            # Check if speed is above 250 km/h (unrealistic)
            if current_float > 250.0:
                validated.iloc[idx] = None
                continue
            
            # Only validate if speed is below 10 km/h
            if current_float < 10.0 and last_valid_value is not None:
                # Calculate percentage change
                change = abs(current_float - last_valid_value)
                percentage_change = change / last_valid_value if last_valid_value > 0 else float('inf')
                
                # If change is more than 50%, set to None
                if percentage_change > 0.5:
                    validated.iloc[idx] = None
                    continue
            
            # Valid value, update last_valid_value
            last_valid_value = current_float
            
        except (ValueError, TypeError):
            # Invalid value, set to None
            validated.iloc[idx] = None
            # Don't reset last_valid_value - keep the last valid value
    
    return validated


def validate_altitude_sequence(altitude_series):
    """
    Validate altitude sequence:
    - If it changes by more than 50 from previous valid value, set to None
    - Uses the last valid value for comparison even if there are None values in between
    - Returns a new series with invalid values set to None
    """
    validated = altitude_series.copy()
    last_valid_value = None
    
    for idx, current_val in enumerate(altitude_series):
        if pd.isna(current_val) or current_val is None:
            # Keep None values as None, but don't reset last_valid_value
            continue
        
        try:
            current_float = float(current_val)
            
            if last_valid_value is not None:
                # Check if change is more than 50
                change = abs(current_float - last_valid_value)
                if change > 50.0:
                    validated.iloc[idx] = None
                    continue
            
            # Valid value, update last_valid_value
            last_valid_value = current_float
            
        except (ValueError, TypeError):
            # Invalid value, set to None
            validated.iloc[idx] = None
            # Don't reset last_valid_value - keep the last valid value
    
    return validated


def validate_battery_voltage_sequence(voltage_series):
    """
    Validate battery voltage sequence:
    - If it changes more than 20% from previous valid value, set to None
    - Uses the last valid value for comparison even if there are None values in between
    - Returns a new series with invalid values set to None
    """
    validated = voltage_series.copy()
    last_valid_value = None
    
    for idx, current_val in enumerate(voltage_series):
        if pd.isna(current_val) or current_val is None:
            # Keep None values as None, but don't reset last_valid_value
            continue
        
        try:
            current_float = float(current_val)
            
            if last_valid_value is not None:
                # Calculate percentage change
                change = abs(current_float - last_valid_value)
                percentage_change = change / last_valid_value if last_valid_value > 0 else float('inf')
                
                # If change is more than 20%, set to None
                if percentage_change > 0.2:
                    validated.iloc[idx] = None
                    continue
            
            # Valid value, update last_valid_value
            last_valid_value = current_float
            
        except (ValueError, TypeError):
            # Invalid value, set to None
            validated.iloc[idx] = None
            # Don't reset last_valid_value - keep the last valid value
    
    return validated


def validate_current_ampere_sequence(ampere_series):
    """
    Validate current ampere sequence:
    - If value is above 240 amps, set to None
    - If previous valid value is >= 10 and change is > 100%, set to None
    - Uses the last valid value for comparison even if there are None values in between
    - Returns a new series with invalid values set to None
    """
    validated = ampere_series.copy()
    last_valid_value = None
    
    for idx, current_val in enumerate(ampere_series):
        if pd.isna(current_val) or current_val is None:
            # Keep None values as None, but don't reset last_valid_value
            continue
        
        try:
            current_float = float(current_val)
            
            # Check if value is above 240 amps
            if current_float > 240.0:
                validated.iloc[idx] = None
                continue
            
            # If previous valid value is >= 10, check for > 100% change
            if last_valid_value is not None and last_valid_value >= 10.0:
                # Calculate percentage change
                change = abs(current_float - last_valid_value)
                percentage_change = change / last_valid_value if last_valid_value > 0 else float('inf')
                
                # If change is more than 100%, set to None
                if percentage_change > 1.0:
                    validated.iloc[idx] = None
                    continue
            
            # Valid value, update last_valid_value
            last_valid_value = current_float
            
        except (ValueError, TypeError):
            # Invalid value, set to None
            validated.iloc[idx] = None
            # Don't reset last_valid_value - keep the last valid value
    
    return validated


def impute_with_linear_interpolation(processed_df):
    """
    Perform linear interpolation on missing values and create a separate dataframe
    with imputation flags.
    
    Args:
        processed_df: DataFrame with cleaned but potentially missing values
    
    Returns:
        DataFrame with imputed values and boolean columns indicating imputation
    """
    imputed_df = processed_df.copy()
    
    # Columns that should be interpolated (numeric columns)
    numeric_columns = ['latitude', 'longitude', 'altitude', 'speed_kmh', 
                      'milliamps', 'current_ampere', 'battery_voltage']
    
    # Track which values were originally missing (before interpolation)
    original_missing = {}
    for col in numeric_columns:
        if col in imputed_df.columns:
            # Track original missing values (None, NaN, or string 'None')
            original_missing[col] = imputed_df[col].apply(
                lambda x: pd.isna(x) or x is None or str(x).strip() == 'None'
            )
    
    # Perform interpolation for each numeric column
    for col in numeric_columns:
        if col not in imputed_df.columns:
            continue
        
        # Convert to numeric, setting invalid values to NaN
        numeric_series = pd.to_numeric(imputed_df[col], errors='coerce')
        
        # Perform linear interpolation
        interpolated = numeric_series.interpolate(method='linear', limit_direction='both')
        
        # Format interpolated values according to column type
        if col == 'milliamps':
            # Milliamps should be integers
            interpolated = interpolated.round().astype('Int64')  # Nullable integer type
        elif col == 'current_ampere':
            # Current ampere should have 2 decimal places
            interpolated = interpolated.round(2)
        # Other columns keep as float
        
        # Update the dataframe with interpolated values
        imputed_df[col] = interpolated
        
        # Create imputation flag column
        imputation_flag_col = f"{col}_imputed"
        if col in original_missing:
            imputed_df[imputation_flag_col] = original_missing[col]
        else:
            imputed_df[imputation_flag_col] = False
    
    # For time column, we don't interpolate but still add a flag column (all False)
    if 'time' in imputed_df.columns:
        imputed_df['time_imputed'] = False
    
    return imputed_df


def create_plots(imputed_df, output_dir, base_name):
    """
    Create plots of altitude, speed, milliamps, current_ampere, and battery_voltage over index.
    Each plot is saved as a 1000x1000 pixel image.
    
    Args:
        imputed_df: DataFrame with imputed values
        output_dir: Directory to save plots
        base_name: Base name for output files
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib is not installed. Skipping plot generation.")
        print("Install matplotlib with: pip install matplotlib")
        return
    
    # Columns to plot
    plot_columns = {
        'altitude': 'Altitude (M)',
        'speed_kmh': 'Speed (km/h)',
        'milliamps': 'Milliamps',
        'current_ampere': 'Current Ampere (A)',
        'battery_voltage': 'Battery Voltage (V)'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set DPI to 100 so 10x10 inches = 1000x1000 pixels
    dpi = 100
    fig_size = (10, 10)  # 10 inches = 1000 pixels at 100 DPI
    
    for col, ylabel in plot_columns.items():
        if col not in imputed_df.columns:
            print(f"Warning: Column '{col}' not found, skipping plot")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        # Convert to numeric, handling any remaining non-numeric values
        plot_data = pd.to_numeric(imputed_df[col], errors='coerce')
        
        # Plot the data
        ax.plot(plot_data.index, plot_data.values, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} over Time (Index)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        valid_data = plot_data.dropna()
        if len(valid_data) > 0:
            stats_text = f'Mean: {valid_data.mean():.2f}\n'
            stats_text += f'Min: {valid_data.min():.2f}\n'
            stats_text += f'Max: {valid_data.max():.2f}\n'
            stats_text += f'Valid points: {len(valid_data)}/{len(plot_data)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        output_file = os.path.join(output_dir, f'{base_name}_{col}_plot.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {output_file}")
    
    print(f"\nAll plots saved to: {output_dir}")


def aggregate_by_seconds(imputed_df):
    """
    Aggregate imputed data by seconds, averaging values for each second.
    Keeps latitude, longitude, altitude, speed, current_ampere, battery_voltage, and milliamps.
    
    Args:
        imputed_df: DataFrame with imputed values
    
    Returns:
        DataFrame indexed by seconds with averaged values
    """
    # Columns to keep and aggregate
    columns_to_keep = ['latitude', 'longitude', 'altitude', 'speed_kmh', 
                       'current_ampere', 'battery_voltage', 'milliamps']
    
    # Check which columns exist
    available_columns = [col for col in columns_to_keep if col in imputed_df.columns]
    
    if not available_columns:
        print("Warning: No valid columns found for aggregation")
        return pd.DataFrame()
    
    # Create a copy with only the columns we need
    df_to_aggregate = imputed_df[available_columns].copy()
    
    # Convert to numeric, handling any non-numeric values
    for col in available_columns:
        df_to_aggregate[col] = pd.to_numeric(df_to_aggregate[col], errors='coerce')
    
    # Calculate seconds from time column or use index
    if 'time' in imputed_df.columns:
        def time_to_seconds(time_str):
            """Convert time string (MM:SS or HH:MM:SS) to total seconds"""
            if pd.isna(time_str) or time_str is None:
                return None
            try:
                time_str = str(time_str).strip()
                parts = time_str.split(':')
                if len(parts) == 2:
                    # MM:SS format
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    # HH:MM:SS format
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
                else:
                    return None
            except (ValueError, AttributeError):
                return None
        
        seconds = imputed_df['time'].apply(time_to_seconds)
        # If we have valid seconds, use them; otherwise fall back to index
        if seconds.notna().any():
            df_to_aggregate['seconds'] = seconds
        else:
            # Use index as seconds (assuming 1 row per frame, frames at some rate)
            df_to_aggregate['seconds'] = df_to_aggregate.index
    else:
        # No time column, use index as seconds
        df_to_aggregate['seconds'] = df_to_aggregate.index
    
    # Remove rows where seconds is None
    df_to_aggregate = df_to_aggregate[df_to_aggregate['seconds'].notna()].copy()
    
    if len(df_to_aggregate) == 0:
        print("Warning: No valid data for aggregation")
        return pd.DataFrame()
    
    # Group by seconds and average the values
    aggregated = df_to_aggregate.groupby('seconds')[available_columns].mean().reset_index()
    
    # Set seconds as index
    aggregated.set_index('seconds', inplace=True)
    aggregated.index.name = 'seconds'
    
    # Round numeric values appropriately
    if 'latitude' in aggregated.columns:
        aggregated['latitude'] = aggregated['latitude'].round(7)  # 7 decimal places for lat/lon
    if 'longitude' in aggregated.columns:
        aggregated['longitude'] = aggregated['longitude'].round(7)
    if 'altitude' in aggregated.columns:
        aggregated['altitude'] = aggregated['altitude'].round(2)  # 2 decimal places for altitude
    if 'speed_kmh' in aggregated.columns:
        aggregated['speed_kmh'] = aggregated['speed_kmh'].round(2)  # 2 decimal places for speed
    if 'current_ampere' in aggregated.columns:
        aggregated['current_ampere'] = aggregated['current_ampere'].round(2)  # 2 decimal places for current
    if 'battery_voltage' in aggregated.columns:
        aggregated['battery_voltage'] = aggregated['battery_voltage'].round(2)  # 2 decimal places for voltage
    if 'milliamps' in aggregated.columns:
        aggregated['milliamps'] = aggregated['milliamps'].round().astype('Int64')  # Integer for milliamps
    
    return aggregated


def aggregate_by_half_seconds(imputed_df):
    """
    Aggregate imputed data by half seconds, averaging values for each half second.
    For frames with the same timestamp, splits them into two groups:
    - First half: X.0 (e.g., 22.0)
    - Second half: X.5 (e.g., 22.5)
    
    Keeps latitude, longitude, altitude, speed, current_ampere, battery_voltage, and milliamps.
    Also includes start_time and end_time for each aggregation in subtitle format (HH:MM:SS.mmm)
    with sub-second precision based on the actual frame positions.
    
    Args:
        imputed_df: DataFrame with imputed values
    
    Returns:
        DataFrame indexed by half_seconds with averaged values and start_time/end_time columns
        in subtitle format (HH:MM:SS.mmm)
    """
    # Columns to keep and aggregate
    columns_to_keep = ['latitude', 'longitude', 'altitude', 'speed_kmh', 
                       'current_ampere', 'battery_voltage', 'milliamps']
    
    # Check which columns exist
    available_columns = [col for col in columns_to_keep if col in imputed_df.columns]
    
    if not available_columns:
        print("Warning: No valid columns found for aggregation")
        return pd.DataFrame()
    
    # Create a copy with only the columns we need
    df_to_aggregate = imputed_df[available_columns].copy()
    
    # Convert to numeric, handling any non-numeric values
    for col in available_columns:
        df_to_aggregate[col] = pd.to_numeric(df_to_aggregate[col], errors='coerce')
    
    # Calculate seconds from time column or use index
    if 'time' in imputed_df.columns:
        def time_to_seconds(time_str):
            """Convert time string (MM:SS or HH:MM:SS) to total seconds"""
            if pd.isna(time_str) or time_str is None:
                return None
            try:
                time_str = str(time_str).strip()
                parts = time_str.split(':')
                if len(parts) == 2:
                    # MM:SS format
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    # HH:MM:SS format
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
                else:
                    return None
            except (ValueError, AttributeError):
                return None
        
        seconds = imputed_df['time'].apply(time_to_seconds)
        # If we have valid seconds, use them; otherwise fall back to index
        if seconds.notna().any():
            df_to_aggregate['seconds'] = seconds
            df_to_aggregate['time'] = imputed_df['time']  # Keep original time for start_time/end_time
        else:
            # Use index as seconds (assuming 1 row per frame, frames at some rate)
            df_to_aggregate['seconds'] = df_to_aggregate.index
            df_to_aggregate['time'] = None
    else:
        # No time column, use index as seconds
        df_to_aggregate['seconds'] = df_to_aggregate.index
        df_to_aggregate['time'] = None
    
    # Remove rows where seconds is None
    df_to_aggregate = df_to_aggregate[df_to_aggregate['seconds'].notna()].copy()
    
    if len(df_to_aggregate) == 0:
        print("Warning: No valid data for aggregation")
        return pd.DataFrame()
    
    # Add a position within each second group (for splitting into halves)
    df_to_aggregate = df_to_aggregate.sort_values('seconds').reset_index(drop=True)
    df_to_aggregate['position_in_second'] = df_to_aggregate.groupby('seconds').cumcount()
    df_to_aggregate['count_in_second'] = df_to_aggregate.groupby('seconds')['seconds'].transform('count')
    
    # Calculate precise timestamp for each frame with sub-second precision
    # For frames within the same second, distribute them evenly across that second
    def calculate_precise_timestamp(row):
        """Calculate precise timestamp with sub-second precision"""
        base_second = row['seconds']
        position = row['position_in_second']
        count = row['count_in_second']
        
        if count > 1:
            # Distribute frames evenly across the second
            # e.g., if 5 frames in second 22: positions 0,1,2,3,4 -> timestamps 22.0, 22.2, 22.4, 22.6, 22.8
            sub_second = position / count
        else:
            sub_second = 0.0
        
        precise_seconds = base_second + sub_second
        return precise_seconds
    
    df_to_aggregate['precise_seconds'] = df_to_aggregate.apply(calculate_precise_timestamp, axis=1)
    
    # Determine which half each row belongs to
    # First half: position < count_in_second / 2
    # Second half: position >= count_in_second / 2
    df_to_aggregate['half'] = (df_to_aggregate['position_in_second'] >= 
                                df_to_aggregate['count_in_second'] / 2).astype(int)
    
    # Create half_seconds: base_second + (half * 0.5)
    df_to_aggregate['half_seconds'] = df_to_aggregate['seconds'] + (df_to_aggregate['half'] * 0.5)
    
    # Convert precise_seconds to subtitle time format (HH:MM:SS.mmm)
    def seconds_to_subtitle_time(total_seconds):
        """Convert total seconds to subtitle time format HH:MM:SS.mmm"""
        if pd.isna(total_seconds):
            return None
        try:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            secs = total_seconds % 60
            seconds_int = int(secs)
            milliseconds = int((secs - seconds_int) * 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"
        except (ValueError, TypeError):
            return None
    
    df_to_aggregate['precise_timestamp'] = df_to_aggregate['precise_seconds'].apply(seconds_to_subtitle_time)
    
    # Group by half_seconds and aggregate
    grouped = df_to_aggregate.groupby('half_seconds')
    
    # Calculate mean for numeric columns
    aggregated = grouped[available_columns].mean()
    
    # Count frames in each group
    frame_counts = grouped.size()
    frame_counts.name = 'frames_aggregated'
    
    # Calculate start_time and end_time for each group with sub-second precision
    time_ranges = []
    for half_sec, group in grouped:
        # Get precise timestamps for this group
        group_timestamps = group['precise_timestamp'].dropna()
        if len(group_timestamps) > 0:
            # Get first and last timestamp in the group
            start_time = group_timestamps.iloc[0]
            end_time = group_timestamps.iloc[-1]
        else:
            # Fallback: convert half_seconds to subtitle format
            start_time = seconds_to_subtitle_time(half_sec)
            end_time = seconds_to_subtitle_time(half_sec + 0.5)
        time_ranges.append({
            'half_seconds': half_sec,
            'start_time': start_time,
            'end_time': end_time
        })
    
    time_ranges_df = pd.DataFrame(time_ranges).set_index('half_seconds')
    
    # Merge time ranges and frame counts with aggregated data
    aggregated = aggregated.join(time_ranges_df)
    aggregated = aggregated.join(frame_counts)
    
    # Reset index to make half_seconds a column, then set it as index again
    aggregated = aggregated.reset_index()
    aggregated.set_index('half_seconds', inplace=True)
    aggregated.index.name = 'half_seconds'
    
    # Round numeric values appropriately
    if 'latitude' in aggregated.columns:
        aggregated['latitude'] = aggregated['latitude'].round(7)  # 7 decimal places for lat/lon
    if 'longitude' in aggregated.columns:
        aggregated['longitude'] = aggregated['longitude'].round(7)
    if 'altitude' in aggregated.columns:
        aggregated['altitude'] = aggregated['altitude'].round(2)  # 2 decimal places for altitude
    if 'speed_kmh' in aggregated.columns:
        aggregated['speed_kmh'] = aggregated['speed_kmh'].round(2)  # 2 decimal places for speed
    if 'current_ampere' in aggregated.columns:
        aggregated['current_ampere'] = aggregated['current_ampere'].round(2)  # 2 decimal places for current
    if 'battery_voltage' in aggregated.columns:
        aggregated['battery_voltage'] = aggregated['battery_voltage'].round(2)  # 2 decimal places for voltage
    if 'milliamps' in aggregated.columns:
        aggregated['milliamps'] = aggregated['milliamps'].round().astype('Int64')  # Integer for milliamps
    
    return aggregated


def post_process_dvr_output(input_file, output_file=None):
    """
    Post-process DVR OCR output CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_file with '_processed' suffix)
    
    Returns:
        tuple: (processed_df, imputed_df, aggregated_df) - Three dataframes:
            - processed_df: Cleaned data with None values for invalid data
            - imputed_df: Data with linear interpolation applied and imputation flags
            - aggregated_df: Data aggregated by seconds (latitude, longitude, altitude, speed only)
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
    
    # Track None values before validation (from parsing)
    none_from_parsing = {}
    
    # Process latitude and longitude
    lat_lon_col = 'Top right of the frame. What is the latitude and longitude shown?'
    if lat_lon_col in df.columns:
        lat_lon_data = df[lat_lon_col].apply(extract_lat_lon)
        processed_df['latitude'] = [x[0] for x in lat_lon_data]
        processed_df['longitude'] = [x[1] for x in lat_lon_data]
        # Count None values from parsing (no validation for these columns)
        none_from_parsing['latitude'] = sum(1 for x in processed_df['latitude'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        none_from_parsing['longitude'] = sum(1 for x in processed_df['longitude'] if pd.isna(x) or x is None or str(x).strip() == 'None')
    
    # Process time
    time_col = 'Upper right corner of the frame. What is the time of the recording shown next to the red dot?'
    if time_col in df.columns:
        processed_df['time'] = df[time_col].apply(clean_time)
        # Count None values from parsing (no validation for time)
        none_from_parsing['time'] = sum(1 for x in processed_df['time'] if pd.isna(x) or x is None or str(x).strip() == 'None')
    
    # Process altitude
    altitude_col = 'Middle right of the frame. What is the altitude of the drone shown? ALT ... M'
    if altitude_col in df.columns:
        processed_df['altitude'] = df[altitude_col].apply(clean_altitude)
        # Count None values from parsing
        none_from_parsing['altitude'] = sum(1 for x in processed_df['altitude'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        # Validate altitude sequence (change > 50 from previous valid value)
        processed_df['altitude'] = validate_altitude_sequence(processed_df['altitude'])
    
    # Process speed
    speed_col = 'Middle right of the frame. What is the speed of the drone in km/h?'
    if speed_col in df.columns:
        processed_df['speed_kmh'] = df[speed_col].apply(clean_speed)
        # Count None values from parsing
        none_from_parsing['speed_kmh'] = sum(1 for x in processed_df['speed_kmh'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        # Validate speed sequence (speed > 250 km/h, or for speeds < 10 km/h check for >50% change)
        processed_df['speed_kmh'] = validate_speed_sequence(processed_df['speed_kmh'])
    
    # Process milliamps
    milliamps_col = 'Middle left of the frame. What is the milliamps used shown? MAh'
    if milliamps_col in df.columns:
        processed_df['milliamps'] = df[milliamps_col].apply(clean_milliamps)
        # Count None values from parsing
        none_from_parsing['milliamps'] = sum(1 for x in processed_df['milliamps'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        # Validate milliamps sequence (must be increasing, no jumps > 100)
        processed_df['milliamps'] = validate_milliamps_sequence(processed_df['milliamps'])
    
    # Process current ampere
    current_col = 'Middle left of the frame. What is the current ampere usage shown? ..... A'
    if current_col in df.columns:
        processed_df['current_ampere'] = df[current_col].apply(clean_current_ampere)
        # Count None values from parsing
        none_from_parsing['current_ampere'] = sum(1 for x in processed_df['current_ampere'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        # Validate current ampere sequence (value > 240 amps)
        processed_df['current_ampere'] = validate_current_ampere_sequence(processed_df['current_ampere'])
    
    # Process battery voltage
    voltage_col = 'Bottom middle of the frame. What is the upper battery voltage? Has format X.YY V'
    if voltage_col in df.columns:
        processed_df['battery_voltage'] = df[voltage_col].apply(clean_battery_voltage)
        # Count None values from parsing
        none_from_parsing['battery_voltage'] = sum(1 for x in processed_df['battery_voltage'] if pd.isna(x) or x is None or str(x).strip() == 'None')
        # Validate battery voltage sequence (change > 20% from previous valid value)
        processed_df['battery_voltage'] = validate_battery_voltage_sequence(processed_df['battery_voltage'])
    
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
        
        # Get counts from parsing and heuristics
        none_from_parsing_count = none_from_parsing.get(col, 0)
        none_from_heuristics_count = none_count - none_from_parsing_count
        
        print(f"  {col:25s}: {none_count:5d} total None values ({percentage:5.1f}%)")
        print(f"    {'':25s}  - {none_from_parsing_count:5d} from parsing")
        print(f"    {'':25s}  - {none_from_heuristics_count:5d} from heuristics")
    print("="*60)
    
    # Second post-processing step: Linear interpolation
    print("\n" + "="*60)
    print("Performing linear interpolation for missing values...")
    print("="*60)
    imputed_df = impute_with_linear_interpolation(processed_df)
    
    # Save imputed dataframe
    if output_file:
        if '_processed.csv' in output_file:
            imputed_output_file = output_file.replace('_processed.csv', '_imputed.csv')
        else:
            base_name = os.path.splitext(output_file)[0]
            imputed_output_file = f"{base_name}_imputed.csv"
    else:
        base_name = os.path.splitext(input_file)[0]
        imputed_output_file = f"{base_name}_imputed.csv"
    
    print(f"\nSaving imputed data to: {imputed_output_file}")
    imputed_df.to_csv(imputed_output_file, index=False)
    
    # Debug output: Count imputed values
    print("\n" + "="*60)
    print("DEBUG: Imputation statistics:")
    print("="*60)
    numeric_columns = ['latitude', 'longitude', 'altitude', 'speed_kmh', 
                      'milliamps', 'current_ampere', 'battery_voltage']
    for col in numeric_columns:
        if col in imputed_df.columns:
            imputation_flag_col = f"{col}_imputed"
            if imputation_flag_col in imputed_df.columns:
                imputed_count = imputed_df[imputation_flag_col].sum()
                percentage = (imputed_count / total_rows * 100) if total_rows > 0 else 0
                print(f"  {col:25s}: {imputed_count:5d} imputed values ({percentage:5.1f}%)")
    print("="*60)
    
    # Show sample of imputed data (excluding boolean imputation flag columns)
    print("\nSample of imputed data:")
    # Filter out imputation flag columns for display
    value_columns = [col for col in imputed_df.columns if not col.endswith('_imputed')]
    imputed_df_display = imputed_df[value_columns]
    print(imputed_df_display.head(200).to_string())
    print('--------------------------------')
    print(imputed_df_display.tail(200).to_string())
    
    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    # Determine output directory and base name for plots
    if output_file:
        plot_output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        plot_base_name = os.path.splitext(os.path.basename(output_file))[0]
    else:
        plot_output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else '.'
        plot_base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Remove _imputed suffix if present for base name
    if plot_base_name.endswith('_imputed'):
        plot_base_name = plot_base_name[:-8]
    
    create_plots(imputed_df, plot_output_dir, plot_base_name)
    
    # Third post-processing step: Aggregate by seconds
    print("\n" + "="*60)
    print("Aggregating data by seconds...")
    print("="*60)
    aggregated_df = aggregate_by_seconds(imputed_df)
    
    if len(aggregated_df) > 0:
        # Save aggregated dataframe
        if output_file:
            if '_processed.csv' in output_file:
                aggregated_output_file = output_file.replace('_processed.csv', '_aggregated_by_seconds.csv')
            elif '_imputed.csv' in output_file:
                aggregated_output_file = output_file.replace('_imputed.csv', '_aggregated_by_seconds.csv')
            else:
                base_name = os.path.splitext(output_file)[0]
                aggregated_output_file = f"{base_name}_aggregated_by_seconds.csv"
        else:
            base_name = os.path.splitext(input_file)[0]
            aggregated_output_file = f"{base_name}_aggregated_by_seconds.csv"
        
        print(f"\nSaving aggregated data to: {aggregated_output_file}")
        aggregated_df.to_csv(aggregated_output_file, index=True)
        
        print(f"Aggregated {len(aggregated_df)} seconds from {len(imputed_df)} frames")
        print(f"Columns: {list(aggregated_df.columns)}")
        
        # Show sample of aggregated data
        print("\nSample of aggregated data:")
        print(aggregated_df.head(50).to_string())
        print('--------------------------------')
        print(aggregated_df.tail(50).to_string())
    else:
        print("Warning: Could not create aggregated dataframe")
        aggregated_df = pd.DataFrame()
    
    # Fourth post-processing step: Aggregate by half seconds
    print("\n" + "="*60)
    print("Aggregating data by half seconds...")
    print("="*60)
    aggregated_half_df = aggregate_by_half_seconds(imputed_df)
    
    if len(aggregated_half_df) > 0:
        # Save aggregated half-second dataframe
        if output_file:
            if '_processed.csv' in output_file:
                aggregated_half_output_file = output_file.replace('_processed.csv', '_aggregated_by_half_seconds.csv')
            elif '_imputed.csv' in output_file:
                aggregated_half_output_file = output_file.replace('_imputed.csv', '_aggregated_by_half_seconds.csv')
            else:
                base_name = os.path.splitext(output_file)[0]
                aggregated_half_output_file = f"{base_name}_aggregated_by_half_seconds.csv"
        else:
            base_name = os.path.splitext(input_file)[0]
            aggregated_half_output_file = f"{base_name}_aggregated_by_half_seconds.csv"
        
        print(f"\nSaving aggregated half-second data to: {aggregated_half_output_file}")
        aggregated_half_df.to_csv(aggregated_half_output_file, index=True)
        
        print(f"Aggregated {len(aggregated_half_df)} half-seconds from {len(imputed_df)} frames")
        print(f"Columns: {list(aggregated_half_df.columns)}")
        
        # Show sample of aggregated half-second data
        print("\nSample of aggregated half-second data:")
        print(aggregated_half_df.head(50).to_string())
        print('--------------------------------')
        print(aggregated_half_df.tail(50).to_string())
    else:
        print("Warning: Could not create aggregated half-second dataframe")
        aggregated_half_df = pd.DataFrame()
    
    return processed_df, imputed_df, aggregated_df


def test_clean_current_ampere():
    """Test cases for clean_current_ampere function"""
    test_cases = [
        # (input, expected_output, description)
        ("1.52", "1.52", "Single dot, no 'A' unit"),
        ("1.40 A", "1.40", "Single dot with 'A' unit"),
        ("33", None, "No dot, no 'A' unit"),
        ("9\n4. 26\n5. 607\n6. 3", None, "Has '6.' but value after is just '3' (no decimal)"),
        ("3\n4. 27\n5. 35\n6. 15.75", "15.75", "Has '6.' followed by decimal number"),
        ("6. 3.52 A", "3.52", "Has '6.' prefix with decimal and 'A'"),
        ("6. 3.52", "3.52", "Has '6.' prefix with decimal, no 'A'"),
        ("", None, "Empty string"),
        (None, None, "None value"),
        ("LAT 46.97, LON 8.58", None, "Coordinates (should be skipped)"),
    ]
    
    print("Testing clean_current_ampere function:")
    print("=" * 60)
    all_passed = True
    
    for input_val, expected, description in test_cases:
        result = clean_current_ampere(input_val)
        passed = result == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"{status} | {description}")
        print(f"  Input:    {repr(input_val)}")
        print(f"  Expected: {repr(expected)}")
        print(f"  Got:      {repr(result)}")
        print()
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    
    return all_passed


if __name__ == "__main__":
    # Run tests if --test flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_clean_current_ampere()
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print("Usage: python post_process_dvr_output.py <input_csv_file> [output_csv_file]")
        print("       python post_process_dvr_output.py --test  (to run tests)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    post_process_dvr_output(input_file, output_file)

