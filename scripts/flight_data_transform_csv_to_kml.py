import simplekml
import pandas as pd
import os

# Read the CSV file - construct path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(
    project_root,
    "output_files",
    "frame_level",
    "dvr",
    "dvr_ocr_results_20260101_144013_rows_4077_aggregated_by_seconds_absolute_altitude.csv"
)
df = pd.read_csv(csv_path)

# Create output directory if it doesn't exist
kml_dir = os.path.join(
    project_root,
    "output_files",
    "frame_level",
    "dvr",
    "kml"
)
os.makedirs(kml_dir, exist_ok=True)

kml = simplekml.Kml()

# Create a 'LineString' (the path of the flight)
# Note: coordinates must be (longitude, latitude, altitude)
path = kml.newlinestring(name="Flight Path")
path.coords = list(zip(df['longitude'], df['latitude'], df['altitude']))

# Styling for 3D
path.altitudemode = simplekml.AltitudeMode.absolute # Flight altitude above sea level
path.extrude = 1  # This draws a "wall" from the path down to the ground
path.style.linestyle.color = simplekml.Color.red # Red line
path.style.linestyle.width = 3

# Save to kml subfolder
output_path = os.path.join(kml_dir, "flight_visualization.kml")
kml.save(output_path)
print(f"KML file saved to: {output_path}")