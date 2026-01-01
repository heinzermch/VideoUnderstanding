import pydeck as pdk
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

# Convert to a list of coordinates [[lon, lat, alt], ...]
path_data = [
    {
        "path": df[['longitude', 'latitude', 'altitude']].values.tolist(),
        "name": "Short Flight",
        "color": [255, 0, 0] # Red path
    }
]

# Create the layer
layer = pdk.Layer(
    "PathLayer",
    path_data,
    get_path="path",
    get_width=5,
    get_color="color",
    width_min_pixels=2,
)

# Set the view (tilted to see 3D altitude)
view_state = pdk.ViewState(
    latitude=df['latitude'].mean(),
    longitude=df['longitude'].mean(),
    zoom=12,
    pitch=45, # Tilt the camera
    bearing=0
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.to_html("flight_path.html")