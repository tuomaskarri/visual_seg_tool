"""
ROS2 Bag → CSV Conversion Script

OVERVIEW
This script automatically:
1. Finds all ROS2 bag folders in a directory
2. Reads messages from each bag
3. Converts them into a structured pandas DataFrame
4. Interpolates data to a fixed frequency (default: 50 Hz)
5. Renames columns to a standardized format
6. Exports each bag as a CSV file

 INPUT
- ROS2 bag folders (must contain .db3 files)
- Located in the same directory (or base_dir)

 OUTPUT
- One CSV per rosbag:
    <bag_folder_name>.csv

 REQUIREMENTS
- ROS2 (rosbag2_py)
- rclpy
- pandas
- numpy

 RUN
    python3 convert_all_rosbags_to_csv.py
"""
import os
import rosbag2_py
import pandas as pd
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict

# HELPER: FLATTEN NESTED DICTIONARIES
def flatten_dict(d, parent_key=""):
    """
    Recursively flatten a nested dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items

# MAIN: CONVERT ROSBAG → DATAFRAME
def bag_to_dataframe(bag_path):
    """
    Read a ROS2 bag and convert it into a pandas DataFrame.
    """

    # Initialize ROS2 bag reader
    reader = rosbag2_py.SequentialReader()

    # Configure storage (SQLite3 is default ROS2 format)
    storage_options = rosbag2_py.StorageOptions(uri=bag_path,
                                                storage_id="sqlite3")
    # Converter options (no format conversion)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    # Get topic types (needed for deserialization)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    timestamps = []         # Store timestamps (ms)
    rows = []               # Store row data
    current_values = {}     # Holds latest values from all topics
    start_time = None       # Reference time (first message)

    # Read all messages
    while reader.has_next():
        topic, data, t = reader.read_next()

        # Initialize start time
        if start_time is None:
            start_time = t

        # Convert timestamp to milliseconds (relative)
        timestamp = int((t - start_time) / 1e6)

        # Get message type and deserialize
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        # Convert ROS message → dictionary
        msg_dict = message_to_ordereddict(msg)

        # Flatten nested dictionary
        flat_msg = flatten_dict(msg_dict)

        # Remove any internal timestamp field (avoid duplication)
        flat_msg.pop("timestamp", None)

        # Update current values with latest message fields
        current_values.update(flat_msg)

        # Store snapshot of all known values at this time
        timestamps.append(timestamp)
        rows.append(current_values.copy())

    # Convert collected data into DataFrame
    df = pd.DataFrame(rows)

    # Insert timestamp as first column
    df.insert(0, "timestamp", timestamps)

    return df

# INTERPOLATION (UNIFORM SAMPLING)
def interpolate_dataframe(df, frequency=50):
    """
    Convert irregular timestamps into uniform sampling.

    Default:
        50 Hz → 20 ms intervals
    """

    # Determine time range
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    # Compute timestep in ms
    dt = int(1000 / frequency)

    # Generate uniform timestamps
    t_uniform = np.arange(t_min, t_max + dt, dt)

    # Create new DataFrame with uniform timestamps
    df_uniform = pd.DataFrame({"timestamp": t_uniform})

    # Interpolate each column
    for col in df.columns:
        if col == "timestamp":
            continue

        # Linear interpolation
        if col in df:
            df_uniform[col] = np.interp(t_uniform, df["timestamp"], df[col])
        else:
            df_uniform[col] = 0  # fill missing column with 0

    # Round floats to 3 decimals
    for col in df_uniform.columns:
        if df_uniform[col].dtype.kind in "fc":
            df_uniform[col] = df_uniform[col].round(3)

    return df_uniform

# COLUMN RENAMING
def rename_columns(df):
    """
    Rename columns to match expected format used in segmentation tool.
    """
    mapping = {
        # Pose
        "x_pos": "X_pos",
        "y_pos": "Y_pos",
        "z_pos": "Z_pos",
        "x_rot": "X_rot",
        "y_rot": "Y_rot",
        "z_rot": "Z_rot",
        "w_rot": "W_rot",
        # Thumb
        "thumb_cmc_spread": "thumb_cmc_spread",
        "thumb_cmc_stretch": "thumb_cmc_stretch",
        "thumb_mcp_stretch": "thumb_mcp_stretch",
        "thumb_ip_stretch": "thumb_ip_stretch",
        "thumb_force": "thumb_force",
        # Index
        "index_mcp_spread": "index_mcp_spread",
        "index_mcp_stretch": "index_mcp_stretch",
        "index_pip_stretch": "index_pip_stretch",
        "index_dip_stretch": "index_dip_stretch",
        "index_force": "index_force",
        # Middle
        "middle_mcp_spread": "middle_mcp_spread",
        "middle_mcp_stretch": "middle_mcp_stretch",
        "middle_pip_stretch": "middle_pip_stretch",
        "middle_dip_stretch": "middle_dip_stretch",
        "middle_force": "middle_force",
        # Ring
        "ring_mcp_spread": "ring_mcp_spread",
        "ring_mcp_stretch": "ring_mcp_stretch",
        "ring_pip_stretch": "ring_pip_stretch",
        "ring_dip_stretch": "ring_dip_stretch",
        "ring_force": "ring_force",
        # Pinky
        "pinky_mcp_spread": "pinky_mcp_spread",
        "pinky_mcp_stretch": "pinky_mcp_stretch",
        "pinky_pip_stretch": "pinky_pip_stretch",
        "pinky_dip_stretch": "pinky_dip_stretch",
        "pinky_force": "pinky_force",
        # Palm
        "palm_force": "palm_force",
    }

    # Keep any unknown columns unchanged
    for col in df.columns:
        if col not in mapping:
            mapping[col] = col  # keep as-is
    df = df.rename(columns=mapping)

    return df

# FIND ALL ROSBAG FOLDERS
def find_rosbags(directory):
    """
    Search directory for ROS2 bag folders.
    """
    bags = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path) and any(f.endswith(".db3") for f in
                                       os.listdir(path)):
            bags.append(path)
    return bags

# MAIN EXECUTION
if __name__ == "__main__":
    base_dir = "." # Current directory

    # Find all bag folders
    bag_folders = find_rosbags(base_dir)
    print(f"Found {len(bag_folders)} bag(s)")

    # Process each bag
    for bag in bag_folders:
        try:
            print(f"Converting {bag}")

            # Convert bag → DataFrame
            df = bag_to_dataframe(bag)

            # Interpolate to fixed frequency
            df = interpolate_dataframe(df, frequency=50)

            # Rename columns for consistency
            df = rename_columns(df)

            # Fill any remaining NaN with 0
            df = df.fillna(0)

            # Output file name
            output_name = os.path.basename(bag) + ".csv"

            # Save CSV
            df.to_csv(output_name, index=False)

            print(f"Saved {output_name}")

        except Exception as e:
            print(f"Failed to convert {bag}")
            print(e)

    print("All conversions finished.")