"""
Script Name:    02_mf4_process.py

Description:    Mf4 data imputation with previous signal.
                
Comment:        Local window average to be considered.
                Decrease parsing frequency to obtain more values.         
"""

import pandas as pd
import os

# Paths to load and save the data
PATH_TO_LOAD = "../Processed_data_new/01_Mf4_Extracted"
PATH_TO_SAVE = "../Processed_data_new/02_Mf4_Filled"

# List of vehicle names to process
vehicle_names = ["SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

# List of context variables to fill missing values
context_to_fill = [
    'temperature_out', 
    'temperature_in',
    'steering_speed', 
    'avg_irradiation',
    'light_sensor_rear', 
    'light_sensor_front', 
    'KBI_speed', 
    'ESP_speed', 
    'soc', 
    'latitude', 
    'longitude',
    'street_category',
    'rain_sensor', 
    'altitude',
    'kickdown', 
    'CHA_ESP_drive_mode', 
    'CHA_MO_drive_mode',
    'MO_drive_mode',
    'seatbelt_codriver',
    'seatbelt_rear_l',
    'seatbelt_rear_r',
    'seatbelt_rear_m',
]

def fill_missing_with_previous(column):
    """
    Fill missing values in a column with the previous value.

    Args:
        column (pd.Series): The column to fill missing values in.

    Returns:
        pd.Series: The column with missing values filled.
    """
    previous_value = None
    for i, value in enumerate(column):
        if pd.isna(value):
            if previous_value is not None:
                value = previous_value
            else:
                value = 0.0
        previous_value = value
        column.iloc[i] = value
    return column

# Process each vehicle
for vehicle in vehicle_names:
    print("%" * 40, "\nProcess: ", vehicle)
    
    # Load the extracted data for the vehicle
    df = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_extracted_mf4.csv"), parse_dates=['datetime'])

    # Keep only context variables that are present in the DataFrame
    context_intersection = list(set(context_to_fill) & set(df.keys()))

    # Counters for seatbelt passengers
    count_front_passenger = 0
    count_passenger_rear_r = 0
    count_passenger_rear_l = 0
    count_passenger_rear_m = 0

    # Loop over sessions
    for i, sess in enumerate(df['session'].unique()):
        if i % 100 == 0:
            print(f"session completed: {i}/{len(df['session'].unique())}")

        # Get session indices
        idx_sess = df['session'] == sess

        # Update seatbelt status for passengers
        if (df.loc[idx_sess, "seatbelt_codriver"] == 1).any():
            df.loc[idx_sess, "seatbelt_codriver"] = 1
            count_front_passenger += 1
        if (df.loc[idx_sess, "seatbelt_rear_l"] == 1).any():
            df.loc[idx_sess, "seatbelt_rear_l"] = 1
            count_passenger_rear_l += 1
        if (df.loc[idx_sess, "seatbelt_rear_r"] == 1).any():
            df.loc[idx_sess, "seatbelt_rear_r"] = 1
            count_passenger_rear_r += 1
        if (df.loc[idx_sess, "seatbelt_rear_m"] == 1).any():
            df.loc[idx_sess, "seatbelt_rear_m"] = 1
            count_passenger_rear_m += 1

        # Loop over context variables and fill missing values
        for context in context_intersection:
            df.loc[idx_sess, context] = fill_missing_with_previous(df.loc[idx_sess, context]).ffill()

    print(f"Num passengers (f/r/l/m) / num_sessions: ({count_front_passenger}/{count_passenger_rear_r}/{count_passenger_rear_l}/{count_passenger_rear_m}) / {len(df['session'].unique())}")

    # Save the filled DataFrame
    df.to_csv(os.path.join(PATH_TO_SAVE, vehicle + "_filled_mf4.csv"), index=False)
