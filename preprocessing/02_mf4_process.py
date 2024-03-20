"""
Script Name:    02_mf4_process.py

Description:    Mf4 data imputation with previous signal.
                
Comment:        Local window average to be considered.
                Decrease parsing frequency to obtain more values.         
"""

import pandas as pd
import os

PATH_TO_LOAD = "../Processed_data_new/01_Mf4_Extracted"
PATH_TO_SAVE = "../Processed_data_new/02_Mf4_Filled"

vehicle_names = ["SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

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
    previous_value = None
    #column = column.copy() 
    for i, value in enumerate(column):
        if pd.isna(value):
            if previous_value != None:
                value = previous_value
            else: 
                value = 0.0
        previous_value = value
        column.iloc[i] = value
    return column

for vehicle in vehicle_names:

    print("%"*40, "\nProcess: ", vehicle)
    
    df = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_extracted_mf4.csv"), parse_dates=['datetime'])

    # Keep only context to process that are in the datarame
    context_intersection = list(set(context_to_fill) & set(df.keys()))

    count_front_passenger = 0
    count_passenger_rear_r = 0
    count_passenger_rear_l = 0
    count_passenger_rear_m = 0

    # Loop over sessions 
    for i, sess in enumerate(df['session'].unique()):
        if i%100 == 0:
            print(f"session completed: {i}/{len(df['session'].unique())}")

        # Get session indeces 
        idx_sess = df['session'] == sess

        # Sealtbelt passengers
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

        # Loop over context variables
        for context in context_intersection:

            # Fill Nan values 
            df.loc[idx_sess, context] = fill_missing_with_previous(df.loc[idx_sess, context]).ffill()

    print(f"Num passegers (f/r/l/m) / num_sessios: ({count_front_passenger}/{count_passenger_rear_r}/{count_passenger_rear_l}/{count_passenger_rear_m}) / {len(df['session'].unique())}")
      
    df.to_csv(os.path.join(PATH_TO_SAVE, vehicle + "_filled_mf4.csv"), index=False)   