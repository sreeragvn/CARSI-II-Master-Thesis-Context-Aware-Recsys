"""
Script Name:    06_event_windows.py

Description:    Create MF4 context windows around labels.
"""
import os
import sys 
import pandas as pd 
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PATH_TO_LOAD = "./Processed_data_new/04_Merged"
PATH_TO_SAVE = "./Processed_data_new/Datasets"

# Odd window size for centric event
WINDOW_SIZE = 31
window_step = WINDOW_SIZE // 2

vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

def zero_pad_dataframe(df, num_pad_rows):
    # Create a DataFrame with the desired number of rows filled with zeros
    zero_pad_data = {
        col: [float("nan")] * (num_pad_rows) for col in df.columns
    }
    zero_pad_df = pd.DataFrame(zero_pad_data)
    # Concatenate the zero pad DataFrame before and after the original DataFrame
    padded_df = pd.concat([zero_pad_df, zero_pad_df, df], ignore_index=True)
    return padded_df

window_id = 0
window_id_no_nan = 0

for vehicle in vehicle_names:

    # Load data for vehicle 
    df = pd.read_csv(os.path.join(PATH_TO_LOAD, f"{vehicle}_merged.csv"), parse_dates=['datetime'], low_memory=False)

    # drop some cols
    cols_to_drop = ["index", "domain", "BeginTime", "ts_normalized", "ID", "FunctionValue", "driving_program"] 
    df = df.drop(columns=cols_to_drop)

    # New dataframe composed of windows
    df_window = pd.DataFrame()
    df_window_no_nan = pd.DataFrame()

    counter_nan = 0 

    # Event-based Windows - iterate over events (labels)
    for sess in tqdm(df.session.unique()):

        df_sess = df[df.session == sess].copy()

        # Pad the session
        df_padded = zero_pad_dataframe(df_sess, window_step)
        
        # Indices events in session 
        ids_events = df_padded[df_padded.Label.notna()].index.tolist()
        df_padded["index_ref"] = df_padded.index # create reference for filtering

        for idx in ids_events:

            # Filter double datetime (labels can overlap)
            datetime_event = df_padded.loc[idx, "datetime"]
            #test = df_padded[(df_padded["datetime"] != datetime_event) | (df_padded.index == idx)].copy()
            df_padded_filtered = df_padded[(df_padded["datetime"] != datetime_event) | (df_padded.index == idx)].copy()
            #df_padded_filtered.loc[df_padded_filtered['datetime'].notna()] = df_padded_filtered.loc[df_padded_filtered['datetime'].notna()].drop_duplicates(subset=['datetime'])
            df_padded_filtered = df_padded_filtered.reset_index(drop=True)
            idx_ref = df_padded_filtered[df_padded_filtered["index_ref"] == idx].index.values[0]

            # Build the window centered around the event
            # df_event = df_padded_filtered.iloc[idx_ref - window_step: idx_ref + window_step+1, :].copy()
            # df_event["Label"] = df_padded_filtered.loc[idx_ref, "Label"]
            # df_event["window_id"] = window_id
            # window_id += 1

            # df_window = pd.concat([df_window, df_event], ignore_index=True)

            # Build the window to the left of the event
            df_event = df_padded_filtered.iloc[idx_ref - WINDOW_SIZE: idx_ref, :].copy()
            df_event["Label"] = df_padded_filtered.loc[idx_ref, "Label"]
            df_event["window_id"] = window_id
            window_id += 1

            df_window = pd.concat([df_window, df_event], ignore_index=True)


            if df_event.isna().any().any():
                counter_nan += 1
            else:
                df_event_no_nan = df_event.copy() 
                df_event_no_nan["window_id"] = window_id_no_nan
                df_window_no_nan = pd.concat([df_window_no_nan, df_event_no_nan], ignore_index=True)
                window_id_no_nan += 1

    print(f"Windows without nan: {len(df_window_no_nan.window_id.unique())}/{len(df_window.window_id.unique())}")

    df_window = df_window.drop(columns=["index_ref"])
    df_window_no_nan = df_window_no_nan.drop(columns=["index_ref"])
    #df_window.to_csv(os.path.join(PATH_TO_SAVE, "dataset_event_windows.csv"), index=False)
    df_window_no_nan.to_csv(os.path.join(PATH_TO_SAVE, f"dataset_windows_{vehicle}.csv"), index=False)

    # # Normalize dataframe
    # cols_to_normalize = ["hour", "month", "temperature_out", "steering_speed", "time_second", "avg_irradiation",
    #                     "KBI_speed", "ESP_speed", "soc", "latitude", "longitude", "altitude", "rain_sensor", "weekday"]


    # # Convert 'col1' to numeric type
    # #df_window[cols_to_normalize] = pd.to_numeric(df_window[cols_to_normalize], errors='coerce')
    # for col in cols_to_normalize:
    #     df_window[col] = pd.to_numeric(df_window[col], errors='coerce')

    # # Min-Max scaling for selected columns
    # min_val = df_window[cols_to_normalize].min()
    # max_val = df_window[cols_to_normalize].max()
    # df_window[cols_to_normalize] = (df_window[cols_to_normalize] - min_val) / (max_val - min_val)

    # df_window.to_csv(os.path.join(PATH_TO_SAVE, "dataset_event_windows_normalized.csv"), index=False)

