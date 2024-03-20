import sys
import pandas as pd
import requests
import json
import numpy as np
import os 
from tqdm import tqdm


weather_feat = [
  "precipitation", # mm
  "wind_speed", # km/h
  "condition", # dry fog rain sleet snow hail thunderstorm null
  "cloud_cover", # %
]

# Load data window
PATH_TO_LOAD = "../Processed_data_new/Datasets/"
PATH_TO_SAVE = "../Processed_data_new/Datasets/"

vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

for vehicle in vehicle_names:
  
  df = pd.read_csv(os.path.join(PATH_TO_LOAD, f"dataset_windows_{vehicle}.csv"))

  # Extract date unique string 
  date_unique = df.datetime.unique()
  date_unique = [datetime.split()[0] for datetime in date_unique]
  date_unique = set(date_unique)

  # Get index in the middle of the window
  middle_index = len(df) // len(df.window_id.unique()) // 2

  weather_list = []

  for id in tqdm(df.window_id.unique()):

    df_curr = df[df.window_id == id]
  
    # Keep values from the middle of the window
    date = df_curr.datetime.iloc[middle_index].split()[0]
    hour = df_curr.datetime.iloc[middle_index].split()[1][:2]
    lon = df_curr.longitude.iloc[middle_index].round(3)
    lat = df_curr.latitude.iloc[middle_index].round(3)

    # API call 
    BASE_URL = f"https://api.brightsky.dev/weather?lat={lat}&lon={lon}&date={date}"
    weather_response = requests.get(BASE_URL).json()

    if "weather" in weather_response:
      weather_response = requests.get(BASE_URL).json()["weather"]

      # from 24 hours select one
      weather_response = weather_response[int(hour)]

      # fill new dataframe
      window_list = []
      for feature in weather_feat:
        window_list.append(weather_response[feature] )
      window_list.append(id)
      weather_list.append(window_list)
    else:
        weather_list.append([np.nan]*len(weather_feat) + id)

  sys.exit()
  weather_df = pd.DataFrame(weather_list, columns=weather_feat + ["window_id"])
  weather_df.to_csv(os.path.join(PATH_TO_SAVE, f"weather_{vehicle}.csv"), index=False)
