"""
Script Name:    06_context_feature_engineering.py

Description:    Context data preprocessing
"""

import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from utils.helper_context_data_prep import context_data_preprocessing, train_test_split, test_method_preprocessing

# %%
# Configuration parameters
PATH_TO_LOAD = './data/04_Merged'
combined_context_path = './data/06_context_feat_engg/data_featue_engineering.csv'
augmentation_folder = 'featengg/'
base_path = './datasets/sequential/'
sequence_augmentation = True
whole_session_context = False
feat_engg = True
pad_to_window_size = True
window = 30  # seconds

# Paths for saving processed data
sequence_context_path = f'{base_path}{augmentation_folder}parameters/sequence_context.csv'
parameter_path = f'{base_path}{augmentation_folder}parameters'
train_session_path = f'{base_path}{augmentation_folder}parameters/train_sessions.pkl'
test_session_path = f'{base_path}{augmentation_folder}parameters/test_sessions.pkl'
train_dynamic_context_path = f'{base_path}{augmentation_folder}dynamic_context/train.csv'
test_dynamic_context_path = f'{base_path}{augmentation_folder}dynamic_context/test.csv'
train_static_context_path = f'{base_path}{augmentation_folder}static_context/train.csv'
test_static_context_path = f'{base_path}{augmentation_folder}static_context/test.csv'
train_sequence_path = f'{base_path}{augmentation_folder}seq/train.tsv'
test_sequence_path = f'{base_path}{augmentation_folder}seq/test.tsv'
train_dense_static_context_path = f'{base_path}{augmentation_folder}dense_static_context/train.csv'
test_dense_static_context_path = f'{base_path}{augmentation_folder}dense_static_context/test.csv'

# %%
# Context variables to be used in processing
dynamic_context_var = [
    'avg_irradiation', 'steering_speed', 'temperature_out', 
    'light_sensor_rear', 'light_sensor_front', 
    'temperature_in', 'KBI_speed', 
    'latitude', 'longitude', 'altitude'
]  # TODO: Remove these features in the future

if feat_engg:
    cat_static_context_var = [
        'car_id', 'month', 'weekday', 'hour', 'season', 'seatbelt_codriver', 
        'seatbelt_rear_l', 'seatbelt_rear_r', 'street_category'
    ]
else:
    cat_static_context_var = [
        'car_id', 'month', 'weekday', 'hour', 'seatbelt_codriver', 
        'seatbelt_rear_l', 'seatbelt_rear_r', 'street_category'
    ]

dense_static_context_var = [
    'distance_driven_benchmark', 'soc', 'time_second'
]  # Dense static context

status_static_context_var = [
    'ess_status', 'current_drive_mode', 'current_clima_mode', 
    'current_media_source', 'nav_guidance_status', 'proximity_to_parking_spot', 
    'phone_status', 'bluetooth_connected', 'phone_os', 'new_bluetooth_device_to_pair'
]

# List of vehicle names to process
vehicles = ['SEB880', 'SEB882', 'SEB883', 'SEB885', 'SEB888', 'SEB889']

# %%
# Load the combined context data
context_data = pd.read_csv(combined_context_path, parse_dates=['datetime'], index_col=0, low_memory=False)

# Load train and test session data
with open(train_session_path, 'rb') as pickle_file:
    train_sessions = pickle.load(pickle_file)
with open(test_session_path, 'rb') as pickle_file:
    test_sessions = pickle.load(pickle_file)

# Preprocess context data
dynamic_context, dense_static_context, static_context = context_data_preprocessing(context_data)

# Split data into training and testing sets
train_dynamic_context, test_dynamic_context, train_static_context, test_static_context, train_dense_static_context, test_dense_static_context = train_test_split(dynamic_context, dense_static_context, static_context)

print('Number of sessions:', 
      len(train_dynamic_context.window_id.unique().tolist()), len(test_dynamic_context.window_id.unique().tolist()),
      len(train_static_context.window_id.unique().tolist()), len(test_static_context.window_id.unique().tolist()),
      len(train_dense_static_context.window_id.unique().tolist()), len(test_dense_static_context.window_id.unique().tolist()))

# Save unnormalized context data
train_static_context.to_csv(f'{base_path}{augmentation_folder}static_context/unnormal/train.csv', index=False)
test_static_context.to_csv(f'{base_path}{augmentation_folder}static_context/unnormal/test.csv', index=False)
train_dynamic_context.to_csv(f'{base_path}{augmentation_folder}dynamic_context/unnormal/train.csv', index=False)
test_dynamic_context.to_csv(f'{base_path}{augmentation_folder}dynamic_context/unnormal/test.csv', index=False)
train_dense_static_context.to_csv(f'{base_path}{augmentation_folder}dense_static_context/unnormal/train.csv', index=False)
test_dense_static_context.to_csv(f'{base_path}{augmentation_folder}dense_static_context/unnormal/test.csv', index=False)

# %%
# Normalization
dynamic_context_to_normalize = [col for col in train_dynamic_context.columns if col not in ['window_id', 'wind_id', 'session_ids', 'datetime', 'session_id', 'session']]

# Apply RobustScaler to dynamic context data
scaler_dynamic_context = RobustScaler()
scaler_dynamic_context.fit(train_dynamic_context[dynamic_context_to_normalize])
train_dynamic_context[dynamic_context_to_normalize] = scaler_dynamic_context.transform(train_dynamic_context[dynamic_context_to_normalize])
test_dynamic_context[dynamic_context_to_normalize] = scaler_dynamic_context.transform(test_dynamic_context[dynamic_context_to_normalize])

# Apply RobustScaler to dense static context data
scaler_dense_static_context = RobustScaler()
scaler_dense_static_context.fit(train_dense_static_context[dense_static_context_var])
train_dense_static_context[dense_static_context_var] = scaler_dense_static_context.transform(train_dense_static_context[dense_static_context_var])
test_dense_static_context[dense_static_context_var] = scaler_dense_static_context.transform(test_dense_static_context[dense_static_context_var])

# Save normalized context data
train_dynamic_context.to_csv(train_dynamic_context_path, index=False)
test_dynamic_context.to_csv(test_dynamic_context_path, index=False)
train_static_context.to_csv(train_static_context_path, index=False)
test_static_context.to_csv(test_static_context_path, index=False)
train_dense_static_context.to_csv(train_dense_static_context_path, index=False)
test_dense_static_context.to_csv(test_dense_static_context_path, index=False)

# %%
# Additional preprocessing for testing
test_method_preprocessing(train_dynamic_context, test_dynamic_context, train_static_context, 
                          test_static_context, train_dense_static_context, test_dense_static_context)
