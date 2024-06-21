"""
Script Name:    05_sequencing.py

Description:    Creation of interaction history input data.
"""

# %%
import pandas as pd
import os
from tqdm import tqdm
from utils.helper_sequence import process_vehicle, data_label_preprocessing, sequence_generation, train_test_split_df

# %%
# Configuration parameters
text_num_mapping_start = 1  # Starting number for text-to-number mapping
sequence_augmentation = True  # Enable sequence augmentation
carsi_labels_only = True  # Use only CARSI labels
remove_consecutive_duplicates_clicks = True  # Remove consecutive duplicate clicks

# Paths to load and save the data
PATH_TO_LOAD = './data/04_Merged'
base_path = './datasets/sequential/'
augmentation_folder = 'fix/'
parameter_path = f'{base_path}{augmentation_folder}parameters'
sequence_path = f'{base_path}{augmentation_folder}seq'
vehicles = ['SEB880', 'SEB882', 'SEB883', 'SEB885', 'SEB888', 'SEB889']

# %%
print("LOADING DATA IN PROGRESS...")

result = []
for vehicle in tqdm(vehicles):
    # Process each vehicle's data
    result.append(process_vehicle(vehicle, carsi_labels_only, PATH_TO_LOAD))

# Concatenate all vehicle data into a single DataFrame
full_df = pd.concat(result, axis=0, ignore_index=True)
print("DATA LOADING COMPLETED. PREPROCESSING BEGINS...")

# Preprocess the loaded data
filt_df_new = data_label_preprocessing(full_df, text_num_mapping_start)
print("Total unique labels:", len(full_df.full_label.unique().tolist()))

print("SEQUENCE GENERATION IN PROGRESS...")

# Generate sequences from the preprocessed data
df_exploded = sequence_generation(filt_df_new, sequence_augmentation)

# Split the data into training and testing sets
train_df, test_df, class_weights = train_test_split_df(df_exploded, sequence_augmentation, remove_consecutive_duplicates_clicks)

# Save the training and testing sets to TSV files
test_df.to_csv(os.path.join(sequence_path, 'test.tsv'), sep='\t', index=False)
train_df.to_csv(os.path.join(sequence_path, 'train.tsv'), sep='\t', index=False)

print("SEQUENCE GENERATION COMPLETED.")
