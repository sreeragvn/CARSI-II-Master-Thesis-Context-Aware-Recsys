"""
Script Name:    03_labeling.py

Description:    Create UI interaction labels from EsoTrace and MF4 data.
"""

import pandas as pd 
import os
import yaml
import ast

import utils.helper_mf4 as helper_mf4
import utils.helper_eso as helper_eso

# Paths to load and save the data
PATH_TO_LOAD_ESO = "./data/01_Eso_Extracted"
PATH_TO_LOAD_MF4 = "./data/02_Mf4_Filled"
PATH_TO_SAVE = "./data/03_Labeled"

# List of vehicle names to process
vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

# Load labeling configuration
with open('./cl_preprocessing/utils/Labels.yaml') as config_file:
    labels = yaml.safe_load(config_file)

# Convert labels from string representations to dictionaries
labels = {key: ast.literal_eval(value) for key, value in labels.items()}

# Process each vehicle
for vehicle in vehicle_names:
    print("%" * 40)
    print(f"[Processing vehicle {vehicle}]\n")

    # Load data for vehicle 
    dfEso = pd.read_csv(os.path.join(PATH_TO_LOAD_ESO, vehicle + "_extracted_eso.csv"), parse_dates=['datetime'])
    dfMf4 = pd.read_csv(os.path.join(PATH_TO_LOAD_MF4, vehicle + "_filled_mf4.csv"), parse_dates=['datetime'])
    
    # Initialize the Label column in the MF4 data
    dfMf4["Label"] = float("nan")
    
    # Apply MF4 labeling using helper function
    dfMf4 = helper_mf4.label_drive_mode(dfMf4)
    num_mf4_label = dfMf4.Label.notna().sum()

    # Apply EsoTrace labeling using helper function
    dfLabeled = helper_eso.label_dataframe(dfEso, labels)
    print(f"Num of interactions (Eso - Mf4): {dfLabeled.shape[0]} - {num_mf4_label} labeled interactions.")
    
    print(f"Labeled interactions Eso:  {dfLabeled.shape[0]}/{dfEso.shape[0]}")
    
    # Ensure the save directory exists
    if not os.path.exists(os.path.join(PATH_TO_SAVE)):
        os.makedirs(os.path.join(PATH_TO_SAVE))

    # Save the labeled data
    dfLabeled.to_csv(os.path.join(PATH_TO_SAVE, vehicle + "_labeled_eso.csv"), index=False)
    dfMf4.to_csv(os.path.join(PATH_TO_SAVE, vehicle + "_labeled_mf4.csv"), index=False)
