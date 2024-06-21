import pandas as pd
import json

# Load the CSV file
csv_file_path = 'sample.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Convert DataFrame to the specified JSON format
json_dict = {col: df[col].values.tolist() for col in df.columns}

# Save the JSON to a file
json_file_path = './prediction method/sample_prg.json'  # Replace with your desired JSON file path
with open(json_file_path, 'w') as json_file:
    json_file.write(json.dumps(json_dict, indent=4))
