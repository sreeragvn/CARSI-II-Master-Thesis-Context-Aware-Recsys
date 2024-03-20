"""
Script Name:    01_eso_parsed_to_csv.py

Description:    For each vehicle, extract parsed EsoTrace data (.pncrec) 
                divided per day to create a tabular format of timestamped logs. 
"""
import json
import os
import pandas as pd
from tqdm import tqdm

class EsoData():
    def __init__(self, 
                 abs_path, 
                 vehicle_name) -> None:
        self.empty_pncrec = 0
        self.pncrec_path = os.path.join(abs_path, vehicle_name, vehicle_name + "_pncrec")

    def load_pncrec_to_dataframe_dic(self, path):
        """ return a dataframe form json"""
        with open(path, 'r') as file:
            contents = file.read()
            return pd.DataFrame(json.loads(contents))
    
    def extract_ItemList(self, row):
        _type = row['Type']
        _time = row['Time']
        _topic = row['Topic']
        payload_value = json.loads(row['Payload']['value'])
        _id = payload_value['id']
        _domain = _id.split("/")[1] if len(_id.split("/")) > 1 else 'None'
        _functionValue = payload_value['functionValue'] if 'functionValue' in payload_value.keys() and payload_value['functionValue']  else "non"
        _timestamp = payload_value['timestamp']
        _sourceDisplay = payload_value['sourceDisplay']
        _sender = row['Payload']['sender']
        return _type, _time, _topic, _id, _functionValue, _timestamp, _sourceDisplay, _sender, _domain
    
    def filter_timestamp_to_beginTime(self, df):
        """ Filter data that have timestamp corrupted (different from BeginTime) """
        try:
            begin_time = pd.to_datetime(df['BeginTime'], errors='coerce').dt.date
        except AttributeError:
            begin_time = pd.to_datetime(df['BeginTime'].str[:10])
        date_time = df.datetime.dt.date
        df = df[begin_time == date_time].reset_index(drop=True)
        return df
    
    def to_datetime(self, ts):
        """ Apply a time delta to correct the time according to mf4 """
        # TODO: Investigate origin of time mismatch, is the same for all recordings? 
        return pd.to_datetime(ts, unit='ms') + pd.Timedelta(minutes=120)

    def get_correct_datetime(self, df):
        df['datetime'] = df.Timestamp.apply(self.to_datetime)
        return df

    def sort_df_temporal(self, df):
        return df.sort_values(by=['datetime'])
    
    def is_pncrec(self, file_name):
        if file_name[-6:] == 'pncrec':
            return True
        else:
            return False
        
    def is_empty(self, df):
        if len(df.keys()) == 0: 
                self.empty_pncrec += 1
                return True
    
    def generate_unique_pncrec_df(self):
        """ Core function:  """
        pncrec_path = self.pncrec_path
        file_names = os.listdir(pncrec_path)
        
        self.num_of_files = len(file_names)

        df_pncrec_all = pd.DataFrame()

        for file_name in tqdm(file_names):
            
            file_path = os.path.join(pncrec_path, file_name)
            if not self.is_pncrec(file_name):
                os.remove(file_path)
                continue
            df_pncrec_dic = self.load_pncrec_to_dataframe_dic(file_path)

            if self.is_empty(df_pncrec_dic): 
                continue

            # Extract relevant content
            content_list = df_pncrec_dic["ItemList"].apply(self.extract_ItemList).tolist()
            df_pncrec = pd.DataFrame(content_list, columns=['Type', 'Time', 'Topic', 'ID', 'FunctionValue', 'Timestamp', 'SourceDisplay', 'Sender', 'domain'])
            df_pncrec['BeginTime'] = df_pncrec_dic.BeginTime

            # Concat dayly pncrec to the final dataframe
            df_pncrec_all = pd.concat([df_pncrec_all, df_pncrec])
        
        df_pncrec_all = self.get_correct_datetime(df_pncrec_all)
        df_pncrec_all = self.filter_timestamp_to_beginTime(df_pncrec_all)
        df_pncrec_all = self.sort_df_temporal(df_pncrec_all)
    
        return df_pncrec_all

if __name__ == "__main__":

    vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]
    ABS_PATH = "../Parsed_data_new"
    SAVE_PATH = "../Processed_data_new/01_Eso_Extracted"

    for vehicle_name in vehicle_names:
        print("%"*40, "\nProcess: ", vehicle_name)

        eso = EsoData(ABS_PATH, vehicle_name)
        df = eso.generate_unique_pncrec_df()
        df.to_csv(os.path.join(SAVE_PATH, vehicle_name + "_extracted_eso.csv"), index=False)    
        print(f"Num of empty pncrec: {eso.empty_pncrec}/{eso.num_of_files}")    
        print(f"vehicle {vehicle_name} saved to {SAVE_PATH}")