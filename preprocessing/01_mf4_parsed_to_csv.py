"""
Script Name:    01_mf4_parsed_to_csv.py

Description:    Parsed Mf4 data extraction.
                
Features:       Bits conversion, signal mapping, and driving session segmentation.
"""
import os
import sys
import pandas as pd
from datetime import datetime

class Mf4Data():
    def __init__(self, abs_path, vheicle_name) -> None:
        self.mf4 = None
        self.mf4_path = os.path.join(abs_path, vheicle_name) 
        self.vehicle_name = vehicle_name

        self.cols_to_remove = [
            'timestamps', 
            'second', 
            'minute', 
            'year', 
            'day', 
            'MO_drive_mode',
            'seatbelt_driver',
            'steering_angle', 
            'latitude_direction', 
            'longitude_direction', 
            'button_event', 
            'gps_timestamp', 
            'bumpy_road', 
            'Drowsiness_driver', 
            'acc_y',
            'acc_x', 
            'time_gap_front_vehicle', 
            'next_route_event',
            'rain_detected',
            'fog_detected',
            'tempolimit',
            'driving_programm'
        ] 

        self.cols_with_byte = [
            'temperature', 
            'temperature_inside',
            "temperature_in",
            "temperature_out",
            'steering_speed', 
            'avg_irradiation',
            'light_sensor_rear', 'light_sensor_front', 
            'KBI_speed', 'ESP_speed', 
            'soc', 
            'odometer', 
            'latitude', 'longitude', 'altitude',
            'rain_sensor',
        ]

        """
        Drive mode mapping:
        0: Normal
        1: Sport
        2: Super Sport
        3: Range
        4: Gravel / Offroad
        """

        self.CHA_MO_drive_mode_mapping = {
            "keine_Funktion": 0,
            "Programm_2": 0,
            "Programm_3": 1,
            "Programm_4": 4, # not in the data 
            "Programm_5": 3,
            "Programm_6": 2
        }
        
        self.CHA_ESP_drive_mode_mapping = {
            "keine_Funktion": 0,
            "Programm_1": 0,
            "Programm_2": 1,
            "Programm_3": 2,
            "Programm_4": 4, # not in the data
            "Programm_10": 3,
        }

        self.seatbelt_passenger_mapping = {
            "nicht_gesteckt": 0,                        
            "gesteckt": 1,
            "nicht_verfügbar (Fehler oder Init)": 0
        }

        self.kickdow_mapping = {
            "Kickdown": 1,
            "kein_Kickdown": 0
        }

        # done in the parser (for reference)
        self.street_category_mapping = {
            b"Rest_Feldweg_Schotterweg_Privatweg": 0,
            b"Ortsstrae": 1,
            b"Ortsstra\xc3\x9fe": 1,
            b"Kreisstrae": 2,
            b"Kreisstra\xc3\x9fe": 2,
            b"Landstrae": 3,
            b"Landstra\xc3\x9fe": 3,
            b"Bundesstrae": 4,
            b"Bundesstra\xc3\x9fe": 4,
            b"Autobahn": 5,
            b"Init": 7,
        }

    def load_mf4_data(self):
        """ Load dataframe mf4 data from parquet file """
        mf4_path = os.path.join(self.mf4_path, self.vehicle_name + "_mf4.parquet")
        self.mf4 = pd.read_parquet(mf4_path)
    
    def remove_empty_datetime(self):
        """ Remove signals with missing timestamp """
        non_nan = self.mf4.year != 'nan'
        self.mf4 = self.mf4[non_nan]
        
    def datetime_to_datetime(self):
        """ Add datetime (y-m-d h:min:sec) from date and time """
        if not 'datetime' in self.mf4.columns:
            columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
            self.mf4['year'] = "20" + self.mf4['year']
            for column in columns:
                self.mf4[column] = self.mf4[column].astype(int)
            self.mf4['datetime'] = self.mf4.apply(lambda row: datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'], row['second'], ), axis=1)
        else:
            self.mf4['datetime'] = pd.to_datetime(self.mf4['datetime'])
    
    def sort_df_temporal(self):
        self.mf4 = self.mf4.sort_values(by="datetime")

    def convert_byte_to_float(self, value):
        if value in ['nan', "b'nan'", "b'Fehler'", "b'Init'", "Init", "Fehler", "nicht_verfuegbar"] or pd.isna(value):
            return float('nan') 
        if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
            numeric_part = value[2:-1]
            return float(numeric_part)       
        else:
            return float(value)

    def convert_bytes(self):
        for var in self.cols_with_byte:
            if var in self.mf4.columns:
                self.mf4[var] = self.mf4[var].apply(self.convert_byte_to_float)
    
    def init_session_count(self, session_count):
        self.session_count = session_count

    def get_session_labels(self):
        """ Generate sessioins based on time-intervals and pause time"""
        date_df = self.mf4.datetime.dt.date
        date_unique = date_df.unique()
        label_add = self.session_count
        for i, day in enumerate(date_unique):
            day_df = self.mf4[date_df == day]
            time_diff = day_df['datetime'].diff()
            groups = (time_diff > pd.Timedelta(minutes=20)).cumsum()
            groups += (label_add + 1)
            label_add = max(groups.unique())
            self.mf4.loc[date_df == day, 'session'] = groups
        self.session_count = label_add

    def last_session_count(self):
        return self.session_count

    def delete_cols(self):
        for column in self.cols_to_remove:
            if column in self.mf4.columns:
                del self.mf4[column]

    def signal_mapping(self):
        signal_dict = {
            "CHA_ESP_drive_mode": self.CHA_ESP_drive_mode_mapping,
            "CHA_MO_drive_mode": self.CHA_MO_drive_mode_mapping,
            "seatbelt_codriver": self.seatbelt_passenger_mapping,
            "seatbelt_rear_l": self.seatbelt_passenger_mapping,
            "seatbelt_rear_m": self.seatbelt_passenger_mapping,
            "seatbelt_rear_r": self.seatbelt_passenger_mapping,
            "kickdown": self.kickdow_mapping
        }
        self.mf4 = self.mf4.replace(signal_dict)
    
    def correct_col_names(self):
        if 'temperature' in self.mf4.columns:
            self.mf4.rename(columns={'temperature': 'temperature_out'}, inplace=True)
        if 'temperature_inside' in self.mf4.columns:
            self.mf4.rename(columns={'temperature_inside': 'temperature_in'}, inplace=True)
        if 'street_class' in self.mf4.columns:
            self.mf4.rename(columns={'street_class': 'street_category'}, inplace=True)

    def get_mf4_data(self):
        self.load_mf4_data()
        self.remove_empty_datetime()
        self.datetime_to_datetime()
        self.correct_col_names()
        self.delete_cols()
        self.signal_mapping()
        self.convert_bytes()
        self.sort_df_temporal()
        self.get_session_labels()
        return self.mf4
        

if __name__ == "__main__":
    
    vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]
    
    ABS_PATH = "../Parsed_data_new"
    SAVE_PATH = "../Processed_data_new/01_MF4_Extracted"

    session_count = 0

    for vehicle_name in vehicle_names:
    
        mf4 = Mf4Data(ABS_PATH, vehicle_name)
        mf4.init_session_count(session_count) 
        df = mf4.get_mf4_data()
        session_count = mf4.last_session_count()

        df.to_csv(os.path.join(SAVE_PATH, vehicle_name + "_extracted_mf4.csv"), index=False)

        print(f"vehicle {vehicle_name} saved to {SAVE_PATH}")

    

    
