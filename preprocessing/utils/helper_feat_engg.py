import pandas as pd
import os
from tqdm import tqdm

regenerate_context_data = True
sequence_augmentation = True
whole_session_context = False
model_test_run = False
data_autoencoder = False
pad_to_window_size = True
feat_engg = True

PATH_TO_LOAD = './data/04_Merged'
if feat_engg:
    combined_context_path = './data/06_context_feat_engg/data_featue_engineering.csv'
    augmentation_folder = 'featengg/' if sequence_augmentation else 'non_aug/'
else:
    combined_context_path = './data/05_Interaction_Sequences/context.csv'
    augmentation_folder = 'fix/' if sequence_augmentation else 'non_aug/'

window = 100 #seconds

base_path = './datasets/sequential/'
augmentation_folder = 'aug/' if sequence_augmentation else 'non_aug/'
if model_test_run:
    augmentation_folder = 'test/aug/' if sequence_augmentation else 'test/non_aug/'

sequence_context_path = f'{base_path}{augmentation_folder}parameters/sequence_context.csv'
parameter_path = f'{base_path}{augmentation_folder}parameters'
train_session_path = f'{base_path}{augmentation_folder}parameters/train_sessions.pkl'
test_session_path = f'{base_path}{augmentation_folder}parameters/test_sessions.pkl'
train_dynamic_context_path = f'{base_path}{augmentation_folder}dynamic_context/train.csv'
test_dynamic_context_path = f'{base_path}{augmentation_folder}dynamic_context/test.csv'
train_static_context_path = f'{base_path}{augmentation_folder}static_context/train.csv'
test_static_context_path = f'{base_path}{augmentation_folder}static_context/test.csv'

all_columns = ['index', 'avg_irradiation', 'steering_speed', 'temperature_out', 'hour',
       'month', 'odometer', 'light_sensor_rear', 'light_sensor_front',
       'temperature_in', 'KBI_speed', 'soc', 'ESP_speed', 'latitude',
       'longitude', 'seatbelt_codriver', 'seatbelt_rear_l', 'seatbelt_rear_m',
       'seatbelt_rear_r', 'CHA_ESP_drive_mode', 'CHA_MO_drive_mode',
       'rain_sensor', 'street_category', 'kickdown', 'altitude',
       'driving_program', 'datetime', 'session', 'Label', 'ID',
       'FunctionValue', 'domain', 'BeginTime', 'time_second',
       'distance_driven', 'ts_normalized', 'weekday']

selected = [ 'avg_irradiation', 'steering_speed', 'temperature_out', 'hour',
       'month', 'light_sensor_rear', 'light_sensor_front',
       'temperature_in', 'KBI_speed', 'soc', 'latitude',
       'longitude', 'seatbelt_codriver', 'seatbelt_rear_l',
       'seatbelt_rear_r', 'street_category', 'altitude',
       'datetime', 'session', 'time_second',
       'distance_driven', 'weekday'
]

bad_quality = ['CHA_ESP_drive_mode', 
             'CHA_MO_drive_mode',
             'rain_sensor',
             'kickdown',
             'ESP_speed',
             'seatbelt_rear_m',
            'driving_program',
            'ts_normalized'
             ]

dynamic_context_var = ['avg_irradiation', 'steering_speed', 'temperature_out', 
                       'light_sensor_rear', 'light_sensor_front', 
                       'temperature_in', 'KBI_speed', 'soc', 'latitude',
                       'longitude',  'street_category', 'altitude','time_second',
                       'distance_driven']
static_context_var = ['car_id', 'month', 'weekday', 'hour', 'seatbelt_codriver', 'seatbelt_rear_l',
                       'seatbelt_rear_r',]

status_static_context_var = ['ess_status', 'current_drive_mode', 'current_clima_mode', 'current_media_source', # status static context
                     'nav_guidance_status', 'proximity_to_parking_spot', 'phone_status',
                     'bluetooth_connected', 'phone_os',
                     'new_bluetooth_device_to_pair']
#Todo take average of these value over a window
constant_context_var = ['avg_irradiation','temperature_out'] #to be filled

south_germany_season_mapping = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Autumn',
    10: 'Autumn',
    11: 'Autumn',
    12: 'Winter'
}

def load_context(vehicle):
    context_lists = dynamic_context_var + static_context_var + ['session', 'datetime', 'Label']
    context_lists.remove('car_id')
    df = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_merged.csv"), parse_dates=['datetime'], usecols=context_lists, low_memory=False)
    df = df.sort_values(by=['session','datetime'])
    df['car_id'] = vehicle
    return df

def data_label_preprocessing(car_df):
    print("PREPROCESSING BEGAN")
    car_df = drive_mode_label_fix(car_df)
    print("DRIVE MODE FIX COMPLETED")
    car_df = encode_feature(car_df)
    car_df = ess_status_encode(car_df)
    print("ESS STATUS FEATURE ADDED")
    car_df = drive_mode_status(car_df)
    print("DRIVE MODE STATUS FEATURE ADDED")
    car_df = clima_status(car_df)
    print("CLIMA MODE STATUS FEATURE ADDED")
    car_df = media_source_status(car_df)
    print("MEDIA SOURCE STATUS FEATURE ADDED")
    car_df = navigation_status(car_df)
    print("ACTIVE ROUTE GUIDANCE STATUS FEATURE ADDED")
    car_df = park_assistants_status(car_df)
    car_df = phone_connected_status(car_df)
    print("CONNECTED PHONE STATUS ADDED")
    car_df = bluethooth_connection_status(car_df)
    print("BLUETOOTH CONNECTED STATUS")
    car_df = connected_phone_os(car_df)
    print("PHONE OS FEATURE STATUS ADDED")
    return car_df

def drive_mode_label_fix(merged_data):
    merged_data['Label'] = merged_data['Label'].replace('car/driveMode/0', 'car/driveMode/0.0')
    merged_data['Label'] = merged_data['Label'].replace('car/driveMode/2', 'car/driveMode/2.0')
    merged_data['Label'] = merged_data['Label'].replace('car/driveMode/3', 'car/driveMode/3.0')

    merged_data['Label'] = merged_data['Label'].replace('car/charismaLevel/Abgesenkt', 'car/charismaLevel/change')
    merged_data['Label'] = merged_data['Label'].replace('car/charismaLevel/Lift', 'car/charismaLevel/change')
    merged_data['Label'] = merged_data['Label'].replace('car/charismaLevel/Mittel', 'car/charismaLevel/change')
    merged_data['Label'] = merged_data['Label'].replace('car/charismaLevel/Tief', 'car/charismaLevel/change')
    return merged_data

def encode_feature(context_data):
    context_data['season'] = context_data['month'].map(south_germany_season_mapping)
    return context_data

def ess_status_encode(context_data):
    context_data = context_data.sort_index()
    ess_info = []
    for session in tqdm(context_data.session.unique().tolist()):
        df = context_data[context_data['session']== session].copy()
        df['ess_status'] = 0
        if df['Label'].str.contains('ESS').any():
            ess_index = df.index[df['Label'] == 'car/ESS/on']
            for idx in reversed(ess_index):
                df.loc[:idx, 'ess_status'] = 0  # Set 'ess_status' to 0 for rows before the 'car/ESS/on' row
                df.loc[idx+1:, 'ess_status'] = 1
        ess_info.append(df)
    data_ess = pd.concat(ess_info, axis=0)
    return data_ess

def drive_mode_status(data_ess):
    drive_mode_info = []
    for session in tqdm(data_ess.session.unique().tolist()):
        df = data_ess[data_ess['session']== session].copy()
        df = df.sort_values(by=['datetime'])
        df['current_drive_mode'] = 'car/driveMode/0.0'
        drive_modes = {'car/driveMode/0.0', 'car/driveMode/1.0', 'car/driveMode/2.0', 'car/driveMode/3.0'}

        # Iterate over DataFrame
        first_drive_mode_interaction = 0
        if df['Label'].str.contains('driveMode').any():
            for i, row in df.iterrows():
                label = row['Label']
                if label in drive_modes:
                    if first_drive_mode_interaction ==0 and label == 'car/driveMode/0.0':
                        df.loc[:i, 'current_drive_mode'] = 'car/driveMode/3.0'
                    # df.at[i, 'current_drive_mode'] = label
                    first_drive_mode_interaction=1
                    df.loc[i+1:, 'current_drive_mode'] = label
                else:
                    continue
        drive_mode_info.append(df)
    data_drivemode = pd.concat(drive_mode_info, axis=0)
    return data_drivemode

def clima_status(data_drivemode):
    clima_mode_info = []
    for session in tqdm(data_drivemode.session.unique().tolist()):
        df = data_drivemode[data_drivemode['session']== session].copy().sort_index().reset_index()
        df['current_clima_mode']  = 'unknown'
        df = df.sort_values(by=['datetime'])
        clima_modes = {'clima/AC/on', 'clima/AC/off', 'clima/AC/ECO'}
        if df['Label'].str.contains('clima').any():
            for i, row in df.iterrows():
                label = row['Label']
                if label in clima_modes:
                    df.loc[i+1:, 'current_clima_mode'] = label
        clima_mode_info.append(df)
    data_clima = pd.concat(clima_mode_info, axis=0)
    return data_clima

def media_source_status(data_clima):
    media_source_info = []
    for session in tqdm(data_clima.session.unique().tolist()):
        df = data_clima[data_clima['session']== session].copy().sort_values(by=['datetime'])
        df = df.reset_index(drop=True)
        df['current_media_source'] = 'media/selectedSource/unavailable'
        media_source = {'media/selectedSource/Bluetooth', 'media/selectedSource/Radio',
        'media/selectedSource/Favorite', 'media/selectedSource/CarPlay'}

        if df['Label'].str.contains('selectedSource').any():
            for i, row in df.iterrows():
                label = row['Label']
                if label in media_source:
                    df.loc[i+1:, 'current_media_source'] = label
                else:
                    continue
        media_source_info.append(df)
    data_media = pd.concat(media_source_info, axis=0)
    data_media = data_media.sort_values(by=['session', 'datetime'])
    data_media = data_media.reset_index(drop=True)
    data_media = data_media.drop(columns=['index'])
    return data_media

def navigation_status(data_media):
    navi_status = []
    for session in tqdm(data_media['session'].unique()):
        df = data_media[data_media['session'] == session].copy().reset_index()
        df['nav_guidance_status'] = 'navi_inactive'
        navi_labels = {'navi/Start/Address', 'navi/Start/Favorite'}
        if df['Label'].str.contains('navi').any():
            for i, row in df.iterrows():
                label = row['Label']
                if label in navi_labels:
                    df.loc[i+1:, 'nav_guidance_status'] = 'navi_active'
                    break
        navi_status.append(df)
    data_navi = pd.concat(navi_status, axis=0)
    return data_navi

def park_assistants_status(data_navi):
    park_assistants_status = []
    for session in tqdm(data_navi['session'].unique()):
        df = data_navi[data_navi['session'] == session].copy().reset_index()
        df['proximity_to_parking_spot'] = 'no_parking_spot_closeby'
        park_assistant = {'car/Start/ParkAssistant'}
        if df['Label'].str.contains('ParkAssistant').any():
            print(session, 'i am in')
            for i, row in df.iterrows():
                label = row['Label']
                if label in park_assistant:
                    df.loc[i-window:i+window, 'proximity_to_parking_spot'] = 'parking_spot_closeby'
                    break
        park_assistants_status.append(df)
    data_parkassistant = pd.concat(park_assistants_status, axis=0)
    data_parkassistant = data_parkassistant.sort_values(by=['session', 'datetime'])
    data_parkassistant = data_parkassistant.reset_index(drop=True)
    return data_parkassistant

def phone_connected_status(data_parkassistant):
    phone_status = []
    for session in tqdm(data_parkassistant['session'].unique()):
        df = data_parkassistant[data_parkassistant['session'] == session].copy().reset_index()
        df['phone_status'] = 'unconnected'

        phone_labels = {'phone/goTo/Favorite', 'phone/Start/CarPlay',
        'phone/Start/AndroidAuto', 'phone/Connect/NewDevice',
        'phone/Call/Favorite', 'phone/Call/PersonX'}
        
        if df['Label'].str.contains('phone').any():

            for i, row in df.iterrows():
                label = row['Label']
                if label in phone_labels:
                    df.loc[i-window:i, 'phone_status'] = 'phone_connected'
        phone_status.append(df)
    data_phone_status = pd.concat(phone_status, axis=0)
    df_phone_status = data_phone_status.sort_values(by=['session', 'datetime'])
    df_phone_status = df_phone_status.reset_index(drop=True)
    df_phone_status = df_phone_status.drop(columns=['index'])
    return data_phone_status

def bluethooth_connection_status(data_phone_status):
    bluetooth_device_status = []
    for session in tqdm(data_phone_status['session'].unique()):
        df = data_phone_status[data_phone_status['session'] == session].copy().reset_index()
        df['bluetooth_connected'] = 'bluetooth_unconnected'
        phone_labels = {'media/selectedSource/Bluetooth'}
        if df['Label'].str.contains('Bluetooth|phone').any():
            for i, row in df.iterrows():
                label = row['Label']
                if label in phone_labels:
                    df.loc[i-window:, 'bluetooth_connected'] = 'bluetooth_connected'
        bluetooth_device_status.append(df)
    data_bluetooth_status = pd.concat(bluetooth_device_status, axis=0)
    df_bluetooth_status = data_bluetooth_status.sort_values(by=['session', 'datetime'])
    df_bluetooth_status = df_bluetooth_status.reset_index(drop=True)
    df_bluetooth_status = df_bluetooth_status.drop(columns=['index'])
    return data_bluetooth_status

def connected_phone_os(df_bluetooth_status):
    phone_os = []
    phone_labels = {'phone/Start/CarPlay', 'media/selectedSource/CarPlay',  'phone/Start/AndroidAuto'}
    Android = {'phone/Start/AndroidAuto'}
    carplay = {'phone/Start/CarPlay', 'media/selectedSource/CarPlay'}
    others = {'phone/goTo/Favorite', 'phone/Connect/NewDevice',
        'phone/Call/Favorite', 'phone/Call/PersonX', 'media/selectedSource/Bluetooth', }

    phone_os = []

    for session in tqdm(df_bluetooth_status['session'].unique()):
        df = df_bluetooth_status[df_bluetooth_status['session'] == session].copy()
        df['phone_os'] = 'unconnected'
        if df['Label'].str.contains('Bluetooth|phone|AndroidAuto|CarPlay').any():
            if df['Label'].str.contains('AndroidAuto').any():
                df['phone_os'] = 'Android'
            elif df['Label'].str.contains('CarPlay').any():
                df['phone_os'] = 'ios'
            else:
                df['phone_os'] = 'unknown'
        phone_os.append(df)

    data_phone_os = pd.concat(phone_os, axis=0)
    return data_phone_os

def newdevice_availability(data_phone_os):
    new_bluetooth_device = []

    for session in tqdm(data_phone_os['session'].unique()):
        df = data_phone_os[data_phone_os['session'] == session].reset_index()
        df['new_bluetooth_device_to_pair'] = 'no bluetooth device to pair'
        
        new_device_indices = df[df['Label'].str.contains('NewDevice', na=False)].index
        if not new_device_indices.empty:
            start_index = max(0, new_device_indices[0] - window)  # Calculate start index
            end_index = new_device_indices[-1]  # Calculate end index
            print(start_index, end_index)
            # Update the new_bluetooth_device_to_pair column for the specified range
            df.loc[start_index:end_index, 'new_bluetooth_device_to_pair'] = 'new bluetooth device available'
        
        new_bluetooth_device.append(df)

    # Concatenate all dataframes in new_bluetooth_device list
    data_new_bluetooth_device = pd.concat(new_bluetooth_device, axis=0)

    data_new_bluetooth_device = data_new_bluetooth_device.drop(columns=['index', 'level_0']).sort_values(by=['session', 'session'])
    data_new_bluetooth_device = data_new_bluetooth_device.reset_index(drop=True)
    return data_new_bluetooth_device