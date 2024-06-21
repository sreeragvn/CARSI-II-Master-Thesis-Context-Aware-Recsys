
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import RobustScaler

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

window = 30 #seconds

base_path = './datasets/sequential/'
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
train_sequence_path = f'{base_path}{augmentation_folder}seq/train.tsv'
test_sequence_path = f'{base_path}{augmentation_folder}seq/test.tsv'
train_dense_static_context_path = f'{base_path}{augmentation_folder}dense_static_context/train.csv'
test_dense_static_context_path = f'{base_path}{augmentation_folder}dense_static_context/test.csv'

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
                       'temperature_in', 'KBI_speed', 
                       'latitude','longitude', 'altitude'] # todo remove these features in the future
if feat_engg:
       cat_static_context_var = ['car_id', 'month', 'weekday', 'hour', 'season', 'seatbelt_codriver', 'seatbelt_rear_l', # categorical static context
                            'seatbelt_rear_r',  'street_category']
else:
       cat_static_context_var = ['car_id', 'month', 'weekday', 'hour', 'seatbelt_codriver', 'seatbelt_rear_l', # categorical static context
                            'seatbelt_rear_r',  'street_category']
dense_static_context_var =  ['distance_driven_benchmark', 'soc', 'time_second']  # dense static context
status_static_context_var = ['ess_status', 'current_drive_mode', 'current_clima_mode', 'current_media_source', # status static context
                     'nav_guidance_status', 'proximity_to_parking_spot', 'phone_status',
                     'bluetooth_connected', 'phone_os',
                     'new_bluetooth_device_to_pair']
#todo i feel street category is higly fluctuating. might be better to ignore
vehicles = ['SEB880','SEB882','SEB883','SEB885','SEB888','SEB889']

def context_data_preprocessing(car_df):
    print("PREPROCESSING STARTED")
    car_df = distance_driven_feature(car_df)
    print("ADDED distance driven feature")
    car_df = sampling_windows(car_df)
    print("SPLIT CONTEXT DATA INTO SAMPLES")
    dynamic_context, dense_static_context = dynamic_context_data_gen(car_df)
    print("GENERATED DYNAMIC AND DENSE CONTEXT DATA")
    static_context = static_context_gen(car_df)
    print("GENERATED STATIC CONTEXT DATA")
    print('number of windows', len(dynamic_context.wind_id.unique().tolist()), len(static_context.wind_id.unique().tolist()))
    print('number of session', len(dynamic_context.session.unique().tolist()), len(static_context.session.unique().tolist()))

    return dynamic_context, dense_static_context, static_context

def distance_driven_feature(context_data):
    context_data_filtered = context_data[context_data['distance_driven'] != 0]
    context_data_filtered['distance_driven_benchmark'] = context_data_filtered.groupby('session')['distance_driven'].transform(lambda x: x - x.min())
    context_data['distance_driven_benchmark'] = context_data_filtered['distance_driven_benchmark']
    context_data['distance_driven_benchmark'].fillna(0, inplace=True)
    context_data_list = []
    for session in tqdm(context_data.session.unique().tolist()):
        context_data_curr = context_data[context_data['session']== session]
        context_data_curr['distance_driven_benchmark'] = context_data_curr['distance_driven_benchmark'].replace(0, method='ffill')
        context_data_list.append(context_data_curr)
    context_data = pd.concat(context_data_list, axis=0)
    return context_data

def sampling_windows(context_data):
    train_sequence = pd.read_csv(train_sequence_path, sep='\t', low_memory=False)
    test_sequence = pd.read_csv(test_sequence_path, sep='\t', low_memory=False)
    selected_sequence = pd.concat([train_sequence, test_sequence], axis=0).sort_values(['session', 'window_id'])
    training_sequence_context = context_data

    if sequence_augmentation == True:
        augmented_frames = []
        for index, row in tqdm(selected_sequence.iterrows(), total=len(selected_sequence)):
            session = row['session']
            window_id = row['window_id']
            timestamp_target_interaction = row['timestamp_target_interaction']
            training_sequence_context_curr = training_sequence_context[(training_sequence_context['session'] == session) &
                                                        (training_sequence_context['datetime'] <= timestamp_target_interaction)].copy()
            if training_sequence_context_curr.empty:
                print(session, window_id)
            if not whole_session_context and window < len(training_sequence_context_curr):
                    training_sequence_context_curr = training_sequence_context_curr.tail(window)
            training_sequence_context_curr['window_id'] = window_id
            augmented_frames.append(training_sequence_context_curr)
            # print(session, window_id, timestamp_target_interaction)
            # break
        training_sequence_context_augmented = pd.concat(augmented_frames, axis=0)
        context_data = training_sequence_context_augmented.reset_index(drop=True)
        context_data['wind_id'] = context_data.groupby(['session', 'window_id']).ngroup()
    else:
        # if sequence_augmentation is set to false
        if not whole_session_context:
            context_data = training_sequence_context.groupby('session').tail(window)
        context_data = training_sequence_context.reset_index(drop=True)
        context_data['window_id'] = context_data.groupby('session').ngroup()
    print('total number of sequence data sessions: ', len(selected_sequence.session.unique().tolist()))
    print('total number of Sequence data windows: ', len(train_sequence.window_id.unique().tolist()) + len(test_sequence.window_id.unique().tolist()))
    print('total number of context data sessions: ', len(context_data.session.unique().tolist()))
    print('total number of context data windows: ', len(context_data.wind_id.unique().tolist()))
    #dont be the bothered about the total number of windows.
    # total number of sequence data sessions:  1634
    # total number of Sequence data windows:  5883
    # total number of context data sessions:  1634
    # total number of context data windows:  5883
    return context_data

def dynamic_context_data_gen(context_data):
    dynamic_context = context_data[dynamic_context_var + ['window_id', 'session', 'datetime', 'wind_id']]
    print('number of dynamic context session', len(dynamic_context[['window_id', 'session']].drop_duplicates()))
    # function to pad first value to fit the window size
    if pad_to_window_size:
        df = dynamic_context.copy()
        session_counts = df.groupby('wind_id').size()
        less_than_100 = session_counts[session_counts < window].index.tolist()
        print(f'Number of window with window length less than {window}: ', len(less_than_100))
        window100_dfs = df[~df['wind_id'].isin(less_than_100)]
        empty_df = []
        for window_id in tqdm(less_than_100):
            sub_df = df[df['wind_id'] == window_id]
            rows_to_pad = window - len(sub_df)
            min_datetime_row = sub_df.loc[sub_df['datetime'].idxmin()]
            pad_df = pd.DataFrame(min_datetime_row, df.columns).transpose()
            pad_df = pd.concat([pad_df] * int(rows_to_pad), ignore_index=True, axis=0)

            padded_df = pd.concat([pad_df, sub_df], axis=0).reset_index(drop=True)
            empty_df.append(padded_df)
        if empty_df:
            df = pd.concat(empty_df, axis=0).reset_index(drop=True)
            df = pd.concat([df, window100_dfs], axis=0).sort_values(by=['window_id']).reset_index(drop=True)
            session_counts = df.groupby('window_id').size()
            less_than_100 = session_counts[session_counts < window].index.tolist()
            print(f'Number of window with window length less than {window}: ', len(less_than_100))
            dynamic_context = df

    dense_static_context = context_data[dense_static_context_var + ['window_id', 'session', 'datetime', 'wind_id']]
    dense_static_context = dense_static_context.sort_values(by=['wind_id','datetime'], ascending=False)
    dense_static_context = dense_static_context.groupby('wind_id').first()
    dense_static_context.reset_index(inplace=True)
    dense_static_context = dense_static_context.sort_values(by='wind_id')
    dense_static_context = dense_static_context.sort_values(by=['wind_id', 'datetime'])
    dense_static_context = dense_static_context.drop(columns=['wind_id'])
    dynamic_context = dynamic_context.sort_values(by=['wind_id', 'datetime'])
    dynamic_context = dynamic_context.drop(columns='wind_id')
    return dynamic_context, dense_static_context

def static_context_gen(context_data):
    cat_static_context = context_data[cat_static_context_var + ['window_id', 'session', 'datetime', 'wind_id']]
    cat_static_context = cat_static_context.groupby('wind_id').apply(lambda x: x.mode().iloc[0]).reset_index(drop=True)
    if feat_engg:
        cat_static_context['season'], _ = pd.factorize(cat_static_context['season'])  

        status_static_context = context_data[status_static_context_var + ['window_id', 'session', 'datetime', 'wind_id']]
        for col in status_static_context_var:
            print(col)
            print(sorted(status_static_context[col].unique().tolist()))  
        for col in status_static_context_var:
            status_static_context[col], _ = pd.factorize(status_static_context[col])  
        status_static_context = status_static_context.sort_values(by=['session', 'datetime'])
        latest_indices = status_static_context.groupby('wind_id')['datetime'].idxmax()
        status_static_context_filt = status_static_context.loc[latest_indices]
        status_static_context = status_static_context_filt.reset_index(drop=True)

        static_context = pd.merge(cat_static_context, status_static_context, on=['wind_id', 'window_id', 'session'], how='inner')
    else:
        static_context = cat_static_context
    static_context = static_context.sort_values(by='wind_id')
    static_context = static_context.drop(columns=['datetime_y', 'datetime_x'])
    # rearrage order of columns
    columns = list(static_context.columns)
    columns.remove('car_id')
    columns.append('car_id')
    static_context = static_context[columns]
    static_context = static_context.drop(columns=['wind_id'])
    return static_context

def train_test_split(dynamic_context, dense_static_context, static_context):
    with open(train_session_path, 'rb') as pickle_file:
        train_sessions = pickle.load(pickle_file)
    with open(test_session_path, 'rb') as pickle_file:
        test_sessions = pickle.load(pickle_file)
    train_dynamic_context = dynamic_context[dynamic_context['session'].isin(train_sessions)].reset_index(drop=True)
    test_dynamic_context = dynamic_context[dynamic_context['session'].isin(test_sessions)].reset_index(drop=True)
    train_static_context = static_context[static_context['session'].isin(train_sessions)].reset_index(drop=True)
    test_static_context = static_context[static_context['session'].isin(test_sessions)].reset_index(drop=True)
    train_dense_static_context = dense_static_context[dense_static_context['session'].isin(train_sessions)].reset_index(drop=True)
    test_dense_static_context = dense_static_context[dense_static_context['session'].isin(test_sessions)].reset_index(drop=True)
    return train_dynamic_context, test_dynamic_context, train_static_context, test_static_context, train_dense_static_context, test_dense_static_context

def test_method_preprocessing(train_dynamic_context, test_dynamic_context, train_static_context, 
                              test_static_context, train_dense_static_context, test_dense_static_context):
    def session_window_mapping(df):
        session_window_dict = {}

        for index, row in df.iterrows():
            session = int(row['session'])
            window_id = int(row['window_id'])
            
            # If the session is not already in the dictionary, initialize an empty set
            if session not in session_window_dict:
                session_window_dict[session] = set()
            
            # Add the window_id to the set corresponding to the session
            session_window_dict[session].add(window_id)

        # Convert sets to lists in the resulting dictionary
        session_window_dict = {session: list(window_ids) for session, window_ids in session_window_dict.items()}
        return session_window_dict

    train_session_win_id_mapping_dc = session_window_mapping(train_dynamic_context)
    test_session_win_id_mapping_dc = session_window_mapping(test_dynamic_context)
    train_session_win_id_mapping_sc = session_window_mapping(train_static_context)
    test_session_win_id_mapping_sc = session_window_mapping(test_static_context)
    train_session_win_id_mapping_dsc = session_window_mapping(train_dense_static_context)
    test_session_win_id_mapping_dsc = session_window_mapping(test_dense_static_context)

    with open(os.path.join(parameter_path, 'session_win_id_mapping.pkl'), 'rb') as pickle_file:
        train_session_win_id_mapping = pickle.load(pickle_file)
        test_session_win_id_mapping = pickle.load(pickle_file)

    print(len(train_session_win_id_mapping_dc), len(train_session_win_id_mapping_sc),  len(train_session_win_id_mapping_dsc), len(train_session_win_id_mapping))

    if train_session_win_id_mapping_dc == train_session_win_id_mapping_sc == train_session_win_id_mapping == train_session_win_id_mapping_dsc:
        print("All training data mapping are exactly identical.")
    if test_session_win_id_mapping_dc == test_session_win_id_mapping_sc == test_session_win_id_mapping == test_session_win_id_mapping_dsc:
        print("All testing data mapping are exactly identical.")

    print(len(train_dynamic_context.session.unique().tolist()), len(test_dynamic_context.session.unique().tolist()))
    print(len(train_dynamic_context.window_id.unique().tolist()), len(test_dynamic_context.window_id.unique().tolist()))
    print(len(train_static_context.session.unique().tolist()), len(test_static_context.session.unique().tolist()))
    print(len(train_static_context.window_id.unique().tolist()), len(test_static_context.window_id.unique().tolist()))
    print(len(train_dense_static_context.session.unique().tolist()), len(test_dense_static_context.session.unique().tolist()))
    print(len(train_dense_static_context.window_id.unique().tolist()), len(test_dense_static_context.window_id.unique().tolist()))