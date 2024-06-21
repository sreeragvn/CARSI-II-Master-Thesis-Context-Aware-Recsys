import os
import pickle
import torch
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split

base_path = './datasets/sequential/'
augmentation_folder = 'fix/'
parameter_path = f'{base_path}{augmentation_folder}parameters'

def load_df(vehicle, carsi_labels_only, PATH_TO_LOAD):
    df = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_merged.csv"), parse_dates=['datetime'], low_memory=False, index_col=0)
    df = df.dropna(subset=['Label'])
    df = df.sort_values(by=['session', 'datetime'])
    if carsi_labels_only:
        df['full_label'] = df['Label']
    return df

def process_vehicle(vehicle, carsi_labels_only, PATH_TO_LOAD):
    df_curr = load_df(vehicle, carsi_labels_only, PATH_TO_LOAD)
    df_curr['vehicle'] = vehicle
    df_curr = df_curr.dropna(subset=['full_label'])
    return df_curr

def data_label_preprocessing(car_df, start_num):
    car_df = drive_mode_label_fix(car_df)
    car_df = df_label_mapping(car_df, start_num)
    car_df = car_id_mapping(car_df, start_num)
    car_df = df_selected_columns(car_df)
    car_df = append_dummy_interaction(car_df)
    return car_df

def drive_mode_label_fix(car_df):
    car_df['full_label'] = car_df['full_label'].replace('car/driveMode/0', 'car/driveMode/0.0')
    car_df['full_label'] = car_df['full_label'].replace('car/driveMode/2', 'car/driveMode/2.0')
    car_df['full_label'] = car_df['full_label'].replace('car/driveMode/3', 'car/driveMode/3.0')
    car_df['full_label'] = car_df['full_label'].replace('car/charismaLevel/Abgesenkt', 'car/charismaLevel/change')
    car_df['full_label'] = car_df['full_label'].replace('car/charismaLevel/Lift', 'car/charismaLevel/change')
    car_df['full_label'] = car_df['full_label'].replace('car/charismaLevel/Mittel', 'car/charismaLevel/change')
    car_df['full_label'] = car_df['full_label'].replace('car/charismaLevel/Tief', 'car/charismaLevel/change')
    return car_df

def df_label_mapping(df, start_num):
    mapping = {category: index + start_num for index, category in enumerate(df['full_label'].unique())}
    mapping['no click'] = len(mapping)+1
    with open(os.path.join(parameter_path, 'label_mapping.pkl'), 'wb') as pickle_file:
        pickle.dump(mapping, pickle_file)
    print("Label - number mapping", mapping)
    df['full_label_num'] = df['full_label'].replace(mapping)
    return df

def car_id_mapping(full_df, car_id_stat_num):
    mapping_vehicle = {category: index + car_id_stat_num for index, category in enumerate(full_df['vehicle'].unique())}
    full_df['vehicle_num'] = full_df['vehicle'].replace(mapping_vehicle)
    return full_df

def plot_label_count(df):
    value_counts = df['full_label'].value_counts(normalize=True) * 100
    plt.figure(figsize=(10, 6))
    value_counts.plot(kind='bar')
    plt.title('Percentage of Unique Values in full_label')
    plt.xlabel('Unique Values')
    plt.ylabel('Percentage')
    plt.show()

def df_selected_columns(full_df):
    print(full_df.columns)
    filt_df = full_df[['session','full_label_num','vehicle_num', 'datetime']].sort_values(by = ['session', 'datetime'])
    filt_df = filt_df.drop_duplicates().reset_index(drop=True)
    filt_df['interaction_time_delta'] = (filt_df.groupby('session')['datetime'].diff().dt.total_seconds()/60).round(1)
    filt_df['interaction_time_delta'] = filt_df['interaction_time_delta'].fillna(0)
    filt_df['interaction_time_delta']   = filt_df['interaction_time_delta'].astype(int)
    return filt_df

def session_with_one_interactions(filt_df):
    # To do add those session with just one interactions. that one interaction can the target and input sequence can be no click along with the context
    # find session with just one interactions
    session_counts = filt_df['session'].value_counts()
    session_with_one_interactions = session_counts[session_counts == 1].index.tolist()
    one_interactions = filt_df[filt_df['session'].isin(session_with_one_interactions)]
    print("Number of session with just one click", len(one_interactions.session.unique().tolist()))
    return one_interactions

def append_dummy_interaction(filt_df):
    one_interactions_df = session_with_one_interactions(filt_df)
    time_delta = timedelta(seconds=30)
    new_rows = []
    for index, row in one_interactions_df.iterrows():
        new_row = {
            'session': row['session'],
            'full_label_num': len(filt_df['full_label_num'].unique().tolist())+1,
            'vehicle_num': row['vehicle_num'],
            'datetime': row['datetime'] - time_delta,
            'interaction_time_delta': 0
        }
        new_rows.append(new_row)
    one_interactions_w_empty_click = pd.concat([one_interactions_df, pd.DataFrame(new_rows)], ignore_index=True)
    one_interactions_w_empty_click.sort_values(by=['session', 'datetime'], inplace=True)
    one_interactions_w_empty_click.reset_index(drop=True, inplace=True)
    session_counts = filt_df['session'].value_counts()
    one_interaction_session = session_counts[session_counts == 1].index.tolist()
    filt_df_new = filt_df[~filt_df['session'].isin(one_interaction_session)]
    filt_df_new = pd.concat([filt_df_new, one_interactions_w_empty_click], ignore_index=True)
    filt_df_new.sort_values(by=['session', 'datetime'], inplace=True)
    filt_df_new.reset_index(drop=True, inplace=True)
    # filt_df_new[['session', 'datetime']].to_csv(os.path.join(parameter_path, 'sequence_context.csv'))
    return filt_df_new

## Generating augmented data
def explode_both(row):
        sequences = row['sequence']
        time_deltas = row['time_delta']
        timestamp_target_interaction = row['timestamp_target_interaction']
        sessions = [row['session']] * len(sequences)
        return pd.DataFrame({'session': sessions, 'sequence': sequences, 
                             'time_delta': time_deltas, 'timestamp_target_interaction': timestamp_target_interaction})

def sequence_generation(df, sequence_augmentation):
    sequence_dict = {
        'session': [],
        'sequence': [],
        'time_delta': [],
        'timestamp_target_interaction': []
    }
    if sequence_augmentation == True:
        for session in df['session'].unique().tolist():
            check_df = df[df['session']== session]

            sequence_list = []
            time_delta_list = []
            timestamp_target_interaction_list = []
            seq_length = len(check_df)
            sequence = check_df['full_label_num'].tolist()
            time_delta = check_df['interaction_time_delta'].tolist()
            timestamp_target_interaction =check_df['datetime'].tolist()
            # print(session)
            # print(seq_length)
            # print(sequence)
            # print(time_delta)
            while seq_length != 1:
                sequence_list.append(sequence)
                time_delta_list.append(time_delta)
                timestamp_target_interaction_list.append(timestamp_target_interaction)
                # print(sequence_list)
                # print(time_delta_list)
                time_delta = time_delta[:-1]
                sequence = sequence[:-1]
                timestamp_target_interaction = timestamp_target_interaction[:-1]
                seq_length = seq_length -1
            sequence_dict['session'].append(session)
            sequence_dict['sequence'].append(sequence_list)
            sequence_dict['time_delta'].append(time_delta_list)
            sequence_dict['timestamp_target_interaction'].append(timestamp_target_interaction_list)
        sequence_df = pd.DataFrame(sequence_dict)
        sequence_df = pd.concat(sequence_df.apply(explode_both, axis=1).tolist(), ignore_index=True)
    else:
        for session in df['session'].unique().tolist():
            check_df = df[df['session']== session]
            
            if len(check_df) == 1:
                 continue
            sequence_list = []
            time_delta_list = []
            seq_length = len(check_df)
            sequence = check_df['full_label_num'].tolist()
            time_delta = check_df['interaction_time_delta'].tolist()
            timestamp_target_interaction = check_df['datetime'].tolist()
            
            sequence_dict['session'].append(session)
            sequence_dict['sequence'].append(sequence)
            sequence_dict['time_delta'].append(time_delta)
            sequence_dict['timestamp_target_interaction'].append(timestamp_target_interaction)
        sequence_df = pd.DataFrame(sequence_dict)
    df_exploded = sequence_df
    df_exploded['timestamp_target_interaction'] = df_exploded['timestamp_target_interaction'].apply(lambda x: x[-1])
    df_exploded['time_delta_list'] = df_exploded['time_delta'].apply(lambda x: x[1:] if isinstance(x, list) and len(x) > 1 else x)
    df_exploded['interaction_time_delta_train'] = df_exploded['time_delta_list'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
    df_exploded['item_id_seq_train'] = df_exploded['sequence'].apply(lambda x: ' '.join(map(str, x[:-1])) if isinstance(x, list) and len(x) > 1 else None)
    df_exploded['item_id_target'] = df_exploded['sequence'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)
    df_exploded = df_exploded.dropna(subset=['item_id_target'])
    df_exploded['item_id_target'] = df_exploded['item_id_target'].astype(int)
    df_exploded = df_exploded.drop(columns=['sequence', 'time_delta', 'time_delta_list'])

    return df_exploded

def train_test_split_df(df_exploded, sequence_augmentation, remove_consecutive_duplicates_clicks):
    if sequence_augmentation == True:
      total_sessions = df_exploded.session.unique().tolist()
      test_sessions, train_sessions = train_test_split(total_sessions, test_size=0.8, shuffle=True, random_state=42)
      train_df = df_exploded[df_exploded['session'].isin(train_sessions)].sort_index()
      test_df = df_exploded[df_exploded['session'].isin(test_sessions)].sort_index()
    else:
        train_df, test_df = train_test_split(df_exploded, test_size=0.2, shuffle=True, random_state=42)
        train_df = train_df.sort_index()
        test_df = test_df.sort_index()
        train_sessions = train_df['session'].unique().tolist()
        test_sessions = test_df['session'].unique().tolist()

    with open(os.path.join(parameter_path, 'train_sessions.pkl'), 'wb') as pickle_file:
            pickle.dump(train_sessions, pickle_file)
    with open(os.path.join(parameter_path, 'test_sessions.pkl'), 'wb') as pickle_file:
            pickle.dump(test_sessions, pickle_file)

    test_df['window_id'] = range(len(test_df))
    test_df['window_id'] = test_df['window_id'].astype(int)
    train_df['window_id'] = range(len(train_df))
    train_df['window_id'] = train_df['window_id'].astype(int)

    if remove_consecutive_duplicates_clicks:
        test_df = seq_duplicate_remove(test_df)
        train_df = seq_duplicate_remove(train_df)
    else:
        train_df = train_df[['window_id', 'item_id_seq_train', 'item_id_target', 'interaction_time_delta_train', 'session', 'timestamp_target_interaction']]
        test_df = test_df[['window_id', 'item_id_seq_train', 'item_id_target', 'interaction_time_delta_train', 'session', 'timestamp_target_interaction']]

    class_weights = compute_class_weights(train_df)

    train_session_win_id_mapping = session_window_mapping(train_df)
    test_session_win_id_mapping = session_window_mapping(test_df)
    with open(os.path.join(parameter_path, 'session_win_id_mapping.pkl'), 'wb') as f:
            pickle.dump(train_session_win_id_mapping, f)
            pickle.dump(test_session_win_id_mapping, f)

    return train_df, test_df, class_weights


def remove_consecutive_duplicates(input_list):
    result = []
    removed_indices = []  # List to store indices of removed elements
    prev = None
    for i, num in enumerate(input_list):
        if num != prev:
            result.append(num)
        else:
            removed_indices.append(i)
        prev = num
    return result, removed_indices

def join_list(lst):
    return ' '.join(map(str, lst))

def create_zeros_list(lst):
    return [0] * len(lst)
    
def seq_duplicate_remove(df):
    df['interaction_time_delta_train_list'] = df['interaction_time_delta_train'].str.split().apply(lambda x: [int(i) for i in x])
    result = df['item_id_seq_train'].str.split().apply(lambda x: remove_consecutive_duplicates([int(i) for i in x]))
    df['seq_list'] = result.apply(lambda x: x[0])
    df['removed_indices'] = result.apply(lambda x: x[1])
    
    df['seq_non_dup'] = df['seq_list'].apply(join_list)

    max_length = df['seq_list'].apply(lambda x: len(x)).max()
    print("Maximum length of lists in seq_list column:", max_length)

    empty_lists_exist = any(df['seq_list'].apply(lambda x: len(x) == 0))
    if empty_lists_exist:
        print("There are empty lists in the 'seq_list' column.")
    else:
        print("There are no empty lists in the 'seq_list' column.")

    df['wrong_time_delta_interaction_list'] = df['seq_list'].apply(create_zeros_list)
    df['wrong_time_delta_interaction'] = df['wrong_time_delta_interaction_list'].apply(join_list)

    df.drop(columns=['seq_list', 'item_id_seq_train'], inplace=True)
    df.rename(columns={'seq_non_dup': 'item_id_seq_train'}, inplace=True)

    desired_column_order = ['window_id', 'item_id_seq_train', 'item_id_target', 'wrong_time_delta_interaction', 'session', 'timestamp_target_interaction']
    df = df[desired_column_order]
    return df

def compute_class_weights(data):
    class_frequencies = data['item_id_target'].value_counts(normalize=True)
    total_samples = len(data)
    class_weights = {label: total_samples / (len(class_frequencies) * freq) for label, freq in class_frequencies.items()}
    class_weights[0] = 0
    class_weights[23] = 0
    sorted_class_weights = dict(sorted(class_weights.items()))
    class_weights_tensor_list = torch.tensor(list(sorted_class_weights.values()))

    with open(os.path.join(parameter_path, 'param.pkl'), 'wb') as f:
            pickle.dump(class_weights_tensor_list, f)

    print("class weights", class_weights)


def session_window_mapping(df):
    session_window_dict = {}
    for index, row in df.iterrows():
        session = int(row['session'])
        window_id = int(row['window_id'])
        if session not in session_window_dict:
            session_window_dict[session] = set()
        session_window_dict[session].add(window_id)
    session_window_dict = {session: list(window_ids) for session, window_ids in session_window_dict.items()}
    return session_window_dict