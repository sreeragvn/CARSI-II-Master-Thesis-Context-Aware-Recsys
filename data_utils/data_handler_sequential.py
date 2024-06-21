import pickle
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_sequential import SequentialDataset
import torch as t
import torch.utils.data as data
from os import path
import pandas as pd
import os
import torch

class DataHandlerSequential:
    def __init__(self):
        """
        Initialize the DataHandlerSequential class.
        Set up file paths and initial configuration parameters.
        """
        if not configs["model"]["inference"]:
            data_name = configs['data']['name']
        else:
            data_name = configs['data']['inference_data_folder']

        predir = f'./datasets/sequential/{data_name}'

        # Set paths for parameter files
        configs['train']['parameter_class_weights_path'] = path.join(predir, 'parameters/param.pkl')
        configs['train']['parameter_label_mapping_path'] = path.join(predir, 'parameters/label_mapping.pkl')

        # Set paths for data files
        self.trn_file = path.join(predir, 'seq/train.tsv')
        self.val_file = path.join(predir, 'seq/test.tsv')
        self.tst_file = path.join(predir, 'seq/test.tsv')

        self.trn_dynamic_context_file = path.join(predir, 'dynamic_context/train.csv')
        self.val_dynamic_context_file = path.join(predir, 'dynamic_context/test.csv')
        self.tst_dynamic_context_file = path.join(predir, 'dynamic_context/test.csv')

        self.trn_static_context_file = path.join(predir, 'static_context/train.csv')
        self.val_static_context_file = path.join(predir, 'static_context/test.csv')
        self.tst_static_context_file = path.join(predir, 'static_context/test.csv')

        self.trn_dense_static_context_file = path.join(predir, 'dense_static_context/train.csv')
        self.val_dense_static_context_file = path.join(predir, 'dense_static_context/test.csv')
        self.tst_dense_static_context_file = path.join(predir, 'dense_static_context/test.csv')

        self.max_item_id = 0
        self.max_dynamic_context_length = configs['data']['dynamic_context_window_length']
        self.static_context_embedding_size = 0

    def _read_tsv_to_user_seqs(self, tsv_file):
        """
        Read TSV file and convert to user sequences.
        
        Args:
            tsv_file (str): Path to the TSV file.

        Returns:
            dict: Dictionary containing user sequences and associated data.
        """
        user_seqs = {"uid": [], "item_seq": [], "item_id": [], "time_delta": []}
        with open(tsv_file, 'r') as f:
            # Skip header
            f.readline()
            line = f.readline()
            while line:
                uid, seq, last_item, time_delta_seq, _, _ = line.strip().split('\t')
                seq = [int(item) for item in seq.split(' ')]
                time_delta_seq = [float(time_delta) for time_delta in time_delta_seq.split(' ')]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                user_seqs["time_delta"].append(time_delta_seq)
                self.max_item_id = max(self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs
    
    def _convert_to_int_list(self, string_list):
        return [list(map(int, x.split())) for x in string_list]
    
    def _read_tsv_to_user_seqs_inference(self, tsv_file):
        seq = pd.read_csv(tsv_file, sep="\t")
        print(seq)
        seq = seq[["uid", "item_seq", "item_id", "time_delta"]].to_dict(orient='list') 
        print(seq)
        seq['item_seq'] = self._convert_to_int_list(seq['item_seq'])
        seq['time_delta'] = self._convert_to_int_list(seq['time_delta'])
        print(seq)
        return seq

    def _sample_context_data(self, data):
        """
        Sample a subset of context data for testing purposes.
        
        Args:
            data (dict): Original context data.

        Returns:
            dict: Sampled context data.
        """
        small_dict = {}
        count = 0
        for key, value in data.items():
            if count < configs['experiment']['test_run_sample_no']:
                small_dict[key] = value
                count += 1
            else:
                break
        return small_dict

    def _read_csv_dynamic_context(self, csv_file):
        """
        Read dynamic context data from CSV file.
        
        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            dict: Dictionary containing dynamic context data.
        """
        try:
            context = pd.read_csv(csv_file, parse_dates=['datetime'])
            max_length = context['window_id'].value_counts().max()
            self.max_dynamic_context_length = min(self.max_dynamic_context_length, max_length)
            context = context.drop(['datetime', 'session'], axis=1)
            context_dict = {}
            for window_id, group in context.groupby('window_id'):
                context_dict[window_id] = {
                    column: group[column].tolist() for column in context.columns.difference(['window_id'])
                }
            if configs['experiment']['model_test_run']:
                context_dict = self._sample_context_data(context_dict)
            return context_dict
        except Exception as e:
            print(f"Error reading dynamic CSV file: {e}")
            return None

    def _read_csv_static_context(self, csv_file):
        """
        Read static context data from CSV file.
        
        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            dict: Dictionary containing static context data.
        """
        try:
            context = pd.read_csv(csv_file)
            context = context.drop(columns=['car_id', 'session'])
            context = context.astype(int)
            static_context_vocab_size = context.drop(columns=['window_id']).max(axis=0).tolist()
            if self.static_context_embedding_size != 0:
                self.static_context_embedding_size = [max(x, y) for x, y in zip(static_context_vocab_size, self.static_context_embedding_size)]
            else:
                self.static_context_embedding_size = static_context_vocab_size
            context_dict = {}
            for index, row in context.iterrows():
                session_key = row['window_id']
                row_dict = row.drop('window_id').to_dict()
                context_dict[session_key] = row_dict
            if configs['experiment']['model_test_run']:
                context_dict = self._sample_context_data(context_dict)
            return context_dict
        except Exception as e:
            print(f"Error reading static context CSV file: {e}")
            return None

    def _read_csv_dense_static_context(self, csv_file):
        """
        Read dense static context data from CSV file.
        
        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            dict: Dictionary containing dense static context data.
        """
        try:
            context = pd.read_csv(csv_file)
            context = context.drop(columns=['session', 'datetime'])
            context_dict = {}
            for index, row in context.iterrows():
                session_key = row['window_id']
                row_dict = row.drop('window_id').to_dict()
                context_dict[session_key] = row_dict
            if configs['experiment']['model_test_run']:
                context_dict = self._sample_context_data(context_dict)
            return context_dict
        except Exception as e:
            print(f"Error reading static context CSV file: {e}")
            return None

    def _set_statistics(self, user_seqs_train, user_seqs_test, dynamic_context_data, static_context_data, dense_static_context_test):
        """
        Set various statistics and configuration parameters based on the loaded data.
        
        Args:
            user_seqs_train (dict): Training user sequences.
            user_seqs_test (dict): Testing user sequences.
            dynamic_context_data (dict): Dynamic context data.
            static_context_data (dict): Static context data.
            dense_static_context_test (dict): Dense static context data.
        """
        user_num = max(max(user_seqs_train["uid"]), max(user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        configs['data']['item_num'] = self.max_item_id
        configs['data']['dynamic_context_window_length'] = self.max_dynamic_context_length
        configs['data']['dynamic_context_feat_num'] = len(dynamic_context_data[next(iter(dynamic_context_data))].keys())
        configs['data']['static_context_feat_num'] = len(static_context_data[next(iter(static_context_data))].keys())
        configs['data']['static_context_features'] = list(static_context_data[0].keys())
        configs['data']['dense_static_context_features'] = list(dense_static_context_test[0].keys())
        configs['data']['dynamic_context_features'] = list(dynamic_context_data[0].keys())
        # print('static context features', configs['data']['static_context_features'])
        # print('dynamic context features', configs['data']['dynamic_context_features'])
        # print('dense static context features', configs['data']['dense_static_context_features'])
        configs['data']['static_context_max'] = self.static_context_embedding_size

    def load_data(self):
        """
        Load and preprocess all data.
        """
        if not configs["model"]["inference"]:
            user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
            user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
            dynamic_context_train = self._read_csv_dynamic_context(self.trn_dynamic_context_file)
            dynamic_context_test = self._read_csv_dynamic_context(self.tst_dynamic_context_file)
            static_context_train = self._read_csv_static_context(self.trn_static_context_file)
            static_context_test = self._read_csv_static_context(self.tst_static_context_file)
            dense_static_context_train = self._read_csv_dense_static_context(self.trn_dense_static_context_file)
            dense_static_context_test = self._read_csv_dense_static_context(self.tst_dense_static_context_file)

            if configs['experiment']['model_test_run']:
                user_seqs_train = {key: value[:configs['experiment']['test_run_sample_no']] for key, value in user_seqs_train.items()}
                user_seqs_test = user_seqs_train
                dynamic_context_test = dynamic_context_train
                static_context_test = static_context_train
                dense_static_context_test = dense_static_context_train

            self._set_statistics(user_seqs_train, user_seqs_test, dynamic_context_test, static_context_test, dense_static_context_test)

            trn_data = SequentialDataset(user_seqs_train, dynamic_context_train, static_context_train, dense_static_context_train)
            tst_data = SequentialDataset(user_seqs_test, dynamic_context_test, static_context_test, dense_static_context_test)

            self.test_dataloader = data.DataLoader(
                tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            self.train_dataloader = data.DataLoader(
                trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        
        else:
            user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
            dynamic_context_test = self._read_csv_dynamic_context(self.tst_dynamic_context_file)
            static_context_test = self._read_csv_static_context(self.tst_static_context_file)
            dense_static_context_test = self._read_csv_dense_static_context(self.tst_dense_static_context_file)

            tst_data = SequentialDataset(user_seqs_test, dynamic_context_test, static_context_test, dense_static_context_test)

            self.test_dataloader = data.DataLoader(
                tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)