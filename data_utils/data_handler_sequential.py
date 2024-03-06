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


class DataHandlerSequential:
    def __init__(self):
        data_name = configs['data']['name']

        if configs['train']['model_test_run']:
            print('fuck u')
            predir = f'./datasets/sequential/test/{data_name}'
        else:
            predir = f'./datasets/sequential/{data_name}'
            
        self.trn_file = path.join(predir, 'seq/train.tsv')
        self.val_file = path.join(predir, 'seq/test.tsv')
        self.tst_file = path.join(predir, 'seq/test.tsv')

        self.trn_dynamic_context_file = path.join(predir, 'dynamic_context/train.csv')
        self.val_dynamic_context_file = path.join(predir, 'dynamic_context/test.csv')
        self.tst_dynamic_context_file = path.join(predir, 'dynamic_context/test.csv')

        self.trn_static_context_file = path.join(predir, 'static_context/train.csv')
        self.val_static_context_file = path.join(predir, 'static_context/test.csv')
        self.tst_static_context_file = path.join(predir, 'static_context/test.csv')

        self.max_item_id = 0
        self.max_dynamic_context_length = 0

    def _read_tsv_to_user_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": [], "time_delta": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            # skip header
            line = f.readline()
            while line:
                uid, seq, last_item, time_delta_seq = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(item) for item in seq]
                time_delta_seq = time_delta_seq.split(' ')
                time_delta_seq = [float(time_delta) for time_delta in time_delta_seq]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                user_seqs["time_delta"].append(time_delta_seq)

                self.max_item_id = max(
                    self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs
    
    def _read_csv_dynamic_context(self, csv_file):
        # Todo currently the context have data that is past the last elements in the sequence. This has to be removed.
        try:
            context = pd.read_csv(csv_file, parse_dates=['datetime'])
            max_length = context['session_id'].value_counts().max()
            self.max_dynamic_context_length = max(self.max_dynamic_context_length, max_length)
            context = context.drop(['datetime'], axis=1)

            context_dict = {}
            for session_id, group in context.groupby('session_id'):
                context_dict[session_id] = {
                    column: group[column].tolist() for column in context.columns.difference(['session_id'])
                }
            return context_dict
        except Exception as e:
            print(f"Error reading dynamic CSV file: {e}")
            return None
        
    def _read_csv_static_context(self, csv_file):
        # Todo currently the context have data that is past the last elements in the sequence. This has to be removed.
        try:
            context = pd.read_csv(csv_file)
            context_dict = {}
            for session_id, group in context.groupby('session'):
                context_dict[session_id] = {
                    column: group[column].tolist()[0] for column in context.columns.difference(['session'])
                }
            return context_dict
        except Exception as e:
            print(f"Error reading static context CSV file: {e}")
            return None
        

    def _set_statistics(self, user_seqs_train, user_seqs_test, dynamic_context_data, static_context_data):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        # item originally starts with 1
        configs['data']['item_num'] = self.max_item_id
        configs['data']['max_context_length'] = self.max_dynamic_context_length
        configs['data']['dynamic_context_feat_num'] = len(list(dynamic_context_data[list(dynamic_context_data.keys())[0]].keys()))
        configs['data']['static_context_feat_num'] = len(list(static_context_data[list(static_context_data.keys())[0]].keys()))

    def _seq_aug(self, user_seqs):
        user_seqs_aug = {"uid": [], "item_seq": [], "item_id": [], "time_delta": []}
        for uid, seq, last_item, time_delta in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"], user_seqs["time_delta"]):
            user_seqs_aug["uid"].append(uid)
            user_seqs_aug["item_seq"].append(seq)
            user_seqs_aug["item_id"].append(last_item)
            user_seqs_aug["time_delta"].append(time_delta)
            for i in range(1, len(seq)-1):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq[:i])
                user_seqs_aug["item_id"].append(seq[i])
                user_seqs_aug["time_delta"].append(time_delta[:i])
        return user_seqs_aug

    def load_data(self):
        if configs['train']['model_test_run']:
            user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
            user_seqs_test = self._read_tsv_to_user_seqs(self.trn_file)
            dynamic_context_train =  self._read_csv_dynamic_context(self.trn_dynamic_context_file)
            dynamic_context_test =  self._read_csv_dynamic_context(self.trn_dynamic_context_file)
            static_context_train =  self._read_csv_static_context(self.trn_static_context_file)
            static_context_test =  self._read_csv_static_context(self.trn_static_context_file)
        else:
            user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
            user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
            dynamic_context_train =  self._read_csv_dynamic_context(self.trn_dynamic_context_file)
            dynamic_context_test =  self._read_csv_dynamic_context(self.tst_dynamic_context_file)
            static_context_train =  self._read_csv_static_context(self.trn_static_context_file)
            static_context_test =  self._read_csv_static_context(self.tst_static_context_file)
        self._set_statistics(user_seqs_train, user_seqs_test, dynamic_context_test, static_context_test)
        print(self.max_item_id)

        # # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        # if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
        #     user_seqs_aug = self._seq_aug(user_seqs_train)
        #     trn_data = SequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug)
        # else:
        #     trn_data = SequentialDataset(user_seqs_train)

        #Only implementing the case of no sequence augmentations _seq_aug
        trn_data = SequentialDataset(user_seqs_train, dynamic_context_train, static_context_train)
        tst_data = SequentialDataset(user_seqs_test, dynamic_context_test, static_context_test, mode='test')
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
