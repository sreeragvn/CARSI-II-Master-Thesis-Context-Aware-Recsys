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
        if configs['data']['name'] == 'ml-20m':
            predir = './datasets/sequential/ml-20m_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'sports':
            predir = './datasets/sequential/sports_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'carsii':
            predir = './datasets/sequential/carsii_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'carsii_delta':
            predir = './datasets/sequential/carsii_timedelta_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'carsii_random_delta':
            predir = './datasets/sequential/carsii_timedelta_rand_seq/'
            configs['data']['dir'] = predir
            
        self.trn_file = path.join(predir, 'train.tsv')
        self.val_file = path.join(predir, 'test.tsv')
        self.tst_file = path.join(predir, 'test.tsv')

        self.trn_context_file = path.join(predir, 'context/train.tsv')
        self.val_context_file = path.join(predir, 'context/test.tsv')
        self.tst_context_file = path.join(predir, 'context/test.tsv')
        self.max_item_id = 0

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

    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        # item originally starts with 1
        configs['data']['item_num'] = self.max_item_id

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
        sequence_split = False
        # if not sequence_split:
        #     random_split = True
        random_split = False
        session_dict_generation = False
        vehicle_ident = True

        user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
        user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
        self._set_statistics(user_seqs_train, user_seqs_test)

        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            user_seqs_aug = self._seq_aug(user_seqs_train)
            trn_data = SequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug)
        else:
            trn_data = SequentialDataset(user_seqs_train)
        tst_data = SequentialDataset(user_seqs_test, mode='test')
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
