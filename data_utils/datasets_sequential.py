from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np
import torch
import pandas as pd
from collections import defaultdict, Counter
import scipy.sparse as sp
from os import path
from sklearn.metrics.pairwise import cosine_similarity

class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, dynamic_context, static_context, dense_static_context, mode='train', user_seqs_aug=None):
        """
        Initialize the SequentialDataset.

        Args:
            user_seqs (dict): User sequences containing user IDs, item sequences, last items, and time deltas.
            dynamic_context (dict): Dynamic context data.
            static_context (dict): Static context data.
            dense_static_context (dict): Dense static context data.
            mode (str): Mode of operation, either 'train' or 'test'. Default is 'train'.
            user_seqs_aug (dict, optional): Augmented user sequences for data augmentation.
        """
        self.mode = mode
        self.max_dynamic_context_length = configs['data']['dynamic_context_window_length']
        self.max_seq_len = configs['model']['sasrec_max_seq_len']

        self.user_history_lists = {user: seq for user, seq in zip(user_seqs["uid"], user_seqs["item_seq"])}
        self.user_history_time_delta_lists = {user: time_delta for user, time_delta in zip(user_seqs["uid"], user_seqs["time_delta"])}
        self.static_context = static_context
        self.dense_static_context = dense_static_context
        self.dynamic_context = dynamic_context

        if user_seqs_aug is not None:
            self.uids = user_seqs_aug["uid"]
            self.seqs = user_seqs_aug["item_seq"]
            self.last_items = user_seqs_aug["item_id"]
            self.time_delta = user_seqs_aug["time_delta"]
        else:
            self.uids = user_seqs["uid"]
            self.seqs = user_seqs["item_seq"]
            self.last_items = user_seqs["item_id"]
            self.time_delta = user_seqs["time_delta"]

        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = np.asarray(self.last_items, dtype=np.int32).reshape(-1, 1).tolist()

    def _pad_seq(self, seq):
        """
        Pad the sequence to the maximum sequence length.

        Args:
            seq (list): The input sequence.

        Returns:
            list: The padded sequence.
        """
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        return seq

    def _pad_time_delta(self, seq, time_delta):
        """
        Pad the time delta sequence to the maximum sequence length.

        Args:
            seq (list): The input sequence.
            time_delta (list): The time delta sequence.

        Returns:
            list: The padded time delta sequence.
        """
        if len(seq) >= self.max_seq_len:
            time_delta = time_delta[-self.max_seq_len:]
        else:
            time_delta = [0] * (self.max_seq_len - len(seq)) + time_delta
        return time_delta

    def _pad_context(self, lst):
        """
        Pad the context list to the maximum dynamic context length.

        Args:
            lst (list): The input context list.

        Returns:
            list: The padded context list.
        """
        if len(lst) > self.max_dynamic_context_length:
            return lst[-self.max_dynamic_context_length:]
        else:
            return lst

    def sample_negs(self):
        """
        Sample negative items for each user if negative sampling is enabled in the configuration.
        """
        if 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
            self.negs = []
            for i in range(len(self.uids)):
                u = self.uids[i]
                seq = self.user_history_lists[u]
                last_item = self.last_items[i]
                while True:
                    i_neg = np.random.randint(1, configs['data']['item_num'])
                    if i_neg not in seq and i_neg != last_item:
                        break
                self.negs.append(i_neg)

    def _process_context(self, context_i, context_type):
        """
        Process context data by padding or formatting it as needed.

        Args:
            context_i (dict): The input context data.
            context_type (str): Type of context ('dynamic' or 'static').

        Returns:
            list: The processed context data.
        """
        context_keys = context_i.keys()
        context_values = [context_i[key] for key in context_keys]
        if context_type == 'dynamic':
            context = [self._pad_context(inner_list) for inner_list in context_values]
            return context
        else:
            return context_values

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.uids)

    def __getitem__(self, idx):
        """
        Get a data sample for the given index.

        Args:
            idx (int): The index of the data sample.

        Returns:
            tuple: The data sample containing user ID, padded sequence, last item, padded time delta, dynamic context, static context, dense static context, sequence length, and optionally negative samples.
        """
        try:
            seq_i = self.seqs[idx]
            time_delta_i = self.time_delta[idx]
            padded_dynamic_context = self._process_context(self.dynamic_context[idx], context_type='dynamic')
            padded_dynamic_context = [torch.DoubleTensor(x) for x in padded_dynamic_context]
            static_context = self._process_context(self.static_context[idx], context_type='static')
            dense_static_context = self._process_context(self.dense_static_context[idx], context_type='static')

            if self.mode == 'train' and 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
                result = (
                    self.uids[idx],
                    torch.LongTensor(self._pad_seq(seq_i)),
                    self.last_items[idx],
                    torch.LongTensor(self._pad_time_delta(seq_i, time_delta_i)),
                    padded_dynamic_context,
                    torch.LongTensor(static_context),
                    dense_static_context,
                    len(seq_i),
                    self.negs[idx]
                )
            else:
                result = (
                    self.uids[idx],
                    torch.LongTensor(self._pad_seq(seq_i)),
                    self.last_items[idx],
                    torch.LongTensor(self._pad_time_delta(seq_i, time_delta_i)),
                    padded_dynamic_context,
                    torch.LongTensor(static_context),
                    dense_static_context,
                    len(seq_i)
                )
                
            return result
        except Exception as e:
            print("Error:", e)
            raise
