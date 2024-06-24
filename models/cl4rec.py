import math
import random
import numpy as np
import torch
from torch import nn
import pickle
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.interaction_encoder.transformer import TransformerLayer
from models.utils import FlattenLayers
from models.dynamic_context_encoder.tcn_model import TCNModel
from models.static_context_encoder.static_context_encoder import StaticContextEncoder
from trainer.loss import loss_function

class CL4Rec(BaseModel):
    def __init__(self, data_handler):
        """
        Initialize the CL4Rec model.
        
        Args:
            data_handler: Data handler for loading and preprocessing data.
        """
        super(CL4Rec, self).__init__(data_handler)
        
        data_config = configs['data']
        model_config = configs['model']
        train_config = configs['train']

        self.item_num = data_config['item_num']
        self.emb_size = model_config['item_embedding_size']
        self.mask_token = self.item_num + 1

        self.dropout_rate_fc_concat = model_config['dropout_rate_fc_concat']
        self.batch_size = train_config['batch_size']

        self._interaction_encoder()
        self._dynamic_context_encoder(model_config)
        self._static_context_encoder()
        self._encoder_correlation()

        self.loss_func, self.cl_loss_func = loss_function()

    def _interaction_encoder(self):
        """
        Initialize the interaction encoder using transformer layers.
        """
        self.interaction_encoder = TransformerLayer()
        
    def _static_context_encoder(self):
        """
        Initialize the static context encoder.
        """
        self.static_embedding = StaticContextEncoder()

    def _dynamic_context_encoder(self, model_config):
        """
        Initialize the dynamic context encoder using TCN model.
        
        Args:
            model_config (dict): Configuration dictionary for the model.
        """
        self.context_encoder = TCNModel()
        self.input_size_fc_concat = 2 * self.emb_size + 32

    def _encoder_correlation(self):
        """
        Initialize the fully connected layers for concatenating encoded features.
        """
        self.fc_layers_concat = FlattenLayers(input_size=self.input_size_fc_concat, 
                                               emb_size=self.emb_size, 
                                               dropout_p=self.dropout_rate_fc_concat)

    def count_parameters(model):
        """
        Count the total number of parameters in the model.
        
        Args:
            model (nn.Module): The model whose parameters are to be counted.
        
        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, batch_seqs, batch_context, batch_static_context, batch_dense_static_context):
        """
        Forward pass for the CL4Rec model.
        
        Args:
            batch_seqs (Tensor): Batch of input sequences.
            batch_context (Tensor): Batch of dynamic context features.
            batch_static_context (Tensor): Batch of static context features.
            batch_dense_static_context (Tensor): Batch of dense static context features.
        
        Returns:
            Tensor: Output of the model after encoding and concatenation.
        """
        sasrec_out = self.interaction_encoder(batch_seqs)
        context_output = self.context_encoder(batch_context)
        static_context = self.static_embedding(batch_static_context, batch_dense_static_context)
        context = torch.cat((context_output, static_context), dim=1)
        out = torch.cat((sasrec_out, context), dim=1)
        out = self.fc_layers_concat(out)
        return out

    def cal_loss(self, batch_data):
        """
        Calculate the loss for the given batch of data.
        
        Args:
            batch_data (tuple): Batch of data including sequences, last items, time deltas, context features, etc.
        
        Returns:
            tuple: Calculated loss and a dictionary with loss details.
        """
        _, batch_seqs, batch_last_items, batch_time_deltas, batch_dynamic_context, batch_static_context, batch_dense_static_context, _ = batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, batch_dense_static_context)

        test_item_emb = self.interaction_encoder.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)

        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': 0,
        }
        return loss, loss_dict

    def full_predict(self, batch_data):
        """
        Perform full prediction for the given batch of data.
        
        Args:
            batch_data (tuple): Batch of data including sequences, context features, etc.
        
        Returns:
            Tensor: Prediction scores for each item.
        """
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context, batch_dense_static_context, _ = batch_data
        logits = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, batch_dense_static_context)
        test_item_emb = self.interaction_encoder.emb_layer.token_emb.weight[:self.item_num + 1]
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
