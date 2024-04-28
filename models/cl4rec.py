import math
import random
import numpy as np
import torch
from torch import nn
import pickle
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel

from models.interaction_encoder.sasrec import sasrec
from models.utils import Flatten_layers
from models.dynamic_context_encoder.lstm import lstm_context_encoder
from models.dynamic_context_encoder.transformer import TransformerEncoder_DynamicContext
from models.dynamic_context_encoder.tcn_model import TCNModel
from models.static_context_encoder.static_context_encoder import static_context_encoder
from trainer.loss import loss_function

class CL4Rec(BaseModel):
    def __init__(self, data_handler):
        super(CL4Rec, self).__init__(data_handler)
        # Extract configuration parameters
        data_config = configs['data']
        model_config = configs['model']
        train_config = configs['train']

        self.item_num = data_config['item_num']
        self.emb_size = model_config['item_embedding_size']
        self.mask_token = self.item_num + 1

        self.dropout_rate_fc_concat = model_config['dropout_rate_fc_concat']
        self.batch_size = train_config['batch_size']
        
        self.lmd = model_config['cl_lmd']
        self.tau = model_config['cl_tau']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

        self._interaction_encoder()
        self._dynamic_context_encoder(model_config)
        self._static_context_encoder()
        self._encoder_correlation()

        self.loss_func, self.cl_loss_func = loss_function()

    def _interaction_encoder(self):
        self.interaction_encoder = sasrec()
        
    def _static_context_encoder(self):
        self.static_embedding  = static_context_encoder()

    def _dynamic_context_encoder(self, model_config):
        if model_config['context_encoder'] == 'lstm':
            self.context_encoder = lstm_context_encoder()
            self.input_size_fc_concat = 88
        elif model_config['context_encoder'] == 'transformer':
            self.context_encoder = TransformerEncoder_DynamicContext()
            self.input_size_fc_concat = 6400 + 2 * self.emb_size
        elif model_config['context_encoder'] == 'tempcnn':
            self.context_encoder = TCNModel()
            self.input_size_fc_concat = 2 * self.embedding_size + 32

    def _encoder_correlation(self):
        if configs['model']['encoder_combine'] == 'concat':
        # FCs after concatenation layer
            self.fc_layers_concat = Flatten_layers(input_size = self.input_size_fc_concat, 
                                                   emb_size = self.emb_size, 
                                                   dropout_p=self.dropout_rate_fc_concat)

        # Combine 3 encoder outputs - Attention or concatenation
        # if configs['model']['encoder_combine'] == 'attention':
        #     self.fc_context_dim_red = nn.Linear(72, 64)
        #     self.multi_head_attention = nn.MultiheadAttention(self.emb_size, self.n_heads)


    
    def count_parameters(model):
        # Count the total number of parameters in the model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, batch_seqs,batch_context, batch_static_context, batch_dense_static_context):
        sasrec_out = self.interaction_encoder(batch_seqs)
        context_output = self.context_encoder(batch_context)

        static_context = self.static_embedding(batch_static_context, batch_dense_static_context)
        context = torch.cat((context_output, static_context), dim=1)
        if configs['model']['encoder_combine'] == 'concat':
            out = torch.cat((sasrec_out, context), dim=1)
            # print('after concat', out.size())
            out = self.fc_layers_concat(out)
            # print('after concat flatten fc', out.size())
        # elif configs['model']['encoder_combine'] == 'attention':
        #     context = self.fc_context_dim_red(context)
        #     out, _ = self.multi_head_attention(sasrec_out, context, context)
        return out

    def cal_loss(self, batch_data):
        _, batch_seqs, batch_last_items, batch_time_deltas, batch_dynamic_context, batch_static_context, batch_dense_static_context, _ = batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, batch_dense_static_context)

        test_item_emb = self.interaction_encoder.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)

        if configs['train']['ssl']:
            aug_seq1, aug_seq2 = self._cl4rec_aug(batch_seqs, batch_time_deltas)
            seq_output1 = self.forward(aug_seq1, batch_dynamic_context, batch_static_context, batch_dense_static_context)
            seq_output2 = self.forward(aug_seq2, batch_dynamic_context, batch_static_context, batch_dense_static_context)
            # Compute InfoNCE Loss (Contrastive Loss):
            # Computes the InfoNCE loss (contrastive loss) between the representations of the augmented sequences. 
            # The temperature parameter (temp) and batch size are specified.
            cl_loss = self.lmd * self.info_nce(
                seq_output1, seq_output2, temp=self.tau, batch_size=aug_seq1.shape[0])
            # Aggregate Losses and Return: Aggregates the recommendation loss and contrastive loss into a total loss. 
            # Returns the total loss along with a dictionary containing individual loss components (rec_loss and cl_loss).
            loss_dict = {
                'rec_loss': loss.item(),
                'cl_loss': cl_loss.item(),
            }
        else:
            cl_loss = 0
            loss_dict = {
                'rec_loss': loss.item(),
                'cl_loss': cl_loss,
            }
        return loss + cl_loss, loss_dict

    def full_predict(self, batch_data):
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context, batch_dense_static_context,  _  = batch_data
        logits = self.forward(batch_seqs, batch_dynamic_context, batch_static_context, batch_dense_static_context)
        test_item_emb = self.interaction_encoder.emb_layer.token_emb.weight[:self.item_num+1]
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores

    def info_nce(self, z_i, z_j, temp, batch_size):
        # The method computes the InfoNCE loss for pairs of embeddings (z_i and z_j) by comparing the positive sample similarities with negative sample similarities, where negative samples are selected based on a mask to ensure they are uncorrelated. The final loss is calculated using a contrastive loss function.
        N = 2 * batch_size
        # Combine Embeddings:
        # Concatenates the embeddings z_i and z_j along dimension 0 to create a single tensor z. This tensor represents the combined embeddings of positive sample pairs.
        z = torch.cat((z_i, z_j), dim=0)
        #Compute Similarity Matrix:
        #Computes the similarity matrix by performing matrix multiplication of z with its transpose. The division by temp is a temperature parameter that scales the similarity values.
        sim = torch.mm(z, z.T) / temp
        # Extract Diagonal Elements:
        # Extracts the diagonal elements of the similarity matrix with a stride of batch_size. These represent the similarities between positive sample pairs.
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        # Concatenate Positive Samples:
        # Concatenates the positive similarity scores along dimension 0 and reshapes the tensor to have a shape of (N, 1).
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # Generate Negative Samples Using Mask:
        # Depending on whether batch_size matches self.batch_size, it either uses a predefined mask (self.mask_default) or generates a new correlated samples mask using self.mask_correlated_samples(batch_size). This mask is then used to extract negative samples from the similarity matrix.
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
        # Prepare Labels and Logits:
        # Creates label tensor with zeros for positive samples.
        # Concatenates positive and negative samples to form the logits tensor.
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # Compute InfoNCE Loss:
        # Computes the InfoNCE loss using a contrastive loss function (self.cl_loss_func), comparing the logits with the labels.
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss

    def _cl4rec_aug(self, batch_seqs, batch_time_deltas_seqs):
        def item_crop(seq, length, eta=0.6):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros_like(seq)
            if crop_begin != 0:
                croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):-crop_begin]
            else:
                croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):]
            return croped_item_seq.tolist(), num_left

        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            # token 0 has been used for semantic masking
            mask_index = [-i-1 for i in mask_index]
            masked_item_seq[mask_index] = self.mask_token
            return masked_item_seq.tolist(), length

        def item_reorder(seq, length, selected_elements, beta=0.6):
            reordered_item_seq = seq.copy()
            random.shuffle(selected_elements)
            for i, index in enumerate(longest_sequence):
                reordered_item_seq[index] = selected_elements[i]

            return reordered_item_seq, length
        
            # num_reorder = math.floor(length * beta)
            # reorder_begin = random.randint(0, length - num_reorder)
            # reordered_item_seq = seq[:]
            # shuffle_index = list(
            #     range(reorder_begin, reorder_begin + num_reorder))
            # random.shuffle(shuffle_index)
            # shuffle_index = [-i for i in shuffle_index]
            # reordered_item_seq[-(reorder_begin + 1 + num_reorder):-(reorder_begin+1)] = reordered_item_seq[shuffle_index]
            # return reordered_item_seq.tolist(), length

        # convert each batch into a list of list
        seqs = batch_seqs.tolist()
        time_delta_seqs = batch_time_deltas_seqs.tolist()
        ## a list of number of non zero elements in each sequence
        lengths = batch_seqs.count_nonzero(dim=1).tolist()

        min_time_reorder = configs['train']['min_time_reorder']

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        #iterating through each sequence with in a batch
        for seq, length, time_delta_seq in zip(seqs, lengths, time_delta_seqs):
            seq = np.asarray(seq.copy(), dtype=np.int64)
            time_delta_seq = np.asarray(time_delta_seq.copy(), dtype=np.float64)
            if length > 1:
                # finding if we have any interactions that happened within min_time_reorder
                available_index = np.where((time_delta_seq != 0) & (time_delta_seq < min_time_reorder))[0].tolist()
                interaction_equality = False
                if len(available_index) != 0:
                    consecutive_sequences = np.split(available_index, np.where(np.diff(available_index) != 1)[0] + 1)
                    consecutive_sequences = [sequence.tolist() for sequence in consecutive_sequences]
                    longest_sequence = max(consecutive_sequences, key=len, default=[])
                    longest_sequence.insert(0, min(longest_sequence)-1)
                    selected_elements = [seq[i] for i in longest_sequence]
                    interaction_equality = all(x == selected_elements[0] for x in selected_elements)

                if len(available_index) == 0 or interaction_equality:
                    switch = random.sample(range(2), k=2)
                else:
                    switch = random.sample(range(3), k=2)
                    if switch[0] == switch[1] == 2:
                        coin  =  random.sample(range(2), k=1)
                        value = random.sample(range(2), k=1)
                        switch[coin[0]] = value[0]
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length, selected_elements)

            if aug_len > 0:
                aug_seq1.append(aug_seq)
                aug_len1.append(aug_len)
            else:
                aug_seq1.append(seq.tolist())
                aug_len1.append(length)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length, selected_elements)

            if aug_len > 0:
                aug_seq2.append(aug_seq)
                aug_len2.append(aug_len)
            else:
                aug_seq2.append(seq.tolist())
                aug_len2.append(length)

        aug_seq1 = torch.tensor(np.array(aug_seq1), dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(np.array(aug_seq2), dtype=torch.long, device=batch_seqs.device)

        return aug_seq1, aug_seq2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask