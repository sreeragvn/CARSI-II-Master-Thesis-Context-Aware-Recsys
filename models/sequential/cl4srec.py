import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
from models.interaction_encoder import  LSTM_interactionEncoder
from models.dynamic_context_encoder import TransformerEncoder_DynamicContext, LSTM_contextEncoder
from models.static_context_encoder import static_context_encoder
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import pickle
from ..TCNN.tcn_model import TCNModel


class CL4SRec(BaseModel):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    """
    def __init__(self, data_handler):
        super(CL4SRec, self).__init__(data_handler)
        # # Todo should we embed everything to same space or different space ? how do we select the embedding size ?
        # Extract configuration parameters
        data_config = configs['data']
        model_config = configs['model']
        train_config = configs['train']
        lstm_config = configs['lstm']

        self.item_num = data_config['item_num']
        self.emb_size = model_config['embedding_size']
        self.max_len = model_config['max_seq_len']
        self.mask_token = self.item_num + 1
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = model_config['dropout_rate']
        self.batch_size = train_config['batch_size']
        self.lmd = model_config['lmd']
        self.tau = model_config['tau']
        self.dynamic_context_feat_num = data_config['dynamic_context_feat_num']
        self.lstm_hidden_size = lstm_config['hidden_size']
        self.lstm_num_layers = lstm_config['num_layers']
        self.static_context_max_token = data_config['static_context_max']
        self.static_context_num = data_config['static_context_feat_num']

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # interaction Encoder( # interaction_encoder options are lstm, sasrec, durorec)
        if model_config['interaction_encoder'] == 'lstm':
            self.interaction_encoder = LSTM_interactionEncoder(self.item_num + 2, 
                                                            self.emb_size, 
                                                            self.lstm_hidden_size, 
                                                            self.lstm_num_layers)
        elif model_config['interaction_encoder'] == 'sasrec':
            self.emb_layer = TransformerEmbedding(self.item_num + 2, 
                                                  self.emb_size, self.max_len)
            self.transformer_layers = nn.ModuleList([TransformerLayer(self.emb_size, 
                                                                      self.n_heads, 
                                                                      self.inner_size, 
                                                                      self.dropout_rate) 
                                                                      for _ in range(self.n_layers)])
            # parameters initialization
            self.apply(self._init_weights)
            self.sasrec_fc_layer1 = nn.Linear((self.max_len)* self.emb_size, 128)
            self.sasrecbn1 = nn.BatchNorm1d(128)
            self.sasrec_fc_layer2 = nn.Linear(128, 128) 
            self.sasrecbn2 = nn.BatchNorm1d(128)
            self.sasrec_fc_layer3 = nn.Linear(128, 64) 
            self.sasrecbn3 = nn.BatchNorm1d(64)
        else:
            print('mention the interaction encoder - sasrec or lstm')
        
        #static context encoder
        self.static_embedding  = static_context_encoder(self.static_context_max_token, 8, 32, 16, 8)

        # dynamic Context Encoder
        if model_config['context_encoder'] == 'lstm':
            self.context_encoder = LSTM_contextEncoder(self.dynamic_context_feat_num, 
                                                       self.lstm_hidden_size, 
                                                       self.lstm_num_layers,
                                                       self.emb_size
                                                       )
            input_size = 88
        elif model_config['context_encoder'] == 'transformer':
            self.context_encoder = TransformerEncoder_DynamicContext(self.dynamic_context_feat_num, # num_features_continuous
                                                                     data_config['dynamic_context_window_length'],
                                                                     hidden_dim=self.emb_size, # d_model
                                                                     num_heads=8,)
            input_size = 6400 + 2 * self.emb_size
        elif model_config['context_encoder'] == 'tempcnn':
            self.context_encoder = TCNModel(self.dynamic_context_feat_num, num_channels=[80, 50, 25], kernel_size=3, dropout=0.25)
            input_size = 136
            output_size = 64

        # FCs after concatenation layer
        fc_layers = []
        for _ in range(2):
            fc_layers.extend([nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(), nn.Dropout(p=0.3)])
            input_size = 64
            output_size = 32
        fc_layers.append(nn.Linear(32, self.emb_size))
        self.fc_layers = nn.Sequential(*fc_layers)

        # Combine 3 encoder outputs - Attention or concatenation
        if configs['model']['encoder_combine'] == 'attention':
            self.fc_context_dim_red = nn.Linear(72, 64)
            self.multi_head_attention = nn.MultiheadAttention(self.emb_size, self.n_heads)

        # Loss Function
        if configs['train']['model_test_run'] or not configs['train']['weighted_loss_fn']:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            with open(configs['train']['parameter_class_weights_path'], 'rb') as f:
                _class_w = pickle.load(f)
                # _class_w = _class_w[1:]
            self.loss_func = nn.CrossEntropyLoss(_class_w)
        self.cl_loss_func = nn.CrossEntropyLoss()
        self.val_loss_func = nn.CrossEntropyLoss()

        

    def count_parameters(self):
        # Count the total number of parameters in the model
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        """ Initialize the weights """
        if module in [self.emb_layer, *self.transformer_layers]:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


    def forward(self, batch_seqs,batch_context, batch_static_context):
        # interaction_encoder options are lstm, sasrec, durorec
        if configs['model']['interaction_encoder'] == 'lstm':
            sasrec_out = self.interaction_encoder(batch_seqs)
        elif configs['model']['interaction_encoder'] == 'sasrec':
            mask = (batch_seqs > 0).unsqueeze(1).repeat(
                1, batch_seqs.size(1), 1).unsqueeze(1)
            x = self.emb_layer(batch_seqs)
            for transformer in self.transformer_layers:
                x = transformer(x, mask)

            # all_tokens_except_last = x[:, :-1, :]
            # last_token = x[:, -1, :]
            # print(all_tokens_except_last.size())
            sasrec_out = x.view(x.size(0), -1)
            
            sasrec_out = self.sasrecbn1(self.dropout(self.relu(self.sasrec_fc_layer1(sasrec_out))))
            sasrec_out = self.sasrecbn2(self.dropout(self.relu(self.sasrec_fc_layer2(sasrec_out))))
            sasrec_out = self.sasrecbn3(self.dropout(self.relu(self.sasrec_fc_layer3(sasrec_out))))
            # sasrec_out = x[:, -1, :]

        batch_context = batch_context.to(sasrec_out.dtype)
        context_output = self.context_encoder(batch_context)

        static_context = self.static_embedding(batch_static_context)

        context = torch.cat((context_output, static_context), dim=1)
        if configs['model']['encoder_combine'] == 'concat':
            out = torch.cat((sasrec_out, context), dim=1)
            out = self.fc_layers(out)
        if configs['model']['encoder_combine'] == 'attention':
            context = self.fc_context_dim_red(context)
            out, _ = self.multi_head_attention(sasrec_out, context, context)
            
        return out

    def cal_loss(self, batch_data):
        _, batch_seqs, batch_last_items, batch_time_deltas, batch_dynamic_context, batch_static_context, _ = batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)

        if configs['model']['interaction_encoder'] == 'lstm':
            test_item_emb = self.emb_layer.weight[:self.item_num+1]
        else:
            test_item_emb = self.emb_layer.token_emb.weight[:self.item_num+1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)

        if configs['train']['ssl']:
            aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs, batch_time_deltas)
            seq_output1 = self.forward(aug_seq1, batch_dynamic_context, batch_static_context)
            seq_output2 = self.forward(aug_seq2, batch_dynamic_context, batch_static_context)
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

    def val_cal_loss(self, val_batch_data):
        _, batch_seqs, batch_last_items, _, batch_dynamic_context, batch_static_context, _ = val_batch_data
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)

        if configs['model']['interaction_encoder'] == 'lstm':
            test_item_emb = self.emb_layer.weight[:self.item_num+1]
        else:
            test_item_emb = self.emb_layer.token_emb.weight[:self.item_num+1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.val_loss_func(logits, batch_last_items)

        cl_loss = 0
        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': cl_loss,
        }
        return loss + cl_loss, loss_dict

    def predict(self, batch_data):
        _, batch_seqs, batch_last_items, _, batch_dynamic_context, batch_static_context, _  = batch_data
        logits = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)
        
        if configs['model']['interaction_encoder'] == 'lstm':
            test_item_emb = self.emb_layer(batch_last_items)
        else:
            test_item_emb = self.emb_layer.token_emb(batch_last_items)
        test_item_emb = self.emb_layer(batch_last_items)

        scores = torch.mul(logits, test_item_emb).sum(dim=1)  
        return scores

    def full_predict(self, batch_data):
        # The method is responsible for generating predictions (scores) for items based on the given input sequences. It uses the learned representations from the model to calculate compatibility scores between the user and each item, providing a ranking of items for recommendation. This method is commonly used during the inference phase of a recommendation system.

        # Input Data:Similar to the cal_loss method, batch_data is expected to be a tuple containing three elements: batch_user, batch_seqs, and an ignored third element (_). These elements likely represent user identifiers, sequences of items, and some additional information.
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context, _  = batch_data
        # Sequential Output:Calls the forward method to obtain the output representation (logits) for the input sequences (batch_seqs).
        logits = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)
        # Compute Logits for All Items:Computes scores by performing matrix multiplication between the sequence output (logits) and the transpose of the embedding weights for items (test_item_emb). This operation calculates the compatibility scores between the user representations and representations of all items.
    
        if configs['model']['interaction_encoder'] == 'lstm':
            test_item_emb = self.emb_layer.weight[:self.item_num+1]
        else:
            test_item_emb = self.emb_layer.token_emb.weight[:self.item_num+1]
        # test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        # Return Predicted Scores:Returns the computed scores, which represent the predicted relevance or preference scores for each item in the vocabulary for the given batch of users and sequences.
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

    def _cl4srec_aug(self, batch_seqs, batch_time_deltas_seqs):
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

        ## TODO
        # set the following parameter as a param loaded from yaml file
        min_time_reorder = 0.5 #min

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