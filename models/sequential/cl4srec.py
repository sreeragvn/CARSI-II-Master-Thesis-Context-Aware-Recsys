import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding, LSTM_contextEncoder, CARSI_contextEncoder
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import pickle


class CL4SRec(BaseModel):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    """

    def __init__(self, data_handler):
        super(CL4SRec, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1
        # load parameters info
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        # the dimensionality in feed-forward layer
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']

        self.batch_size = configs['train']['batch_size']
        self.lmd = configs['model']['lmd']
        self.tau = configs['model']['tau']

        with open(configs['train']['parameter_path'], 'rb') as f:
            _class_w = pickle.load(f)

        self.lstm_input_size = configs['lstm']['input_size']
        self.lstm_hidden_size = configs['lstm']['hidden_size']
        self.lstm_num_layers = configs['lstm']['num_layers']
        # self.lstm_output_size = configs['lstm']['output_size']

        # Todo should we embed everything to same space or different space ? how do we select the embedding size ?
        out_emb = 64
        self.static_embedding = nn.Embedding(self.emb_size, out_emb)

        # input_size_cont = configs['data']['input_size']# num_features_continuous
        # input_size_cat = configs['data']['input_size']
        # output_size = configs['carsii']['input_size']
        # seq_len = configs['lstm']['input_size']
        # num_embedding = configs['model']['embedding_size']
        # hidden_dim=512, # d_model
        # num_heads=8

        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.emb_size, self.max_len)

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss(weight =_class_w)

        if configs['model']['context_encoder'] == 'lstm':
            self.context_encoder = LSTM_contextEncoder(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers, self.batch_size)
        
        # if configs['model']['context_encoder'] == 'carsi_context_encoder':
        #     self.context_encoder = CARSI_contextEncoder(input_size_cont, # num_features_continuous
        #                                                     input_size_cat,
        #                                                     output_size,
        #                                                     seq_len,
        #                                                     num_embedding,
        #                                                     hidden_dim=512, # d_model
        #                                                     num_heads=8)
                                                        
        self.fc1 = nn.Linear(192, 128) # number of static and dynamic context variables
        self.fc2 = nn.Linear(128, self.emb_size)
        self.relu = nn.ReLU()

        self.mask_default = self.mask_correlated_samples(
            batch_size=self.batch_size)
        self.cl_loss_func = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
        min_time_reorder = 3 #min

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

        # aug_seq1 = torch.tensor(
        #     aug_seq1, dtype=torch.long, device=batch_seqs.device)
        # aug_seq2 = torch.tensor(
        #     aug_seq2, dtype=torch.long, device=batch_seqs.device)
        # return aug_seq1, aug_seq2
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

    def forward(self, batch_seqs,batch_context, batch_static_context):
        # print("Input Shapes:")
        # print("batch_seqs:", batch_seqs.shape)
        # print("batch_context:", batch_context.shape)
        # method processes input sequences through an embedding layer and a stack of transformer layers, and the final output is the representation of the sequence, typically extracted from the last position. The mask is used to prevent the model from attending to padding elements during the transformer layers' computations.

        # batch_seqs > 0 creates a boolean tensor indicating non-padding elements.
        # .unsqueeze(1) adds a singleton dimension to the tensor to make it compatible for broadcasting.
        # .repeat(1, batch_seqs.size(1), 1) replicates the tensor along the sequence dimension, essentially creating a 3D mask with the same shape as batch_seqs.
        # .unsqueeze(1) adds another singleton dimension at the beginning of the tensor. This is often used for compatibility with transformer models that expect a mask with dimensions [batch_size, 1, sequence_length, sequence_length].
        # Todo - This has to be done for the context as well. Ensure the padding is done with a negative number. not zero. since zero speed itself is relevant.
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        # Embedding Layer:
        # Passes the input sequence batch_seqs through an embedding layer (self.emb_layer). This layer converts integer indices into dense vectors.
        x = self.emb_layer(batch_seqs)

        # Transformer Layers:
        # Iterates through a series of transformer layers (self.transformer_layers) and applies each one to the input tensor x. The transformer layers are expected to take the input tensor and the mask as arguments.
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        # Extracts the output from the last position of the sequence (-1). This is a common practice in transformer-based models, where the output corresponding to the last position is often used as a representation of the entire sequence.
        sasrec_out = x[:, -1, :]

        batch_context = batch_context.to(sasrec_out.dtype)
        batch_context = batch_context.transpose(1, 2)
        context_output = self.context_encoder(batch_context)
        
        static_context = self.static_embedding(batch_static_context)
        static_context = static_context.view(batch_static_context.size(0), -1)

        # print("sasrec_out Rank:", context_output.dim(), context_output.ndim)
        # print("context_output Rank:",  sasrec_out.dim(), sasrec_out.ndim)

        # print("sasrec_out size:", sasrec_out.size())
        # print("context_output size:", context_output.size())
        # print("context_output size:", static_context.size())
        out = torch.cat((sasrec_out, context_output, static_context), dim=1)

        # Concatenate along the feature dimension (axis=1)
        # final_out = torch.cat((sasrec_out, context_output), dim=1)
        # print("Final Output Shape:", final_out.shape)
        # print("output size:", out.size())
        # out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        output = self.relu(out)
        # print("output size:", output.size())

        return output  # [B H]

    def cal_loss(self, batch_data):
        # The method computes the total loss for a recommendation system, which includes a recommendation loss based on the last items in sequences and a contrastive loss using augmented sequences for contrastive learning. This approach aims to learn meaningful representations for recommendation by leveraging both sequential patterns and contrastive learning principles.

        # Input Data:The input batch_data is assumed to be a tuple containing three elements: batch_user, batch_seqs, and batch_last_items. These likely represent user identifiers, sequences of items, and the last items in those sequences, respectively.
        _, batch_seqs, batch_last_items, batch_time_deltas, batch_dynamic_context, batch_static_context = batch_data
        # Sequential Output:Calls the forward method (previously explained) to obtain the output representation (seq_output) for the input sequences (batch_seqs).
        seq_output = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)
        # Compute Logits:Computes logits by performing matrix multiplication between the sequence output (seq_output) and the transpose of the embedding weights for items (test_item_emb). This operation is often used in recommendation systems to calculate the compatibility scores between user representations and item representations.
        # Todo why you are adding + 1 to  item_num when slicing
        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # print(batch_last_items.size(), seq_output.size(), test_item_emb.size(), logits.size())
        # Compute Recommendation Loss:Computes the recommendation loss using a specified loss function (self.loss_func). This loss measures the discrepancy between the predicted logits and the actual last items in the sequences.
        loss = self.loss_func(logits, batch_last_items)
        # Contrastive Learning (NCE):Generates augmented sequences (aug_seq1 and aug_seq2) using the _cl4srec_aug method (not provided). These augmented sequences are then processed through the model to obtain representations (seq_output1 and seq_output2).
        # NCE
        if configs['train']['ssl']:
            aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs, batch_time_deltas)
            seq_output1 = self.forward(aug_seq1, batch_dynamic_context, batch_static_context)
            seq_output2 = self.forward(aug_seq2, batch_dynamic_context, batch_static_context)
            # Compute InfoNCE Loss (Contrastive Loss):Computes the InfoNCE loss (contrastive loss) between the representations of the augmented sequences. The temperature parameter (temp) and batch size are specified.
            cl_loss = self.lmd * self.info_nce(
                seq_output1, seq_output2, temp=self.tau, batch_size=aug_seq1.shape[0])
            # Aggregate Losses and Return: Aggregates the recommendation loss and contrastive loss into a total loss. Returns the total loss along with a dictionary containing individual loss components (rec_loss and cl_loss).
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
        # The method is responsible for generating predictions (scores) for items based on the given input sequences. It uses the learned representations from the model to calculate compatibility scores between the user and each item, providing a ranking of items for recommendation. This method is commonly used during the inference phase of a recommendation system.

        # Input Data:Similar to the cal_loss method, batch_data is expected to be a tuple containing three elements: batch_user, batch_seqs, and an ignored third element (_). These elements likely represent user identifiers, sequences of items, and some additional information.
        _, batch_seqs, _, _, batch_dynamic_context, batch_static_context = batch_data
        # Sequential Output:Calls the forward method to obtain the output representation (logits) for the input sequences (batch_seqs).
        logits = self.forward(batch_seqs, batch_dynamic_context, batch_static_context)
        # Compute Logits for All Items:Computes scores by performing matrix multiplication between the sequence output (logits) and the transpose of the embedding weights for items (test_item_emb). This operation calculates the compatibility scores between the user representations and representations of all items.
        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        # Return Predicted Scores:Returns the computed scores, which represent the predicted relevance or preference scores for each item in the vocabulary for the given batch of users and sequences.
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
