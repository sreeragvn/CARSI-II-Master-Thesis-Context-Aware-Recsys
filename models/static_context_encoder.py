import torch as t
from torch import nn
from config.configurator import configs
import math

def weights_init(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

class static_context_encoder(nn.Module):
    def __init__(self, vocab_sizes):
        super(static_context_encoder, self).__init__()
        self.embedding_layers = nn.ModuleList() 
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=12, month = True)) #  months
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=7)) #  weekday
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=24)) #  hour
        self.embedding_layers.extend([nn.Embedding(num_embeddings=max_val+1, 
                                                    embedding_dim=2) 
                                                    for max_val in  vocab_sizes[-3:]])# last 3 values are the non cyclical one
        
        self.bn = nn.BatchNorm1d(12) 
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)]
        output = t.cat(embedded, dim=1)
        output = self.bn(output)
        output = self.relu(output)
        output = self.dropout(output)
        return output

class CyclicalEmbedding(nn.Module):
    def __init__(self, max_value_scale, month = False):
        super(CyclicalEmbedding, self).__init__()
        self.max_value = max_value_scale
        self.month = month

    def forward(self, x):
        if self.month:
            x = x - 1
        # Convert input to radians
        x = (2. * math.pi * x) / self.max_value
        emb = t.stack((t.sin(x), t.cos(x)), dim=1)
        return emb

    
# import torch as t
# from torch import nn
# from config.configurator import configs

# def weights_init(m):
#     if isinstance(m, nn.Embedding):
#         nn.init.uniform_(m.weight.data, -0.1, 0.1)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight.data)
#         nn.init.constant_(m.bias.data, 0.0)

# class static_context_encoder(nn.Module):
#     def __init__(self, vocab_sizes, embedding_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(static_context_encoder, self).__init__()
#         self.embedding_layers = nn.ModuleList([nn.Embedding(num_embeddings=max_val + 1, 
#                                                             embedding_dim=embedding_dim) 
#                                                             for max_val in vocab_sizes])
#         total_embedding_dim = embedding_dim * len(vocab_sizes)
#         self.fc1 = nn.Linear(total_embedding_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second fully connected layer
#         self.output_layer = nn.Linear(hidden_dim2, output_dim)
#         self.dropout = nn.Dropout(p=0.2)
#         self.relu = nn.ReLU()
#         self.apply(weights_init)

#     # def forward(self, x):
#     #     embedded = [
#     #         emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)
#     #     ]
#     #     embedded = t.cat(embedded, dim=1)  # Concatenate embeddings along dimension 1
#     #     fc1_out = t.relu(self.fc1(embedded))
#     #     fc2_out = t.relu(self.fc2(fc1_out))  # Pass through the second FC layer
#     #     output = self.output_layer(fc2_out)
#     #     return output
    
#     def forward(self, x):
#         embedded = [
#             emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)
#         ]
#         embedded = t.cat(embedded, dim=1)
#         fc1_out = self.relu(self.fc1(embedded))
#         fc1_out = self.dropout(fc1_out) 
#         fc2_out = self.relu(self.fc2(fc1_out))
#         fc2_out = self.dropout(fc2_out) 
#         output = self.relu(self.output_layer(fc2_out))
#         output = self.dropout(output) 
#         return output
