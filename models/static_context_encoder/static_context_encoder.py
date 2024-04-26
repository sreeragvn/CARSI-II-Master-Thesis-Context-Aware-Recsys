import torch as t
from torch import nn
from config.configurator import configs
import math
from models.utils import weights_init


class static_context_encoder(nn.Module):
    def __init__(self):
        super(static_context_encoder, self).__init__()

        dropout_rate_fc_static = configs['model']['dropout_rate_fc_static']
        vocab_sizes = configs['data']['static_context_max']
        self.embedding_layers = nn.ModuleList() 
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=12, month = True)) #  months
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=7)) #  weekday
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=24)) #  hour
        self.embedding_layers.extend([nn.Embedding(num_embeddings=max_val+1, 
                                                    embedding_dim=2) 
                                                    for max_val in  vocab_sizes[3:]])# last 3 values are the non cyclical one
        output_size = 32
        self.linear = nn.Linear(3,3)
        self.linear2 = nn.Linear(len(vocab_sizes)*2+3,output_size)
        self.bn1 = nn.BatchNorm1d(len(vocab_sizes)*2+3) 
        self.bn2 = nn.BatchNorm1d(output_size) 
        self.dropout = nn.Dropout(p=dropout_rate_fc_static) 
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x, z):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)]
        embedded = t.cat(embedded, dim=1)
        linear = self.linear(z)
        output = t.cat((embedded, linear), dim=1)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.bn2(output)
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