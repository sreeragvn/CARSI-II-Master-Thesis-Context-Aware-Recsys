import torch as t
from torch import nn
from config.configurator import configs
import math
from models.utils import weights_init

class static_context_encoder(nn.Module):
    def __init__(self):
        """
        Static context encoder to handle static contextual features using embedding layers.
        """
        super(static_context_encoder, self).__init__()

        dropout_rate_fc_static = configs['model']['dropout_rate_fc_static']
        vocab_sizes = configs['data']['static_context_max']
        
        # Embedding layers for cyclical and non-cyclical features
        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=12, month=True))  # months
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=7))  # weekday
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=24))  # hour
        self.embedding_layers.append(CyclicalEmbedding(max_value_scale=4))  # season
        self.embedding_layers.extend([nn.Embedding(num_embeddings=max_val + 1, embedding_dim=2) 
                                      for max_val in vocab_sizes[4:]])  # non-cyclical features
        
        output_size = 32
        self.linear = nn.Linear(3, 3)
        self.linear2 = nn.Linear(len(vocab_sizes) * 2 + 3, output_size)
        self.bn1 = nn.BatchNorm1d(len(vocab_sizes) * 2 + 3)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(p=dropout_rate_fc_static)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.apply(weights_init)

    def forward(self, x, z):
        """
        Forward pass for the static context encoder.
        
        Args:
            x (Tensor): Input tensor containing static context features.
            z (Tensor): Additional input tensor.
        
        Returns:
            Tensor: Output tensor after encoding static context features.
        """
        # Apply embedding layers to each feature in x
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)]
        embedded = t.cat(embedded, dim=1)
        
        # Process additional input z through a linear layer
        linear = self.linear(z)
        
        # Concatenate embedded features and linear output
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
    def __init__(self, max_value_scale, month=False):
        """
        Cyclical embedding layer for cyclical features such as month, weekday, hour, and season.
        
        Args:
            max_value_scale (int): Maximum value of the cyclical feature.
            month (bool): Whether the feature is month (1-12). Default is False.
        """
        super(CyclicalEmbedding, self).__init__()
        self.max_value = max_value_scale
        self.month = month

    def forward(self, x):
        """
        Forward pass for the cyclical embedding layer.
        
        Args:
            x (Tensor): Input tensor containing cyclical feature values.
        
        Returns:
            Tensor: Output tensor with sine and cosine embeddings.
        """
        if self.month:
            x = x - 1  # Adjust month to 0-11 range
        # Convert input to radians
        x = (2. * math.pi * x) / self.max_value
        emb = t.stack((t.sin(x), t.cos(x)), dim=1)
        
        return emb
