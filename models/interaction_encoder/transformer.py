import math
import torch as t
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
from models.utils import FlattenLayers

class IntTransformerLayer(nn.Module):
    def __init__(self):
        """
        Initialize the SASRec model.
        """
        super(IntTransformerLayer, self).__init__()

        data_config = configs['data']
        model_config = configs['model']

        self.item_num = data_config['item_num']
        self.emb_size = model_config['item_embedding_size']
        self.max_len = model_config['sasrec_max_seq_len']
        self.n_layers = model_config['sasrec_n_layers']
        self.n_heads = model_config['sasrec_n_heads']
        self.inner_size = 4 * self.emb_size

        self.dropout_rate_trans = model_config['dropout_rate_sasrec']
        self.dropout_rate_fc_trans = model_config['dropout_rate_fc_sasrec']
        
        self.emb_layer = TransformerEmbedding(item_num=self.item_num + 1, 
                                              emb_size=self.emb_size, 
                                              max_len=self.max_len)
            
        self.transformer_layers = nn.ModuleList([TransformerLayer(hidden_size=self.emb_size, 
                                                                  num_heads=self.n_heads, 
                                                                  feed_forward_size=self.inner_size, 
                                                                  dropout_rate=self.dropout_rate_trans) 
                                                 for _ in range(self.n_layers)])
        
        self.fc_layers = FlattenLayers(input_size=(self.max_len) * self.emb_size, 
                                        emb_size=self.emb_size, 
                                        dropout_p=self.dropout_rate_fc_trans)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if module in [self.emb_layer, *self.transformer_layers]:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch_seqs):
        """
        Forward pass for the SASRec model.
        
        Args:
            batch_seqs (Tensor): Batch of input sequences.
        
        Returns:
            Tensor: Output of the model.
        """
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        sasrec_out = x.view(x.size(0), -1)
        sasrec_out = self.fc_layers(sasrec_out)
        return sasrec_out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        """
        Multi-head attention mechanism.
        
        Args:
            num_heads (int): Number of attention heads.
            hidden_size (int): Size of the hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        assert hidden_size % num_heads == 0

        self.d_k = hidden_size // num_heads
        self.n_h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def _cal_attention(self, query, key, value, mask=None, dropout=None):
        """
        Calculate attention scores and apply attention to values.
        
        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            mask (Tensor, optional): Mask tensor.
            dropout (function, optional): Dropout function.
        
        Returns:
            Tensor: Attention output.
            Tensor: Attention scores.
        """
        scores = t.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return t.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            mask (Tensor, optional): Mask tensor.
        
        Returns:
            Tensor: Output tensor after attention.
        """
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

        return self.output_linear(x)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        """
        Position-wise feed-forward network.
        
        Args:
            hidden_size (int): Size of the hidden layer.
            d_ff (int): Size of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass for the position-wise feed-forward network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after feed-forward processing.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        """
        Residual connection module with layer normalization and dropout.
        
        Args:
            hidden_size (int): Size of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        
        Args:
            x (Tensor): Input tensor.
            sublayer (function): Sublayer to apply.
        
        Returns:
            Tensor: Output tensor after applying residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_size, dropout_rate):
        """
        Transformer layer consisting of multi-head attention and position-wise feed-forward network.
        
        Args:
            hidden_size (int): Size of the hidden layer.
            num_heads (int): Number of attention heads.
            feed_forward_size (int): Size of the feed-forward layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout_rate)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, d_ff=feed_forward_size, dropout=dropout_rate)
        self.input_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.output_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        """
        Forward pass for the transformer layer.
        
        Args:
            x (Tensor): Input tensor.
            mask (Tensor): Mask tensor.
        
        Returns:
            Tensor: Output tensor after transformer layer processing.
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class TransformerEmbedding(nn.Module):
    def __init__(self, item_num, emb_size, max_len, dropout=0.1):
        """
        Transformer embedding layer with token and positional embeddings.
        
        Args:
            item_num (int): Number of items.
            emb_size (int): Embedding size.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.token_emb = nn.Embedding(item_num, emb_size, scale_grad_by_freq=True, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

    def forward(self, batch_seqs):
        """
        Forward pass for the embedding layer.
        
        Args:
            batch_seqs (Tensor): Batch of input sequences.
        
        Returns:
            Tensor: Output tensor after applying embeddings and dropout.
        """
        batch_size = batch_seqs.size(0)
        pos_emb = self.position_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        return self.dropout(x)
