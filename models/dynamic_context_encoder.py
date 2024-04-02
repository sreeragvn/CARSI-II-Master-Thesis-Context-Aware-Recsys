
import torch as t
from torch import nn
from config.configurator import configs
import torch.nn.functional as F


class LSTM_contextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, emb_size):
        super(LSTM_contextEncoder, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # Half the dimension
        self.fc2 = nn.Linear(hidden_size // 2, emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.apply(weights_init)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        _,(h_n, _) = self.lstm(x)
        h_n = h_n[-1]
        out = self.relu(self.fc1(h_n))
        out = self.dropout(out) 
        out = self.relu(self.fc2(out))
        out = self.dropout(out) 
        return out

# class TransformerEncoder_DynamicContext(nn.Module):
#     def __init__(
#             self, 
#             input_size_cont, # num_features_continuous
#             seq_len,
#             hidden_dim, # d_model
#             num_heads=8,
#     ):
#         super(TransformerEncoder_DynamicContext, self).__init__()

#         feed_forward_size = 1024
#         self.seq_len = seq_len
#         self.hidden_dim = hidden_dim 

#         # Use linear layer instead of embedding 
#         self.input_embedding = nn.Linear(input_size_cont, seq_len)
#         self.pos_enc = self.positional_encoding()

#         # Multi-Head Attention
#         self.multihead = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
#         self.dropout_1 = nn.Dropout(0.1)
#         self.dropout_2 = nn.Dropout(0.1)
#         self.layer_norm_1 = nn.LayerNorm(hidden_dim)
#         self.layer_norm_2 = nn.LayerNorm(hidden_dim)

#         # position-wise Feed Forward
#         self.feed_forward = nn.Sequential(
#             nn.Linear(hidden_dim, feed_forward_size),
#             nn.ReLU(),
#             nn.Linear(feed_forward_size, hidden_dim)
#         )
#         self.fc_out1 = nn.Linear(hidden_dim, 64)
#         self.apply(weights_init)

#     def positional_encoding(self):
#         pe = t.zeros(self.seq_len, self.hidden_dim) # positional encoding 
#         pos = t.arange(0, self.seq_len, dtype=t.float32).unsqueeze(1)
#         _2i = t.arange(0, self.hidden_dim, step=2).float()
#         pe[:, 0::2] = t.sin(pos / (10000 ** (_2i / self.hidden_dim)))
#         pe[:, 1::2] = t.cos(pos / (10000 ** (_2i / self.hidden_dim)))
#         return pe
        
#     def forward(self, x_cont):
        
#         # Embedding + Positional
#         # print(x_cont.size())
#         # print(self.input_embedding.weight.shape)
#         x = self.input_embedding(x_cont)
#         self.pos_enc = self.pos_enc.to(x.device)
#         x += self.pos_enc

#         # Multi-Head Attention
#         x_, _ = self.multihead(x,x,x)
#         x_ = self.dropout_1(x_)

#         # Add and Norm 1
#         x = self.layer_norm_1(x_ + x)

#         # Feed Forward
#         x_ = self.feed_forward(x)
#         x_ = self.dropout_2(x_)

#         # Add and Norm 2
#         x = self.layer_norm_2(x_ + x)
#         # Output (customized flatten)
#         x = self.fc_out1(x)
#         # shape: N, num_features, 64
#         x = t.flatten(x, start_dim=1)
#         self.output_dim = x.size(1)
#         # print(x.size())
#         # x = F.adaptive_avg_pool1d(x.transpose(1, 2), 64).transpose(1, 2)
#         return x
    

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)