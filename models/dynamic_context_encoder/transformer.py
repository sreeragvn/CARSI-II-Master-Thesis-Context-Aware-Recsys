
import torch as t
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
from models.utils import weights_init

class TransformerEncoder_DynamicContext(nn.Module):
    def __init__(self):
        super(TransformerEncoder_DynamicContext, self).__init__()

        data_config = configs['data']
        model_config = configs['model']
        input_size_cont = data_config['dynamic_context_feat_num'] 
        self.seq_len = data_config['dynamic_context_window_length']
        self.hidden_dim = model_config['item_embedding_size']

        feed_forward_size = 1024
        num_heads=8

        # Use linear layer instead of embedding 
        self.input_embedding = nn.Linear(input_size_cont, self.seq_len)
        self.pos_enc = self.positional_encoding()

        # Multi-Head Attention
        self.multihead = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)

        # position-wise Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, self.hidden_dim)
        )
        self.fc_out1 = nn.Linear(self.hidden_dim, 64)
        self.apply(weights_init)

    def positional_encoding(self):
        pe = t.zeros(self.seq_len, self.hidden_dim) # positional encoding 
        pos = t.arange(0, self.seq_len, dtype=t.float32).unsqueeze(1)
        _2i = t.arange(0, self.hidden_dim, step=2).float()
        pe[:, 0::2] = t.sin(pos / (10000 ** (_2i / self.hidden_dim)))
        pe[:, 1::2] = t.cos(pos / (10000 ** (_2i / self.hidden_dim)))
        return pe
        
    def forward(self, x_cont):
        
        # Embedding + Positional
        # print(x_cont.size())
        # print(self.input_embedding.weight.shape)
        x = self.input_embedding(x_cont)
        self.pos_enc = self.pos_enc.to(x.device)
        x += self.pos_enc

        # Multi-Head Attention
        x_, _ = self.multihead(x,x,x)
        x_ = self.dropout_1(x_)

        # Add and Norm 1
        x = self.layer_norm_1(x_ + x)

        # Feed Forward
        x_ = self.feed_forward(x)
        x_ = self.dropout_2(x_)

        # Add and Norm 2
        x = self.layer_norm_2(x_ + x)
        # Output (customized flatten)
        x = self.fc_out1(x)
        # shape: N, num_features, 64
        x = t.flatten(x, start_dim=1)
        self.output_dim = x.size(1)
        # print(x.size())
        # x = F.adaptive_avg_pool1d(x.transpose(1, 2), 64).transpose(1, 2)
        return x