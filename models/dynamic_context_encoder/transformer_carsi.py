
import torch as t
import torch
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
from models.utils import weights_init
from models.utils import Flatten_layers

class TransformerEncoder_DynamicContext(nn.Module):
    def __init__(self):
        super(TransformerEncoder_DynamicContext, self).__init__()

        data_config = configs['data']
        model_config = configs['model']
        input_size_cont = data_config['dynamic_context_feat_num'] 
        output_size = model_config['item_embedding_size']
        seq_len = data_config['dynamic_context_window_length']
        hidden_dim=512, # d_model
        num_heads=8,


        model_config = configs['model']
        dropout_fc = model_config['dropout_rate_fc_tcn']
        

        feed_forward_size = 1024

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # print(input_size_cont)
    
        # Use linear layer instead of embedding 
        self.input_embedding = nn.Linear(10, 512)
        dtype = torch.float  # This is typically default but setting explicitly for clarity
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
        # self.input_embedding.weight = nn.Parameter(torch.empty((hidden_dim, input_size_cont), dtype=dtype, device=device))
        self.pos_enc = self.positional_encoding()

        # Multi-Head Attention
        self.multihead = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(512)
        self.layer_norm_2 = nn.LayerNorm(512)

        # position-wise Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.fc_out1 = nn.Linear(512, 64)

        self.fc1 = nn.Linear(64 * 30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc = Flatten_layers(1920, 64, dropout_p=dropout_fc)
        
    def positional_encoding(self):
        pe = torch.zeros(30, 512) # positional encoding 
        pos = torch.arange(0, 30, dtype=torch.float32).unsqueeze(1)
        _2i = torch.arange(0, 512, step=2).float()
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / 512)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / 512)))
        return pe

    def forward(self, x_cont):
        x_cont = x_cont.transpose(1,2)
        # Embedding + Positional
        x = self.input_embedding(x_cont)
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
        x_cont = x_cont.transpose(1,2)
        # shape: N, num_features, 64
        x = torch.flatten(x, 1)
        
        # Combine static and continupus
        # out = torch.cat((x, out_emb), dim=1)
        # out = self.relu(self.fc1(x))
        # out = self.fc2(out)
        out = self.fc(x)
        return out