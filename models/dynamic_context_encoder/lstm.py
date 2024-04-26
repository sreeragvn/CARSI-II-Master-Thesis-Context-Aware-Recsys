
import torch as t
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
from models.utils import weights_init

class lstm_context_encoder(nn.Module):
    def __init__(self):
        super(lstm_context_encoder, self).__init__()

        data_config = configs['data']
        model_config = configs['model']
        lstm_config = configs['lstm']

        input_size = data_config['dynamic_context_feat_num']
        hidden_size = lstm_config['lstm_hidden_size']
        num_layers = lstm_config['num_layers']
        emb_size = model_config['item_embedding_size']

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
