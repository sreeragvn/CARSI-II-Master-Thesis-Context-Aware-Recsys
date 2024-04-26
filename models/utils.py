from torch import nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight.data, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

class Flatten_layers(nn.Module):
    def __init__(self, input_size,  emb_size, dropout_p=0.4):
        super(Flatten_layers, self).__init__()
        self.emb_size = emb_size
        self.dropout_p = dropout_p
        layers = []
        if input_size//2 > self.emb_size and input_size > self.emb_size :
            layers.append(nn.Linear(input_size, input_size // 2))
            layers.append(nn.BatchNorm1d(input_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_p))
            input_size = input_size // 2
            while input_size > self.emb_size:
                output_size = max(self.emb_size, input_size // 2)
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.BatchNorm1d(output_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout_p))
                input_size = output_size
        else:
            layers.append(nn.Linear(input_size, self.emb_size))
            layers.append(nn.BatchNorm1d(self.emb_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_p))
        self.layers = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        x = self.layers(x)
        return x