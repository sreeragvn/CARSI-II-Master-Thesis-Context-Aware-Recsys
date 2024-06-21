import torch.nn as nn

class Flatten_layers(nn.Module):
    def __init__(self, input_size, emb_size, dropout_p=0.4):
        """
        Initialize the Flatten_layers module which progressively reduces the dimensionality
        of the input feature vector to the desired embedding size.

        Args:
            input_size (int): Size of the input feature vector.
            emb_size (int): Size of the output embedding.
            dropout_p (float): Dropout probability. Default is 0.4.
        """
        super(Flatten_layers, self).__init__()
        self.emb_size = emb_size
        self.dropout_p = dropout_p
        layers = []

        if input_size // 2 > self.emb_size and input_size > self.emb_size:
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
        """
        Forward pass for the Flatten_layers module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the sequential layers.
        """
        return self.layers(x)

def weights_init(m):
    """
    Initialize the weights of the neural network layers.

    Args:
        m (nn.Module): Neural network layer.
    """
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
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
