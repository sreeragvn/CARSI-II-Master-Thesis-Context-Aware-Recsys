import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_input, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            num_input, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # self.decoder = nn.Linear(num_channels[-1], 1)
        self.fc = FlattenLinear(2500, [1024, 512, 256, 128], 64, dropout_p=0.4)

    def forward(self, x):
        # out = self.tcn(x)[:, :, -1]
        # x = x.permute(0, 2, 1)
        # print(x.size())
        out = self.tcn(x).view(x.size(0), -1)
        out = self.fc(out)
        # out = self.dropout(out)
        # out = self.decoder(out)
        return out
        
class FlattenLinear(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.1):
        super(FlattenLinear, self).__init__()
        layers = []
        # Add input layer with batch normalization and dropout
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Add batch normalization
        layers.append(nn.ReLU())  # Add activation function
        layers.append(nn.Dropout(dropout_p))  # Add dropout
        
        # Add hidden layers with batch normalization and dropout
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))  # Add batch normalization
            layers.append(nn.ReLU())  # Add activation function
            layers.append(nn.Dropout(dropout_p))  # Add dropout
        
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        return self.model(x)
