import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from config.configurator import configs
from models.utils import Flatten_layers
from models.utils import weights_init

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x = x.permute(0, 2, 1) # Changing (batch_size, features, num_time_instances) to (batch_size, num_time_instances, features)
        x = self.network(x)
        # x = x[:, :, -1] # Taking the last slice from the sequence dimension
        # x = self.linear(x)
        return x
    
class TCNModel(nn.Module):
    def __init__(self):
        super(TCNModel, self).__init__()

        model_config = configs['model']
        data_config = configs['data']
        emb_size = model_config['item_embedding_size']
        num_channels = model_config['tcn_num_channels']
        kernel_size = model_config['tcn_kernel_size']
        num_input = data_config['dynamic_context_feat_num']
        dynamic_context_window_size = data_config['dynamic_context_window_length']
        dropout = model_config['dropout_rate_tcn']
        dropout_fc = model_config['dropout_rate_fc_tcn']

        # self.tcn = TemporalConvNet(num_input, num_channels[-1], num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        # self.tcn = TemporalConvNet(
        #     num_input, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = Flatten_layers(num_channels[-1]*dynamic_context_window_size, emb_size, dropout_p=dropout_fc)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # print(out.size())
        # print('tcn ouput', out.size())
        # out =  F.avg_pool1d(out, kernel_size=4)
        # out = out.view(x.size(0), -1)
        out = torch.flatten(out, 1) # flatten all dimensions except batch
        # print('flat tcn out size', out.size())
        out = self.fc(out)
        # print('tcn out after fc', out.size())
        return out