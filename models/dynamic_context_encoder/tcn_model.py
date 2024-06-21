import torch
import torch.nn as nn
from torch.nn.utils import parametrizations
import torch.nn.functional as F
from config.configurator import configs
from models.utils import Flatten_layers
from models.utils import weights_init

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        Chomps (truncates) the last elements in the sequence to ensure causality.
        
        Args:
            chomp_size (int): Number of elements to remove from the end of the sequence.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Temporal block consisting of two convolutional layers, each followed by Chomp1d, ReLU, and Dropout layers.
        
        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            dilation (int): Dilation factor of the convolution.
            padding (int): Padding size for the convolution.
            dropout (float): Dropout rate.
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
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
        """
        Initialize weights of the convolutional layers.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass of the temporal block.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying temporal block layers.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        """
        Temporal Convolutional Network (TCN) consisting of multiple TemporalBlocks.
        
        Args:
            input_size (int): Number of input channels.
            num_channels (list): List of output channels for each temporal block.
            kernel_size (int): Size of the convolutional kernel. Default is 2.
            dropout (float): Dropout rate. Default is 0.2.
        """
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

    def forward(self, x):
        """
        Forward pass of the TCN.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying TCN layers.
        """
        return self.network(x)
    
class TCNModel(nn.Module):
    def __init__(self):
        """
        Temporal Convolutional Network (TCN) model for sequential data.
        """
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

        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = Flatten_layers(num_channels[-1] * dynamic_context_window_size, emb_size, dropout_p=dropout_fc)

    def forward(self, x):
        """
        Forward pass of the TCN model.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying TCN and fully connected layers.
        """
        out = self.tcn(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
