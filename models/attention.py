import torch as t
import torch
from torch import nn
from config.configurator import configs
from models.interaction_encoder.sasrec import sasrec
import torch.nn.functional as F
    
class OA(nn.Module):
    """
    Offset-Attention Module.
    """
   
    def __init__(self, channels):
        super(OA, self).__init__()
        # channels = configs['model']['item_embedding_size']
        self.linear_static = nn.Conv1d(39, channels , 1, bias=False)
        self.linear_dynamic = nn.Conv1d(10, channels , 1, bias=False)
        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.s_conv = nn.Conv1d(channels, channels,  1, bias=False) 
        self.trans_conv = nn.Conv1d(channels , channels , 1, bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2
        # self.interaction_encoder = sasrec()
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, out):
        # print(dynamic.size())
        """
        Input:
            static: [B, 21, 1]
            dynamic: [B, 64, 30]
            interaction: [B, 10, 1]
       
        Output:
            x: [B, de, N]
        """
        # print(static.size(), dynamic.size(), interaction.size())
        # interaction = self.interaction_encoder(interaction)
        # static_l = F.pad(static, (0, 32))
        # static_l = static_l.unsqueeze(2)
        # static_l = static_l.transpose(1,2)
        # static_l=self.linear_static(static_l)
        # # dynamic = self.linear_dynamic(dynamic)
        # interaction_l = interaction.transpose(1,2)
        # # print(static_l.size(), dynamic.size(), interaction_l.size())
        # interaction = interaction.unsqueeze(2).float()
        #interaction_l=self.linear_interaction(interaction)
        #static_l: [B, 64, 1]
        #interaction_l: [B, 64, 1]
        #dynamic: [B, 64, 30]
        
        # [B, 64, 32]
        attention_matrix = self.attention(out)
        out = attention_matrix + out
        attention_matrix = self.attention(out)
        attention_matrix = attention_matrix + out
        return attention_matrix

    def attention(self, Stake):
        x_q = self.q_conv(Stake)
        x_k = self.k_conv(Stake)

        x_v = self.v_conv(Stake)
        energy = torch.bmm(x_q.transpose(1, 2), x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here
 
        x = torch.bmm(x_v, attention.transpose(1, 2))
        x = self.act(self.after_norm(self.trans_conv(x)))
 
        return x