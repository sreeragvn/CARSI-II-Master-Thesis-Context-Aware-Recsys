import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.special import lambertw
import numpy as np
from config.configurator import configs
import pickle

def loss_function():
    """
    Select and return the appropriate loss function based on the configuration.

    Returns:
        loss_func (nn.Module): The primary loss function.
        cl_loss_func (nn.Module): The contrastive learning loss function.
    """
    gamma = configs['train']['focal_loss_gamma']
    
    if not configs['train']['weighted_loss_fn']:
        if configs['train']['focal_loss']:
            loss_func = FocalLossAdaptive(gamma=gamma)
        else:
            loss_func = nn.CrossEntropyLoss()
    else:
        with open(configs['train']['parameter_class_weights_path'], 'rb') as f:
            _class_w = pickle.load(f)
        if configs['train']['focal_loss']:
            loss_func = FocalLossAdaptive(gamma=gamma)
        else:
            loss_func = nn.CrossEntropyLoss(_class_w)
    
    cl_loss_func = nn.CrossEntropyLoss()
    return loss_func, cl_loss_func

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        """
        Initialize FocalLoss module.

        Args:
            gamma (float): Focusing parameter gamma.
            alpha (list or Tensor): Class weights.
            size_average (bool): Whether to average the loss.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        """
        Forward pass for the FocalLoss.

        Args:
            input (Tensor): Input predictions.
            target (Tensor): Target labels.

        Returns:
            Tensor: Calculated loss.
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

def get_gamma(p=0.2):
    """
    Calculate the gamma for a given pt where the function g(p, gamma) = 1.

    Args:
        p (float): Probability threshold.

    Returns:
        float: Calculated gamma.
    """
    y = ((1 - p) ** (1 - (1 - p) / (p * np.log(p))) / (p * np.log(p))) * np.log(1 - p)
    gamma_complex = (1 - p) / (p * np.log(p)) + lambertw(-y + 1e-12, k=-1) / np.log(1 - p)
    return np.real(gamma_complex)

# Predefined gamma values for different probabilities
ps = [0.2, 0.5]
gammas = [5.0, 3.0]
gamma_dic = {p: gammas[i] for i, p in enumerate(ps)}

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        """
        Initialize FocalLossAdaptive module.

        Args:
            gamma (float): Base focusing parameter gamma.
            size_average (bool): Whether to average the loss.
        """
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = configs['device']

    def get_gamma_list(self, pt):
        """
        Generate a list of gamma values based on the prediction probabilities.

        Args:
            pt (Tensor): Prediction probabilities.

        Returns:
            Tensor: Tensor of gamma values.
        """
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if pt_sample >= 0.5:
                gamma_list.append(self.gamma)
            else:
                for key in sorted(gamma_dic.keys()):
                    if pt_sample < key:
                        gamma_list.append(gamma_dic[key])
                        break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        """
        Forward pass for the FocalLossAdaptive.

        Args:
            input (Tensor): Input predictions.
            target (Tensor): Target labels.

        Returns:
            Tensor: Calculated loss.
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1 - pt) ** gamma * logpt
        return loss.mean() if self.size_average else loss.sum()
