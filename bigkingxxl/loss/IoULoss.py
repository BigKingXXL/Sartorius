"""
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
LICENSE: APACHE 2.0 - https://www.apache.org/licenses/LICENSE-2.0
"""

import torch.nn as nn
from torch import sigmoid

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = sigmoid(inputs)
        size = inputs.size()
        #flatten label and prediction tensors
        inputs = inputs.reshape(size[0], -1)
        targets = targets.reshape(size[0], -1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU