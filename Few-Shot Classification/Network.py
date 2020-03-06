# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

import torch.nn as nn
from SAAM import SAM, SAAM
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        SAM(),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_block2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels)
    )

def conv_block3():
    return nn.Sequential(
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class SimpleNetwork(nn.Module):
    def __init__(self, in_dim=3, hid_dim=64, out_dim=64):
        super(SimpleNetwork, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block2(hid_dim, out_dim)
        )
        self.ssam = SAAM(gate_channels=64)
        self.rear = conv_block3()

    def forward(self, x, x_emb=None, x_emb_targets=None, wordEmbedding=False):
        x = self.encoder(x)
        if not wordEmbedding:
            x = self.ssam(x)
            x = self.rear(x)
            output = x.view(x.size(0), -1)
            return output
        else:
            x, loss = self.ssam(x, x_emb, x_emb_targets, wordEmbedding)
            x = self.rear(x)
            output = x.view(x.size(0), -1)
        return output, loss

class SAAMNetwork(nn.Module):

    def __init__(self):
        super(SAAMNetwork, self).__init__()
        self.simpleNetwork = SimpleNetwork()

    def forward(self, x, y, x_emb, n_class, n_support):
        x_emb_targets = torch.stack([x_emb[i:i + n_support].mean(0) for i in range(0, len(x), n_support)])
        x, loss = self.simpleNetwork(x, x_emb, x_emb_targets, wordEmbedding=True)
        y = self.simpleNetwork(y, wordEmbedding=False)
        return x, y, loss
