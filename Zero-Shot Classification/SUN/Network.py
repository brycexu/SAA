# Author : Bryce Xu
# Time : 2020/2/19
# Function: 

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

class FeatureNetwork(nn.Module):
    def __init__(self, network='googlenet'):
        super(FeatureNetwork, self).__init__()
        self.network = nn.Sequential(
            GoogleNet(network),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.network(x)
        return output

class GoogleNet(nn.Module):
    def __init__(self, network):
        super(GoogleNet, self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)
        net_list = list(net.children())
        self.encoder = nn.Sequential(*net_list[:-2])

    def forward(self, x):
        x = self.encoder(x)
        output = x.view(x.size(0), -1)
        return output

class SynthesisNetwork(nn.Module):
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(202, 613),
            nn.BatchNorm1d(613),
            nn.ReLU(),
            nn.Linear(613, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, attribute, embedding):
        x = torch.cat((attribute, embedding), 1)
        x = self.network(x)
        output = F.normalize(x, dim=1)
        return output
