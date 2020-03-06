# Author : Bryce Xu
# Time : 2020/1/17
# Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.SpatialGate(x)
        return x_out

class SAAM(nn.Module):
    def __init__(self, gate_channels):
        super(SAAM, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.linear = nn.Sequential(
            nn.Linear(100, gate_channels),
            nn.Dropout(0.5),
            nn.Linear(gate_channels, gate_channels),
            nn.BatchNorm1d(gate_channels),
            nn.ReLU()
        )
        self.gate_channels = gate_channels

    def forward(self, x, x_emb=None, x_emb_targets=None, wordEmbedding=False):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)  # (N,1,f,f)
        if not wordEmbedding:
            scale = torch.sigmoid(x_out)  # broadcasting
            return x * scale
        else:
            # x:(N,gate_chanels,filter_size,filter_size)
            x_process = x.view(x.size(0), self.gate_channels, -1)
            x_process = x_process.view(x.size(0), -1, self.gate_channels)  # (N,fxf,gate_channels)
            x_process = F.normalize(x_process, dim=2)
            x_emb_targets = self.linear(x_emb_targets)  # (C,gate_channels)
            x_emb_targets = F.normalize(x_emb_targets, dim=1)
            distance = torch.matmul(x_process, x_emb_targets.t())
            distance = distance.view(x.size(0), x_emb_targets.size(0), distance.size(1))  # (N,C,fxf)
            positive_distance = torch.stack(
                [distance[i][i * x_emb_targets.size(0) // x.size(0)] for i in range(0, x.size(0))])  # (N,fxf)
            positive_distance_softmax = F.softmax(positive_distance, dim=1)  # (N,fxf)
            positive_distance_softmax = positive_distance_softmax.view(x.size(0), 1, x.size(2), x.size(2))  # (N,1,f,f)
            x_out = x_out.mul(positive_distance_softmax)

            alpha = 0.4
            distance -= positive_distance.view(x.size(0), 1, -1)  # (N,C,fxf)
            distance = torch.clamp(distance, min=alpha)
            distance = distance.mul(positive_distance_softmax.view(x.size(0), 1, x.size(2) * x.size(2)))
            loss = torch.mean(torch.sum(torch.sum(distance, dim=2), dim=1), dim=0)

            scale = torch.sigmoid(x_out)  # broadcasting
            return x * scale, loss
