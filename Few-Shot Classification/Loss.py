# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

import torch
import torch.nn as nn
from torch.nn import functional as F

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def loss_fn(support_input, query_input, targets, n_support):
    classes = torch.unique(targets)
    n_classes = len(classes)
    n_query = targets.eq(classes[0].item()).sum().item() - n_support
    prototypes = torch.stack([support_input[i:i+n_support].mean(0) for i in range(0, len(support_input), n_support)])
    dists = euclidean_dist(query_input, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes).cuda()
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    loss = loss_val
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc_val

def loss_fn2(support_input, query_input, targets, n_support):
    classes = torch.unique(targets)
    n_classes = len(classes)
    n_query = targets.eq(classes[0].item()).sum().item() - n_support
    prototypes = torch.stack([support_input[i:i+n_support].mean(0) for i in range(0, len(support_input), n_support)])
    dists = euclidean_dist(query_input, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes).cuda()
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val
