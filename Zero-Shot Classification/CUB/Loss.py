# Author : Bryce Xu
# Time : 2020/2/19
# Function: 

import torch
import torch.nn.functional as F

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def loss_fn1(parser, features, prototypes):
    dists = euclidean_dist(features, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(50, parser.num_per_class, -1)
    target_inds = torch.arange(0, 50).cuda()
    target_inds = target_inds.view(50, 1, 1)
    target_inds = target_inds.expand(50, parser.num_per_class, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val

def loss_fn2(embeddings, prototypes):
    dists = euclidean_dist(embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(50, 1, -1)
    target_inds = torch.arange(0, 50).cuda()
    target_inds = target_inds.view(50, 1, 1)
    target_inds = target_inds.expand(50, 1, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    return loss_val