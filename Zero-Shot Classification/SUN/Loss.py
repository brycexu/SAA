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
    log_p_y = F.log_softmax(-dists, dim=1).view(parser.classes_per_tr, parser.num_per_tr, -1)
    target_inds = torch.arange(0, parser.classes_per_tr).cuda()
    target_inds = target_inds.view(parser.classes_per_tr, 1, 1)
    target_inds = target_inds.expand(parser.classes_per_tr, parser.num_per_tr, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val

def loss_fn2(parser, features, prototypes):
    dists = euclidean_dist(features, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(parser.classes_per_ts, parser.num_per_ts, -1)
    target_inds = torch.arange(0, parser.classes_per_ts).cuda()
    target_inds = target_inds.view(parser.classes_per_ts, 1, 1)
    target_inds = target_inds.expand(parser.classes_per_ts, parser.num_per_ts, 1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val
