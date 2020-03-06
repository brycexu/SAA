# Author : Bryce Xu
# Time : 2020/2/1
# Function: 

from Dataset import TieredImageNet
from DataSampler import TieredImageNetBatchSampler
import torch
import numpy as np

def dataloader(parser, mode):
    dataset = TieredImageNet(mode=mode, root=parser.dataset_root)
    print(mode)
    print(len(dataset))
    print(len(np.unique(dataset.label_specific)))
    if 'train' in mode:
        classes_per_it = parser.classes_per_it_tr
        num_samples = parser.num_support_tr + parser.num_query_tr
    else:
        classes_per_it = parser.classes_per_it_val
        num_samples = parser.num_support_val + parser.num_query_val
    sampler = TieredImageNetBatchSampler(labels=dataset.label_specific, classes_per_it=classes_per_it, num_samples=num_samples,
                                   iterations=parser.iterations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader