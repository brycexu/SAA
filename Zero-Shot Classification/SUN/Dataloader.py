# Author : Bryce Xu
# Time : 2020/2/29
# Function: 

from Dataset import SUNDataset
from Datasampler import SUNBatchSampler
import torch
import numpy as np

def dataloader(parser, mode):
    dataset = SUNDataset(mode=mode, data_root=parser.image_dir, attribute_root=parser.attributes_file,
                         wmodel_root=parser.word_dir)
    print(mode)
    print(len(np.unique(dataset.label)))
    if 'train' in mode:
        sampler = SUNBatchSampler(labels=dataset.label, classes_per_it=parser.classes_per_tr,
                                  num_samples=parser.num_per_tr, iterations=parser.iterations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    else:
        sampler = SUNBatchSampler(labels=dataset.label, classes_per_it=parser.classes_per_ts,
                                  num_samples=parser.num_per_ts, iterations=parser.iterations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader
