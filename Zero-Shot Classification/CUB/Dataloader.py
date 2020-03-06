# Author : Bryce Xu
# Time : 2020/2/18
# Function: 

from torchvision import transforms
from torch.utils import data
from Dataset import CUBDataset
from Datasampler import CUBBatchSampler
import torch
import numpy as np

def dataloader(parser, examples, labels, attributes, names, is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08, 1), (0.5, 4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        dataset = CUBDataset(parser, examples, labels, attributes, names, tr_transforms)
        sampler = CUBBatchSampler(labels=dataset.label, classes_per_it=parser.classes_per_it,
                                  num_samples=parser.num_per_class, iterations=parser.iterations)
        data_loader = data.DataLoader(dataset, batch_sampler=sampler)
    else:
        dataset = CUBDataset(parser, examples, labels, attributes, names, ts_transforms)
        sampler = CUBBatchSampler(labels=dataset.label, classes_per_it=parser.classes_per_it,
                                  num_samples=parser.num_per_class, iterations=parser.iterations)
        data_loader = data.DataLoader(dataset, batch_sampler=sampler)
    return data_loader

def image_load(class_file, label_file):
    with open(class_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    class_map = {}
    for i,l in enumerate(class_names):
        items = l.split()
        class_map[items[-1]] = i
    #print(class_map)
    examples = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        examples.append(items[0])
        labels[items[0]] = int(items[1])
    return examples,labels, class_map

def split_byclass(parser, examples,labels, attributes, class_map):
    with open(parser.train_classes, 'r') as f:
        train_lines = [l.strip() for l in f.readlines()]
    with open(parser.test_classes, 'r') as f:
        test_lines = [l.strip() for l in f.readlines()]
    train_attr = []
    test_attr = []
    train_name = []
    test_name = []
    train_class_set = {}
    for i,name in enumerate(train_lines):
        idx = class_map[name]
        train_class_set[idx] = i
        # idx is its real label
        train_attr.append(attributes[idx])
        train_name.append(name[4:].lower())
    test_class_set = {}
    for i,name in enumerate(test_lines):
        idx = class_map[name]
        test_class_set[idx] = i
        test_attr.append(attributes[idx])
        test_name.append(name[4:].lower())
    train = []
    test = []
    label_map = {}
    for ins in examples:
        v = labels[ins]
        # inital label
        if v in train_class_set:
            train.append(ins)
            label_map[ins] = train_class_set[v]
        else:
            test.append(ins)
            label_map[ins] = test_class_set[v]
    train_attr = torch.from_numpy(np.array(train_attr,dtype='float')).float()
    test_attr = torch.from_numpy(np.array(test_attr,dtype='float')).float()
    return [(train,test,label_map,train_attr,test_attr,train_name,test_name)]