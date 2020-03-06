# Author : Bryce Xu
# Time : 2020/2/21
# Function: 

# Author : Bryce Xu
# Time : 2020/2/18
# Function:

import argparse
from Logger import Logger
from Dataloader import *
from Network import FeatureNetwork, SynthesisNetwork
from Loss import loss_fn1, loss_fn2
import torch.nn as nn
import os

import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_dir', '--image_dir',
                        type=str,
                        default='/mnt/datadev_2/std/SUN/')

    parser.add_argument('-word_dir', '--word_dir',
                        type=str,
                        default='/mnt/datadev_2/std/MiniImageNet/glove.6B.100d.txt')

    parser.add_argument('-attributes_file', '--attributes_file',
                        type=str,
                        default='/mnt/datadev_2/std/SUN/attributes.txt')

    parser.add_argument('-model_dir', '--model_dir',
                        type=str,
                        default='/home/std-1/brycexu/ZSL/SUN/SAAM/model/')

    parser.add_argument('-classes_per_tr', '--classes_per_tr',
                        type=int,
                        default=50)

    parser.add_argument('-num_per_tr', '--num_per_tr',
                        type=int,
                        default=5)

    parser.add_argument('-classes_per_ts', '--classes_per_ts',
                        type=int,
                        default=10)

    parser.add_argument('-num_per_ts', '--num_per_ts',
                        type=int,
                        default=5)

    parser.add_argument('-iterations', '--iterations',
                        type=int,
                        default=10)

    return parser

logger = Logger('./logs')
parser = get_parser().parse_args()

def splitAttributes(inputs, targets, attributes, embeddings):
    classes = torch.unique(targets)
    idxs = list(map(lambda c: targets.eq(c).nonzero()[:10].squeeze(1), classes))
    embeddings_split = []
    attributes_split = []
    inputs_split = []
    for idx in idxs:
        attributes_split.append(attributes[idx[0]])
        embeddings_split.append(embeddings[idx[0]])
        for i in idx:
            inputs_split.append(inputs[i])
    inputs_split = torch.stack(inputs_split, dim=0)
    attributes_split = torch.stack(attributes_split, dim=0)
    embeddings_split = torch.stack(embeddings_split, dim=0)
    return inputs_split, attributes_split, embeddings_split

def main(parser, logger):
    print('--> Preparing Dataset:')
    trainloader = dataloader(parser, 'train')
    testloader = dataloader(parser, 'test')
    print('--> Building Model:')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    featureNetwork = FeatureNetwork()
    featureNetwork = nn.DataParallel(featureNetwork).to(device)
    prototypeNetwork = SynthesisNetwork()
    prototypeNetwork = nn.DataParallel(prototypeNetwork).to(device)
    print('--> Initializing Optimizer and Scheduler')
    optimizer1 = torch.optim.Adam(params=featureNetwork.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(params=prototypeNetwork.parameters(), lr=1e-4, weight_decay=1e-5)
    train_loss = []
    train_acc = []
    test_acc = []
    best_acc = 0
    best_prototypeNetwork_path = os.path.join(parser.model_dir, 'best_prototypeNetwork.pth')
    for epoch in range(50):
        print('\nEpoch: %d' % epoch)
        # Training
        featureNetwork.train()
        prototypeNetwork.train()
        for batch_index, (inputs, targets, attributes, embeddings) in enumerate(trainloader):
            inputs_split, attributes_split, embeddings_split = splitAttributes(inputs, targets, attributes, embeddings)
            inputs_split = inputs_split.to(device)
            attributes_split = attributes_split.to(device)
            embeddings_split = embeddings_split.to(device)
            features_output = featureNetwork(inputs_split)
            prototypes_output = prototypeNetwork(attributes_split, embeddings_split)
            loss, acc = loss_fn1(parser, features_output, prototypes_output)
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
        avg_loss = np.mean(train_loss[-parser.iterations:])
        avg_acc = 100. * np.mean(train_acc[-parser.iterations:])
        print('Training Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        info = {'train_loss': avg_loss, 'train_accuracy': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        featureNetwork.eval()
        prototypeNetwork.eval()
        for batch_index, (inputs, targets, attributes, embeddings) in enumerate(testloader):
            inputs_split, attributes_split, embeddings_split = splitAttributes(inputs, targets, attributes, embeddings)
            inputs_split = inputs_split.to(device)
            attributes_split = attributes_split.to(device)
            features_output = featureNetwork(inputs_split)
            prototypes_output = prototypeNetwork(attributes_split, embeddings_split)
            _, acc = loss_fn2(parser, features_output, prototypes_output)
            test_acc.append(acc.item())
        avg_acc = 100. * np.mean(test_acc[-parser.iterations:])
        print('Testing: {}'.format(avg_acc))
        info = {'test_accuracy': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        if avg_acc > best_acc:
            best_prototypeNetwork_state = prototypeNetwork.state_dict()
            torch.save(best_prototypeNetwork_state, best_prototypeNetwork_path)
            best_acc = avg_acc

if __name__ == '__main__':
    main(parser, logger)
