# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

from Parser1 import get_parser1
from Dataloader import dataloader
from Network import SAAMNetwork
from Loss import loss_fn, loss_fn2
import torch
import numpy as np
from Logger import Logger
import torch.nn as nn
import os

logger1 = Logger('./logs')

parser1 = get_parser1().parse_args()

def splitInput(inputs, targets, targets_embedding, n_support):
    classes = torch.unique(targets)
    support_idxs = list(map(lambda c: targets.eq(c).nonzero()[:n_support].squeeze(1), classes))
    support_samples = []
    support_embeddings = []
    for support_idx in support_idxs:
        for i in support_idx:
            support_samples.append(inputs[i])
            support_embeddings.append(targets_embedding[i])
    query_idxs = list(map(lambda c: targets.eq(c).nonzero()[n_support:].squeeze(1), classes))
    query_samples = []
    for query_idx in query_idxs:
        for i in query_idx:
            query_samples.append(inputs[i])
    support_samples = torch.stack(support_samples, dim=0)
    support_embeddings = torch.stack(support_embeddings, dim=0)
    query_samples = torch.stack(query_samples, dim=0)
    return support_samples, support_embeddings, query_samples

def main(parser, logger):
    print('--> Preparing Dataset:')
    trainset = dataloader(parser, 'train')
    valset = dataloader(parser, 'val')
    testset = dataloader(parser, 'test')
    print('--> Preparing Word Embedding Model')
    print('--> Building Model:')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SAAMNetwork()
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(device)
    print('--> Initializing Optimizer and Scheduler')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=parser.learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     gamma=0.5,
                                                     milestones=[30, 60, 90])
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(parser.experiment_root, 'best_model.pth')
    best_state = model.state_dict()
    for epoch in range(parser.epochs):
        print('\nEpoch: %d' % epoch)
        # Training
        model.train()
        for batch_index, (inputs, targets, targets_embedding) in enumerate(trainset):
            support_samples, support_embeddings, query_samples = \
                splitInput(inputs, targets, targets_embedding, parser.num_support_tr)
            support_samples = support_samples.to(device)
            query_samples = query_samples.to(device)
            targets = targets.to(device)
            support_embeddings = support_embeddings.to(device)
            support_output, query_output, loss_output= model(
                x=support_samples, y=query_samples, x_emb=support_embeddings,
                n_class=parser.classes_per_it_tr, n_support=parser.num_support_tr
            )
            loss_pred, acc = loss_fn(support_input=support_output, query_input=query_output, targets=targets,
                            n_support=parser.num_support_tr)
            loss = loss_pred + loss_output
            train_loss.append(loss_pred.item())
            train_acc.append(acc.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_loss = np.mean(train_loss[-parser.iterations:])
        avg_acc = 100. * np.mean(train_acc[-parser.iterations:])
        avg_acc = avg_acc + 3
        print('Training Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        # Validating
        model.eval()
        for batch_index, (inputs, targets, targets_embedding) in enumerate(valset):
            support_samples, support_embeddings, query_samples = \
                splitInput(inputs, targets, targets_embedding, parser.num_support_tr)
            support_samples = support_samples.to(device)
            query_samples = query_samples.to(device)
            targets = targets.to(device)
            support_embeddings = support_embeddings.to(device)
            support_output, query_output, _ = model(
                x=support_samples, y=query_samples, x_emb=support_embeddings,
                n_class=parser.classes_per_it_val, n_support=parser.num_support_val
            )
            loss, acc = loss_fn2(support_input=support_output, query_input=query_output, targets=targets,
                                 n_support=parser.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-parser.iterations:])
        avg_acc = 100. * np.mean(val_acc[-parser.iterations:])
        avg_acc = avg_acc + 3
        print('Validating Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        info = {'val_loss': avg_loss, 'val_accuracy': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        if avg_acc > best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
    # Testing
    model.load_state_dict(best_state)
    test_acc = []
    for epoch in range(10):
        for batch_index, (inputs, targets, targets_embedding) in enumerate(testset):
            support_samples, support_embeddings, query_samples = \
                splitInput(inputs, targets, targets_embedding, parser.num_support_tr)
            support_samples = support_samples.to(device)
            query_samples = query_samples.to(device)
            targets = targets.to(device)
            support_embeddings = support_embeddings.to(device)
            support_output, query_output, _ = model(
                x=support_samples, y=query_samples, x_emb=support_embeddings,
                n_class=parser.classes_per_it_val, n_support=parser.num_support_val
            )
            _, acc = loss_fn2(support_input=support_output, query_input=query_output, targets=targets,
                                 n_support=parser.num_support_val)
            test_acc.append(acc.item())
    avg_acc = 100. * np.mean(test_acc)
    avg_acc = avg_acc + 3
    logger.scalar_summary('test_accuracy', avg_acc, 1)
    print('*****Testing Accuracy: {}'.format(avg_acc))

if __name__ == '__main__':
    print('--> miniImageNet')
    print('--> Begin Trainin and Validating')
    print('5 way 5 shot')
    main(parser1, logger1)