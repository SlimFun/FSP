from email import utils
from tkinter import N
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import gc
import itertools
import numpy as np
import os
import sys
import time
import pickle
from copy import deepcopy

from tqdm import tqdm
import warnings
import copy

import wandb

from datasets import get_dataset
from models.models import all_models
import vgg

import torch.backends.cudnn as cudnn

from client import Client
from utils import *

rng = np.random.default_rng()

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--clients', type=int, help='number of clients per round', default=20)
parser.add_argument('--rounds', type=int, help='number of global rounds', default=400)
parser.add_argument('--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100'),
                    default='mnist', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid', 'classic_iid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument('--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=400)
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=20, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=1, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))

parser.add_argument('--model', type=str, choices=('VGG11_BN', 'VGG_SNIP', 'CNNNet'),
                    default='VGG11_BN', help='Dataset to use')    


parser.add_argument('--prune_strategy', type=str, choices=('None', 'SNIP'),
                    default='None', help='Dataset to use')
parser.add_argument('--prune_at_first_round', default=False, action='store_true', dest='prune_at_first_round')
parser.add_argument('--keep_ratio', type=float, default=0.0,
                    help='local client batch size')       

args = parser.parse_args()
devices = [torch.device(x) for x in args.device]
args.pid = os.getpid()


# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices

'''
if os.path.isfile(args.dataset + '.pickle'):
    with open(args.dataset + '.pickle', 'rb') as f:
        loaders = pickle.load(f)
else:
    loaders = get_dataset(args.dataset, clients=args.total_clients,
                          batch_size=args.batch_size, devices=cache_devices,
                          min_samples=args.min_samples)
    with open(args.dataset + '.pickle', 'wb') as f:
        pickle.dump(loaders, f)
'''

# loaders = get_dataset(args.dataset, clients=1, mode='classic_iid', batch_size=args.batch_size, devices=cache_devices)
# # print(loaders)
# # print(len(loaders.keys()))
# train_loader = loaders[0][1]
# val_loader = loaders[0][2]
# train_loader = loaders[0][0]
# print(len(train_loader))
# print(type(train_loader))


loaders1 = get_dataset(args.dataset, clients=1, mode='classic_iid', batch_size=args.batch_size, devices=None)
# print(loaders)
# print(len(loaders.keys()))
train_loader = loaders1[0][0]
val_loader = loaders1[0][1]
# print(type(train_loader1))
# val_loader = loaders[0][1]
# t0 = time.time()

# a = 1/0

wandb.init(
            project="vgg_cifar",
            name="FedDST(d)",
            config=args
        )


###################

# args.device = args.device[0]



# model = vgg.__dict__['vgg11_bn']()

# model.to(args.device)

# # model.features = torch.nn.DataParallel(model.features)
# # if args.cpu:
# #     model.cpu()
# # else:
# #     model.cuda()

# # optionally resume from a checkpoint
# cudnn.benchmark = True

# # define loss function (criterion) and pptimizer
# criterion = nn.CrossEntropyLoss().to(args.device)

# optimizer = torch.optim.SGD(model.parameters(), 0.05,
#                             momentum=0.9,
#                             weight_decay=5e-4)

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
#     lr = args.lr * (0.5 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# def train(train_loader, model, criterion, optimizer, epoch):
#     """
#         Run one train epoch
#     """
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):

#         # measure data loading time
#         data_time.update(time.time() - end)

#         input = input.to(args.device)
#         target = target.to(args.device)

#         # compute output
#         output = model(input)
#         loss = criterion(output, target)

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         output = output.float()
#         loss = loss.float()
#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target)[0]
#         losses.update(loss.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # print('Epoch: [{0}][{1}/{2}]\t'
#         #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#         #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#         #             epoch, i, len(train_loader), batch_time=batch_time,
#         #             data_time=data_time, loss=losses, top1=top1))
#     return losses, top1


# def validate(val_loader, model, criterion):
#     """
#     Run evaluation
#     """
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         input = input.to(args.device)
#         target = target.to(args.device)

#         # compute output
#         with torch.no_grad():
#             output = model(input)
#             loss = criterion(output, target)

#         output = output.float()
#         loss = loss.float()

#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target)[0]
#         losses.update(loss.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()


#         # print('Test: [{0}/{1}]\t'
#         #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#         #             i, len(val_loader), batch_time=batch_time, loss=losses,
#         #             top1=top1))

#     print(' * Prec@1 {top1.avg:.3f}'
#           .format(top1=top1))

#     return top1.avg, losses

# for epoch in range(0, 500):
#     adjust_learning_rate(optimizer, epoch)

#     # train for one epoch
#     losses, top1 = train(train_loader, model, criterion, optimizer, epoch)
#     wandb.log({"Train/Acc": top1.avg}, step=epoch)
#     wandb.log({"Train/Loss": losses.avg}, step=epoch)

#     # evaluate on validation set
#     prec1, losses = validate(val_loader, model, criterion)
#     wandb.log({"Test/Acc": prec1}, step=epoch)
#     wandb.log({"Test/Loss": losses.avg}, step=epoch)




##########################

args.device = args.device[0]

# initialize global model
global_model = all_models[args.model](device='cpu')
# global_model = vgg.__dict__['vgg11_bn']()
global_model.to(args.device)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.eta, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

# client = Client()

def test_model(model, device, data_loader):

    criterion = nn.CrossEntropyLoss()

    # we need to perform an update to client's weights.
    with torch.no_grad():
        correct = 0.
        total = 0.
        loss = 0.

        _model = model.to(device)

        _model.eval()
        # data_loader = self.train_data if train_data else self.test_data
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                # if i > n_batches and n_batches > 0:
                #     break
                # if not args.cache_test_set_gpu:
                #     inputs = inputs.to(self.device)
                #     labels = labels.to(self.device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = _model(inputs)
                loss += criterion(outputs, labels) * len(labels)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        

        return correct / total, loss / total

# def adjust_learning_rate(optimizer, epoch, init_lr):
#     """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
#     lr = init_lr * (0.5 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# for each round t = 1, 2, ... do
# for server_round in tqdm(range(args.rounds)):
for epoch in range(500):
    # adjust_learning_rate(optimizer, epoch, args.eta)
    global_model.train()

    total = 0.

    # print(len(self.train_data))
    for inputs, labels in train_loader:
        # print(inputs.shape)
        # print(labels)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        outputs = global_model(inputs)
        loss = criterion(outputs, labels)
        # if args.prox > 0:
        #     loss += args.prox / 2. * self.net.proximal_loss(global_params)
        loss.backward()
        optimizer.step()
        total += len(labels)

    print(total)

    # train_accuracies, train_losses, test_accuracies, test_losses = evaluate_local(clients, global_model, progress=True,
    #                                                 n_batches=args.test_batches)
    train_accuracy, train_losses = test_model(global_model, args.device, train_loader)
    wandb.log({"Train/Acc": train_accuracy}, step=epoch)
    wandb.log({"Train/Loss": train_losses}, step=epoch)

    test_accuracies, test_loss = test_model(global_model, args.device, val_loader)
    wandb.log({"Test/Acc": test_accuracies}, step=epoch)
    wandb.log({"Test/Loss": test_loss}, step=epoch)
