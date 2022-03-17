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

client_loaders = get_dataset(args.dataset, clients=1, mode='classic_iid', batch_size=args.batch_size, devices=cache_devices)
# print(loaders)
# print(len(loaders.keys()))
# train_loader = loaders[0][0]
# val_loader = loaders[0][1]
# t0 = time.time()

wandb.init(
            project="vgg_cifar",
            name="FedDST(d)",
            config=args
        )

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


args.device = args.device[0]

# initialize global model
global_model = all_models[args.model](device='cpu')
# global_model = vgg.__dict__['vgg11_bn']()
global_model.to(args.device)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.eta, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

client = Client(0, *client_loaders[0], net=all_models[args.model],
                learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)

for server_round in tqdm(range(args.rounds)):
    # print(clients)

    # sample clients
    # client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)

    # global_params = global_model.cpu().state_dict()
    global_params = deepcopy(global_model.state_dict())
    total_sampled = 0

    train_result = client.train(global_params=global_params, initial_global_params=None, sparsity=1-args.keep_ratio)
    cl_params = train_result['state']
    global_model.load_state_dict(cl_params) 

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0:
        train_accuracy, train_losses = test_model(global_model, global_model.device, client.train_data)
        wandb.log({"Train/Acc": train_accuracy}, step=server_round*10)
        wandb.log({"Train/Loss": train_losses}, step=server_round*10)

        test_accuracies, test_loss = test_model(global_model, global_model.device, client.test_data)
        wandb.log({"Test/Acc": test_accuracies}, step=server_round*10)
        wandb.log({"Test/Loss": test_loss}, step=server_round*10)
