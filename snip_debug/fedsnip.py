from email import utils
from tkinter import N

from scipy import test
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

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))
print(sys.path)

from tqdm import tqdm
import warnings
import copy

import wandb

# from datasets import get_dataset

from utils import *
import random
from data_loader import load_partition_data_cifar10

import json
from snip_debug.snip import SNIP
from snip_debug.vgg import vgg11_bn
# import ..snip


def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
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
parser.add_argument('--prune_vote', type=int, default=1,
                    help='local client batch size')

# rng = np.random.default_rng()

class Client:

    def __init__(self, id, device, train_data, test_data, prune_strategy=None, net=None,
                 local_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=0.001, prune_at_first_round=False):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''
        # print('init client')
        # print(id)
        # print(device)

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.prune_strategy = prune_strategy

        self.device = device
        self.net = net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.prune_at_first_round = prune_at_first_round
        self.pruned = False

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None

        self.train_size = self._get_train_size()
        self.keep_masks = None
        
        # self.init_model()

    def init_model(self):
        print(f'client {self.id} init model !!!!')
        self.net.load_state_dict(torch.load('init_model.pt'))

    def _get_train_size(self):
        return sum(len(x) for x in self.train_data)

    def check_prunable(self):
        return self.net.is_maskable and (self.prune_strategy != 'None')

    def record_keep_masks(self, keep_masks):
        masks = torch.cat([torch.flatten(x) for x in keep_masks]).to('cpu').tolist()
        with open(f'./{self.id}_keep_masks.txt', 'a+') as f:
            f.write(json.dumps(masks))
            f.write('\n')

    def train(self, global_params=None, initial_global_params=None, sparsity=0):
        '''Train the client network for a single round.'''

        
        # self.reset_weights(global_params)
        
        print(f'client: {self.id} **************')

        # prune_criterion = SNIP.SNIP(model=self.net, limit=sparsity, start=None, steps=None, device=self.device)
        # # prune_criterion = SNAP(model=net, device=device)
        # prune_criterion.prune(sparsity, train_loader=self.train_data, manager=None)
        # self.pruned = True
    
        # self.init_model()
        self.keep_masks = SNIP(self.net, sparsity, self.train_data, self.device, self.id)  # TODO: shuffle?
        self.record_keep_masks(self.keep_masks)
        handlers = apply_prune_mask(self.net, self.keep_masks)

        return self.keep_masks

def merge_masks(keep_masks_dict):
    for i in range(1, len(keep_masks_dict[0])):
        for j in range(len(keep_masks_dict.keys())):
            keep_masks_dict[0][i] += keep_masks_dict[j][i]
    return keep_masks_dict[0]


def main(args):

    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    devices = [torch.device(x) for x in args.device]
    args.pid = os.getpid()


    # Fetch and cache the dataset
    dprint('Fetching dataset...')

    path = os.path.join('../..', 'data', args.dataset)

    # train_data_dicts = []
    # test_data_dicts = []
    # for i in range(args.total_clients):
    train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.dataset, path, 'homo', None, args.total_clients, args.batch_size)
        # train_data_dicts.append(train_data_local_dict)
        # test_data_dicts.append(test_datda_local_dict)

    # initialize clients
    dprint('Initializing clients...')
    clients = {}
    client_ids = []

    # for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    for i in range(args.total_clients):
        cl = Client(id=i, device=devices[0], train_data=train_data_local_dict[i], test_data=test_data_local_dict[i], net=vgg11_bn,
                    learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
        clients[i] = cl
        client_ids.append(i)
        # torch.cuda.empty_cache()

    global_model = vgg11_bn()
    # init_net(global_model)
    global_model = global_model.to(devices[0])
    global_model.load_state_dict(torch.load('init_model.pt'))

    mask_dict = {}

    global_params = deepcopy(global_model.state_dict())
    
    for client_id in clients.keys():
        print(f'client {client_id} start !!!')
        client = clients[client_id]
        i = client_ids.index(client_id)

        # wandb.watch(client.net, log='all')

        t0 = time.time()
        keep_masks = client.train(global_params=global_params, 
                                    initial_global_params=None, 
                                    sparsity=1-args.keep_ratio)
        mask_dict[client_id] = keep_masks

    global_mask = merge_masks(mask_dict)
    masks = torch.cat([torch.flatten(x) for x in global_mask]).to('cpu').tolist()
    with open(f'./global_mask_keep_masks.txt', 'a+') as f:
        f.write(json.dumps(masks))
        f.write('\n')

    print('done')

if __name__ == "__main__":
    # print('????')
    args = parser.parse_args()
    main(args)
    # print('!!!!')