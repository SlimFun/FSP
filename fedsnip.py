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

from tqdm import tqdm
import warnings
import copy

from debug.debug_info import DebugInfo

import wandb

from datasets import get_dataset
from models.models import all_models

from client import Client
from utils import *
import random

from data_loader import load_partition_data_cifar10
import json

from server import Server


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

parser.add_argument('--clip_grad', default=False, action='store_true', dest='clip_grad')

parser.add_argument('--model', type=str, choices=('VGG11_BN', 'VGG_SNIP', 'CNNNet'),
                    default='VGG11_BN', help='Dataset to use')

parser.add_argument('--prune_strategy', type=str, choices=('None', 'SNIP'),
                    default='None', help='Dataset to use')
parser.add_argument('--prune_at_first_round', default=False, action='store_true', dest='prune_at_first_round')
parser.add_argument('--keep_ratio', type=float, default=0.0,
                    help='local client batch size')         
parser.add_argument('--prune_vote', type=int, default=1,
                    help='local client batch size')

parser.add_argument('--single_shot_pruning', default=False, action='store_true', dest='single_shot_pruning')


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

rng = np.random.default_rng()


def main(args):
    
    devices = [torch.device(x) for x in args.device]
    args.pid = os.getpid()


    # Fetch and cache the dataset
    dprint('Fetching dataset...')
    

    path = os.path.join('..', 'data', args.dataset)
    train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.dataset, path, 'homo', None, args.total_clients, args.batch_size)

    # t0 = time.time()

    # initialize clients
    dprint('Initializing clients...')
    clients = {}
    client_ids = []

    wandb.init(
                project="fedsnip",
                name="FedAVG(d)"+ str(args.prune_strategy) + str(args.keep_ratio) + '-' + str(args.distribution) + "-prune_at_1r" + str(args.prune_at_first_round) + "-single_shot_pruning" + str(args.single_shot_pruning) + "-lr" + str(
                    args.eta) + '-clip_grad' + str(args.clip_grad),
                config=args
            )
    
    for i in range(args.total_clients):
        cl = Client(id=i, device=devices[0], train_data=train_data_local_dict[i], test_data=test_data_local_dict[i], net=all_models[args.model],
                    learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
                    
        clients[i] = cl
        client_ids.append(i)
        torch.cuda.empty_cache()

    server = Server(all_models[args.model], devices[0])
    # server = Server(vgg11_bn, devices[0])
    # server = Server(vgg11_bn, devices[0])

    # server.init_global_model()
    for round in range(args.rounds):
        keep_masks_dict = {}
        model_list = []
        client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)
        global_params = server.get_global_params()

        for client_id in client_indices:
            client = clients[client_id]

            train_result = client.train(global_params=global_params,
                                        sparsity=1-args.keep_ratio,
                                        single_shot_pruning=args.single_shot_pruning,
                                        clip_grad=args.clip_grad)
            cl_params = train_result['state']
            cl_mask_prarms = train_result['mask']

            keep_masks_dict[client_id] = cl_mask_prarms
            # print(client.train_size)
            model_list.append((client.train_size, cl_params))


        server.aggregate(keep_masks_dict, model_list, round)

        # yield DebugInfo('', (model_list, server))

        if round % args.eval_every == 0:
            server.test_global_model_on_all_client(clients, round)

if __name__ == "__main__":
    # print('????')
    args = parser.parse_args()
    main(args)
    # print('!!!!')