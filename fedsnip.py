from email import utils
from pydoc import cli
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
parser.add_argument('--l2', default=1e-5, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=1, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))

parser.add_argument('--clip_grad', default=False, action='store_true', dest='clip_grad')

parser.add_argument('--model', type=str, choices=('VGG11_BN', 'VGG_SNIP', 'CNNNet', 'CIFAR10Net'),
                    default='VGG11_BN', help='Dataset to use')

parser.add_argument('--prune_strategy', type=str, choices=('None', 'SNIP', 'SNAP', 'random_masks', 'Iter-SNIP', 'layer_base_SNIP', 'Grasp'),
                    default='None', help='Dataset to use')
parser.add_argument('--prune_at_first_round', default=False, action='store_true', dest='prune_at_first_round')
parser.add_argument('--keep_ratio', type=float, default=0.0,
                    help='local client keep ratio')         
parser.add_argument('--prune_vote', type=int, default=1,
                    help='local client batch size')

parser.add_argument('--single_shot_pruning', default=False, action='store_true', dest='single_shot_pruning')

parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                    help='partition alpha (default: 0.5)')

parser.add_argument('--target_keep_ratio', default=0.1, type=float, help='server target keep ratio')

parser.add_argument('--num_pruning_steps', type=int, help='total number of pruning steps')
parser.add_argument('--pruning_steps_decay_mode', type=str, default='linear', choices=('linear', 'exp'), help='pruning steps decay mode')
parser.add_argument('--saliency_mode', type=str, choices=('saliency', 'mask'))
parser.add_argument('--sparsity_distribution', type=str, choices=('erk', 'uniform'), default='uniform')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

rng = np.random.default_rng()

def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=args.outfile)
    print(*arg, **kwargs)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

def main(args):
    
    devices = [torch.device(x) for x in args.device]
    args.pid = os.getpid()


    # Fetch and cache the dataset
    dprint('Fetching dataset...')
    

    path = os.path.join('..', 'data', args.dataset)
    train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.dataset, path, args.partition_method, args.partition_alpha, args.total_clients, args.batch_size)

    # t0 = time.time()

    # initialize clients
    dprint('Initializing clients...')
    clients = {}
    client_ids = []

    wandb.init(
                project="fedsnip",
                name="FedAVG(d)"+ str(args.prune_strategy) + '-target_kr' + str(args.target_keep_ratio) + '-' + str(args.keep_ratio) + '-' + str(args.distribution) + "-prune_at_1r" + str(args.prune_at_first_round) + "-single_shot_pruning" + str(args.single_shot_pruning) + "-lr" + str(
                    args.eta) + '-clip_grad' + str(args.clip_grad),
                config=args
            )
    
    for i in range(args.total_clients):
        cl = Client(id=i, device=devices[0], train_data=train_data_local_dict[i], test_data=test_data_local_dict[i], net=all_models[args.model],
                    learning_rate=args.eta, momentum=args.momentum, weight_decay=args.l2, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
                    
        clients[i] = cl
        client_ids.append(i)
        torch.cuda.empty_cache()

    if args.prune_strategy not in ['Iter-SNIP', 'layer_base_SNIP', 'Grasp']:
        assert args.num_pruning_steps == 1, 'Non Iter pruning method num_pruning_steps must be 1'


    server = Server(all_models[args.model], devices[0], train_data_local_dict[0], prune_strategy=args.prune_strategy, target_keep_ratio=args.target_keep_ratio)
    # server = Server(vgg11_bn, devices[0])
    print(f'server model param size: {server.model.param_size}')
    compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
    download_cost = np.zeros(len(clients))
    upload_cost = np.zeros(len(clients))

    global_masks = server.masks
    if args.prune_strategy == 'random_masks':
        global_masks = server.generate_random_masks(sparsity=1-args.target_keep_ratio, sparsity_distribution=args.sparsity_distribution)
        # server.masks is not None, will not merge and change server masks
        server.masks = global_masks
        server._prune_global_model(global_masks)
    elif args.prune_strategy in ['SNIP', 'SNAP', 'Iter-SNIP', 'layer_base_SNIP', 'Grasp']:
        # target_pruning_ratio
        if args.pruning_steps_decay_mode == 'linear':
            keep_ratio_steps = [1 - ((x + 1) * (1 - args.target_keep_ratio) / args.num_pruning_steps) for x in range(args.num_pruning_steps)]
            
        elif args.pruning_steps_decay_mode == 'exp':
            keep_ratio_steps = [np.exp(0 - ((x + 1) * (0 - np.log(args.target_keep_ratio)) / args.num_pruning_steps)) for x in range(args.num_pruning_steps)]

        print('keep_ratio_steps: {}'.format(keep_ratio_steps))
        local_sparsity = 1-args.keep_ratio
        for keep_ratio in keep_ratio_steps:
            keep_masks_dict = {}
            model_list = []
            client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)
            global_params = server.get_global_params()

            num_training_data = [0 for _ in range(len(client_indices))]

            for client_id in client_indices:
                client = clients[client_id]

                # need process SNAP
                pass
                
                # train_result = client.train(global_params=global_params,
                #                         sparsity=1-args.keep_ratio,
                #                         single_shot_pruning=args.single_shot_pruning,
                #                         test_on_each_round=True,
                #                         clip_grad=args.clip_grad,
                #                         global_sparsity=server.model.sparsity_percentage(),
                #                         global_masks=global_masks)
                train_result = client.local_mask(global_params=global_params, sparsity=local_sparsity, global_masks=global_masks, saliency_mode=args.saliency_mode)

                cl_mask_prarms = train_result['mask']
                download_cost[client_id] = train_result['dl_cost']
                upload_cost[client_id] = train_result['ul_cost']

                keep_masks_dict[client_id] = cl_mask_prarms
                num_training_data[client_id] = client.train_size
                # print(client.train_size)
            server.target_keep_ratio = keep_ratio
            # yield keep_masks_dict
            if args.saliency_mode == 'mask':
                num_training_data = None
            server.merge_masks(keep_masks_dict, keep_ratio, download_cost, upload_cost, compute_times, last_masks=global_masks, num_training_data=num_training_data)

            # only keep 10% of pruned_global_model. e.g. 1 - 0.23 * 0.1, 1 - 0.56 * 0.1
            # local_sparsity = (1-args.keep_ratio) * keep_ratio
            local_sparsity = 1 - args.keep_ratio * keep_ratio
            
            global_masks = server.masks
            server._prune_global_model(server.masks)
            print('server masked {}% params'.format(server.masked_percent() * 100))
        
    # server._prune_global_model(server.masks)
    
    torch.save(server.model, 'server_model.pt')
    torch.save(server.masks, 'server_masks.pt')
    # 1/0

    # server.init_global_model()
    for round in range(args.rounds):
        keep_masks_dict = {}
        model_list = []
        client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)
        global_params = server.get_global_params()

        for client_id in client_indices:
            client = clients[client_id]

            if (round == 1) and (args.prune_strategy == 'SNAP'):
                del client.net
                torch.cuda.empty_cache()
                client.net = copy.deepcopy(server.model)
            t0 = time.time()
            train_result = client.train(global_params=global_params,
                                        sparsity=1-args.keep_ratio,
                                        single_shot_pruning=args.single_shot_pruning,
                                        test_on_each_round=True,
                                        clip_grad=args.clip_grad,
                                        global_sparsity=server.model.sparsity_percentage(),
                                        global_masks=global_masks)
            cl_params = train_result['state']
            cl_mask_prarms = train_result['mask']
            download_cost[client_id] = train_result['dl_cost']
            upload_cost[client_id] = train_result['ul_cost']
            compute_times[client_id] = time.time() - t0

            keep_masks_dict[client_id] = cl_mask_prarms
            # print(client.train_size)
            model_list.append((client.train_size, cl_params))


        server.aggregate(keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times)

        # global_masks = server.masks

        print('server masked {}% params'.format(server.masked_percent() * 100))
        # yield DebugInfo('', (server))

        if round % args.eval_every == 0:
            train_accuracies, test_accuracies = server.test_global_model_on_all_client(clients, round)
            # for client_id in clients:
            #     print_csv_line(pid=args.pid,
            #                 dataset=args.dataset,
            #                 clients=args.clients,
            #                 total_clients=len(clients),
            #                 round=round,
            #                 batch_size=args.batch_size,
            #                 epochs=args.epochs,
            #                 target_sparsity=server.model.sparsity_percentage(),
            #                 pruning_rate=0,
            #                 initial_pruning_threshold='',
            #                 final_pruning_threshold='',
            #                 pruning_threshold_growth_method='',
            #                 pruning_method='',
            #                 lth=False,
            #                 client_id=client_id,
            #                 accuracy=test_accuracies[client_id],
            #                 sparsity=0,
            #                 compute_time=compute_times[client_id],
            #                 download_cost=download_cost[client_id],
            #                 upload_cost=upload_cost[client_id])
            print(f'download cost: {download_cost}')
            print(f'upload cost: {upload_cost}')

if __name__ == "__main__":
    # print('????')
    args = parser.parse_args()
    main(args)
    # print('!!!!')