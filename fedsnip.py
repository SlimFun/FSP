from email import utils
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

import wandb

from datasets import get_dataset
from models.models import all_models

from client import Client
from utils import *

rng = np.random.default_rng()

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
parser.add_argument('--eval-every', default=2, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))

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

loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

# t0 = time.time()

# initialize clients
dprint('Initializing clients...')
clients = {}
client_ids = []

wandb.init(
            project="feddst",
            name="FedDST(d)",
            config=args
        )

model = 'VGG_SNIP'
# model = 'CNNNet'

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, net=all_models[model],
                learning_rate=args.eta, local_epochs=args.epochs, prune_strategy='SNIP')
    # cl = Client(client_id, *client_loaders, net=all_models[model],
    #             learning_rate=args.eta, local_epochs=args.epochs)
    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# initialize global model
global_model = all_models[model](device='cpu')
# init_net(global_model)


initial_global_params = deepcopy(global_model.state_dict())


# t1 = time.time()
# print(f'init cost: {t1 - t0}')

# for each round t = 1, 2, ... do
for server_round in tqdm(range(args.rounds)):
    # print(clients)

    # sample clients
    client_indices = rng.choice(list(clients.keys()), size=args.clients)

    # global_params = global_model.cpu().state_dict()
    global_params = deepcopy(global_model.state_dict())
    aggregated_params = {}
    for name, param in global_params.items():
        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')

    # for each client k \in S_t in parallel do
    total_sampled = 0
    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)

        # wandb.watch(client.net, log='all')

        train_result = client.train(global_params=global_params, initial_global_params=initial_global_params, sparsity=0.3)
        cl_params = train_result['state']

        client.net.clear_gradients() # to save memory

        cl_weight_params = {}

        # first deduce masks for the received weights
        for name, cl_param in cl_params.items():
            cl_weight_params[name] = cl_param.to(device='cpu', copy=True)

        # at this point, we have weights and masks (possibly all-ones)
        # for this client. we will proceed by applying the mask and adding
        # the masked received weights to the aggregate, and adding the mask
        # to the aggregate as well.
        for name, cl_param in cl_weight_params.items():
            aggregated_params[name].add_(client.train_size() * cl_param)

    # t2 = time.time()
    # print(f'one round training cost: {t2 - t1}')

    # at this point, we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_params.items():
        aggregated_params[name] /= sum(clients[i].train_size() for i in client_indices)
        assert (global_params[name].shape == aggregated_params[name].shape)
        # aggregated_params[name].type_as(global_params[name])
        global_params[name] = global_params[name].type_as(aggregated_params[name])
        global_params[name] += aggregated_params[name]
    global_model.load_state_dict(global_params) 
    
    # global_model.load_state_dict(aggregated_params)

    if server_round % 1 == 0:
        compare_model(initial_global_params, global_model.state_dict())

    # t3 = time.time()
    # print(f'aggregate cost: {t3 - t2}')

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0:
        accuracies = evaluate_local(clients, global_model, progress=True,
                                                    n_batches=args.test_batches)

        wandb.log({"Test/Acc": sum(accuracies.values())/len(accuracies.values())}, step=server_round)

    for client_id in clients:
        i = client_ids.index(client_id)
        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params

    
    # wandb.log({"Test/Sparsity": sum(sparsities.values())/len(sparsities.values())}, step=server_round)

