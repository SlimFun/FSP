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

parser.add_argument('--single_shot_pruning', default=False, action='store_true', dest='single_shot_pruning')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

rng = np.random.default_rng()


class Server:
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model(self.device)

    def get_global_params(self):
        return self.model.cpu().state_dict()

    def set_global_params(self, params_dict):
        self.model.load_state_dict(params_dict)

    def _merge_local_masks(self, keep_masks_dict):
    #keep_masks_dict[clients][params]
        for m in range(len(keep_masks_dict[0])):
            for client_id in keep_masks_dict.keys():
            # for j in range(0, len(keep_masks_dict.keys())):
                if client_id != 0:
                    keep_masks_dict[0][m] += keep_masks_dict[client_id][m]
        # for i in range(len(keep_masks_dict[0])):
        #     for j in range(1, len(keep_masks_dict.keys())):
        #         # params = self.keep_masks_dict[0][j].to('cpu')
        #         keep_masks_dict[0][i] += keep_masks_dict[j][i]
        #         # if j == len(self.keep_masks_dict.keys())-1:
        #         #     diff = self.keep_masks_dict[0][i].view(-1).cpu().numpy()
        #         #     self.keep_masks_dict[0][i] = np.where(diff, 0, 1)
        return keep_masks_dict[0]

    def _prune_global_model(self, masks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.model.modules())

        for layer, keep_mask in zip(prunable_layers, masks):
            assert (layer.weight.shape == keep_mask.shape)

            layer.weight.data[keep_mask == 0.] = 0.

    def aggregate(self, keep_masks_dict, model_list, round):
        last_params = self.get_global_params()

        training_num = sum(local_train_num for (local_train_num, _) in model_list)

        (num0, averaged_params) = model_list[0]
        if (averaged_params is not None) and (round != 0):
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    # print(training_num)
                    # print(w)
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # for name, param in averaged_params.items():
            for name in last_params:
                assert (last_params[name].shape == averaged_params[name].shape)
                last_params[name] = last_params[name].type_as(averaged_params[name])
                last_params[name] += averaged_params[name]
            self.set_global_params(last_params)

        if keep_masks_dict[0] is not None:
            masks = self._merge_local_masks(keep_masks_dict)

            self._prune_global_model(masks)

    def test_global_model_on_all_client(self, clients, round):
        # pruned_c = 0.0
        # total = 0.0
        # for name, param in self.model.state_dict().items():
        #     a = param.view(-1).to(device='cpu', copy=True).numpy()
        #     pruned_c +=sum(np.where(a, 0, 1))
        #     total += param.numel()
        # print(f'global model zero params: {pruned_c / total}')

        train_accuracies, train_losses, test_accuracies, test_losses = evaluate_local(clients, self.model, progress=True,
                                                    n_batches=args.test_batches)
        wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=round)
        wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=round)
        print(f'round: {round}')
        print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')

        wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=round)
        wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=round)
        print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')



def main(args):
    
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
                project="feddst",
                name="FedDST(d)",
                config=args
            )
    
    for i in range(args.total_clients):
        cl = Client(id=i, device=devices[0], train_data=train_data_local_dict[i], test_data=test_data_local_dict[i], net=all_models[args.model],
                    learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
                    
        clients[i] = cl
        client_ids.append(i)
        torch.cuda.empty_cache()

    server = Server(all_models[args.model], devices[0])

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
                                        single_shot_pruning=args.single_shot_pruning)
            cl_params = train_result['state']
            cl_mask_prarms = train_result['mask']

            keep_masks_dict[client_id] = cl_mask_prarms
            # print(client.train_size)
            model_list.append((client.train_size, cl_params))

        # yield DebugInfo('', (model_list, server))

        server.aggregate(keep_masks_dict, model_list, round)

        if round % args.eval_every == 0:
            server.test_global_model_on_all_client(clients, round)





    # global_model = all_models[args.model](device=devices[0])
    # # init_net(global_model)
    # global_model.load_state_dict(torch.load('init_model.pt'))
    # global_model = global_model.to(devices[0])
    # # global_model.load_state_dict(torch.load('init_model.pt'))


    # init_model = deepcopy(global_model)
    # initial_global_params = deepcopy(global_model.state_dict())

    # global_params = global_model.cpu().state_dict()

    # # for each round t = 1, 2, ... do
    # for server_round in tqdm(range(args.rounds)):
    #     client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)
        
    #     # global_params = deepcopy(global_model.state_dict())
    #     global_params = global_model.cpu().state_dict()
    #     global_model = global_model.to(devices[0])

    #     keep_masks_dict = {}
    #     model_list = []
    #     total_sampled = 0
    #     for client_id in client_indices:
    #         print(f'client {client_id} start !!!')
    #         client = clients[client_id]
    #         i = client_ids.index(client_id)

    #         t0 = time.time()
    #         train_result = client.train(global_params=copy.deepcopy(global_params), 
    #                                     sparsity=1-args.keep_ratio)
    #         print(f'cost: {time.time()}')
    #         cl_params = train_result['state']
    #         cl_mask_prarms = train_result['mask']

    #         # client.net.clear_gradients() # to save memory

    #         # keep_masks_dict[client_id] = [c.cpu() for c in cl_mask_prarms.values()]
    #         keep_masks_dict[client_id] = cl_mask_prarms
    #         model_list.append((client.train_size, cl_params))

    #     last_params = global_model.cpu().state_dict()
    #     global_model = global_model.to(devices[0])
    #     # global_model = global_model.to(devices[0])
    #     # last_params = global_params

    #     training_num = sum(clients[i].train_size for i in client_indices)

    #     (num0, averaged_params) = model_list[0]
    #     # if averaged_params is not None:
    #     for k in averaged_params.keys():
    #         # print(k)
    #         for i in range(0, len(model_list)):
    #             local_sample_number, local_model_params = model_list[i]
    #             w = local_sample_number / training_num
    #             # w = 1.
    #             # print(w)
    #             if i == 0:
    #                 averaged_params[k] = local_model_params[k] * w
    #             else:
    #                 averaged_params[k] += local_model_params[k] * w

    #     # for name, param in averaged_params.items():
    #     for name in last_params:
    #         assert (last_params[name].shape == averaged_params[name].shape)
    #         last_params[name] = last_params[name].type_as(averaged_params[name])
    #         last_params[name] += averaged_params[name]
    #     global_model.load_state_dict(last_params)

    #     masks = merge_local_masks(keep_masks_dict)

        # apply_global_mask(global_model, masks)

        # if server_round % 1 == 0:
        #     compare_model(initial_global_params, global_model.state_dict())

        #     pruned_c = 0.0
        #     total = 0.0
        #     for name in global_model.mask:
        #         a = global_model.mask[name].view(-1).to(device='cpu', copy=True).numpy()
        #         pruned_c +=sum(np.where(a, 0, 1))
        #         total += global_model.mask[name].numel()
        #     print(f'masked : {pruned_c / total}')
            
        # if server_round % args.eval_every == 0:
        #     # global_model_cp = copy.deepcopy(global_model)
        #     pruned_c = 0.0
        #     total = 0.0
        #     for name in global_model.mask:
        #         a = global_model.mask[name].view(-1).to(device='cpu', copy=True).numpy()
        #         pruned_c +=sum(np.where(a, 0, 1))
        #         total += global_model.mask[name].numel()
        #     print(f'check global_model masked : {pruned_c / total}')

        #     pruned_c = 0.0
        #     total = 0.0
        #     for name, param in global_model.state_dict().items():
        #         a = param.view(-1).to(device='cpu', copy=True).numpy()
        #         pruned_c +=sum(np.where(a, 0, 1))
        #         total += param.numel()
        #     print(f'global model zero params: {pruned_c / total}')

        #     train_accuracies, train_losses, test_accuracies, test_losses = evaluate_local(clients, global_model, progress=True,
        #                                                 n_batches=args.test_batches)
        #     # train_accuracy, train_losses = test_model(global_model_cp, devices[0], clients.train_data)
        #     wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=server_round)
        #     wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=server_round)
        #     print(f'round: {server_round}')
        #     print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')
        #     # wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=server_round)
        #     # wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=server_round)


        #     # test_accuracies, test_loss = test_model(global_model_cp, devices[0], clients.test_data)
        #     wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=server_round)
        #     wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=server_round)
        #     print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')
        #     # wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=server_round)
        #     # wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=server_round)

        #     print('-'*10)

        #     train_accuracies, train_losses, test_accuracies, test_losses = evaluate_local(clients, None, progress=True,
        #                                                 n_batches=args.test_batches)
        #     # train_accuracy, train_losses = test_model(global_model_cp, devices[0], clients.train_data)
        #     print(f'round: {server_round}')
        #     print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')
        #     # wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=server_round)
        #     # wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=server_round)


        #     # test_accuracies, test_loss = test_model(global_model_cp, devices[0], clients.test_data)
        #     print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')
        # global_params = global_model.state_dict()

        # global_params = last_params

if __name__ == "__main__":
    # print('????')
    args = parser.parse_args()
    main(args)
    # print('!!!!')