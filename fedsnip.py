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

    loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution, batch_size=args.batch_size, devices=None)

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

    # model = 'VGG11_BN'
    # prune_strategy = 'SNIP'
    # prune_at_first_round = False
    # keep = 0.9
    # model = 'CNNNet'

    # def clients_local_train(clients, client_indices, global_params, initial_global_params, args):

        

    for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
        cl = Client(id=client_id, device=devices[0], train_data=client_loaders[0], test_data=client_loaders[1], net=all_models[args.model],
                    learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
        # cl = Client(client_id, *client_loaders, net=all_models[args.model],
        #             learning_rate=args.eta, local_epochs=args.epochs, prune_strategy=args.prune_strategy, prune_at_first_round=args.prune_at_first_round)
        # cl = Client(client_id, *client_loaders, net=all_models[model],
        #             learning_rate=args.eta, local_epochs=args.epochs)
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    # initialize global model
    # global_model = all_models[args.model](device='cpu')args.device = args.device[0]
    global_model = all_models[args.model](device=devices[0])
    # init_net(global_model)
    global_model = global_model.to(global_model.device)


    initial_global_params = deepcopy(global_model.state_dict())

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


            print(f'test toatl: {total}')
            # remove copies if needed
            if model is not _model:
                del _model

            

            return correct / total, loss / total


    # t1 = time.time()
    # print(f'init cost: {t1 - t0}')

    # for each round t = 1, 2, ... do
    for server_round in tqdm(range(args.rounds)):
        # print(clients)

        # sample clients
        client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)

        # global_params = global_model.cpu().state_dict()
        global_params = deepcopy(global_model.state_dict())
        aggregated_params = {}
        aggregated_masks = {}
        for name, param in global_params.items():
            aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device=devices[0])
            # if isinstance(module, (nn.Linear, nn.Conv2d))
            if name in global_model.mask.keys():
                aggregated_masks[name] = torch.zeros_like(param, dtype=torch.float, device=devices[0])


        # clients_local_train()
        # for each client k \in S_t in parallel do
        total_sampled = 0
        for client_id in client_indices:
            print(f'client {client_id} start !!!')
            client = clients[client_id]
            i = client_ids.index(client_id)

            # wandb.watch(client.net, log='all')

            t0 = time.time()
            train_result = client.train(global_params=global_params, 
                                        initial_global_params=initial_global_params, 
                                        sparsity=1-args.keep_ratio)
            print(f'cost: {time.time()}')
            cl_params = train_result['state']
            cl_mask_prarms = train_result['mask']

            client.net.clear_gradients() # to save memory

            # cl_weight_params = {}
            # global_model.load_state_dict(cl_params) 

            # # first deduce masks for the received weights
            # for name, cl_param in cl_params.items():
            #     cl_weight_params[name] = cl_param.to(device='cpu', copy=True)

            # at this point, we have weights and masks (possibly all-ones)
            # for this client. we will proceed by applying the mask and adding
            # the masked received weights to the aggregate, and adding the mask
            # to the aggregate as well.
            for name, cl_param in cl_params.items():
                aggregated_params[name].add_(client.train_size * cl_param.to(device=devices[0]))

                if name in global_model.mask.keys():
                    aggregated_masks[name].add_(cl_mask_prarms[name].to(device=devices[0]))
        

        # t2 = time.time()
        # print(f'one round training cost: {t2 - t1}')

        # at this point, we have the sum of client parameters
        # in aggregated_params, and the sum of masks in aggregated_masks. We
        # can take the average now by simply dividing...


        # print(f'start aggregate #############')
        for name, param in aggregated_params.items():
            aggregated_params[name] /= sum(clients[i].train_size for i in client_indices)
            assert (global_params[name].shape == aggregated_params[name].shape)
            # aggregated_params[name].type_as(global_params[name])
            global_params[name] = global_params[name].type_as(aggregated_params[name])
            global_params[name] += aggregated_params[name]

            if name in global_model.mask.keys():
                global_model.mask[name] = torch.where(aggregated_masks[name]>=args.prune_vote, 1, 0)
        global_model.load_state_dict(global_params) 

        # yield DebugInfo('', (aggregated_masks, cl_mask_prarms))

        if server_round % 1 == 0:
            # initial_global_params.to(global_model.device)
            # global_model = global_model.to(global_model.device)
            compare_model(initial_global_params, global_model.state_dict())

            pruned_c = 0.0
            total = 0.0
            for name in global_model.mask:
                a = global_model.mask[name].view(-1).to(device='cpu', copy=True).numpy()
                pruned_c +=sum(np.where(a, 0, 1))
                total += aggregated_masks[name].numel()
            print(f'masked : {pruned_c / total}')

        # t3 = time.time()
        # print(f'aggregate cost: {t3 - t2}')

        # evaluate performance
        torch.cuda.empty_cache()
        for name in global_model.mask:
            global_model.mask[name] = torch.zeros_like(global_model.mask[name], dtype=torch.float, device=devices[0])
            # # a = global_model.mask[name].view(-1).to(device='cpu', copy=True).numpy()
            # # pruned_c +=sum(np.where(a, 0, 1))
            # total += aggregated_masks[name].numel()

        if server_round % args.eval_every == 0:
            global_model_cp = copy.deepcopy(global_model)
            if args.prune_at_first_round:
                global_model_cp.apply_weight_mask()

                pruned_c = 0.0
                total = 0.0
                for name in global_model_cp.mask:
                    a = global_model_cp.mask[name].view(-1).to(device='cpu', copy=True).numpy()
                    pruned_c +=sum(np.where(a, 0, 1))
                    total += aggregated_masks[name].numel()
                print(f'check global_model_cp masked : {pruned_c / total}')
            # if args.prune_at_first_round:
            #     global_model.apply_weight_mask()
            train_accuracies, train_losses, test_accuracies, test_losses = evaluate_local(clients, global_model_cp, progress=True,
                                                        n_batches=args.test_batches)
            # train_accuracy, train_losses = test_model(global_model_cp, devices[0], clients.train_data)
            wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=server_round)
            wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=server_round)
            print(f'round: {server_round}')
            print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')
            # wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=server_round)
            # wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=server_round)


            # test_accuracies, test_loss = test_model(global_model_cp, devices[0], clients.test_data)
            wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=server_round)
            wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=server_round)
            print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')
            # wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=server_round)
            # wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=server_round)

        # global_params = global_model.state_dict()

        # for client_id in clients:
        #     i = client_ids.index(client_id)
        #     # if we didn't send initial global params to any clients in the first round, send them now.
        #     # (in the real world, this could be implemented as the transmission of
        #     # a random seed, so the time and place for this is not a concern to us)
        #     if server_round == 0:
        #         clients[client_id].initial_global_params = initial_global_params

        # if server_round == 0:
        #     initial_global_params = deepcopy(global_model.state_dict())

        
        # wandb.log({"Test/Sparsity": sum(sparsities.values())/len(sparsities.values())}, step=server_round)

if __name__ == "__main__":
    # print('????')
    args = parser.parse_args()
    main(args)
    # print('!!!!')