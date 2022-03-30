from logging import handlers
import re
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import time
import copy
import json

# import cv_models.models as models
# from snip import SNIP
import models.models as models
# from SNIP import SNIP
# from snip import SNIP
import SNIP
import SNAP
from utils import apply_global_mask, apply_prune_mask, compare_model, count_model_zero_params, apply_grad_mask
import wandb

import random
import numpy as np

class Client:

    def __init__(self, id, device, train_data, test_data, prune_strategy=None, net=models.CNN2,
                 local_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=1e-5, prune_at_first_round=False):
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

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.prune_strategy = prune_strategy

        self.device = device
        self.net = net(device=self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.reset_optimizer()

        self.prune_at_first_round = prune_at_first_round
        self.pruned = False

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None

        self.train_size = self._get_train_size()
        # self.keep_masks = None


    def init_model(self):
        print(f'client {self.id} init model !!!!')
        self.net.load_state_dict(torch.load('init_model.pt'))


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


    def reset_weights(self, global_state_dict):
        self.net.load_state_dict(global_state_dict)


    def _get_train_size(self):
        return sum(len(x) for x in self.train_data)

    def check_prunable(self):
        return self.net.is_maskable and (self.prune_strategy != 'None')

    def record_keep_masks(self, keep_masks):
        masks = torch.cat([torch.flatten(x) for x in keep_masks]).to('cpu').tolist()
        with open(f'./{self.id}_keep_masks.txt', 'a+') as f:
            f.write(json.dumps(masks))
            f.write('\n')

    def get_net_params(self):
        return self.net.cpu().state_dict()

    def train(self, global_params=None, initial_global_params=None, sparsity=0, single_shot_pruning=False, test_on_each_round=False, clip_grad=False, global_sparsity=None):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0

        handlers = None
        if self.prune_strategy == 'SNIP':
            if not (self.prune_at_first_round and self.pruned):
                print(f'client: {self.id} **************')
            
                # self.keep_masks = snip.SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
                prune_criterion = SNIP.SNIP(model=self.net, device=self.device)
                prune_criterion.prune_masks(percentage=sparsity, train_loader=self.train_data)
                self.pruned = True
                # self.record_keep_masks(keep_masks)

                self.net.mask = [m for m in self.net.mask.values()]
                # transmit local masks
                ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size()

                if single_shot_pruning:
                    ret = dict(state=None, running_loss=None, mask=copy.deepcopy(self.net.mask), ul_cost=ul_cost, dl_cost=dl_cost)
                    # handlers = apply_prune_mask(self.net, self.net.mask)
                    return ret

                # handlers = apply_prune_mask(self.net, self.net.mask)
            # single shot snip pruning
            # downloads partial global model(do we need masks now ???)
            # print(global_sparsity)
            dl_cost += (1-global_sparsity) * self.net.param_size
                
            handlers = apply_grad_mask(self.net, self.net.mask)
        elif self.prune_strategy == 'SNAP':
            if not self.pruned:
                prune_criterion = SNAP.SNAP(model=self.net, device=self.device)
                neural_masks = prune_criterion.prune(percentage=sparsity, train_loader=self.train_data)

                self.net.mask = neural_masks
                ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size() * 32

                ret = dict(state=None, running_loss=None, mask=neural_masks, ul_cost=0., dl_cost=0.)
                self.pruned = True
                return ret
            self.net.init_param_sizes()
            dl_cost += self.net.param_size
        elif self.prune_strategy == 'None':
            dl_cost += self.net.param_size

        
        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            self.reset_weights(global_state_dict=global_params)

            # Try to reset the optimizer state.
            self.reset_optimizer()

        init_params = copy.deepcopy(self.get_net_params())


        if test_on_each_round:
            print('**********test before train*************')
            zero_params_num = count_model_zero_params(self.net.state_dict())
            print(f'global model zero params: {zero_params_num}')

            accuracy, loss = self.test(self.net, train_data=True)
            print(f'accuracy: {accuracy}; loss: {loss}')
            print('**********test before train*************')

        # print('*********************model_size**********************')
        # for name, params in self.net.state_dict().items():
        #     print(f'name: {name}; shape: {params.shape}')
        # print('*********************model_size**********************')

        t0 = time.time()
        self.net.to(self.device)
        total = 0.
        
        for epoch in range(self.local_epochs):
            # break

            self.net.train()
            running_loss = 0.
            i = 0
            for inputs, labels in self.train_data:
                # print(f'i : {i}')
                # i += 1
                # if self.check_prunable():
                #     self.net.apply_weight_mask()
                # print(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # self.optimizer.zero_grad() 
                self.net.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                # if args.prox > 0:
                #     loss += args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()

                # if clip_grad:
                #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)

                self.optimizer.step()
                # total += len(labels)
                # if self.check_prunable():
                #     self.net.apply_weight_mask()

                running_loss += loss.item()
                # if i >= 5: 
                # break

            print(f'running loss: {running_loss / len(self.train_data)}')
            

            self.curr_epoch += 1
        print(f'total: {total}; time: {time.time() - t0}')

        finish_params = copy.deepcopy(self.get_net_params())
        for param_tensor in finish_params:
            finish_params[param_tensor] -= init_params[param_tensor]

        zero_params_percetage = count_model_zero_params(finish_params)
        print(f'client {self.id} finish params zeros: {zero_params_percetage}')
        if self.prune_strategy == 'SNIP':
            ul_cost += (1-self.net.sparsity_percentage()) * self.net.param_size
        elif self.prune_strategy == 'None':
            ul_cost += self.net.param_size

        if handlers is not None:
            for h in handlers:
                h.remove()
        
        print(f'dl_cost: {dl_cost}; ul_cost: {ul_cost}')
        ret = dict(state=finish_params, running_loss=None, mask=None, ul_cost=ul_cost, dl_cost=dl_cost)
        # ret = dict(state=finish_params, running_loss=None, mask=self.net.mask, ul_cost=ul_cost, dl_cost=dl_cost)

        return ret

    def test(self, model=None, n_batches=0, train_data=False):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        correct = 0.
        total = 0.
        loss = 0.

        _model.eval()
        data_loader = self.train_data if train_data else self.test_data
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                if i > n_batches and n_batches > 0:
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = _model(inputs)
                loss += self.criterion(outputs, labels) * len(labels)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        # if model is not _model:
        #     del _model

        print(f'Test client {self.id}: Accuracy: {correct / total}; Loss: {loss / total}; Total: {total};')

        return correct / total, loss / total

