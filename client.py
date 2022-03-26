from email import utils
from logging import handlers
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
import snip
from utils import apply_global_mask, apply_prune_mask, compare_model
import wandb

import random
import numpy as np

class Client:

    def __init__(self, id, device, train_data, test_data, prune_strategy=None, net=models.CNN2,
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

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.prune_strategy = prune_strategy

        self.device = device
        self.net = net(device=self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

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
        self.keep_masks = None


    def init_model(self):
        print(f'client {self.id} init model !!!!')
        self.net.load_state_dict(torch.load('init_model.pt'))


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
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

    def train(self, global_params=None, initial_global_params=None, sparsity=0, single_shot_pruning=False, test_on_each_round=False, clip_grad=False):
        '''Train the client network for a single round.'''

        dl_cost = 0

        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            self.reset_weights(global_state_dict=global_params)

            # Try to reset the optimizer state.
            self.reset_optimizer()

            # if mask_changed:
            #     dl_cost += self.net.mask_size # need to receive mask

            # if not self.initial_global_params:
            #     self.initial_global_params = initial_global_params
            #     # no DL cost here: we assume that these are transmitted as a random seed
            # else:
            #     # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
            #     # all parameters that don't have a mask (e.g. biases in this case)
            #     dl_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        # self.init_model()

        
        
        handlers = None
        if self.prune_strategy == 'SNIP':
            if not (self.prune_at_first_round and self.pruned):
                print(f'client: {self.id} **************')
            
                self.keep_masks = snip.SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
                self.pruned = True
                # self.record_keep_masks(keep_masks)

                if single_shot_pruning:
                    ret = dict(state=None, running_loss=None, mask=self.keep_masks)
                    # handlers = apply_prune_mask(self.net, self.keep_masks)
                    return ret

                handlers = apply_prune_mask(self.net, self.keep_masks)

        init_params = copy.deepcopy(self.get_net_params())


        if test_on_each_round:
            print('**********test before train*************')

            pruned_c = 0.0
            total = 0.0

            for name, param in self.net.state_dict().items():
                a = param.view(-1).to(device='cpu', copy=True).numpy()
                pruned_c +=sum(np.where(a, 0, 1))
                total += param.numel()
            print(f'global model zero params: {pruned_c / total}')

            accuracy, loss = self.test(self.net, train_data=True)
            print(f'accuracy: {accuracy}; loss: {loss}')
            print('**********test before train*************')

        self.net.to(self.device)
        total = 0.
        for epoch in range(self.local_epochs):
            # break
            
            self.net.train()

            running_loss = 0.
            i = 0
            for inputs, labels in self.train_data:
                # print(f'i : {i}')
                i += 1
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

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)

                self.optimizer.step()
                total += len(labels)
                # if self.check_prunable():
                #     self.net.apply_weight_mask()

                running_loss += loss.item()
                # if i >= 5:
                # break

            print(f'running loss: {running_loss / len(self.train_data)}')
            

            self.curr_epoch += 1
        print('total: ', total)

        finish_params = copy.deepcopy(self.get_net_params())
        for param_tensor in finish_params:
            finish_params[param_tensor] -= init_params[param_tensor]

        if handlers is not None:
            for h in handlers:
                h.remove()
        
        ret = dict(state=finish_params, running_loss=None, mask=self.keep_masks)

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

