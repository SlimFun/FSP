from email import utils
from logging import handlers
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import time
import copy

import cv_models.models as models
from snip import SNIP
from utils import apply_prune_mask, compare_model
import wandb

class Client:

    def __init__(self, id, device, train_data, test_data, prune_strategy=None, net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=0.001):
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

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None
        


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


    # def reset_weights(self, *args, **kwargs):
    #     return self.net.reset_weights(*args, **kwargs)
    def reset_weights(self, global_state_dict):
        self.net.load_state_dict(global_state_dict)


    # def sparsity(self, *args, **kwargs):
    #     return self.net.sparsity(*args, **kwargs)


    def train_size(self):
        return sum(len(x) for x in self.train_data)


    def train(self, global_params=None, initial_global_params=None, sparsity=0):
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
        handlers = None
        if self.prune_strategy == 'SNIP':
            keep_masks = SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
            handlers = apply_prune_mask(self.net, keep_masks)

        # init_params = self.net.state_dict()
        init_model = copy.deepcopy(self.net)

        #pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        # t0 = time.time()
        for epoch in range(self.local_epochs):

            self.net.train()

            running_loss = 0.
            # print(len(self.train_data))
            for inputs, labels in self.train_data:
                # print(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                # if args.prox > 0:
                #     loss += args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()
                self.optimizer.step()

                # self.net

                # self.reset_weights() # applies the mask

                running_loss += loss.item()
            
            # print(running_loss)

            self.curr_epoch += 1

        finish_params = self.net.state_dict()
        for param_tensor in finish_params:
            finish_params[param_tensor] -= init_model.state_dict()[param_tensor]

        if handlers is not None:
            for h in handlers:
                h.remove()

        # compare_model(self.net.state_dict(), init_model.state_dict())
        # print(f'local training cost: {time.time() - t0}')

        # # we only need to transmit the masked weights and all biases
        # if args.fp16:
        #     ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 16 + (self.net.param_size - self.net.mask_size * 16)
        # else:
        #     ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        ret = dict(state=finish_params, running_loss=running_loss)

        #dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        #dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])
        return ret

    def test(self, model=None, n_batches=0):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                if i > n_batches and n_batches > 0:
                    break
                # if not args.cache_test_set_gpu:
                #     inputs = inputs.to(self.device)
                #     labels = labels.to(self.device)
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        return correct / total

