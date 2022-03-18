from email import utils
from logging import handlers
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import time
import copy

# import cv_models.models as models
# from snip import SNIP
import models.models as models
# from SNIP import SNIP
# from snip import SNIP
import SNIP
import snip
from utils import apply_prune_mask, compare_model
import wandb

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
        # print('init client')
        # print(id)
        # print(device)

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
        


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


    def reset_weights(self, global_state_dict):
        self.net.load_state_dict(global_state_dict)

    # def reset_weights(self, *args, **kwargs):
    #     return self.net.reset_weights(*args, **kwargs)
    def reset_weights_and_mask(self, global_state_dict):
        self.net.load_state_dict(global_state_dict)

        self.net.rebuild_mask()


    # def sparsity(self, *args, **kwargs):
    #     return self.net.sparsity(*args, **kwargs)


    def _get_train_size(self):
        return sum(len(x) for x in self.train_data)

    def check_prunable(self):
        return self.net.is_maskable and (self.prune_strategy != 'None')


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
            if not (self.prune_at_first_round and self.pruned):
                print(f'client: {self.id} **************')
                prune_criterion = SNIP.SNIP(model=self.net, limit=sparsity, start=None, steps=None, device=self.device)
                # prune_criterion = SNAP(model=net, device=device)
                prune_criterion.prune(sparsity, train_loader=self.train_data, manager=None)
                self.pruned = True
            
        #     # keep_masks = snip.SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
        #     # handlers = apply_prune_mask(self.net, keep_masks)

        if self.check_prunable():
            self.net.apply_weight_mask()

        # init_params = self.net.state_dict()
        init_model = copy.deepcopy(self.net)

        #pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        # t0 = time.time()
        total = 0.
        for epoch in range(self.local_epochs):
            
            self.net.train()

            running_loss = 0.
            # print(len(self.train_data))
            # print(epoch)
            i = 0
            for inputs, labels in self.train_data:
                # print(f'i : {i}')
                # i += 1
                if self.check_prunable():
                    self.net.apply_weight_mask()
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
                total += len(labels)
                if self.check_prunable():
                    self.net.apply_weight_mask()

                # self.net

                # self.reset_weights() # applies the mask

                running_loss += loss.item()

            print(f'running loss: {running_loss / len(self.train_data)}')
            
            # print(running_loss)

            self.curr_epoch += 1
        print('total: ', total)

        # ret = dict(state=self.net.state_dict(), running_loss=running_loss, mask=self.net.mask)

        finish_params = self.net.state_dict()
        for param_tensor in finish_params:
            finish_params[param_tensor] -= init_model.state_dict()[param_tensor]

        if handlers is not None:
            for h in handlers:
                h.remove()

        compare_model(self.net.state_dict(), init_model.state_dict())
        # print(f'local training cost: {time.time() - t0}')

        # # we only need to transmit the masked weights and all biases
        # if args.fp16:
        #     ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 16 + (self.net.param_size - self.net.mask_size * 16)
        # else:
        #     ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        
        
        ret = dict(state=finish_params, running_loss=running_loss, mask=self.net.mask)

        return ret

        #dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        #dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])

    def test(self, model=None, n_batches=0, train_data=False):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.
        loss = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        data_loader = self.train_data if train_data else self.test_data
        # _model.apply_weight_mask()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                if i > n_batches and n_batches > 0:
                    break
                # if not args.cache_test_set_gpu:
                #     inputs = inputs.to(self.device)
                #     labels = labels.to(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = _model(inputs)
                loss += self.criterion(outputs, labels) * len(labels)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        print(f'Test client {self.id}: Accuracy: {correct / total}; Loss: {loss / total}; Total: {total};')

        return correct / total, loss / total

