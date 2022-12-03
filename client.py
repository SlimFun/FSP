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
from grasp import Grasp
from force import Force

from collections import OrderedDict

import random
import numpy as np

def check_random_masks(random_masks):
    for mask in random_masks:
        print('total {}'.format(mask.numel()))

class Client:

    def __init__(self, id, device, train_data, test_data, prune_strategy=None, net=models.CNN2,
                 local_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=1e-5, prune_at_first_round=False, bandwidth=0.):
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

        self.bandwidth = bandwidth
        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None

        self.train_size = self._get_train_size()
        # self.keep_masks = None
        self.output_channels = None
        self.first_time = True

        self.param_idx = None

    # def init_model_from_server(self, model, device, server_params, output_channels):
    #     self.net = model(device, output_channels=output_channels).to(device)
    #     # for k, p in server_params.items():
    #     for k, p in self.net.state_dict():
    #         dim = len(p.shape)
    #         if dim == 4:
    #             self.net.copy_(server_params[k])
            

    def init_model(self):
        print(f'client {self.id} init model !!!!')
        self.net.load_state_dict(torch.load('init_model.pt'))


    def reset_optimizer(self, learning_rate=None):
        # if self.id != 8:
        #     learning_rate = 0.001
        if learning_rate != None:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


        # if learning_rate != None:
        #     self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate, weight_decay=self.weight_decay)
        # else:
        #     self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.001, weight_decay=self.weight_decay)
        
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


    def reset_weights(self, global_state_dict):
        print(f'reset_weights: {self.output_channels}')
        # if self.output_channels == None:
        # self.net.load_state_dict(global_state_dict)
        # return
        
        print('===================')
        prio_channel = 3
        local_params = self.net.state_dict()
        idx = -1
        for k in local_params.keys():
            
            shape_dim = len(local_params[k].shape)
            # print(local_params[k].shape)
            if 'shortcut' in k:
                if k not in global_state_dict.keys():
                    continue
                # print(f'{k} {global_state_dict[k].shape}')
                if shape_dim == 3:
                    local_params[k].copy_(global_state_dict[k][:self.output_channels[idx],:,:])
                elif shape_dim == 4:
                    local_params[k].copy_(global_state_dict[k][:self.output_channels[idx],:self.output_channels[idx-2],:,:])
                elif shape_dim == 1:
                    local_params[k].copy_(global_state_dict[k][:self.output_channels[idx]])
            elif shape_dim == 4:
                print(k)
                idx += 1
                # print(f'{k}: {local_params[k].shape} == {self.output_channels[idx]}, {prio_channel}')
                local_params[k].copy_(global_state_dict[k][:self.output_channels[idx],:prio_channel,:,:])
                prio_channel = self.output_channels[idx]
            elif shape_dim == 1:
                # print(f'{k}: {local_params[k].shape} == {self.output_channels[idx]}')
                # if 'running' not in k:
                local_params[k].copy_(global_state_dict[k][:self.output_channels[idx]])
            elif shape_dim == 2:
                # print(f'{k}: {local_params[k].shape} == {self.output_channels[idx]}')
                local_params[k].copy_(global_state_dict[k][:,:self.output_channels[idx]])
            else:
                local_params[k].copy_(global_state_dict[k])
                # print(f'{k}: ERROR')
        
    # def split_model(self, global_state_dict):
    #     # idx_i = [None for _ in range(len(user_idx))]
    #     # idx = [OrderedDict() for _ in range(len(user_idx))]
    #     idx_i = None
    #     idx = OrderedDict()
    #     output_weight_name = [k for k in global_state_dict.keys() if 'weight' in k][-1]
    #     output_bias_name = [k for k in global_state_dict.keys() if 'bias' in k][-1]
    #     print(f'output_weight_name: {output_weight_name}; output_bias_name: {output_bias_name}')
    #     i = 0
    #     for k, v in global_state_dict.items():
    #         parameter_type = k.split('.')[-1]
    #         if 'weight' in parameter_type or 'bias' in parameter_type:
    #             if parameter_type == 'weight':
    #                 if v.dim() > 1:
    #                     input_size = v.size(1)
    #                     output_size = v.size(0)
    #                     # if idx_i[m] is None:
    #                     #     idx_i[m] = torch.arange(input_size, device=v.device)
    #                     if idx_i is None:
    #                         idx_i = torch.arange(input_size, device=v.device)
    #                     input_idx_i_m = idx_i
    #                     if k == output_weight_name:
    #                         output_idx_i_m = torch.arange(output_size, device=v.device)
    #                     else:
    #                         # scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
    #                         # local_output_size = int(np.ceil(output_size * scaler_rate))
    #                         local_output_size = self.output_channels[i]
    #                         i += 1
    #                         output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
    #                     idx[k] = output_idx_i_m, input_idx_i_m
    #                     idx_i = output_idx_i_m
    #                 else:
    #                     print(f'weight dim<=1: {k}')
    #                     input_idx_i_m = idx_i
    #                     idx[k] = input_idx_i_m
    #             else:
    #                 print(f'non weight: {k}')
    #                 if k == output_bias_name:
    #                     input_idx_i_m = idx_i
    #                     idx[k] = input_idx_i_m
    #                 else:
    #                     input_idx_i_m = idx_i
    #                     idx[k] = input_idx_i_m
    #         else:
    #             print(f'others: {k}')
    #             # input_idx_i_m = idx_i
    #             # idx[k] = input_idx_i_m
    #             pass
    #     return idx

    # def reset_weights(self, global_state_dict):
    #     # self.make_model_rate()
    #     if self.param_idx == None:
    #         self.param_idx = self.split_model(global_state_dict)
    #     # local_parameters = [OrderedDict() for _ in range(len(user_idx))]
    #     local_parameters = OrderedDict()
    #     for k, v in global_state_dict.items():
    #         parameter_type = k.split('.')[-1]
    #         if 'weight' in parameter_type or 'bias' in parameter_type:
    #             if 'weight' in parameter_type:
    #                 if v.dim() > 1:
    #                     local_parameters[k] = copy.deepcopy(v[torch.meshgrid(self.param_idx[k])])
    #                 else:
    #                     local_parameters[k] = copy.deepcopy(v[self.param_idx[k]])
    #             else:
    #                 local_parameters[k] = copy.deepcopy(v[self.param_idx[k]])
    #         else:
    #             if v.dim() == 1:
    #                 local_parameters[k] = copy.deepcopy(v[self.param_idx[k]])
    #             else:
    #                 local_parameters[k] = copy.deepcopy(v)
    #     self.net.load_state_dict(local_parameters)

    def _get_train_size(self):
        return sum(len(x) for x,_ in self.train_data)

    def check_prunable(self):
        return self.net.is_maskable and (self.prune_strategy != 'None')

    def record_keep_masks(self, keep_masks):
        masks = torch.cat([torch.flatten(x) for x in keep_masks]).to('cpu').tolist()
        with open(f'./{self.id}_keep_masks.txt', 'a+') as f:
            f.write(json.dumps(masks))
            f.write('\n')

    def get_net_params(self):
        return self.net.cpu().state_dict()

    def local_mask(self, global_params=None, sparsity=0, global_masks=None, saliency_mode=None):
        ul_cost = 0.
        dl_cost = 0.

        self.reset_weights(global_state_dict=global_params)
        if self.prune_strategy in ['SNIP', 'Iter-SNIP', 'layer_base_SNIP']:
            print(f'client: {self.id} **************')
                
            # self.keep_masks = snip.SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
            prune_criterion = SNIP.SNIP(model=self.net, device=self.device)
            prune_criterion.prune_masks(percentage=sparsity, train_loader=self.train_data, last_masks=global_masks, layer_based=self.prune_strategy=='layer_base_SNIP', saliency_mode=saliency_mode)
            self.pruned = True
            # self.record_keep_masks(keep_masks)

            # self.net.mask = [m for m in self.net.mask.values()]
            local_mask = [m for m in self.net.mask.values()]
            # transmit local masks
            ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size()

            ret = dict(state=None, running_loss=None, mask=local_mask, ul_cost=ul_cost, dl_cost=dl_cost)
            # handlers = apply_prune_mask(self.net, self.net.mask)
            return ret
        # elif self.prune_strategy == 'SNAP':
        #     if not self.pruned:
        #         prune_criterion = SNAP.SNAP(model=self.net, device=self.device)
        #         neural_masks = prune_criterion.prune(percentage=sparsity, train_loader=self.train_data)

        #         self.net.mask = neural_masks
        #         ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size() * 32

        #         ret = dict(state=None, running_loss=None, mask=neural_masks, ul_cost=ul_cost, dl_cost=dl_cost)
        #         self.pruned = True
        #         return ret
        #     self.net.init_param_sizes()
        #     dl_cost += self.net.param_size
        elif self.prune_strategy == 'SNAP':
            pass
        elif self.prune_strategy == 'Grasp':
            prune_criterion = Grasp(model=self.net, device=self.device)
            # local_mask = prune_criterion.prune_masks(percentage=sparsity, train_loader=self.train_data, last_masks=global_masks, saliency_mode=saliency_mode)
            masks = prune_criterion.GraSP(ratio=sparsity, train_dataloader=self.train_data, saliency_mode=saliency_mode)
            local_mask = [m for m in masks.values()]
            # local_mask = masks

            ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size()
            ret = dict(state=None, running_loss=None, mask=local_mask, ul_cost=ul_cost, dl_cost=dl_cost)
            return ret
        elif self.prune_strategy == 'Grasp_it':
            prune_criterion = Force(model=self.net, device=self.device)
            prune_criterion.prune_masks(ratio=sparsity, train_dataloader=self.train_data, saliency_mode=saliency_mode)
            local_mask = [m for m in self.net.mask.values()]
            # print(self.net.mask.keys())
            # print(f'local_mask: {len(local_mask)}')

            ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size()

            ret = dict(state=None, running_loss=None, mask=local_mask, ul_cost=ul_cost, dl_cost=dl_cost)
            # handlers = apply_prune_mask(self.net, self.net.mask)
            return ret

    def proximal_loss(self, last_state):

        loss = torch.tensor(0.).to(self.device)

        state = self.get_net_params()
        prio_channel = 3
        local_params = self.net.state_dict()
        idx = -1
        for i, (name, param) in enumerate(state.items()):
            if name.endswith('_mask'):
                continue
            shape_dim = len(param.shape)
            if shape_dim == 4:
                idx += 1
                gpu_param = last_state[name][:self.output_channels[idx],:prio_channel,:,:].to(self.device)
                prio_channel = self.output_channels[idx]
            elif shape_dim == 1:
                gpu_param = last_state[name][:self.output_channels[idx]].to(self.device)
            elif shape_dim == 2:
                gpu_param = last_state[name][:,:self.output_channels[idx]].to(self.device)
            loss += torch.sum(torch.square(param.to(self.device) - gpu_param))
            if gpu_param.data_ptr != last_state[name].data_ptr:
                del gpu_param

        self.net.to(self.device)
        return loss

    def train(self, global_params=None, initial_global_params=None, sparsity=0, single_shot_pruning=False, test_on_each_round=False, clip_grad=False, global_sparsity=None, global_masks=None, learning_rate=None,global_model=None, round=0, prox=0.):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0

        handlers = None
        if self.prune_strategy in ['SNIP', 'Iter-SNIP', 'layer_base_SNIP', 'Grasp', 'Grasp_it', 'synflow', 'given']:
            # if not (self.prune_at_first_round and self.pruned):
            #     print(f'client: {self.id} **************')
            
            #     # self.keep_masks = snip.SNIP(self.net, sparsity, self.train_data, self.device)  # TODO: shuffle?
            #     prune_criterion = SNIP.SNIP(model=self.net, device=self.device)
            #     prune_criterion.prune_masks(percentage=sparsity, train_loader=self.train_data)
            #     self.pruned = True
            #     # self.record_keep_masks(keep_masks)

            #     self.net.mask = [m for m in self.net.mask.values()]
            #     # transmit local masks
            #     ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size()

            #     if single_shot_pruning:
            #         ret = dict(state=None, running_loss=None, mask=copy.deepcopy(self.net.mask), ul_cost=ul_cost, dl_cost=dl_cost)
            #         # handlers = apply_prune_mask(self.net, self.net.mask)
            #         return ret

                # handlers = apply_prune_mask(self.net, self.net.mask)
            # single shot snip pruning
            # downloads partial global model(do we need masks now ???)
            # print(global_sparsity)
            dl_cost += (1-global_sparsity) * self.net.param_size
            self.net.mask = global_masks
                
            handlers = apply_grad_mask(self.net, self.net.mask)
        elif self.prune_strategy == 'SNAP':
            if not self.pruned:
                prune_criterion = SNAP.SNAP(model=self.net, device=self.device)
                neural_masks = prune_criterion.prune(percentage=sparsity, train_loader=self.train_data)

                self.net.mask = neural_masks
                ul_cost += (1-self.net.sparsity_percentage()) * self.net.mask_size() * 32

                ret = dict(state=None, running_loss=None, mask=neural_masks, ul_cost=ul_cost, dl_cost=dl_cost)
                self.pruned = True
                return ret
            self.net.init_param_sizes()
            dl_cost += self.net.param_size
        elif self.prune_strategy == 'None':
            dl_cost += self.net.param_size
        elif self.prune_strategy == 'random_masks':
            # print('==========random_masks===============')
            # print(random_masks)
            # check_random_masks(random_masks)
            dl_cost += (1-global_sparsity) * self.net.param_size
            handlers = apply_grad_mask(self.net, global_masks)


        
        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            # if self.first_time:
            self.reset_weights(global_state_dict=global_params)
                # self.first_time = False

            # Try to reset the optimizer state.
            self.reset_optimizer(learning_rate)

        init_params = copy.deepcopy(self.get_net_params())


        if test_on_each_round:
            print('**********test before train*************')
            zero_params_num = count_model_zero_params(self.net.state_dict())
            print(f'global model zero params: {zero_params_num}')

            accuracy, loss = self.test(self.net, train_data=True)
            print(f'Train accuracy: {accuracy}; loss: {loss}')

            # accuracy, loss = self.test(self.net, train_data=False)
            # print(f'Test accuracy: {accuracy}; loss: {loss}')
            print('**********test before train*************')

        # print('*********************model_size**********************')
        # for name, params in self.net.state_dict().items():
        #     print(f'name: {name}; shape: {params.shape}')
        # print('*********************model_size**********************')

        self.net.to(self.device)
        total = 0.
        
        self.net.train()
        t0 = time.time()
        total_trianing_time = []
        total_time = []

        # server_model = copy.deepcopy(global_model)
        # server_model = server_model.to(self.device)
        # divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
        # temp = 1

        for epoch in range(self.local_epochs):
            # break

            running_loss = 0.
            tc = 0.
            n = 0
            total_norm = 0.
            gradnorms = []
            for inputs, labels in self.train_data:
                # pass
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                t1 = time.time()
                self.net.zero_grad()
                outputs = self.net(inputs)

                # with torch.no_grad():
                #     teacher_preds = server_model(inputs)
                # ditillation_loss = divergence_loss_fn(
                #     F.softmax(outputs / temp, dim=1),
                #     F.softmax(teacher_preds / temp, dim=1)
                # )
                if prox > 0:
                    loss += prox / 2. * self.proximal_loss(global_params)
                # ditillation_loss.backward()

                # loss = 0.2 * self.criterion(outputs, labels) + 0.8 * ditillation_loss

                loss = self.criterion(outputs, labels)
                loss.backward()

                # if clip_grad:
                #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                # if round >= 10:
                # grad_norm = 9 - round if 9 - round > 0 else 0.5
                # print(f'grad_norm: {grad_norm}')
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm)
                # if round > 7:
                # if self.id in [0,1,2,3,4,5,6,7,9]:
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                # for p in self.net.parameters():
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # gradnorms.append(total_norm ** (1. / 2))
                

                self.optimizer.step()

                running_loss += loss.item()
                # print(f'{time.time() - t0}')
                total += inputs.size(0)
                n += inputs.size(0)
                tc += time.time() - t1
            total_trianing_time.append(tc)
            # print(float(sum(tc))/len(tc))
            # print(f'client {self.id} grad_norm: {sum(gradnorms)/len(gradnorms)} ')
            print(f'running loss: {running_loss / len(self.train_data)}')
            self.curr_epoch += 1
        print(f'total: {total}; time: {(time.time() - t0)/self.local_epochs}')
        print(f'training time: {float(sum(total_trianing_time))}')
        # 1/0


        if test_on_each_round:
            print('**********test after train*************')
            zero_params_num = count_model_zero_params(self.net.state_dict())
            print(f'global model zero params: {zero_params_num}')

            accuracy, loss = self.test(self.net, train_data=True)
            print(f'accuracy: {accuracy}; loss: {loss}')

            # accuracy, loss = self.test(self.net, train_data=False)
            # print(f'Test accuracy: {accuracy}; loss: {loss}')
            print('**********test after train*************')

        finish_params = copy.deepcopy(self.get_net_params())
        for param_tensor in finish_params:
            finish_params[param_tensor] -= init_params[param_tensor]

        zero_params_percetage = count_model_zero_params(finish_params)
        print(f'client {self.id} finish params zeros: {zero_params_percetage}')
        if self.prune_strategy in ['SNIP', 'Iter-SNIP', 'layer_base_SNIP', 'Grasp', 'Grasp_it', 'synflow', 'given']:
            ul_cost += (1-self.net.sparsity_percentage()) * self.net.param_size
        elif self.prune_strategy == 'None':
            ul_cost += self.net.param_size
        elif self.prune_strategy == 'SNAP':
            ul_cost += self.net.param_size
        elif self.prune_strategy == 'random_masks':
            ul_cost += (1-global_sparsity) * self.net.param_size

        if handlers is not None:
            for h in handlers:
                h.remove()
        
        print(f'dl_cost: {dl_cost}; ul_cost: {ul_cost}')
        ## personalized target32
        training_times = [30.118340253829956,25.567391633987427,31.870572566986084,30.326955556869507,29.87587594985962,27.24259901046753,30.010392665863037,31.961384773254395,33.668541431427,29.92017436027527]
        comm_times = [1.7159310612372631,6.803562243905361,2.4722066286463735,2.1826581373991143,1.9637254110061555,4.967846687173153,2.245153596359483,2.589699972950448,0.8584250011028982,2.2392911638165636]

                
        ret = dict(state=finish_params, running_loss=None, mask=None, ul_cost=ul_cost, dl_cost=dl_cost, training_time=training_times[self.id] + comm_times[self.id])
        # ret = dict(state=finish_params, running_loss=None, mask=None, ul_cost=ul_cost, dl_cost=dl_cost, training_time=float(sum(total_trianing_time)))
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

