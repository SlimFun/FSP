import copy
from enum import Flag
from torch import nn
import utils
from collections import Counter
import numpy as np
import torch
import SNAP
import torch.nn as nn
from models.vgg import VGG
import math
import random
import torch.nn.functional as F
from collections import OrderedDict

import wandb

def needs_mask(layer_name):
    return layer_name.endswith('weight') and ('bn' not in layer_name)

class Server:
    def __init__(self, model, device, train_data=None, prune_strategy='None', target_keep_ratio=1., clients=None) -> None:
        self.device = device
        self.model = model(self.device).to(self.device)
        self._init_masks()
        self.train_data = train_data
        self.pruned = False

        self.transmission_cost = 0.
        self.compute_time = 0.
        self.prune_strategy = prune_strategy
        self.target_keep_ratio = target_keep_ratio
        self.vote = None
        self.output_channels = None
        self.min_channels = None
        self.clients = clients
        # torch.save(self.model.state_dict(), 'ori_init_model.pt')

    def _init_masks(self):
        masks = []
        with torch.no_grad():
            if isinstance(self.model, VGG):
                for name, layer in self.model.named_children():
                    for n, l in layer.named_children():
                        if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                            continue

                        for pname, param in l.named_parameters():
                            if not needs_mask(pname):
                                continue
                            
                            mask = torch.ones_like(param.data)
                            print('params.data.shape: {}; mask.shape: {}'.format(param.data.shape, mask.shape))
                            masks.append(mask)
            else:
                for name, layer in self.model.named_children():
                    for pname, param in layer.named_parameters():
                        if not needs_mask(pname):
                            continue

                        mask = torch.ones_like(param.data)
                        masks.append(mask)

        self.masks = masks

    def get_global_params(self):
        return self.model.cpu().state_dict()

    def generate_random_masks(self, sparsity=0.8, sparsity_distribution='uniform'):
        masks = []
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            print(f'weights_by_layer: {weights_by_layer}')
            if isinstance(self.model, VGG):
                for name, layer in self.model.named_children():
                    for n, l in layer.named_children():
                        if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                            continue
                        # We need to figure out how many to prune
                        n_total = 0
                        # for bname, buf in layer.named_buffers():
                        for pname, param in l.named_parameters():
                            if needs_mask(pname):
                                n_total += param.numel()
                        n_prune = int(n_total - weights_by_layer[n])
                        print('total: {}; n_prune: {}'.format(n_total, n_prune))
                        if n_prune >= n_total or n_prune < 0:
                            continue
                        #print('prune out', n_prune)

                        for pname, param in l.named_parameters():
                            if not needs_mask(pname):
                                continue

                            # Determine smallest indices
                            _, prune_indices = torch.topk(torch.abs(param.data.flatten()),
                                                        n_prune, largest=False)

                            mask = torch.ones_like(param.data.flatten())
                            mask[prune_indices] = 0
                            mask = torch.reshape(mask, param.data.shape)
                            print('params.data.shape: {}; mask.shape: {}'.format(param.data.shape, mask.shape))
                            masks.append(mask)
            else:
                for name, layer in self.model.named_children():

                    # We need to figure out how many to prune
                    n_total = 0
                    # for bname, buf in layer.named_buffers():
                    for pname, param in layer.named_parameters():
                        if needs_mask(pname):
                            n_total += param.numel()
                    n_prune = int(n_total - weights_by_layer[name])
                    print('total: {}; n_prune: {}'.format(n_total, n_prune))
                    if n_prune >= n_total or n_prune < 0:
                        continue
                    #print('prune out', n_prune)

                    for pname, param in layer.named_parameters():
                        if not needs_mask(pname):
                            continue

                        # Determine smallest indices
                        _, prune_indices = torch.topk(torch.abs(param.data.flatten()),
                                                    n_prune, largest=False)

                        mask = torch.ones_like(param.data.flatten())
                        mask[prune_indices] = 0
                        masks.append(torch.reshape(mask, param.data.shape))
                        # Write and apply mask
                        # param.data.view(param.data.numel())[prune_indices] = 0
                        # for bname, buf in layer.named_buffers():
                        #     if bname == pname + '_mask':
                        #         buf.view(buf.numel())[prune_indices] = 0
                #print('pruned sparsity', self.sparsity())
        # return torch.Tensor(masks).to(self.device)
        return masks

    def _weights_by_layer(self, sparsity=0.8, sparsity_distribution='uniform'):
        with torch.no_grad():
            layer_names = []
            count = 0
            for i, (name, layer) in enumerate(self.model.named_children()):
                for i, (n, l) in enumerate(layer.named_children()):
                    if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                            continue
                    count += 1
            sparsities = np.empty(count)
            n_weights = np.zeros_like(sparsities, dtype=np.int)
            
            if isinstance(self.model, VGG):
                index = 0
                for i, (name, layer) in enumerate(self.model.named_children()):
                    for i, (n, l) in enumerate(layer.named_children()):
                        if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                            continue
                        layer_names.append(n)
                        for pname, param in l.named_parameters():
                            if needs_mask(pname):
                                n_weights[index] += param.numel()
                        if sparsity_distribution == 'uniform':
                            sparsities[index] = sparsity
                            index += 1
                            continue
                        kernel_size = None
                        if isinstance(l, nn.modules.conv._ConvNd):
                            neur_out = l.out_channels
                            neur_in = l.in_channels
                            kernel_size = l.kernel_size
                        elif isinstance(l, nn.Linear):
                            neur_out = l.out_features
                            neur_in = l.in_features
                        else:
                            raise ValueError('Unsupported layer type ' + type(l))
                        
                        if sparsity_distribution == 'er':
                            sparsities[index] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                        elif sparsity_distribution == 'erk':
                            # print('erk')
                            # print(l)
                            if isinstance(l, nn.modules.conv._ConvNd):
                                sparsities[index] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                            else:
                                # print('er')
                                sparsities[index] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                        else:
                            raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
                        index += 1

                # Now we need to renormalize sparsities.
                # We need global sparsity S = sum(s * n) / sum(n) equal to desired
                # sparsity, and s[i] = C n[i]
                # print(n_weights)
                sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
                # print(f'n_weights: {n_weights}; sparsities: {sparsities}')
                n_weights = np.floor((1-sparsities) * n_weights)

                return {layer_names[i]: n_weights[i] for i in range(len(layer_names))}
                        
            else:
                layer_names = []
                sparsities = np.empty(len(list(self.model.named_children())))
                n_weights = np.zeros_like(sparsities, dtype=np.int)

                for i, (name, layer) in enumerate(self.model.named_children()):

                    layer_names.append(name)
                    for pname, param in layer.named_parameters():
                        if needs_mask(pname):
                            n_weights[i] += param.numel()

                    if sparsity_distribution == 'uniform':
                        sparsities[i] = sparsity
                        continue
                    
                    kernel_size = None
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        neur_out = layer.out_channels
                        neur_in = layer.in_channels
                        kernel_size = layer.kernel_size
                    elif isinstance(layer, nn.Linear):
                        neur_out = layer.out_features
                        neur_in = layer.in_features
                    else:
                        raise ValueError('Unsupported layer type ' + type(layer))

                    if sparsity_distribution == 'er':
                        sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                    elif sparsity_distribution == 'erk':
                        if isinstance(layer, nn.modules.conv._ConvNd):
                            sparsities[i] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                        else:
                            sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                    else:
                        raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
                    
                # Now we need to renormalize sparsities.
                # We need global sparsity S = sum(s * n) / sum(n) equal to desired
                # sparsity, and s[i] = C n[i]
                sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
                n_weights = np.floor((1-sparsities) * n_weights)

                return {layer_names[i]: n_weights[i] for i in range(len(layer_names))}

    def set_global_params(self, params_dict):
        self.model.load_state_dict(params_dict)

    def _count_vote(self, masks, vote):
        tc = 0.
        keeped = 0.
        for i in range(len(masks)):
            tc += masks[i].numel()
            m = masks[i].clone().detach()
            keeped += torch.sum(torch.where(m >= vote, 1, 0))
        return keeped / tc

    def decide_vote_num(self, keep_masks):
        for vote in range(10):
            if self._count_vote(keep_masks, vote) <= self.target_keep_ratio:
                return vote

    def keep_topk_masks(self, masks, last_masks):
        flat_masks = torch.cat([m.flatten() for m in masks])
        flat_last_masks = torch.cat([m.flatten() for m in last_masks])
        print(f'{flat_last_masks.shape} == {flat_masks.shape}')
        flat_masks[flat_last_masks == 0] = -1
        keep_num = math.ceil(len(flat_masks) * self.target_keep_ratio) if random.random() > 0.5 else math.floor(len(flat_masks) * self.target_keep_ratio)
        thd, indices = flat_masks.topk(keep_num)
        print(f'threshold : {thd[-1]}')
        topk_masks = torch.zeros_like(flat_masks)
        topk_masks[indices] = 1
        idx = 0
        ms = []
        for i in range(len(masks)):
            m = topk_masks[idx:idx+masks[i].numel()].reshape(masks[i].size())
            idx += masks[i].numel()
            ms.append(m)
        return ms

    def keep_topk_masks_with_erk(self, masks, last_masks):
        weights_by_layer = list(self._weights_by_layer(1 - self.target_keep_ratio, sparsity_distribution='erk').values())
        ret = []
        idx = 0
        for num_of_keep in weights_by_layer:
            masks[idx][last_masks[idx] == 0] = -1
            flat_masks = masks[idx].flatten()
            # print(num_of_keep / m.numel())
            thd, indices = flat_masks.topk(math.ceil(num_of_keep) if random.random() > 0.5 else math.floor(num_of_keep))
            print(f'flat_masks.shape: {flat_masks.shape}; threshold : {thd[-1]}')
            topk_masks = torch.zeros_like(flat_masks)
            topk_masks[indices] = 1
            ret.append(topk_masks.reshape(masks[idx].shape))
            idx += 1
        return ret
        # flat_masks = torch.cat([m.flatten() for m in masks])
        # keep_num = math.ceil(len(flat_masks) * self.target_keep_ratio) if random.random() > 0.5 else math.floor(len(flat_masks) * self.target_keep_ratio)
        # _, indices = flat_masks.topk(keep_num)
        # topk_masks = torch.zeros_like(flat_masks)
        # topk_masks[indices] = 1
        # idx = 0
        # ms = []
        # for i in range(len(masks)):
        #     m = topk_masks[idx:idx+masks[i].numel()].reshape(masks[i].size())
        #     idx += masks[i].numel()
        #     ms.append(m)
        # return ms

    def _merge_local_masks(self, keep_masks_dict, last_masks, num_training_data):
    #keep_masks_dict[clients][params]
        total_training_data = sum(num_training_data) if num_training_data != None else 0.
        print('merge local masks')
        for m in range(len(keep_masks_dict[0])):
            for client_id in keep_masks_dict.keys():
                w = float(num_training_data[client_id]) / total_training_data if num_training_data != None else 1.
                print(f'merge mask weight: {w}')
            # for j in range(0, len(keep_masks_dict.keys())):
                if client_id == 0:
                    keep_masks_dict[0][m] = keep_masks_dict[client_id][m] * w
                else:
                    keep_masks_dict[0][m] += keep_masks_dict[client_id][m] * w

        keep_masks_dict[0] = self.keep_topk_masks(keep_masks_dict[0], last_masks)

        if self.prune_strategy == 'SNAP':
            for i in range(len(keep_masks_dict[0])):
                keep_masks_dict[0][i] = torch.where(keep_masks_dict[0][i]>=6, 1, 0)
        
        return keep_masks_dict[0]

    def _prune_global_model(self, masks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.model.modules())

        for layer, keep_mask in zip(prunable_layers, masks):
            assert (layer.weight.shape == keep_mask.shape)

            layer.weight.data[keep_mask == 0.] = 0.

    def round_trans_cost(self, download_cost, upload_cost):
        trans_cost = 0.
        for c in range(len(download_cost)):
            trans_cost += (download_cost[c] + upload_cost[c]) / (1024 * 1024)
        return trans_cost

    def round_compute_time(self, compute_times):
        comp_t = 0.
        for c in range(len(compute_times)):
            comp_t += compute_times[c]
        return comp_t

    def masked_percent(self):
        pruned_c = 0.0
        total = 0.0

        for name, param in self.model.state_dict().items():
            a = param.view(-1).to(device='cpu', copy=True).numpy()
            pruned_c +=sum(np.where(a, 0, 1))
            total += param.numel()
        # for m in self.masks:
        #     a = m.view(-1).to(device='cpu', copy=True).numpy()
        #     pruned_c += sum(np.where(a, 0, 1))
        #     total += m.numel()
        return pruned_c / total

    def merge_masks(self, keep_masks_dict, keep_ratio, download_cost, upload_cost, compute_times, last_masks, num_training_data=None):
        self.transmission_cost += self.round_trans_cost(download_cost, upload_cost)
        # self.compute_time += self.round_compute_time(compute_times)
        # self.compute_time += max(compute_times)

        # self.compute_time += 48.326801575719195

        # print(f'keep masks dict[0]: {keep_masks_dict[0]}')
        assert keep_masks_dict[0] is not None, 'pruning stage local keep masks can not be None'
        # if (keep_masks_dict[0] is not None):
        # if self.masks is None:
        self.masks = self._merge_local_masks(keep_masks_dict, last_masks, num_training_data)
            # self.model.mask = self.masks

        # applyed_masks = copy.deepcopy(self.masks)
        # for i in range(len(applyed_masks)):
        #     applyed_masks[i] = applyed_masks[i].cpu().numpy().tolist()
        # with open(f'./applyed_masks.txt', 'a+') as f:
        #     f.write(json.dumps(applyed_masks))
        #     f.write('\n')

        # if self.prune_strategy in ['SNIP', 'Iter-SNIP']:
        #     self._prune_global_model(self.masks)
        # elif self.prune_strategy == 'SNAP':
        #     if not self.pruned:
        #         print('global model prune')
        #         # self.model = self.model.to(self.device)
        #         SNAP.SNAP(model=self.model, device=self.device).prune_global_model(self.masks, self.train_data)
        #         self.pruned = True
        self.model.mask = self.masks



    # def aggregate(self, keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times):
    #     self.transmission_cost += self.round_trans_cost(download_cost, upload_cost)
    #     self.compute_time += self.round_compute_time(compute_times)
    #     last_params = self.get_global_params()

    #     training_num = sum(local_train_num for (local_train_num, _) in model_list)

    #     (num0, averaged_params) = model_list[0]
    #     if (averaged_params is not None):
    #         for k in averaged_params.keys():
    #             for i in range(0, len(model_list)):
    #                 local_sample_number, local_model_params = model_list[i]
    #                 w = local_sample_number / training_num
    #                 if i == 0:
    #                     averaged_params[k] = local_model_params[k] * w
    #                 else:
    #                     averaged_params[k] += local_model_params[k] * w

    #         # for name, param in averaged_params.items():
    #         for name in last_params:
    #             assert (last_params[name].shape == averaged_params[name].shape)
    #             averaged_params[name] = averaged_params[name].type_as(last_params[name])
    #             # last_params[name] = last_params[name].type_as(averaged_params[name])
    #             last_params[name] += averaged_params[name]
    #         self.set_global_params(last_params)

    # def training_nums(self, model_list, last_params):
    #     training_nums = {}
    #     idx = -1
    #     for k, p in last_params.items():
    #         dim = len(p.data.shape)
    #         if dim == 0:
    #             continue
    #         if dim == 4:
    #             idx += 1
    #             max_channel = p.data.shape[0]
    #         elif dim == 2:
    #             max_channel = p.data.shape[1]
    #         tn = torch.zeros(max_channel)
    #         # i = 0
    #         for local_sample_number, local_model_params, output_channel in model_list:
    #             # print(f'client {i} output_channel[{idx}]: {output_channel[idx]}')
    #             # i += 1
    #             tn[:output_channel[idx]] += local_sample_number
    #             # print(f'local_sample_number: {local_sample_number}')
    #         # training_nums.append(tn)
    #         # print(f'{k} -- {idx}')
    #         training_nums[k] = tn
    #     # print(training_nums)
    #     return training_nums

    def training_nums(self, model_list, last_params):
        training_nums = {}
        idx = -1
        last_layer_dim = 0
        for k, p in last_params.items():
            dim = len(p.data.shape)
            if dim == 4 and 'shortcut' not in k:
                idx += 1
            #     max_channel = p.data.shape[0]
            # elif dim == 2:
            #     max_channel = p.data.shape[1]
            tn = torch.zeros(p.data.shape)
            # i = 0
            for local_sample_number, local_model_params, output_channel, _ in model_list:
                # print(f'client {i} output_channel[{idx}]: {output_channel[idx]}')
                # i += 1
                skip_bn = False
                for oc in output_channel:
                    if oc not in self.output_channels:
                        skip_bn = True

                if 'shortcut' in k:
                    # print(f'{k} {global_state_dict[k].shape}')
                    if dim == 3:
                        tn[:output_channel[idx],:,:] += local_sample_number
                    elif dim == 4:
                        tn[:output_channel[idx],:output_channel[idx-2],:,:] += local_sample_number
                    elif dim == 1:
                        tn[:output_channel[idx]] += local_sample_number
                elif dim == 4:
                    prio_channel = 3 if idx == 0 else output_channel[idx-1]
                    print(f'output_channel[{idx}] -- {output_channel[idx]}, prio_channel: {prio_channel}')
                    tn[:output_channel[idx],:prio_channel,:,:] += local_sample_number
                elif dim == 1:
                    # if not skip_bn:
                    # if last_layer_dim == 4:
                    #     tn[:output_channel[idx]] += local_sample_number
                    # elif output_channel[idx] == self.output_channels[idx]:
                    # if output_channel[idx] == self.output_channels[idx]:
                    tn[:output_channel[idx]] += local_sample_number 
                    # else:
                    #     if 'running' not in k:
                    #         tn[:output_channel[idx]] += local_sample_number
                elif dim == 2:
                    tn[:,:output_channel[idx]] += local_sample_number
                else:
                    tn += local_sample_number
            last_layer_dim = len(p.data.shape)
                # print(f'local_sample_number: {local_sample_number}')
            # training_nums.append(tn)
            # print(f'{k} -- {idx}')
            training_nums[k] = tn
        # print(training_nums)
        return training_nums

    # def reorder_params(self, params):

    def exchange_dim(self, model_params):
        prio_exchange_idx = [0,1,2]
        for k, p in model_params.items():
            if len(p.shape) == 4:
    #             exchange_idx = []
    #             for i in range(p.shape[0])
                exchange_idx = np.argsort([p[i,:,:,:].mean().cpu() for i in range(p.shape[0])])
    #             print(exchange_idx)
                model_params[k] = model_params[k][:,prio_exchange_idx,:,:]
                model_params[k] = model_params[k][exchange_idx,:,:,:]
                prio_exchange_idx = exchange_idx
            elif len(p.shape) == 1:
    #             print(p.shape)
    #             print(prio_exchange_idx)
                model_params[k] = model_params[k][prio_exchange_idx]
            elif len(p.shape) == 2:
                model_params[k] = model_params[k][:,prio_exchange_idx]
                prio_exchange_idx = [i for i in range(10)]

    # def aggregate(self, keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times):
    #     count = OrderedDict()
    #     global_params = self.get_global_params()
    #     output_weight_name = [k for k in global_params.keys() if 'weight' in k][-1]
    #     output_bias_name = [k for k in global_params.keys() if 'bias' in k][-1]
    #     for k, v in global_params.items():
    #         parameter_type = k.split('.')[-1]
    #         count[k] = v.new_zeros(v.size(), dtype=torch.float32)
    #         tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
    #         for m in range(len(model_list)):
    #             local_sample_number, local_parameters, output_channel, param_idx = model_list[m]
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if parameter_type == 'weight':
    #                     if v.dim() > 1:
    #                         if k == output_weight_name:
    #                             # label_split = self.label_split[user_idx[m]]
    #                             param_idx[k] = list(param_idx[k])
    #                             # param_idx[m][k][0] = param_idx[m][k][0][label_split]
    #                             tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k] * local_sample_number
    #                             count[k][torch.meshgrid(param_idx[k])] += local_sample_number

    #                             # tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k] 
    #                             # count[k][torch.meshgrid(param_idx[k])] += 1
    #                         else:
    #                             tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k] * local_sample_number
    #                             count[k][torch.meshgrid(param_idx[k])] += local_sample_number

    #                             # tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k] 
    #                             # count[k][torch.meshgrid(param_idx[k])] += 1
    #                     else:
    #                         tmp_v[param_idx[k]] += local_parameters[k]  * local_sample_number
    #                         count[k][param_idx[k]] += local_sample_number

    #                         # tmp_v[param_idx[k]] += local_parameters[k] 
    #                         # count[k][param_idx[k]] += 1
    #                 else:
    #                     if k == output_bias_name:
    #                         # label_split = self.label_split[user_idx[m]]
    #                         # param_idx[m][k] = param_idx[m][k][label_split]
    #                         tmp_v[param_idx[k]] += local_parameters[k]  * local_sample_number
    #                         count[k][param_idx[k]] += local_sample_number
                            
    #                         # tmp_v[param_idx[k]] += local_parameters[k] 
    #                         # count[k][param_idx[k]] += 1
    #                     else:
    #                         tmp_v[param_idx[k]] += local_parameters[k]  * local_sample_number
    #                         count[k][param_idx[k]] += local_sample_number

    #                         # tmp_v[param_idx[k]] += local_parameters[k] 
    #                         # count[k][param_idx[k]] += 1
    #             else:
    #                 # if v.dim() == 1:
    #                 #     tmp_v[param_idx[k]] += local_parameters[k] 
    #                 #     count[k][param_idx[k]] += 1
    #                 # else:
    #                 tmp_v += local_parameters[k]  * local_sample_number
    #                 count[k] += local_sample_number
                    
    #                 # tmp_v += local_parameters[k] 
    #                 # count[k] += 1
    #         # tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
    #         tmp_v[count[k] > 0] = tmp_v[count[k] > 0] / 50000.
    #         v[count[k] > 0] += tmp_v[count[k] > 0].to(v.dtype)
        
    #     self.set_global_params(global_params)

    def aggregate(self, keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times):
        self.transmission_cost += self.round_trans_cost(download_cost, upload_cost)
        # self.compute_time += self.round_compute_time(compute_times)

        # self.compute_time += 48.326801575719195
        self.compute_time += 34.55108474620484

        last_params = self.get_global_params()

        training_num = sum(local_train_num for (local_train_num, _, _, _) in model_list)

        # for _, model_params, _ in model_list:
        #     self.exchange_dim(model_params)

        prio_channel = 3
        if model_list[0][2] != None:
            # torch.Size([64, 3, 3, 3]) == 64 == torch.Size([64, 3, 3, 3])
            # torch.Size([64, 3, 3, 3]) == 128 == torch.Size([64, 3, 3, 3])
            # torch.Size([64, 3, 3, 3]) == 95 == torch.Size([64, 3, 3, 3])
            # torch.Size([64, 3, 3, 3]) == 256 == torch.Size([64, 3, 3, 3])
            # torch.Size([64, 3, 3, 3]) == 60 == torch.Size([64, 3, 3, 3])

            training_nums = self.training_nums(model_list, last_params)

            idx = -1
            last_layer_dim = 0
            for k in last_params.keys():
                if len(last_params[k].shape) == 4 and 'shortcut' not in k:
                    idx += 1
                # prio_channel = [3] * len(model_list)
                count = 0
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params, output_channel, _ = model_list[i]
                    # if local_sample_number != 8078:
                    #     local_sample_number = local_sample_number / 2
                    w = local_sample_number / training_num
                    skip_bn = False
                    for oc in output_channel:
                        if oc not in self.output_channels:
                            skip_bn = True
                    if 'num_batches_tracked' in k:
                        w = local_sample_number / training_num
                    else:
                        # print(f'training_nums[{k}][output_channel[{idx}]-1]: {training_nums[k][output_channel[idx]-1]}')
                        # print(f'{k}: output_channel {output_channel[idx]}, training_num {training_nums[k][output_channel[idx]-1]}')
                        # w = local_sample_number / training_nums[k][output_channel[idx]-1]
                        pass
                    shape_dim = len(last_params[k].shape)

                    if 'shortcut' in k:
                        if k not in local_model_params.keys():
                            continue
                        # print(f'{k} {global_state_dict[k].shape}')
                        if shape_dim == 3:
                            last_params[k][:output_channel[idx],:,:] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx],:,:])
                        elif shape_dim == 4:
                            last_params[k][:output_channel[idx],:output_channel[idx-2],:,:] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx],:output_channel[idx-2],:,:])
                        elif shape_dim == 1:
                            last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])
                    elif shape_dim == 4:
                        prio_channel = 3 if idx == 0 else output_channel[idx-1]
                        min_prio_channel = 3 if idx == 0 else self.min_channels[idx-1]
                        # print(f'{last_params[k].shape} == {output_channel[idx]} == {local_model_params[k].shape}')
                        # if output_channel[idx] > self.min_channels[idx]:
                        #     local_model_params[k][:self.min_channels[idx],:min_prio_channel, :, :] *= 2
                        last_params[k][:output_channel[idx],:prio_channel,:,:] += local_model_params[k] * w
                        # last_params[k][:output_channel[idx],:prio_channel,:,:] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx],:prio_channel,:,:])

                        trip = True
                        # prio_channel[i] = output_channel[idx]
                    elif shape_dim == 1:
                        # print(f'{last_params[k].shape} == {output_channel[idx]} == {local_model_params[k].shape}')
                        # last_params[k][:output_channel[idx]] += local_model_params[k] * w
                        # if not skip_bn :
                        # if last_layer_dim == 4:
                        #     last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])

                        # elif output_channel[idx] == self.output_channels[idx]:
                            # count += 1
                        # if output_channel[idx] == self.output_channels[idx]:
                        # last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])
                        # if output_channel[idx] > self.min_channels[idx]:
                        #     local_model_params[k][:self.min_channels[idx]] *= 2
                        last_params[k][:output_channel[idx]] += local_model_params[k] * w
                        # else:
                        #     if 'running' not in k:
                        #         last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])
                        #     last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])
                            # last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/26120.)
                        # if 'running' not in k:
                        #     last_params[k][:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:output_channel[idx]])
                    elif shape_dim == 2:
                        # print(f'{last_params[k].shape} == {output_channel[idx]} == {local_model_params[k].shape}')
                        # if output_channel[idx] > self.min_channels[idx]:
                        #     local_model_params[k][:,:self.min_channels[idx]] *= 2
                        last_params[k][:,:output_channel[idx]] += local_model_params[k] * w
                        # last_params[k][:,:output_channel[idx]] += local_model_params[k] * (local_sample_number/training_nums[k][:,:output_channel[idx]])
                    # else:
                    #     last_params[k] += (local_model_params[k] * w).type_as(last_params[k])
                        # last_params[k] += (local_model_params[k] * (local_sample_number/training_nums[k])).type_as(last_params[k])
                last_layer_dim = len(last_params[k].shape)
                # print(f'{k} aggregate {count} clients')
                # if shape_dim == 4:
                #     idx += 1

            self.set_global_params(last_params)

        # else:
        # (num0, averaged_params, _, _) = model_list[0]
        # if (averaged_params is not None):
        #     for k in averaged_params.keys():
        #         for i in range(0, len(model_list)):
        #             local_sample_number, local_model_params, _, _ = model_list[i]
        #             w = local_sample_number / training_num
        #             if i == 0:
        #                 averaged_params[k] = local_model_params[k] * w
        #             else:
        #                 averaged_params[k] += local_model_params[k] * w

        #     # for name, param in averaged_params.items():
        #     for name in last_params:
        #         assert (last_params[name].shape == averaged_params[name].shape)
        #         averaged_params[name] = averaged_params[name].type_as(last_params[name])
        #         # last_params[name] = last_params[name].type_as(averaged_params[name])
        #         last_params[name] += averaged_params[name]
        #     self.set_global_params(last_params)

        # self.distill_submodel()
  

    def distill_submodel(self):
        # for idx in self.clients.keys():
        idx = random.randint(0,9)
        c = self.clients[idx]
        c.reset_weights(self.get_global_params())
        submodel = c.net.to(self.device)
        init_params = copy.deepcopy(submodel.state_dict())
        divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
        temp = 1
        self.model = self.model.to(self.device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, submodel.parameters()), lr=0.01, weight_decay=1e-5)
        for inputs, labels in c.train_data:
            # pass
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            submodel.zero_grad()
            with torch.no_grad():
                teacher_preds = self.model(inputs)
            student_preds = submodel(inputs)
            # loss = self.criterion(outputs, labels)
            ditillation_loss = divergence_loss_fn(
                F.softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1)
            )
            # if args.prox > 0:
            #     loss += args.prox / 2. * self.net.proximal_loss(global_params)
            ditillation_loss.backward()

            # if clip_grad:
            #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)

            optimizer.step()

            # c.reset_weights(submodel.cpu().state_dict())
        # finish_params = submodel.state_dict()
        # for param_tensor in finish_params:
        #     finish_params[param_tensor] -= init_params[param_tensor]

        self.reset_global_model(submodel.state_dict(), c)

    def reset_global_model(self, submodel_state_dict, client):
        prio_channel = 3
        local_params = self.model.state_dict()
        output_channels = client.output_channels
        idx = -1

        for k in local_params.keys():
            
            shape_dim = len(local_params[k].shape)
            # print(local_params[k].shape)
            if 'shortcut' in k:
                print(f'{k} {submodel_state_dict[k].shape}')
                if shape_dim == 3:
                    local_params[k].copy_(submodel_state_dict[k][:output_channels[idx],:,:])
                elif shape_dim == 4:
                    local_params[k].copy_(submodel_state_dict[k][:output_channels[idx],:output_channels[idx-2],:,:])
                elif shape_dim == 1:
                    local_params[k].copy_(submodel_state_dict[k][:output_channels[idx]])
            elif shape_dim == 4:
                print(k)
                idx += 1
                # print(f'{local_params[k].shape} == {self.output_channels[idx]}')
                local_params[k][:output_channels[idx],:prio_channel,:,:].copy_(submodel_state_dict[k])
                # local_params[k].copy_(submodel_state_dict[k][:self.output_channels[idx],:prio_channel,:,:])
                prio_channel = output_channels[idx]
            elif shape_dim == 1:
                # print(local_params[k].shape)
                if 'running' not in k:
                    local_params[k][:output_channels[idx]].copy_(submodel_state_dict[k])
                # local_params[k].copy_(submodel_state_dict[k][:self.output_channels[idx]])
            elif shape_dim == 2:
                # print(f'{local_params[k].shape} == {self.output_channels[idx]}')
                local_params[k][:,:output_channels[idx]].copy_(submodel_state_dict[k])

    def test_global_model_on_all_client(self, clients, round):
        # pruned_c = 0.0
        # total = 0.0
        # for name, param in self.model.state_dict().items():
        #     a = param.view(-1).to(device='cpu', copy=True).numpy()
        #     pruned_c +=sum(np.where(a, 0, 1))
        #     total += param.numel()
        # print(f'global model zero params: {pruned_c / total}')

        train_accuracies, train_losses, test_accuracies, test_losses = utils.evaluate_local(clients, self.model, progress=True,
                                                    n_batches=0)
        wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values()), 'round': round, 'comm_cost': int(self.transmission_cost), 'training_time': self.compute_time})
        wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values()), 'round': round, 'comm_cost': int(self.transmission_cost), 'training_time': self.compute_time})
        print(f'round: {round}')
        print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')

        wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values()), 'round': round, 'comm_cost': int(self.transmission_cost), 'training_time': self.compute_time})
        wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values()), 'round': round, 'comm_cost': int(self.transmission_cost), 'training_time': self.compute_time})
        print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')

        # wandb.log({"Trans_Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=int(self.transmission_cost))
        # wandb.log({"Trans_Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=int(self.transmission_cost))
        # print(f'round: {round}')
        # print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')

        # wandb.log({"Trans_Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=int(self.transmission_cost))
        # wandb.log({"Trans_Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=int(self.transmission_cost))
        # print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')

        return train_accuracies, test_accuracies

