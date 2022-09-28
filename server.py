import copy
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

import wandb

def needs_mask(layer_name):
    return layer_name.endswith('weight') and ('bn' not in layer_name)

class Server:
    def __init__(self, model, device, train_data=None, prune_strategy='None', target_keep_ratio=1.) -> None:
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
        self.compute_time += self.round_compute_time(compute_times)

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



    def aggregate(self, keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times):
        self.transmission_cost += self.round_trans_cost(download_cost, upload_cost)
        self.compute_time += self.round_compute_time(compute_times)
        last_params = self.get_global_params()

        training_num = sum(local_train_num for (local_train_num, _) in model_list)

        (num0, averaged_params) = model_list[0]
        if (averaged_params is not None):
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # for name, param in averaged_params.items():
            for name in last_params:
                assert (last_params[name].shape == averaged_params[name].shape)
                averaged_params[name] = averaged_params[name].type_as(last_params[name])
                # last_params[name] = last_params[name].type_as(averaged_params[name])
                last_params[name] += averaged_params[name]
            self.set_global_params(last_params)
  

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

