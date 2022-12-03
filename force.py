import os

import torch
import torch.nn.functional as F

from models.criterions.General import General
# from utils.constants import RESULTS_DIR, OUTPUT_DIR, SNIP_BATCH_ITERATIONS
from utils import *
import copy
import types
import random
from models.vgg import VGG

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


class Force(General):

    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(Force, self).__init__(*args, **kwargs)
        self.first = True

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune_masks(self, ratio, train_dataloader=None, last_masks=None, layer_based=False, saliency_mode=None, **kwargs):

        saliency = self.get_weight_saliencies(train_dataloader, last_masks)

        if saliency_mode == 'mask':
            self.handle_pruning(saliency, ratio)
        elif saliency_mode == 'saliency':
            for m, s in saliency.items():
                self.model.mask[m + '.weight'] = s

    def handle_pruning(self, saliency, percentage):

        # don't prune more or less than possible
        total_params = sum([s.numel() for s in saliency.values()])
        num_params_to_keep = int(total_params * (1 - percentage))
        print(f'num_params_to_keep: {num_params_to_keep}, percentage: {(1 - percentage)}')
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > total_params:
            num_params_to_keep = total_params

        flatten_saliency = torch.cat([s.flatten() for s in saliency.values()])
        # threshold
        threshold, _ = torch.topk(flatten_saliency, num_params_to_keep, sorted=True)
        # print(all_scores.shape)
        # print(float(torch.nonzero(all_scores).shape[0])/all_scores.shape[0])
        acceptable_score = threshold[-1]

        for name, s in saliency.items():
            name = name + '.weight'
            self.model.mask[name] = (s > acceptable_score).float().to(self.device)

            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()
        
            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        print("final percentage after snip:", self.model.sparsity_percentage())

    def _get_average_gradients(self, train_dataloader, num_batches=-1):
        gradients = []
        net = copy.deepcopy(self.model)
        for layer in net.modules():
            # Select only prunable layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gradients.append(0)

        count_batch = 0
        for batch_idx in range(len(train_dataloader)):
            inputs, targets = next(iter(train_dataloader))
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Compute gradients (but don't apply them)
            net.zero_grad()
            outputs = net.forward(inputs)
            loss = F.nll_loss(outputs, targets)
            loss.backward()
            
            # Store gradients
            counter = 0
            for layer in net.modules():
                # Select only prunable layers
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    gradients[counter] += layer.weight.grad
                    counter += 1
            count_batch += 1
            if batch_idx == num_batches - 1:
                break
        avg_gradients = [x / count_batch for x in gradients] 
            
        return avg_gradients

    def get_weight_saliencies(self, train_loader, last_masks):
        gradients = self._get_average_gradients(train_loader, 5)

        saliency = {}
        idx = 0
        for name, layer in self.model.named_modules():
            # print(name)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                saliency[name] = gradients[idx]**2
                idx += 1
        # print(f'len of gradients: {len(gradients)}; idx: {idx}')
        return saliency
