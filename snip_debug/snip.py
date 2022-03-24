import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

import logging
import json

import random
import numpy as np


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def record_targets(targets):
    masks = torch.cat([torch.flatten(x) for x in targets]).to('cpu').tolist()
    with open(f'./targets.txt', 'a+') as f:
        f.write(json.dumps(masks))
        f.write('\n')

def SNIP(net, keep_ratio, train_dataloader, device, id):
    print('snip debug snip')
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    record_targets(targets)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # net.load_state_dict(torch.load(f'after_init_{id}.pt'))

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    # loss = F.nll_loss(outputs, targets)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    grads_abs = []
    for name, layer in net.named_modules():
        if "Norm" in str(layer): continue
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            grads_abs.append(torch.abs(layer.weight_mask.grad))
                # grads_abs[name + ".weight"] = torch.abs(layer.weight_mask.grad)
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    print(f'acceptable_score: {acceptable_score}')

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) > acceptable_score).float())

    # print(f"all params num: {len(all_scores)}; num_params_to_keep: {num_params_to_keep}")
    logging.info('all params num: {0}; num_params_to_keep: {1}'.format(len(all_scores), num_params_to_keep))
    logging.info(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)