import os

import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from models.criterions.General import General
# from utils.constants import RESULTS_DIR, OUTPUT_DIR, SNIP_BATCH_ITERATIONS
from utils import *
import copy
import types
import random
from models.vgg import VGG
import math

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


class GivenRatio(General):

    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(GivenRatio, self).__init__(*args, **kwargs)
        self.first = True

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def layer_keep_ratio(self, ratio, layer_params):
        total_params = sum(layer_params)
        def ineq_func(x):
            prune_params = total_params*ratio
            for i in range(len(x)):
                prune_params = prune_params - x[i] * layer_params[i]
        #     print(f'{total_params*0.005 - prune_params}')
        #     print(prune_params)
            return prune_params

        def func(x, sign=1.0):
        #     for pi in x:
        #         print(pi)
            return sign*(sum([np.log(pi) for pi in x]))

        e = 0

        cons = ({
            'type': 'ineq',
            'fun': ineq_func,
        },
        )
        def tmp_fun(x, i):
        #     print(i)
            return x[i] - e
        def tmp_fun2(x, i):
        #     print(i)
            return 1 - x[i]
        for i in range(len(layer_params)):
            cons = cons + ({
                'type': 'ineq',
                'fun': tmp_fun,
                'args': (i,)
            },{
                'type': 'ineq',
                'fun': tmp_fun2,
                'args': (i,)
            },)
        res = minimize(func, [1.0/len(layer_params) for i in range(len(layer_params))], args=(-1.0,),
               constraints=cons, method='SLSQP', options={'disp': True})
        print(f'layer ratio: {res.x}')
        return res.x

    def generate_random_masks(self, weights_by_layer):
        masks = []
        with torch.no_grad():
            print(f'weights_by_layer: {weights_by_layer}')
            if isinstance(self.model, VGG):
                idx = 0
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
                        n_prune = int(n_total - weights_by_layer[idx])
                        idx += 1
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
                idx = 0
                for name, layer in self.model.named_children():

                    # We need to figure out how many to prune
                    n_total = 0
                    # for bname, buf in layer.named_buffers():
                    for pname, param in layer.named_parameters():
                        if needs_mask(pname):
                            n_total += param.numel()
                    n_prune = int(n_total - weights_by_layer[idx])
                    idx += 1
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

    def prune_masks(self, ratio, train_dataloader=None, last_masks=None, layer_based=False, epochs=1, pruning_schedule='exp', **kwargs):

        net = copy.deepcopy(self.model)
        # for name, param in net.state_dict().items():
        #     param.abs_()
        layer_params = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer_params.append(layer.weight.data.numel())
        print(layer_params)

        layer_ratio = self.layer_keep_ratio(ratio, layer_params)
        weights_by_layer = []
        for i in range(len(layer_ratio)):
            weights_by_layer.append(layer_ratio[i] * layer_params[i])
        net.mask = self.generate_random_masks(weights_by_layer)
        # self.reorder_pruning_weights_2(net)
        return net.mask

    def reorder_pruning_weights(self, net):
        idx = 0
        for m in net.mask:
            layer_total = m.numel()
            layer_keep_ratio = math.sqrt(float(m.sum()) / layer_total)
            # num_params_to_keep = 
            mask = torch.ones(m.shape).to(self.device)
            if len(m.shape) == 4:
                if idx == 0:
                    mask[:,int(m.shape[1]):,:,:] = 0
                    mask[int(m.shape[0] * layer_keep_ratio * layer_keep_ratio):,:,:,:] = 0
                    print(f'cnn: in_channel: {m.shape[1]}->{int(m.shape[1])}; out_channel: {m.shape[0]}->{int(m.shape[0] * layer_keep_ratio * layer_keep_ratio)}')
                else:
                    mask[:,int(m.shape[1] * layer_keep_ratio):,:,:] = 0
                    mask[int(m.shape[0] * layer_keep_ratio):,:,:,:] = 0
                    print(f'cnn: in_channel: {m.shape[1]}->{int(m.shape[1] * layer_keep_ratio)}; out_channel: {m.shape[0]}->{int(m.shape[0] * layer_keep_ratio)}')
            if len(m.shape) == 2:
                mask[int(m.shape[0] * layer_keep_ratio):,:] = 0
                mask[:,int(m.shape[1] * layer_keep_ratio):] = 0
            net.mask[idx] = mask
            idx += 1
            # print(m.shape)

    def reorder_pruning_weights_2(self, net):
        idx = 0
        input_channel = 3
        prio_ratio = 1.
        for m in net.mask:
            layer_total = m.numel()
            # layer_keep_ratio = math.sqrt(float(m.sum()) / layer_total)
            layer_keep_ratio = float(m.sum()) / layer_total
            # num_params_to_keep = 
            mask = torch.ones(m.shape).to(self.device)
            if len(m.shape) == 4:
                output_channel = int(m.shape[0] * (layer_keep_ratio / prio_ratio))
                prio_ratio = layer_keep_ratio / prio_ratio
                mask[:,input_channel:,:,:] = 0
                mask[output_channel:,:,:,:] = 0
                print(f'cnn: in_channel: {m.shape[1]}->{input_channel}; out_channel: {m.shape[0]}->{output_channel}')
                input_channel = output_channel
            if len(m.shape) == 2:
                mask[int(m.shape[0] * layer_keep_ratio):,:] = 0
                mask[:,int(m.shape[1] * layer_keep_ratio):] = 0
            net.mask[idx] = mask
            idx += 1