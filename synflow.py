import os
from pyexpat import model
from statistics import mode

from pkg_resources import yield_lines

import torch
import torch.nn.functional as F

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


class SynFlow(General):

    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(SynFlow, self).__init__(*args, **kwargs)
        self.first = True
        self.scores = {}

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune_masks(self, ratio, train_dataloader=None, last_masks=None, layer_based=False, epochs=1, pruning_schedule='exp', **kwargs):

        self.masked_parameters = {}
        if isinstance(self.model, VGG):
            for name, layer in self.model.named_children():
                for n, l in layer.named_children():
                    if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                        continue
                    print(n)
                    mask = torch.ones_like(l.weight)
                    self.masked_parameters[mask] = l.weight
        else:
            for name, layer in self.model.named_children():
                for pname, param in layer.named_parameters():
                    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                        continue
                    mask = torch.ones_like(layer.weight)
                    self.masked_parameters[mask] = layer.weight

        for epoch in tqdm(range(epochs)):
            self.score(self.model, train_dataloader, self.device, [m for m in self.masked_parameters.keys()])
            if pruning_schedule == 'exp':
                sparse = ratio**((epoch + 1) / epochs)
            elif pruning_schedule == 'linear':
                sparse = 1.0 - (1.0 - ratio)*((epoch + 1) / epochs)
            self.mask(sparse)
        self.model.mask = [m for m in self.masked_parameters.keys()]

        self.reorder_pruning_weights_2(self.model)

        for m in self.model.mask:
            print(m.mean())

        return self.model.mask

        # net = copy.deepcopy(self.model)
        # # for name, param in net.state_dict().items():
        # #     param.abs_()
        # # for layer in net.modules():
        # #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        # #         layer.weight.data.abs_()
        # # net.mask = copy.deepcopy([m for m in self.model.mask.values()])
        # for epoch in range(epochs): 
        #     if pruning_schedule == 'linear':
        #         r = 1.0 - (1.0 - ratio)*((epoch + 1) / epochs)
        #     else:
        #         r = ratio **((epoch + 1) / epochs)

        #     # handlers = apply_prune_mask(net, net.mask)
        #     saliency = self.get_weight_saliencies(net, train_dataloader)
        #     # yield saliency
        #     self.handle_pruning(net, saliency, r)
            
        #     # if handlers is not None:
        #     #     for h in handlers:
        #     #         h.remove()
        #     # print(self.model.mask.keys())
        # net.mask = [m for m in net.mask.values()]
        # # self.reorder_pruning_weights_2(net)
        # # return [m for m in net.mask.values()]
        # return net.mask

    def masked_percent(self, net):
        pruned_c = 0.0
        total = 0.0

        for name, param in net.state_dict().items():
            a = param.view(-1).to(device='cpu', copy=True).numpy()
            pruned_c +=sum(np.where(a, 0, 1))
            total += param.numel()
        return pruned_c / total

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
                output_channel = math.ceil(m.shape[0] * (layer_keep_ratio / prio_ratio))
                prio_ratio = layer_keep_ratio / prio_ratio
                mask[:,input_channel:,:,:] = 0
                mask[output_channel:,:,:,:] = 0
                print(f'cnn: in_channel: {m.shape[1]}->{input_channel}; out_channel: {m.shape[0]}->{output_channel}')
                input_channel = output_channel
            if len(m.shape) == 2:
                layer_keep_ratio = math.sqrt(layer_keep_ratio)
                mask[math.ceil(m.shape[0] * layer_keep_ratio):,:] = 0
                mask[:,math.ceil(m.shape[1] * layer_keep_ratio):] = 0
                print(f'linear: in: {m.shape[1]}->{math.ceil(m.shape[1] * layer_keep_ratio)}; out: {m.shape[0]}->{math.ceil(m.shape[0] * layer_keep_ratio)}')
            net.mask[idx] = mask
            idx += 1

    def reorder(self):
        prio_ratio = 1.
        input_channel = 3
        for mask, param in self.masked_parameters:
            layer_keep_ratio = float(mask.sum()) / param.numel()

            mask = torch.ones(param.shape).to(param.device)
            if len(param.shape) == 4:
                output_channel = int(mask.shape[0] * (layer_keep_ratio / prio_ratio))
                prio_ratio = layer_keep_ratio / prio_ratio
                mask[:,input_channel:,:,:] = 0
                mask[output_channel:,:,:,:] = 0
                print(f'cnn: in_channel: {param.shape[1]}->{input_channel}; out_channel: {param.shape[0]}->{output_channel}')
                input_channel = output_channel
            if len(mask.shape) == 2:
                layer_keep_ratio = math.sqrt(layer_keep_ratio)
                mask[int(mask.shape[0] * layer_keep_ratio):,:] = 0
                mask[:,int(mask.shape[1] * layer_keep_ratio):] = 0

    # def handle_pruning(self, net, saliency, percentage):

    #     # don't prune more or less than possible
    #     last_mask = []
    #     for m in net.mask.values():
    #         last_mask.append(torch.where(m==0, -1, 1))
    #     idx = 0
    #     for n, s in saliency.items():
    #         saliency[n] = saliency[n] * last_mask[idx]
    #         idx += 1


    #     total_params = sum([s.numel() for s in saliency.values()])
    #     num_params_to_keep = int(total_params * (percentage))
    #     print(f'num_params_to_keep: {num_params_to_keep}, percentage: {(percentage)}')
    #     if num_params_to_keep < 1:
    #         num_params_to_keep += 1
    #     elif num_params_to_keep > total_params:
    #         num_params_to_keep = total_params

    #     flatten_saliency = torch.cat([s.flatten() for s in saliency.values()])
    #     # threshold
    #     threshold, _ = torch.topk(flatten_saliency, num_params_to_keep, sorted=True)
    #     # print(all_scores.shape)
    #     # print(float(torch.nonzero(all_scores).shape[0])/all_scores.shape[0])
    #     acceptable_score = threshold[-1]

    #     for name, s in saliency.items():
    #         name = name  + '.weight'
    #         net.mask[name] = (s > acceptable_score).float().to(self.device)

    #         length_nonzero = float(net.mask[name].flatten().shape[0])

    #         cutoff = (net.mask[name] == 0).sum().item()
        
    #         print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero, "; remaind params:", length_nonzero - cutoff)
    #     print("final percentage after snip:", net.sparsity_percentage())
    #     # self.model.mask = [m for m in self.model.mask.values()]

    # def check_mask(self, masks):
    #     flatten_mask = torch.cat(masks)
    #     print(f'keeped: {flatten_mask.mean()}')

    # def _get_average_gradients(self, net, train_dataloader):
    #     # net = copy.deepcopy(net)
    #     self.check_mask([m.flatten() for m in net.mask.values()])
    #     handlers = apply_global_mask(net, [m for m in net.mask.values()])
    #     print(self.masked_percent(net))

    #     gradients = []
    #     # for layer in net.modules():
    #     #     # Select only prunable layers
    #     #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #     #         gradients.append(0)
        

    #     (data, _) = next(iter(train_dataloader))
    #     input_dim = list(data[0,:].shape)
    #     input = torch.ones([1] + input_dim).to(self.device)
    #     output = net.forward(input)
    #     net.zero_grad()
    #     torch.sum(output).backward()
    #     for layer in net.modules():
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             gradients.append(torch.clone(layer.weight.grad).detach().abs_())
    #             # layer.weight.grad.data.zero_()
            
    #     return gradients

    # def get_weight_saliencies(self, net, train_loader):
    #     gradients = self._get_average_gradients(net, train_loader)

    #     saliency = {}
    #     idx = 0
    #     masks = [m for m in net.mask.values()]
    #     for name, layer in net.named_modules():
    #         # print(name)
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             saliency[name] = torch.abs(layer.weight * gradients[idx])
    #             remain_parasm = masks[idx].sum()
    #             idx += 1
    #             # remain_parasm = torch.cat([m.flatten() for m in net.mask.values()]).sum()
    #             print(f'{name}: {saliency[name].sum() / remain_parasm}; {saliency[name].var()}; {saliency[name].sum()}; {remain_parasm}')
    #     # print(f'len of gradients: {len(gradients)}; idx: {idx}')
    #     return saliency

    def score(self, net, dataloader, device, mask):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        model = copy.deepcopy(net)
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        apply_global_mask(model, mask)
        output = model(input)
        torch.sum(output).backward()
        
        scores = []
        if isinstance(model, VGG):
            for name, layer in model.named_children():
                for n, l in layer.named_children():
                    if not (isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)):
                        continue
                    # if not (isinstance(l, nn.Conv2d)):
                    #     continue
                    scores.append(torch.clone(l.weight.grad * l.weight).detach().abs_())
                    l.weight.grad.data.zero_()
            self.scores = scores
            for s in self.scores:
                print(f'{id(l.weight)}: {s.mean()}; {s.var()}; {s.sum()}')
        else:
            for name, layer in model.named_children():
                for pname, param in layer.named_parameters():
                    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                        continue
                    self.scores[id(layer.weight)] = torch.clone(layer.weight.grad * layer.weight).detach().abs_()
                    layer.weight.grad.data.zero_()
        # for _, p in self.masked_parameters.items():
        #     self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
        #     p.grad.data.zero_()
        #     print(f'{id(p)}: {self.scores[id(p)].mean()}; {self.scores[id(p)].var()}; {self.scores[id(p)].sum()}')

        nonlinearize(model, signs)

    def mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        # global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        # global_scores = torch.cat([torch.flatten(self.scores[i]) for i in range(len(self.scores) - 1)])
        global_scores = torch.cat([torch.flatten(v) for v in self.scores])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            idx = 0
            for mask, param in self.masked_parameters.items():
                score = self.scores[idx] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one).to(mask.device))
                idx += 1
                if idx == len(self.scores) - 1:
                    break