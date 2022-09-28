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


class SNIP(General):

    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(SNIP, self).__init__(*args, **kwargs)
        self.first = True

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune_masks(self, percentage, train_loader=None, manager=None, last_masks=None, layer_based=False, saliency_mode=None, **kwargs):

        all_scores, grads_abs, log10, norm_factor = self.get_weight_saliencies(train_loader, last_masks)

        all_scores = self._mask_all_scores(all_scores, last_masks)

        # yield all_scores
        # if percentage != 0.9:
        #     total = all_scores.numel()
        #     masked_score = sum(torch.where(all_scores==0, 1, 0))
        #     print('masked {} score'.format(masked_score/float(total)))
        #     print('min of score : {}'.format(min(all_scores)))
        if saliency_mode == 'mask':
            if layer_based:
                self.handle_layer_based_pruning(all_scores, grads_abs, log10, manager, norm_factor, percentage)
            else:
                self.handle_pruning(all_scores, grads_abs, log10, manager, norm_factor, percentage)
        elif saliency_mode == 'saliency':
            start_idx = 0
            for m, g in grads_abs.items():
                # keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
                self.model.mask[m] = all_scores[start_idx:start_idx + g.numel()].reshape(g.shape)
                start_idx += g.numel()

    def _mask_all_scores(self, all_scores, last_masks):
        flatten_masks = torch.cat([torch.flatten(m) for m in last_masks])
        assert all_scores.shape == flatten_masks.shape
        return all_scores * flatten_masks

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
                            if isinstance(layer, nn.modules.conv._ConvNd):
                                sparsities[index] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                            else:
                                sparsities[index] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                        else:
                            raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
                        index += 1

                # Now we need to renormalize sparsities.
                # We need global sparsity S = sum(s * n) / sum(n) equal to desired
                # sparsity, and s[i] = C n[i]
                # print(n_weights)
                sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
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

    # reshape tensor list a to tensor list b shape
    def reshape_like(self, ta, tb):
        ret = []
        idx = 0
        for t in tb:
            tc = t.numel()
            ret.append(torch.reshape(ta[idx:tc+idx], t.shape))
            idx += tc
        return ret

    def handle_layer_based_pruning(self, all_scores, grads_abs, log10, manager, norm_factor, percentage):
        with torch.no_grad():
            weights_by_layer = list(self._weights_by_layer(sparsity=percentage, sparsity_distribution='erk').values())
            print(weights_by_layer)
            all_scores = self.reshape_like(all_scores, grads_abs.values())
            idx = 0
            for layer_score, (name, grad) in zip(all_scores, grads_abs.items()):
                print('layer_score.shape: {}; grad.shape: {}'.format(layer_score.shape, grad.shape))
                threshold, _ = torch.topk(layer_score.flatten(), int(weights_by_layer[idx]), sorted=True)
                idx += 1
                acceptable_score = threshold[-1]
                
                self.model.mask[name] = ((grad / norm_factor) > acceptable_score).float().to(self.device)
                length_nonzero = float(self.model.mask[name].flatten().shape[0])

                # print(f'zero masks count: {self.model.mask[name].flatten().sum()}, shape: {self.model.mask[name].flatten().shape[0]}')
                cutoff = (self.model.mask[name] == 0).sum().item()

                print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        print("final percentage after snip:", self.model.sparsity_percentage())

    def handle_pruning(self, all_scores, grads_abs, log10, manager, norm_factor, percentage):
        if manager is not None:
            manager.save_python_obj(all_scores.cpu().numpy(),
                                    os.path.join(RESULTS_DIR, manager.stamp, OUTPUT_DIR, f"scores"))

        # don't prune more or less than possible
        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        print(f'num_params_to_keep: {num_params_to_keep}, percentage: {(1 - percentage)}')
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > len(all_scores):
            num_params_to_keep = len(all_scores)

        # threshold
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        # print(all_scores.shape)
        # print(float(torch.nonzero(all_scores).shape[0])/all_scores.shape[0])
        acceptable_score = threshold[-1]
        
        # prune
        for name, grad in grads_abs.items():
            # self.model.mask[name] = ((grad / norm_factor) >= acceptable_score).float().to(self.device)
            # print(self.model.mask[name].sum().item())
            # self.model.mask[name] = ((grad / norm_factor) > acceptable_score).__and__(
            #     self.model.mask[name].bool()).float().to(self.device)
            self.model.mask[name] = ((grad / norm_factor) > acceptable_score).float().to(self.device)

            # self.model.mask[name] = tmp.__and__(
            #     self.model.mask[name].type(tmp.dtype)).float().to(self.device)
            # self.model.mask[name] = torch.logical_and(tmp, self.model.mask[name].type(torch.bool)).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            # print(f'zero masks count: {self.model.mask[name].flatten().sum()}, shape: {self.model.mask[name].flatten().shape[0]}')
            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        # self.model.apply_weight_mask()
        # self.model.apply_mask()
        print("final percentage after snip:", self.model.sparsity_percentage())
        # self.cut_lonely_connections()
    
    def handle_pruning_with_rank(self, all_scores, grads_abs, log10, manager, norm_factor, percentage):
        ranks = self.trans_to_rank(all_scores)
        start_idx = 0
        for name, g in grads_abs.items():
            self.model.mask[name] = ranks[start_idx:start_idx + g.numel()].reshape(g.shape)
            start_idx += g.numel()

    def get_weight_saliencies(self, train_loader, last_masks):

        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Let's create a fresh copy of the network so that we're not worried about
        # affecting the actual training-phase
        net = copy.deepcopy(self.model)

        # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
        # instead of the weights
        # for layer in net.modules():
        # for layer, mask in zip(net.modules(), last_masks):
        idx = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # print('{} == {}'.format(layer.weight.shape, last_masks[idx].shape))
                assert layer.weight.shape == last_masks[idx].shape
                # layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight_mask = nn.Parameter(last_masks[idx])
                idx += 1

                # if self.first:
                #     random.seed(0)
                #     np.random.seed(0)
                #     torch.manual_seed(0)
                #     torch.cuda.manual_seed_all(0)
                #     self.first = False

                # nn.init.xavier_normal_(layer.weight)
                # layer.weight.requires_grad = False

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

        grads_abs = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads_abs[name + ".weight"] = copy.deepcopy(torch.abs(layer.weight_mask.grad))
        # for layer in net.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         grads_abs.append(torch.abs(layer.weight_mask.grad))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        # net = self.model.eval()

        # iterations = SNIP_BATCH_ITERATIONS
        # # iterations = 1

        # # accumalate gradients of multiple batches
        # net.zero_grad()
        # loss_sum = torch.zeros([1]).to(self.device)
        # for i, (x, y) in enumerate(train_loader):

        #     if i == iterations: break

        #     inputs = x.to(self.model.device)
        #     targets = y.to(self.model.device)
        #     outputs = net.forward(inputs)
        #     loss = F.nll_loss(outputs, targets) / iterations
        #     loss.backward()
        #     loss_sum += loss.item()

        # get elasticities
        # grads_abs = {}
        # for name, layer in net.named_modules():
        #     if "Norm" in str(layer): continue
        #     if name + ".weight" in self.model.mask:
        #         # grads_abs[name + ".weight"] = torch.abs(
        #             # layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item())))
        #         grads_abs[name + ".weight"] = torch.abs(
        #             layer.weight.grad * (layer.weight.data))
        # all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        # norm_factor = 1
        # log10 = all_scores.sort().values.log10()
        # all_scores.div_(norm_factor)

        # self.model = self.model.train()

        return all_scores, grads_abs, None, norm_factor

    def trans_to_rank(self, all_scores, num_of_rank=10):
        min_score = all_scores.min()
        max_score = all_scores.max()
        step = abs(max_score) + abs(min_score) / num_of_rank
        rank = torch.zeros_like(all_scores)
        for i in range(num_of_rank):
            b = min_score + step
            rank += torch.where(all_scores <= b, i, 0)
        return rank
