from models.criterions.General import General
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import numpy as np

def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total

def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

class Grasp(General):
    def __init__(self, *args, **kwargs):
        super(Grasp, self).__init__(*args, **kwargs)
        self.first = True

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_average_gradients(self, train_loader):
        net = copy.deepcopy(self.model)
        # Prepare list to store gradients
        gradients = []
        for layer in net.modules():
            # Select only prunable layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gradients.append(0)
        
        # Take a whole epoch
        count_batch = 0
        for batch_idx in range(len(train_loader)):
            inputs, targets = next(iter(train_loader))
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
            # if batch_idx == num_batches - 1:
            #     break
        avg_gradients = [x / count_batch for x in gradients] 
            
        return avg_gradients

    # def pruning_criteria(self, )
    def get_average_saliency(self, gradients):
        saliency = []
        idx = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                saliency.append(gradients[idx] ** 2)
                idx += 1
        return saliency

    def handle_pruning(self, saliency, percentage):
        # don't prune more or less than possible
        all_scores = torch.cat([torch.flatten(x) for x in saliency])
        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        print(f'num_params_to_keep: {num_params_to_keep}, percentage: {(1 - percentage)}')
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > len(saliency):
            num_params_to_keep = len(saliency)

        # threshold
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        
        masks = []
        for m in saliency:
            masks.append((m >= acceptable_score).float().to(self.device))
        return masks
        # # prune
        # for name, grad in grads_abs.items():
        #     self.model.mask[name] = ((grad / norm_factor) > acceptable_score).float().to(self.device)

        #     length_nonzero = float(self.model.mask[name].flatten().shape[0])

        #     # print(f'zero masks count: {self.model.mask[name].flatten().sum()}, shape: {self.model.mask[name].flatten().shape[0]}')
        #     cutoff = (self.model.mask[name] == 0).sum().item()

        #     print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        # print("final percentage after snip:", self.model.sparsity_percentage())

    def prune_masks(self, percentage, train_loader=None, manager=None, last_masks=None, saliency_mode=None, **kwargs):
        gradients = self.get_average_gradients(train_loader)
        # pruning_criteria()
        saliency = self.get_average_saliency(gradients)
        if saliency_mode == 'saliency':
            return saliency
        return self.handle_pruning(saliency, percentage)

    def GraSP(self, ratio, train_dataloader, num_classes=10, samples_per_class=25, num_iters=1, T=1, reinit=True, saliency_mode=None):
        eps = 1e-10
        keep_ratio = 1-ratio
        old_net = self.model

        net = copy.deepcopy(self.model)  # .eval()
        net.zero_grad()

        weights = []
        # total_parameters = count_total_parameters(self.model)
        # fc_parameters = count_fc_parameters(self.model)

        # rescale_weights(net)
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # if isinstance(layer, nn.Linear) and reinit:
                #     # if self.first:
                #     #     random.seed(0)
                #     #     np.random.seed(0)
                #     #     torch.manual_seed(0)
                #     #     torch.cuda.manual_seed_all(0)
                #     #     self.first = False
                #     nn.init.xavier_normal(layer.weight)
                weights.append(layer.weight)

        inputs_one = []
        targets_one = []

        grad_w = None
        for w in weights:
            w.requires_grad_(True)

        print_once = False
        dataloader_iter = iter(train_dataloader)
        for it in range(num_iters):
            print("(1): Iterations %d/%d." % (it, num_iters))
            # inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
            inputs, targets = next(dataloader_iter)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            inputs_one.append(din[:N//2])
            targets_one.append(dtarget[:N//2])
            inputs_one.append(din[N // 2:])
            targets_one.append(dtarget[N // 2:])
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = net.forward(inputs[:N//2])/T
            if print_once:
                # import pdb; pdb.set_trace()
                x = F.softmax(outputs)
                print(x)
                print(x.max(), x.min())
                print_once = False
            loss = F.cross_entropy(outputs, targets[:N//2])
            # ===== debug ================
            grad_w_p = autograd.grad(loss, weights)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            outputs = net.forward(inputs[N // 2:])/T
            loss = F.cross_entropy(outputs, targets[N // 2:])
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

        ret_inputs = []
        ret_targets = []

        for it in range(len(inputs_one)):
            print("(2): Iterations %d/%d." % (it, num_iters))
            inputs = inputs_one.pop(0).to(self.device)
            targets = targets_one.pop(0).to(self.device)
            ret_inputs.append(inputs)
            ret_targets.append(targets)
            outputs = net.forward(inputs)/T
            loss = F.cross_entropy(outputs, targets)
            # ===== debug ==============

            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        grads = dict()
        old_modules = list(old_net.modules())
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
        norm_factor = torch.abs(torch.sum(all_scores)) + eps
        print("** norm factor:", norm_factor)
        all_scores.div_(norm_factor)

        if saliency_mode == 'saliency':
            ret_masks = dict()
            start_idx = 0
            for m, g in grads.items():
                # keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
                ret_masks[m] = all_scores[start_idx:start_idx + g.numel()].reshape(g.shape)
                start_idx += g.numel()

            return ret_masks
        elif saliency_mode == 'mask':

            # num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
            num_params_to_rm = int(len(all_scores) * keep_ratio)
            threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
            # import pdb; pdb.set_trace()
            acceptable_score = threshold[-1]
            print('** accept: ', acceptable_score)
            keep_masks = dict()
            for m, g in grads.items():
                # keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
                keep_masks[m] = ((g / norm_factor) > acceptable_score).float()

            print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))
            return keep_masks
        
    # def trans_to_rank(self, all_scores):
    #     min_score = all_scores.min()
    #     max_score = all_scores.max()
    #     step = abs(max_score) + abs(min_score) / 5
    #     rank = torch.zeros_like(all_scores)
    #     for i in range(5):
    #         b = min_score + step
    #         rank += torch.where(all_scores <= b, i, 0)
    #     return rank
        
        