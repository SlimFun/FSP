from models.criterions.General import General
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SNIP(General):
    def __init__(self, *args, **kwargs):
        super(SNIP, self).__init__(*args, **kwargs)
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

    def prune_masks(self, percentage, train_loader=None, manager=None, last_masks=None, **kwargs):
        gradients = self.get_average_gradients(train_loader)
        # pruning_criteria()
        saliency = self.get_average_saliency(gradients)
        return self.handle_pruning(saliency, percentage)

        
        