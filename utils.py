import copy
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import sys


from tqdm import tqdm

PROJ_NAME = "SNIP-it"
WORKING_DIR_PATH = "."

# output
RESULTS_DIR = "results"
DATA_DIR = "data"
GITIGNORED_DIR = "gitignored"

IMAGENETTE_DIR = os.path.join(".", "gitignored", "data", "imagenette-320")
IMAGEWOOF_DIR = os.path.join(".", "gitignored", "data", "imagewoof-320")
TINY_IMAGNET_DIR = os.path.join(".", "gitignored", "data", "tiny_imagenet")

CODEBASE_DIR = "codebase"
SUMMARY_DIR = "summary"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
PROGRESS_DIR = "progress"
OUTPUT_DIRS = [OUTPUT_DIR, SUMMARY_DIR, CODEBASE_DIR, MODELS_DIR, PROGRESS_DIR]

# DATA_MANAGER = DataManager(os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR))
DATASET_PATH = os.path.join(GITIGNORED_DIR, DATA_DIR)
# RESULTS_PATH = os.path.join(DATA_MANAGER.directory, RESULTS_DIR)

# printing
PRINTCOLOR_PURPLE = '\033[95m'
PRINTCOLOR_CYAN = '\033[96m'
PRINTCOLOR_DARKCYAN = '\033[36m'
PRINTCOLOR_BLUE = '\033[94m'
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_YELLOW = '\033[93m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_UNDERLINE = '\033[4m'
PRINTCOLOR_END = '\033[0m'

MODELS_DIR = "models"
LOSS_DIR = "losses"
CRITERION_DIR = "criterions"
NETWORKS_DIR = "networks"
TRAINERS_DIR = "trainers"
TESTERS_DIR = "testers"
OPTIMS = "optim"
DATASETS = "datasets"
types = [LOSS_DIR, NETWORKS_DIR, CRITERION_DIR, TRAINERS_DIR, TESTERS_DIR]

TEST_SET = "test"
VALIDATION_SET = "validation"
TRAIN_SET = "train"

ZERO_SIGMA = -1 * 1e6

SNIP_BATCH_ITERATIONS = 5

HOYER_THERSHOLD = 1e-3

SMALL_POOL = (2, 2)
PROD_SMALL_POOL = np.prod(SMALL_POOL)
MIDDLE_POOL = (3, 3)
PROD_MIDDLE_POOL = np.prod(MIDDLE_POOL)
BIG_POOL = (5, 5)
PROD_BIG_POOL = np.prod(BIG_POOL)
NUM_WORKERS = 6
FLIP_CHANCE = 0.2
STRUCTURED_SINGLE_SHOT = [
    "SNAP",
    "SNAPit",
    "StructuredRandom",
    "StructuredGRASP",
    "GateDecorators",
    "CNIP",
    "CNIPit",
]
SINGLE_SHOT = [
    "SNIP",
    "SNIPit",
    "GRASP",
    "IterativeGRASP",
    "UnstructuredRandom"
]
SINGLE_SHOT += STRUCTURED_SINGLE_SHOT
DURING_TRAINING = [
    "SNAPitDuring",
    "GateDecorators",
    "CNIPitDuring",
    "GroupHoyerSquare",
    "EfficientConvNets"
]

TIMEOUT = int(60 * 60 * 1.7)  # one hour and a 45 minutes max
STACK_NAME = "command_stack"
alias = {
    "SNAPit": "SNAP-it (before)",
    "SNIPit": "SNIP-it (before)",
    "UnstructuredRandom": "Random (before)",
    "StructuredRandom": "Random (before) ",
    "SNIPitDuring": "SNIP-it (during)",
    "IMP": "IMP-global (during)",
    "L0": "L0 (during)",
    "GRASP": "GraSP (before)",
    "SNIP": "SNIP (before)",
    "GateDecorators": "GateDecorators (after)",
    "CNIPit": "CNIP-it (before)",
    "CNIPitDuring": "CNIP-it (during)",
    "HoyerSquare": "HoyerSquare (after)",
    "GroupHoyerSquare": "Group-HS (after)",
    "EfficientConvNets": "EfficientConvNets (after)"
}
SKIP_LETTERS = 19

linestyles_times = {
    "before": "solid",
    "during": "solid",
    "after": "solid",
}

timing_names = {
    "SNAPit": "before",
    "SNIPit": "before",
    "UnstructuredRandom": "before",
    "StructuredRandom": "before",
    "SNIPitDuring": "during",
    "IMP": "during",
    "L0": "during",
    "GRASP": "before",
    "SNIP": "before",
    "GateDecorators": "after",
    "CNIPit": "before",
    "CNIPitDuring": "during",
    "HoyerSquare": "after",
    "GroupHoyerSquare": "after",
    "EfficientConvNets": "after"
}

color_per_method = {
    "SNAPit": "#1f77b4",
    "SNIPit": "#d62728",
    "UnstructuredRandom": "#8c564b",
    "StructuredRandom": "#8c564b",
    "SNIPitDuring": "#1f77b4",
    "IMP": "#9467bd",
    "L0": "#d62728",
    "GRASP": "#ff7f0e",
    "SNIP": "#7f7f7f",
    "GateDecorators": "#ff7f0e",
    "CNIPit": "#ff7f0e",
    "CNIPitDuring": "#2ca02c",
    "HoyerSquare": "#2ca02c",
    "GroupHoyerSquare": "#2ca02c",
    "EfficientConvNets": "#9467bd"
}


#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type='xavier', init_gain=1.0):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     model.to(gpu_ids[0])
    #     model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

def evaluate_global(clients, global_model, progress=False, n_batches=0):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            accuracies[client_id] = client.test(model=global_model).item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        train_accuracies = {}
        train_losses = {}
        # sparsities = {}

        test_accuracies = {}
        test_losses = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            # client.reset_weights(global_state_dict=global_model.state_dict())
            # global_model = None
            accuracy, loss = client.test(model=global_model, train_data=True)
            train_accuracies[client_id] = accuracy
            train_losses[client_id] = loss
            # accuracies[client_id] = client.test().item()
            # sparsities[client_id] = client.sparsity()

            accuracy, loss = client.test(model=global_model, train_data=False)
            test_accuracies[client_id] = accuracy
            test_losses[client_id] = loss

    return train_accuracies, train_losses, test_accuracies, test_losses


def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=args.outfile)
    print(*arg, **kwargs)

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def needs_mask(name):
    return name.endswith('weight')

def compare_model(model_a, model_b):
    diff_c = 0
    total_p = 0
    d1 = None
    d2 = None
    for name, params in model_a.items():
        # if needs_mask(name):
        d1 = params.device
        d2 = model_b[name].device
        diff = params - model_b[name]
        diff_c += sum(np.where(diff.to('cpu', copy=True).view(-1).numpy(), 0, 1))
        total_p += params.numel()
    print(f'non changed: {float(diff_c) / total_p}, total_p: {total_p}, {d1}-{d2}')

def apply_global_mask(net, keep_masks):
    prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        layer.weight.data[keep_mask == 0.] = 0.

def count_model_zero_params(params):
    zero_c = 0.0
    total = 0.0

    for name, param in params.items():
        a = param.view(-1).to(device='cpu', copy=True).numpy()
        zero_c +=sum(np.where(a, 0, 1))
        total += param.numel()

    return zero_c / total

def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    handles = []

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        # layer.weight.register_hook(hook_factory(keep_mask))
        handles.append(layer.weight.register_hook(hook_factory(keep_mask)))

    return handles

def apply_grad_mask(net, keep_masks):
    # print('apply grad mask')
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    handles = []

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)
        # print(layer.weight.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # layer.weight.data[keep_mask == 0.] = 0.
        # layer.weight.register_hook(hook_factory(keep_mask))
        handles.append(layer.weight.register_hook(hook_factory(keep_mask)))

    return handles