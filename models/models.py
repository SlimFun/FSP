import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

import warnings

from .vgg import VGG_SNIP, vgg11_bn, vgg11, vgg4_bn
from .Pruneable import Pruneable
from .resnet import *
from .conv import conv

# class CNN2(PrunableNet):
#     def __init__(self, device='cpu', in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
#         super(CNN2, self).__init__(device=device)
#         # self.activation = nn.ReLU(True)

#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        
#         # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
#         # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
#         # self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear((hidden_channels * 2) * (8 * 8), num_hiddens, bias=True)
#         self.fc2 = nn.Linear(num_hiddens, num_classes, bias=True)

#         self.init_param_sizes()

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x), inplace=True), kernel_size=2, padding=1)

#         # x = self.activation(self.conv1(x))
#         # x = F.max_pool2d(F.relu(self.bn1(self.conv1(x)), inplace=True), kernel_size=2, stride=2)
#         # x = nn.MaxPool2d()
#         x = F.max_pool2d(F.relu(self.conv2(x), inplace=True), kernel_size=2, padding=1)
#         # x = self.activation(self.conv2(x))
#         # x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         # x = self.flatten(x)
    
#         # x = self.activation(self.fc1(x))
#         x = F.relu(self.fc1(x), inplace=True)
#         x = self.fc2(x)
        
#         return x

class CNN2(Pruneable):
    def __init__(self, device='cpu', in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(CNN2, self).__init__(device=device)
        # self.activation = nn.ReLU(True)

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        # self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        
        # # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        # # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        # # self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(input_dim=(hidden_channels * 2) * (8 * 8), output_dim=num_hiddens, bias=True)
        # self.fc2 = nn.Linear(input_dim=num_hiddens, output_dim=num_classes, bias=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear((hidden_channels * 2) * (8 * 8), num_hiddens, bias=True)
        self.fc2 = nn.Linear(num_hiddens, num_classes, bias=True)

        self.init_param_sizes()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x), inplace=True), kernel_size=2, padding=1)

        # x = self.activation(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.bn1(self.conv1(x)), inplace=True), kernel_size=2, stride=2)
        # x = nn.MaxPool2d()
        x = F.max_pool2d(F.relu(self.conv2(x), inplace=True), kernel_size=2, padding=1)
        # x = self.activation(self.conv2(x))
        # x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
    
        # x = self.activation(self.fc1(x))
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        
        return x

class CIFAR10Net(Pruneable):

    def __init__(self, *args, **kwargs):
        super(CIFAR10Net, self).__init__(*args, **kwargs)

        self.conv1 = self.Conv2d(3, 6, 5)
        self.conv2 = self.Conv2d(6, 16, 5)

        self.fc1 = self.Linear(16 * 20 * 20, 120)
        self.fc2 = self.Linear(120, 84)
        self.fc3 = self.Linear(84, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def needs_mask(name):
    return name.endswith('weight')

def initialize_mask(model, dtype=torch.bool):
    layers_to_prune = (layer for _, layer in model.named_children())
    for layer in layers_to_prune:
        for name, param in layer.named_parameters():
            if name.endswith('weight'):
                if hasattr(layer, name + '_mask'):
                    warnings.warn(
                        'Parameter has a pruning mask already. '
                        'Reinitialize to an all-one mask.'
                    )
                name = name.replace('.', '_')
                layer.register_buffer(name + '_mask', torch.ones_like(param, dtype=dtype))
                continue
                parent = name[:name.rfind('.')]

                for mname, module in layer.named_modules():
                    if mname != parent:
                        continue
                    module.register_buffer(name[name.rfind('.')+1:] + '_mask', torch.ones_like(param, dtype=dtype))

all_models = {
        'CNNNet': CNN2,
        'VGG_SNIP': VGG_SNIP,
        'VGG11_BN': vgg11_bn,
        'VGG11': vgg11,
        'CIFAR10Net': CIFAR10Net,
        'resnet18': resnet18,
        'VGG4_BN': vgg4_bn,
        'conv': conv,
}