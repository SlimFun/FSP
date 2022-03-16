import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from .vgg import VGG_SNIP
from .Pruneable import Pruneable

class CNN2(Pruneable):
    def __init__(self, device='cpu', in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(CNN2, self).__init__(device=device)
        self.activation = nn.ReLU(True)

        self.conv1 = self.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = self.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = self.Linear(input_dim=(hidden_channels * 2) * (8 * 8), output_dim=num_hiddens, bias=False)
        self.fc2 = self.Linear(input_dim=num_hiddens, output_dim=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

all_models = {
        'CNNNet': CNN2,
        'VGG_SNIP': VGG_SNIP
}