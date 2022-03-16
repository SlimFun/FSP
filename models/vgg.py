"Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Pruneable import Pruneable

VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_SNIP(Pruneable):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm
    """

    def __init__(self, device='cpu', config='D', num_classes=10):
        super().__init__(device=device)

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)

        self.classifier = nn.Sequential(
            self.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
            # nn.LeakyReLU(leak, True),
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            self.Linear(512, 512),
            # nn.LeakyReLU(leak, True),
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            self.Linear(512, num_classes),
        )

        self._initialize_weights()

    def make_layers(self, config, batch_norm=False):  # TODO: BN yes or no?
        layers = []
        in_channels = 3
        # leak = 0.05

        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = self.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        # nn.LeakyReLU(leak, inplace=True)
                        nn.ReLU(True),
                    ]
                else:
                    # layers += [conv2d, nn.LeakyReLU(leak, inplace=True)]
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            # if isinstance(m, self.Conv2d) or isinstance(m, self.Linear):
            #     nn.init.xavier_normal_(m.weight)
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = F.log_softmax(x, dim=1)
        return x