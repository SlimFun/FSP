"Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Pruneable import Pruneable
from models.networks.assisting_layers.ContainerLayers import ContainerLinear, ContainerConv2d
from .layers import *


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(Pruneable):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        output_channel: int=512,
        init_weights: bool = True, device='cpu'
    ) -> None:
        super(VGG, self).__init__(device=device)
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.MaxPool2d(2)
        # self.classifier = nn.Sequential(OrderedDict([
        #     # nn.Linear(512 * 7 * 7, 4096),
        #     # nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(4096, 4096),
        #     # nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(4096, num_classes),
        #     # nn.BatchNorm1d(512),
        #     # self.Linear(512, 512),
        #     # nn.ReLU(True),
        #     # nn.BatchNorm1d(512),
        #     # self.Linear(512, 512),
        #     # nn.ReLU(True),
        #     # self.Linear(512, 10),
        #     ('fc1', self.Linear(512, 512)),  # 512 * 7 * 7 in the original VGG
        #     # nn.LeakyReLU(leak, True),
        #     ('relu1', nn.ReLU(True)),
        #     # ('bn1', nn.BatchNorm1d(512)),  # instead of dropout
        #     ('fc2', self.Linear(512, 512)),
        #     # nn.LeakyReLU(leak, True),
        #     ('relu2', nn.ReLU(True)),
        #     # ('bn2', nn.BatchNorm1d(512)),  # instead of dropout
        #     ('fc3', self.Linear(512, num_classes)),
        # ]))
        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', Linear(output_channel, num_classes)),
        ]))
        # if init_weights:
        # self._initialize_weights()

        self.init_param_sizes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             v = int(v)
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)

def make_layers(cfg, batch_norm=False, layer_ratio=None, track_running_stats=True):
    layers = []
    in_channels = 3
    idx = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if layer_ratio != None:
                print(f'{idx} : {layer_ratio[idx]}')
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1, ratio=layer_ratio[idx])
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, momentum=None, track_running_stats=track_running_stats), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, momentum=None, track_running_stats=track_running_stats), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            idx += 1
    return nn.Sequential(*layers)

cfgs ={
    'O': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg4_bn(device='cpu', output_channels=None, layer_ratio=None):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if output_channels == None:
        print(layer_ratio)
        return VGG(make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M'], batch_norm=True, layer_ratio=layer_ratio), output_channel=512, device=device)
    else:
        print(layer_ratio)
        return VGG(make_layers([output_channels[0], 'M', output_channels[1], 'M', output_channels[2], 'M', output_channels[3], 'M'], batch_norm=True, layer_ratio=layer_ratio), output_channel=output_channels[3], device=device)


def vgg11(device='cpu', output_channels=None, layer_ratio=None):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if output_channels == None:
        return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']), output_channel=512, device=device)
    else:
        print(layer_ratio)
        return VGG(make_layers([output_channels[0], 'M', output_channels[1], 'M', output_channels[2], output_channels[3], 'M', output_channels[4], output_channels[5], 'M', output_channels[6], output_channels[7], 'M'], layer_ratio=layer_ratio), output_channel=output_channels[7], device=device)

def vgg11_bn(device='cpu', output_channels=None, layer_ratio=None):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if output_channels == None:
        print(layer_ratio)
        return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], batch_norm=True, layer_ratio=layer_ratio), output_channel=512, device=device)
    else:
        print(layer_ratio)
        return VGG(make_layers([output_channels[0], 'M', output_channels[1], 'M', output_channels[2], output_channels[3], 'M', output_channels[4], output_channels[5], 'M', output_channels[6], output_channels[7], 'M'], batch_norm=True, layer_ratio=layer_ratio), output_channel=output_channels[7], device=device)


def vgg13():
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['B']))


def vgg13_bn():
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['A'], batch_norm=True))



def vgg16():
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['D']))


def vgg16_bn():
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['D'], batch_norm=True))


def vgg19():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['E']))

def vgg19_bn():
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs['E'], batch_norm=True))

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