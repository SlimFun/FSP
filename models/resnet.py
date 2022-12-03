# Based on code taken from https://github.com/weiaicunzai/pytorch-cifar100

"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from .layers import *
import numpy as np
from .Pruneable import Pruneable

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        # print(f'{out_channels[0]}; {out_channels[1]}')

        #residual function
        self.residual_function = nn.Sequential(
            Conv2d(in_channels, out_channels[0], kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            Conv2d(out_channels[0], out_channels[1], kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels[1])
        )

        #shortcut
        self.shortcut = Identity2d(in_channels)

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels[1]:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels[1], kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels[1])
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.))
        self.residual_function = nn.Sequential(
            Conv2d(in_channels, width, kernel_size=1, bias=False),
            BatchNorm2d(width),
            nn.ReLU(inplace=True),
            Conv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(width),
            nn.ReLU(inplace=True),
            Conv2d(width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = Identity2d(in_channels)

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
# output_channels =[63, 12, 52, 15, 48, 14, 48, 13, 49, 12, 60, 10, 71, 9, 88, 6, 217]
# output_channels = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]

class ResNet(Pruneable):

    def __init__(self, block, num_block, base_width, output_channels, num_classes=200, dense_classifier=False, device='cpu', layer_ratio=None):
        super().__init__()

        self.in_channels = output_channels[0]
        self.device = device

        self.conv1 = nn.Sequential(
            Conv2d(3, output_channels[0], kernel_size=3, padding=1, bias=False),
            BatchNorm2d(output_channels[0]),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, [output_channels[1],output_channels[2],output_channels[3],output_channels[4]], num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, [output_channels[5],output_channels[6],output_channels[7],output_channels[8]], num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, [output_channels[9],output_channels[10],output_channels[11],output_channels[12]], num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, [output_channels[13],output_channels[14],output_channels[15],output_channels[16]], num_block[3], 2, base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(output_channels[16] * block.expansion, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(output_channels[16] * block.expansion, num_classes)
        self._initialize_weights()

        self.init_param_sizes()

    def init_param_sizes(self):
        # bits required to transmit mask and parameters?
        self.param_size = 0
        for _, layer in self.named_children():
            for name, param in layer.named_parameters():
                param_size = np.prod(param.size())
                self.param_size += param_size * 32 # FIXME: param.dtype.size?

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        idx = 0
        for stride in strides:
            layer_list.append(block(self.in_channels, [out_channels[idx], out_channels[idx+1]], stride, base_width))
            # self.in_channels = out_channels * block.expansion
            self.in_channels = out_channels[idx+1]
            idx += 2
        
        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

def _resnet(arch, block, num_block, base_width, num_classes, dense_classifier, pretrained, output_channels, device, layer_ratio):
    model = ResNet(block, num_block, base_width, output_channels, num_classes, dense_classifier, device, layer_ratio)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet18(device, output_channels=None, layer_ratio=None, num_classes=10, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    if output_channels == None:
        output_channels = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained, output_channels, device, layer_ratio)

def resnet34(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 34 object
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained)

def resnet50(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained)

def resnet101(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 101 object
    """
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], 64, num_classes, dense_classifier, pretrained)

def resnet152(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 152 object
    """
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], 64, num_classes, dense_classifier, pretrained)



def wide_resnet18(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64 * 2, num_classes, dense_classifier, pretrained)

def wide_resnet34(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 34 object
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], 64 * 2, num_classes, dense_classifier, pretrained)

def wide_resnet50(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64 * 2, num_classes, dense_classifier, pretrained)

def wide_resnet101(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 101 object
    """
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], 64 * 2, num_classes, dense_classifier, pretrained)

def wide_resnet152(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 152 object
    """
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], 64 * 2, num_classes, dense_classifier, pretrained)



