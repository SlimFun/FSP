import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .utils import init_param
# from modules import Scaler

def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output

class Conv(nn.Module):
    def __init__(self, data_shape, output_channels, classes_size, rate=None, track=False):
        super().__init__()
        norm = nn.BatchNorm2d(output_channels[0], momentum=None, track_running_stats=track)
        if rate == None:
            scaler = nn.Identity()
        else:
            scaler = Scaler(rate[0])
        blocks = [nn.Conv2d(3, output_channels[0], 3, 1, 1),
                  scaler,
                  norm,
                  nn.ReLU(inplace=True),
                #   nn.MaxPool2d(2, 2)
                    nn.MaxPool2d(2)
                  ]
        for i in range(len(output_channels) - 1):
            norm = nn.BatchNorm2d(output_channels[i + 1], momentum=None, track_running_stats=track)
            if rate == None:
                scaler = nn.Identity()
            else:
                scaler = Scaler(rate[i+1])
            blocks.extend([nn.Conv2d(output_channels[i], output_channels[i + 1], 3, 1, 1),
                           scaler,
                           norm,
                           nn.ReLU(inplace=True),
                        #    nn.MaxPool2d(2, 2)]
                            nn.MaxPool2d(2)]
                           )
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(output_channels[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

        self.apply(init_param)

        self.init_param_sizes()

    def sparsity_percentage(self):
        if isinstance(self.mask, dict):
            masks = self.mask.values()
        elif isinstance(self.mask, list):
            masks = self.mask
        zero_c = 0.
        total = 0.
        for m in masks:
            a = m.view(-1).to(device='cpu', copy=True)
            a = torch.where(a>=1, 1, 0)
            zero_c +=sum(np.where(a.numpy(), 0, 1))
            total += m.numel()
        return zero_c/total

    def init_param_sizes(self):
        # bits required to transmit mask and parameters?
        self.param_size = 0
        for _, layer in self.named_children():
            for name, param in layer.named_parameters():
                param_size = np.prod(param.size())
                self.param_size += param_size * 32 # FIXME: param.dtype.size?

    def forward(self, input):
        # output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        # x = input['img']
        out = self.blocks(input)
        return out

def conv(device, output_channels=[64, 128, 256, 512], layer_ratio=None):
    model = Conv([3, 32, 32], output_channels, 10, rate=layer_ratio, track=False).to(device)
    return model
# def conv(model_rate=1, track=False):
#     data_shape = cfg['data_shape']
#     hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['conv']['hidden_size']]
#     classes_size = cfg['classes_size']
#     scaler_rate = model_rate / cfg['global_model_rate']
#     model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track)
#     model.apply(init_param)
#     return model