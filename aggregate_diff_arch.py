import copy
from turtle import shape
from torch import nn
import utils
from collections import Counter
import numpy as np
import torch
import SNAP
import torch.nn as nn
from models.vgg import VGG
import math
import random
from models.models import all_models

import wandb

def needs_mask(layer_name):
    return layer_name.endswith('weight') and ('bn' not in layer_name)


def aggregate(self, keep_masks_dict, model_list, round, download_cost, upload_cost, compute_times, client_outputs=None):
        self.transmission_cost += self.round_trans_cost(download_cost, upload_cost)
        self.compute_time += self.round_compute_time(compute_times)
        last_params = self.get_global_params()

        training_num = sum(local_train_num for (local_train_num, _) in model_list)

        if client_outputs != None:
            for k in last_params.keys():
                idx = 0
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    shape_dim = len(last_params[k].shape)
                    if shape_dim == 4:
                        idx += 1
                        print(f'{last_params[k].shape} == {output_channels_1[idx]} == {local_model_params[k].shape}')
                        last_params[k][:output_channels_1[idx],:prio_channel,:,:] += local_model_params[k] * w
                        prio_channel = output_channels_1[idx]
                    elif shape_dim == 1:
                        print(last_params[k].shape)
                        last_params[k][:output_channels_1[idx]] += local_model_params[k] * w
                    elif shape_dim == 2:
                        print(f'{last_params[k].shape} == {output_channels_1[idx]} == {local_model_params[k].shape}')
                        last_params[k][:,:output_channels_1[idx]] += local_model_params[k] * w

            self.set_global_params(last_params)

        else:
            (num0, averaged_params) = model_list[0]
            if (averaged_params is not None):
                for k in averaged_params.keys():
                    for i in range(0, len(model_list)):
                        local_sample_number, local_model_params = model_list[i]
                        w = local_sample_number / training_num
                        if i == 0:
                            averaged_params[k] = local_model_params[k] * w
                        else:
                            averaged_params[k] += local_model_params[k] * w

                # for name, param in averaged_params.items():
                for name in last_params:
                    assert (last_params[name].shape == averaged_params[name].shape)
                    averaged_params[name] = averaged_params[name].type_as(last_params[name])
                    # last_params[name] = last_params[name].type_as(averaged_params[name])
                    last_params[name] += averaged_params[name]
                self.set_global_params(last_params)
  
output_channels_1 = [63, 127, 94, 203, 59, 321, 37, 511]
net1 = all_models['VGG11_BN'](output_channels=output_channels_1)
net2 = all_models['VGG11_BN'](output_channels=[64, 128, 256, 256, 512, 512, 512, 512])

last_params = net2.cpu().state_dict()
local_sample_number = 500.
local_model_params = net1.cpu().state_dict()
server_model_params = net2.cpu().state_dict()

w = local_sample_number / 5000
idx = -1
prio_channel = 3
for k in local_model_params.keys():
    shape_dim = len(server_model_params[k].shape)
    if shape_dim == 4:
        idx += 1
        print(f'{server_model_params[k].shape} == {output_channels_1[idx]} == {local_model_params[k].shape}')
        server_model_params[k][:output_channels_1[idx],:prio_channel,:,:] += local_model_params[k] * w
        prio_channel = output_channels_1[idx]
    elif shape_dim == 1:
        print(server_model_params[k].shape)
        server_model_params[k][:output_channels_1[idx]] += local_model_params[k] * w
    elif shape_dim == 2:
        print(f'{server_model_params[k].shape} == {output_channels_1[idx]} == {local_model_params[k].shape}')
        server_model_params[k][:,:output_channels_1[idx]] += local_model_params[k] * w
    print(server_model_params[k].shape)
for k in local_model_params.keys():
    for i in range(0, len([local_model_params])):
        w = local_sample_number / 5000


