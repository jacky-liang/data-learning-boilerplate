import os
from collections import OrderedDict
import numpy as np
import random

import torch.nn as nn
from async_savers import load_shards


def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)


def load_datas_from_cfg(data_dir, tags):
    datas = []
    for tag in tags:
        tag_path = data_dir / tag
        all_data_paths = list(filter(lambda p : p.is_dir(), list(tag_path.iterdir())))
        all_data_paths.sort()
        latest_data_path = all_data_paths[-1]
        datas.extend(load_shards(latest_data_path / 'mock_data'))

    return datas


def make_mlp(in_size, layer_sizes, act='relu', last_act=True, dropout=0, prefix=''):
    if act =='tanh':
        act_f = nn.Tanh()
    elif act == 'relu':
        act_f = nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
        act_f = nn.LeakyReLU(inplace=True)
    else:
        raise ValueError(f'Unknown act: {act}')

    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append((f'{prefix}_linear{i}', nn.Linear(in_size, layer_size)))
        if i < len(layer_sizes) - 1:
            if dropout > 0:
                layers.append((f'{prefix}_dropout{i}', nn.Dropout(dropout)))
            layers.append((f'{prefix}_{act}{i}', act_f))
        else:
            if last_act:
                layers.append((f'{prefix}_{act}{i}', act_f))
        in_size = layer_size
    return nn.Sequential(OrderedDict(layers))
