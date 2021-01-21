from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningModule
import wandb


def _make_mlp(in_size, layer_sizes, act='relu', last_act=True, dropout=0, prefix=''):
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


class MlpModel(LightningModule):

    def __init__(self, cfg, ds=None):
        super().__init__()

        # Parse data 
        self._cfg = cfg
        if ds is not None:
            self._ds = ds

        self._mlp = _make_mlp(cfg['x_dim'], cfg['hidden_layers'] + [cfg['y_dim']], last_act=False)

    def forward(self, x):
        y_hat = self._mlp(x)

        return {
            'y_hat': y_hat
        }

    def _criterion(self, y_hat, y):
        loss = F.mse_loss(y_hat, y)
        return {
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch['x'])
        criterion_outs = self._criterion(outs['y_hat'], batch['y'])

        for k, v in criterion_outs.items():
            if v.dim() == 0:
                self.log(f'tr/{k}', v, on_step=True)

        return {'loss': criterion_outs['loss']}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._cfg['lr'])

    def train_dataloader(self):
        return DataLoader(self._ds, batch_size=self._cfg['batch_size'], sampler=SubsetRandomSampler(self._ds.train_idxs))

    def val_dataloader(self):
        return DataLoader(self._ds, batch_size=self._cfg['batch_size'], sampler=SubsetRandomSampler(self._ds.test_idxs))

    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch['x'])
        criterion_outs = self._criterion(outs['y_hat'], batch['y'])
        return criterion_outs

    def validation_epoch_end(self, all_outputs):
        for k, v in all_outputs[0].items():
            if v.dim() == 0:
                vs = [outputs[k] for outputs in all_outputs]
                mean_v = torch.mean(torch.stack(vs))
                self.log(f'val/{k}', mean_v)
