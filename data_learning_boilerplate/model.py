from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningModule
import wandb


def _make_mlp(in_size, layer_sizes, act='tanh', last_act=True, dropout=0, prefix=''):
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

        self._make_model()

    def _make_model(self):
        self._encoder = _make_mlp(self._cfg['n_data'] + self._cfg['n_cond'],
                                    self._cfg['encoder'], prefix='encoder')

        self._hidden_dim = self._cfg['encoder'][-1]
        self._fc_mu = _make_mlp(self._hidden_dim, [self._cfg['n_latent']], prefix='fc_mu', last_act=False)
        self._fc_var = _make_mlp(self._hidden_dim, [self._cfg['n_latent']], prefix='fc_var', last_act=False)

        self._decoder = _make_mlp(self._cfg['n_latent'] + self._cfg['n_cond'], 
                        list(self._cfg['decoder']) + [self._cfg['n_data']],
                        prefix='encoder', last_act=False)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def encode(self, data_cond):
        result = self.encoder(data_cond)
        
        mu = self._fc_mu(result)
        log_var = self._fc_var(result)

        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, eps

    def decode(self, z):
        result = self._decoder(z)
        return result

    def forward(self, x, cond):
        data_cond = torch.cat([x, cond], dim=1)
        mu, log_var = self.encode(data_cond)

        z, eps = self.reparameterize(mu, log_var)
        z_cond = torch.cat([z, cond], dim=1)

        x_hat = self.decode(z_cond)

        nll = -torch.sum(self._normal_distr.log_prob(eps), dim=1)

        return {
            'x_hat': x_hat,
            'mu': mu,
            'log_var': log_var,
            'nll': nll
        }

    def sample(self, cond, num_samples, truncate=1):
        rvs = truncnorm.rvs(-truncate, truncate, size=(num_samples, self._cfg['n_latent']))
        nll = from_numpy(-np.sum(truncnorm.logpdf(rvs, -3, 3), axis=1)).type_as(cond)
        
        z = from_numpy(rvs).type_as(cond)
        z_cond = torch.cat([z, cond.repeat(num_samples, 1)], dim=1)
        x_hats = self.decode(z_cond)
        return {
            'x_hats': x_hats,
            'nlls': nll
        }

    def generate(self, x, cond):
        return self.forward(x, cond)['x_hat']

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch['x'], batch['cond'])
        criterion_outs = self._criterion(batch['x'], outs['x_hat'], outs['mu'], outs['log_var'])

        for k, v in criterion_outs.items():
            if v.dim() == 0:
                self.log(f'tr/{k}', v, on_step=True)

        return {'loss': criterion_outs['loss']}

    def _criterion_pred(self, x, x_hat):
        recons_loss = F.mse_loss(x_hat, x)
        
        n_state = self._cfg['n_state']
        state_loss = F.mse_loss(x_hat[:, :n_state], x[:, :n_state])
        exec_time_loss = F.mse_loss(x_hat[:, n_state], x[:, n_state])
        exec_cost_loss = F.mse_loss(x_hat[:, n_state + 1], x[:, n_state + 1])
        plan_time_loss = F.mse_loss(x_hat[:, n_state + 2], x[:, n_state + 2])

        return {
            'recons_loss': recons_loss,
            'state_loss': state_loss,
            'exec_time_loss': exec_time_loss,
            'exec_cost_loss': exec_cost_loss,
            'plan_time_loss': plan_time_loss
        }

    def _criterion(self, x, x_hat, mu, log_var):
        pred_losses = self._criterion_pred(x, x_hat)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) / x.shape[0]

        loss = pred_losses['recons_loss'] + kld_loss * self._cfg['klscale']

        criterion_outs = {
            'loss': loss, 
            'kld_loss': kld_loss
        }
        criterion_outs.update(pred_losses)
        return criterion_outs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._cfg['lr'])

    def train_dataloader(self):
        return DataLoader(self._ds, batch_size=self._cfg['batch_size'], sampler=SubsetRandomSampler(self._ds.train_idxs))

    def val_dataloader(self):
        return DataLoader(self._ds, batch_size=self._cfg['batch_size'], sampler=SubsetRandomSampler(self._ds.test_idxs))

    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch['x'], batch['cond'])
        criterion_outs = self._criterion(batch['x'], outs['x_hat'], outs['mu'], outs['log_var'])
        criterion_outs.update(batch)
        criterion_outs.update(outs)

        return criterion_outs

    def validation_epoch_end(self, all_outputs):
        for k, v in all_outputs[0].items():
            if v.dim() == 0:
                vs = [outputs[k] for outputs in all_outputs]
                mean_v = torch.mean(torch.stack(vs))
                self.log(f'val/{k}', mean_v)
