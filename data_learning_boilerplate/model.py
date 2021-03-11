import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningModule

from .utils import make_mlp


class MlpModel(LightningModule):

    def __init__(self, cfg_dict, ds_gen=None):
        super().__init__()

        self._ds_gen = ds_gen
        self.save_hyperparameters(cfg_dict)

        self._mlp = make_mlp(
            self.hparams.x_dim, self.hparams.hidden_layers + [self.hparams.y_dim], 
            last_act=False
        )

    @property
    def ds(self):
        return self._ds_gen()

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.hparams.batch_size, sampler=SubsetRandomSampler(self.ds.train_idxs))

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.hparams.batch_size, sampler=SubsetRandomSampler(self.ds.test_idxs))

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

                if k == 'loss':
                    # To get around this problem:
                    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4012
                    self.log(f'val_{k}', mean_v)
