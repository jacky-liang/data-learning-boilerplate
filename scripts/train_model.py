import os
from pathlib import Path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_learning_boilerplate.model import MlpModel
from data_learning_boilerplate.data import MockDataset
from data_learning_boilerplate.callbacks import WandbUploadCallback
from data_learning_boilerplate.utils import set_seed


@hydra.main(config_path='../cfg', config_name='train_model.yaml')
def main(cfg):
    set_seed(cfg['seed'], torch=True)

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'

    callbacks = []
    wandb_logger = WandbLogger(name=cfg['tag'], config=cfg['train'], **cfg['wandb']['logger'])
    _ = wandb_logger.experiment # to explicitly trigger wandb init
    callbacks.append(WandbUploadCallback(
        checkpoint_path, hydra_dir, cfg['wandb']['saver']['upload']
    ))
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=checkpoint_path / '{epoch:04d}-{val_loss:.4f}',
        save_top_k=3
    )
    trainer = Trainer(
        gpus=[0],
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=cfg['train']['max_epochs'],
        callbacks=callbacks,
    )

    ds = MockDataset(Path(cfg['root_dir']), cfg['data'])
    ds_gen = lambda : ds
    model = MlpModel(cfg['train'], ds_gen=ds_gen)
    trainer.fit(model)


if __name__ == '__main__':
    main()
