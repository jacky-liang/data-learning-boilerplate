import os
from pathlib import Path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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

    # Make logger 
    wandb_logger = WandbLogger(name=cfg['tag'], config=cfg['train'], **cfg['wandb']['logger'])
    _ = wandb_logger.experiment # to explicitly trigger wandb init

    # Make callbacks
    wandb_upload_callback = WandbUploadCallback(
        checkpoint_path, hydra_dir, cfg['wandb']['saver']['upload']
    )
    early_stop_callback = EarlyStopping(
        monitor=cfg['wandb']['saver']['monitor'],
        min_delta=0.0,
        patience=3,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=str(checkpoint_path),
        filename='{epoch:04d}-{val_loss:.4f}',
        save_top_k=3,
        save_last=True
    )

    # Make trainer and start straining
    trainer = Trainer(
        gpus=[0],
        fast_dev_run=cfg['debug'],
        max_epochs=cfg['train']['max_epochs'],
        logger=wandb_logger,
        callbacks=[
            wandb_upload_callback,
            early_stop_callback,
            checkpoint_callback
        ],
    )

    ds = MockDataset(Path(cfg['root_dir']), cfg['data'])
    ds_gen = lambda : ds
    model = MlpModel(cfg['train'], ds_gen=ds_gen)
    trainer.fit(model)


if __name__ == '__main__':
    main()
