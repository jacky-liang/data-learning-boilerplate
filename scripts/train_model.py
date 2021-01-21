import os
from pathlib import Path

import hydra
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_learning_boilerplate.model import MlpModel
from data_learning_boilerplate.data import MockDataset
from data_learning_boilerplate.utils import set_seed


@hydra.main(config_path='../cfg', config_name='train_model.yaml')
def main(cfg):
    ds = MockDataset(Path(cfg['root_dir']), cfg['data'])
    set_seed(cfg['seed'], torch=True)

    wandb_logger = WandbLogger(name=cfg['tag'], config=cfg['train'], **cfg['wandb']['logger'])
    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'
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
        max_epochs=cfg['train']['max_epochs']
    )

    ds = MockDataset(Path(cfg['root_dir']), cfg['data'])
    model = MlpModel(cfg['train'], ds)
    trainer.fit(model)

    if cfg['wandb']['saver']['upload']:
        wandb.save(str(checkpoint_path / '*.ckpt'), base_path=str(hydra_dir))
        wandb.save(str(hydra_dir / '.hydra' / '*.yaml'), base_path=str(hydra_dir))


if __name__ == '__main__':
    main()
