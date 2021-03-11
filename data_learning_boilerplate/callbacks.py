from pytorch_lightning.callbacks import Callback
import wandb


class WandbUploadCallback(Callback):

    def __init__(self, checkpoint_path, hydra_dir, upload, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_path = checkpoint_path
        self._hydra_dir = hydra_dir
        self._upload = upload

    def on_epoch_start(self, trainer, pl_module):
        if self._upload:
            wandb.save(str(self._hydra_dir / '.hydra' / '*.yaml'), base_path=str(self._hydra_dir))

    def on_train_end(self, trainer, pl_module):
        if self._upload:
            wandb.save(str(self._checkpoint_path / '*.ckpt'), base_path=str(self._hydra_dir))
