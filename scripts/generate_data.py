import os
from time import sleep

import numpy as np
import hydra
from async_savers import AsyncSaver
from tqdm import trange

from data_learning_boilerplate.utils import set_seed


@hydra.main(config_path='../cfg', config_name='generate_data')
def main(cfg):
    set_seed(cfg['seed'])

    hydra_dir = os.getcwd()
    saver = AsyncSaver(hydra_dir, 'data', save_every=cfg['save_every'])
    saver.start()

    for _ in trange(cfg['n_samples']):
        x = np.random.uniform([-1, -1], [1, 1])
        eps = np.random.normal(scale=0.1)
        y = x + eps

        saver.save({
            'x': x,
            'y': y
        })

        sleep(1e-3)

    saver.stop()
    

if __name__ == "__main__":
    main()
