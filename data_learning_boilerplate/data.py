from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from async_savers import load_shards


def _load_datas_from_cfg(data_dir, tags):
    datas = []
    for tag in tags:
        tag_path = data_dir / tag
        all_data_paths = list(filter(lambda p : p.is_dir(), list(tag_path.iterdir())))
        all_data_paths.sort()
        latest_data_path = all_data_paths[-1]
        datas.extend(load_shards(latest_data_path / 'mock_data'))

    return datas


class MockDataset(Dataset):

    def __init__(self, root_dir, cfg):
        data_dir = root_dir / cfg['rel_data_dir']
        datas = _load_datas_from_cfg(data_dir, cfg['tags'])

        X = np.array([d['x'] for d in datas])
        Y = np.array([d['y'] for d in datas])

        X_tr, X_t, Y_tr, Y_t = train_test_split(X, Y, test_size=cfg['test_size'])

        self._X_tr, self._X_t = torch.from_numpy(X_tr).float().cuda(), torch.from_numpy(X_t).float().cuda()
        self._Y_tr, self._Y_t = torch.from_numpy(Y_tr).float().cuda(), torch.from_numpy(Y_t).float().cuda()

        self._len = len(X)
        self._test_idxs = np.arange(len(X_tr), self._len)
        self._train_idxs = np.arange(len(X_tr))

    @property
    def train_idxs(self):
        return self._train_idxs.copy()

    @property
    def test_idxs(self):
        return self._test_idxs.copy()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < len(self._train_idxs):
            x, y = self._X_tr[idx], self._Y_tr[idx]
        else:
            idx -= len(self._train_idxs)
            x, y = self._X_t[idx], self._Y_t[idx]

        return {
            'x': x,
            'y': y
        }
