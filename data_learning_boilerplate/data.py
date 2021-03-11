import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .utils import load_datas_from_cfg


class MockDataset(Dataset):

    def __init__(self, root_dir, cfg):
        data_dir = root_dir / cfg['rel_data_dir']
        datas = load_datas_from_cfg(data_dir, cfg['tags'])

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

    def get_train_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.train_idxs))

    def get_val_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.test_idxs))

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
