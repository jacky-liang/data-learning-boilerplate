import os
import numpy as np
import random


def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)
