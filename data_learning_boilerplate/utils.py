import os
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)
