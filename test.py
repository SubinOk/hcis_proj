import pandas as pd
import torch
from scipy import interpolate
import numpy as np
import scipy.stats as ss

import dataset
from torch.utils.data import Dataset, DataLoader

dictionary = torch.FloatTensor([.1, .2, .3, .4])

if (type(dictionary) == np.ndarray) or (type(dictionary) == torch.Tensor):
    dictionary = dictionary.tolist()

print(type(dictionary))