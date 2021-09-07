import torch
import argparse
from copy import deepcopy

import model

# ====== Random Seed Initialization ====== #
seed = 666
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
<<<<<<< Updated upstream
args.batch_size = 1
=======
# args.batch_size = 1
>>>>>>> Stashed changes
args.x_frames = 1
args.y_frames = 3 # the number of classes

# ====== Model Capacity ===== #
args.input_dim = 62
args.hid_dim = 10
# args.n_layers = 1
# args.n_filters = 64
args.filter_size = 1
args.str_len = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.5
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam'
<<<<<<< Updated upstream
args.model = 'CNN'
args.lr = 0.001
args.epoch = 2
=======
args.model = 'ConvLSTM'
# args.lr = (0.0001, 0.001)
# args.epoch = 2
>>>>>>> Stashed changes

# ====== Experiment Variable ====== #
args.batch_size = (16, 128)
args.n_layers = 1
args.n_filters = 64
args.lr = (0.0001, 0.001)
args.epoch = 2
# args.l2

<<<<<<< Updated upstream
=======
args.init_points = 2
args.n_iter = 8
# ================================= #
>>>>>>> Stashed changes

md_num = int(input("Enter the number of setting(1: train, 2: val, 3: test): "))

if md_num == 1:
    args.mode = 'train'
elif md_num == 2:
    args.mode = 'val'
elif md_num == 3:
    args.mode = 'test'
else:
    raise ValueError('In-valid mode choice')

setting, result = model.experiment(args.mode, deepcopy(args))

print('Settings:', setting)
print('Results:', result)
