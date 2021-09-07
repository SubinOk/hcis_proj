import numpy as np
import torch
import argparse
from copy import deepcopy

import model
import exp
import dataset

# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
#args.batch_size = 1
args.batch_size = (16,128)
args.x_frames = 1
args.y_frames = 3 # the number of classes

# ====== Model Capacity ===== #
args.input_dim = 62
args.hid_dim = 10
args.n_layers = 1
args.n_filters = 64
args.filter_size = 1
args.str_len = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.5
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam'
args.model = 'Conv1D'
#args.lr = 0.001
args.lr = (0.0001,0.001)
args.epoch = 2

# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'n_layers'
list_var1 = [0.001, 0.0001, 0.00001]
list_var2 = [1, 2, 3]

args.init_points = 2
args.n_iter = 8
# ================================= #

trainset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames, args.str_len)
trainset, valset = torch.utils.data.random_split(trainset, [362, 90])
valset, testset = torch.utils.data.random_split(valset, [45, 45])
"""
valset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames)
testset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames)
"""
partition = {'train': trainset, 'val': valset, 'test': testset}
setting, result = model.experiment(partition, deepcopy(args))
print('Settings:', setting)
print('Results:', result)
"""
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)

        setting, result = model.experiment(partition, deepcopy(args))
        exp.save_exp_result(setting, result)
        
var1 = 'lr'
var2 = 'n_layers'
df = exp.load_exp_result('exp1')

exp.plot_acc(var1, var2, df)
exp.plot_loss_variation(var1, var2, df, sharey=False)
# sharey를 True로 하면 모둔 subplot의 y축의 스케일이 같아집니다.

exp.plot_acc_variation(var1, var2, df, margin_titles=True, sharey=True)
"""