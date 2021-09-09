import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms

class NumDataset(Dataset):

    def __init__(self, file_path, x_frames, y_frames, stride):
        """
        Args:
            file_path (string): Path to the csv file with annotations
            x_frames (integer): Length of input value (time scale)
            y_frames (integer): Length of output value (time scale)
        """

        # if data are .csv file
        raw_data = pd.read_csv(file_path)
        raw_data = raw_data.fillna(-5) # 결측값을 -5로 대체하기
        self.data = raw_data

        self.x_frames = x_frames
        self.y_frames = y_frames
        self.stride = stride

    def __len__(self):
        max_idx = len(self.data) - self.x_frames
        num_of_idx = max_idx // self.stride
        return num_of_idx

    def __getitem__(self, idx):
        idx = idx * self.stride
        data = self.data.values
        X = data[idx : idx + self.x_frames, 1:-1]
        X.astype(float)
        y = data[idx + self.x_frames, -1]
        y.astype(float)
        y = np.expand_dims(y, axis=0)

        return X, y
