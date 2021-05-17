import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import random


class BSD500(Dataset):

    def __init__(self, data_dir, noise):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.noise = noise
        data_file = os.listdir(self.data_dir)
        data_path = os.path.join(self.data_dir, data_file[0])
        h5f = h5py.File(data_path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.data_dir, 'r')
        key = self.keys[index]
        data = torch.Tensor(np.array(h5f[key]))
        noise = torch.FloatTensor(data.size()).normal_(mean=0, std=self.noise / 255.)
        noisy_data = data + noise
        h5f.close()
        return noisy_data, data

