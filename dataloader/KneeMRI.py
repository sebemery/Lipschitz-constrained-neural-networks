import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class KneeMRI(Dataset):
    def __init__(self, data_dir, sigma, architecture):

        self.data_dir = data_dir
        self.sigma = sigma
        self.architecture = architecture
        self.files = []
        self._set_files()

    def _set_files(self):
        self.files = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.files[index])
        slice = cv2.imread(path, 0)
        slice = slice / 255
        slice = torch.from_numpy(slice)
        noise = torch.randn(slice.shape)
        noisy_slice = slice + noise * self.sigma
        noisy_slice = torch.unsqueeze(noisy_slice, 0)
        cropp1 = noisy_slice[:, 0:160, 0:160]
        cropp2 = noisy_slice[:, 160:320, 0:160]
        cropp3 = noisy_slice[:, 0:160, 160:320]
        cropp4 = noisy_slice[:, 160:320, 160:320]
        if self.architecture == "residual":
            noise = torch.unsqueeze(noise, 0)
            noise1 = noise[:, 0:160, 0:160]
            noise2 = noise[:, 160:320, 0:160]
            noise3 = noise[:, 0:160, 160:320]
            noise4 = noise[:, 160:320, 160:320]
            return cropp1.float(), cropp2.float(), cropp3.float(), cropp4.float(), noise1.float(), noise2.float(), \
                   noise3.float(), noise4.float(), self.files[index]
        else:
            slice = torch.unsqueeze(slice, 0)
            slice1 = slice[:, 0:160, 0:160]
            slice2 = slice[:, 160:320, 0:160]
            slice3 = slice[:, 0:160, 160:320]
            slice4 = slice[:, 160:320, 160:320]
            return cropp1.float(), cropp2.float(), cropp3.float(), cropp4.float(), slice1.float(), slice2.float(), \
                   slice3.float(), slice4.float(), self.files[index]


