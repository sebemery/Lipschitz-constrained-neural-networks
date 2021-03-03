import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class KneeMRI(Dataset):
    def __init__(self, target_dir, noise_dirs):

        self.target_dir = target_dir
        self.noise_dirs = noise_dirs
        self.target_files = []
        self.noise1_files = []
        if len(noise_dirs) > 1:
            self.noise2_files = []
        self._set_files()

    def _set_files(self):
        self.target_files = os.listdir(self.target_dir)
        self.noise1_files = os.listdir(self.noise_dirs[0])
        if len(self.noise_dirs) > 1:
            self.noise2_files = os.listdir(self.noise_dirs[1])

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, index):
        path_target = os.path.join(self.target_dir, self.target_files[index])
        slice = torch.load(path_target)
        if len(self.noise_dirs) == 1:
            path_noise = os.path.join(self.noise_dirs[0], self.noise1_files[index])
            noisy_slice = torch.load(path_noise)
            noisy_slice = torch.unsqueeze(noisy_slice, 0)
            cropp1 = noisy_slice[:, 0:160, 0:160]
            cropp2 = noisy_slice[:, 160:320, 0:160]
            cropp3 = noisy_slice[:, 0:160, 160:320]
            cropp4 = noisy_slice[:, 160:320, 160:320]
            slice = torch.unsqueeze(slice, 0)
            slice1 = slice[:, 0:160, 0:160]
            slice2 = slice[:, 160:320, 0:160]
            slice3 = slice[:, 0:160, 160:320]
            slice4 = slice[:, 160:320, 160:320]
            return cropp1.float(), cropp2.float(), cropp3.float(), cropp4.float(), slice1.float(), slice2.float(), \
                   slice3.float(), slice4.float(), self.target_files[index]

        if len(self.noise_dirs) == 2:
            path_noise1 = os.path.join(self.noise_dirs[0], self.noise1_files[index])
            path_noise2 = os.path.join(self.noise_dirs[1], self.noise2_files[index])
            noisy_slice1 = torch.load(path_noise1)
            noisy_slice1 = torch.unsqueeze(noisy_slice1, 0)
            noisy_slice2 = torch.load(path_noise2)
            noisy_slice2 = torch.unsqueeze(noisy_slice2, 0)
            cropp1_1 = noisy_slice1[:, 0:160, 0:160]
            cropp2_1 = noisy_slice1[:, 160:320, 0:160]
            cropp3_1 = noisy_slice1[:, 0:160, 160:320]
            cropp4_1 = noisy_slice1[:, 160:320, 160:320]
            cropp1_2 = noisy_slice2[:, 0:160, 0:160]
            cropp2_2 = noisy_slice2[:, 160:320, 0:160]
            cropp3_2 = noisy_slice2[:, 0:160, 160:320]
            cropp4_2 = noisy_slice2[:, 160:320, 160:320]
            slice = torch.unsqueeze(slice, 0)
            slice1 = slice[:, 0:160, 0:160]
            slice2 = slice[:, 160:320, 0:160]
            slice3 = slice[:, 0:160, 160:320]
            slice4 = slice[:, 160:320, 160:320]
            return cropp1_1.float(), cropp2_1.float(), cropp3_1.float(), cropp4_1.float(), cropp1_2.float(), \
                   cropp2_2.float(), cropp3_2.float(), cropp4_2.float(), slice1.float(), slice2.float(), \
                   slice3.float(), slice4.float(), self.target_files[index]


