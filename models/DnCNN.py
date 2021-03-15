import torch
import torch.nn.functional as F
import utils
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, depth=7, n_channels=64, image_channels=1, kernel_size=3, padding=1, architecture="residual", spectral_norm ="Chen"):
        super(DnCNN, self).__init__()
        layers = []
        self.architecture = architecture

        if spectral_norm == "Chen":
            layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth - 2):
                layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
                layers.append(nn.ReLU(inplace=True))
            layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding)))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "Normal":
            layers.append(utils.Spectral_Normalize.spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth - 2):
                layers.append(utils.Spectral_Normalize.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
                layers.append(nn.ReLU(inplace=True))
            layers.append(utils.Spectral_Normalize.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding)))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "None":
            layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth - 2):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "Perseval":
            layers.append(PersevalConv(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth - 2):
                layers.append(PersevalConv(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
                layers.append(nn.ReLU(inplace=True))
            layers.append(PersevalConv(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding))
            self.dncnn = nn.Sequential(*layers)
        else:
            raise ValueError("The normalization specified is not provided choose : None, Normal, Chen or Perseval")

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        if self.architecture == "residual":
            return y-out
        elif self.architecture == "direct":
            return out
        else:
            raise ValueError("The architecture specified is not provided choose : residual or direct")

    def perseval_normalization(self, beta):
        # access the weight kernels
        for name, param in self.dncnn.named_parameters():
            # check convolutional layer
            if "weight" in name:
                W = param.data
                W_tmp = torch.reshape(W, (W.shape[0], (W.shape[2]*W.shape[3])*W.shape[1]))
                W_tmp_T = torch.transpose(W_tmp, 0, 1)
                W_new = (1+beta)*W_tmp - beta*torch.matmul(torch.matmul(W_tmp, W_tmp_T), W_tmp)
                param.data = torch.reshape(W_new, (W.shape[0], W.shape[1], W.shape[2], W.shape[3]))


class PersevalConv(nn.Module):
    """
     A basic 2d convolution with output scaled for perseval networks
    """

    def __init__(self, in_channels, out_channels, kernel_size,padding):
        super(PersevalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.scale = 1/kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = x*self.scale
        return x
