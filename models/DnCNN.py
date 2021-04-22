import torch
import torch.nn.functional as F
import utils
from torch import nn
from models.SplineActivations.basemodel import BaseModel


class DnCNN(BaseModel):
    def __init__(self, config, depth=7, n_channels=64, image_channels=1, kernel_size=3, padding=1, architecture="residual", spectral_norm ="Chen", device="cpu"):
        super().__init__(config,device)
        layers = []
        self.architecture = architecture
        self.activation = config["model"]["activation_type"]

        if spectral_norm == "Chen":
            layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == "prelu":
                layers.append(nn.PReLU(num_parameters=n_channels))
            else:
                layers.append(self.init_activation(('conv', n_channels)))
            for _ in range(depth - 2):
                layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
                if self.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(inplace=True))
                elif self.activation == "prelu":
                    layers.append(nn.PReLU(num_parameters=n_channels))
                else:
                    layers.append(self.init_activation(('conv', n_channels)))
            layers.append(utils.Spectral_Normalize_chen.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding)))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "Normal":
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
            if self.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == "prelu":
                layers.append(nn.PReLU(num_parameters=n_channels))
            else:
                layers.append(self.init_activation(('conv', n_channels)))
            for _ in range(depth - 2):
                layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding)))
                if self.activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif self.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(inplace=True))
                elif self.activation == "prelu":
                    layers.append(nn.PReLU(num_parameters=n_channels))
                else:
                    layers.append(self.init_activation(('conv', n_channels)))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding)))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "None":
            layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            if self.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == "prelu":
                layers.append(nn.PReLU(num_parameters=n_channels))
            else:
                layers.append(self.init_activation(('conv', n_channels)))
            for _ in range(depth - 2):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
                if self.activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif self.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(inplace=True))
                elif self.activation == "prelu":
                    layers.append(nn.PReLU(num_parameters=n_channels))
                else:
                    layers.append(self.init_activation(('conv', n_channels)))
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding))
            self.dncnn = nn.Sequential(*layers)
        elif spectral_norm == "Perseval":
            layers.append(PersevalConv(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            if self.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == "prelu":
                layers.append(nn.PReLU(num_parameters=n_channels))
            else:
                layers.append(self.init_activation(('conv', n_channels)))
            for _ in range(depth - 2):
                layers.append(PersevalConv(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
                if self.activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif self.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(inplace=True))
                elif self.activation == "prelu":
                    layers.append(nn.PReLU(num_parameters=n_channels))
                else:
                    layers.append(self.init_activation(('conv', n_channels)))
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
        elif self.architecture == "halfaveraged":
            return (y+out)/2.0
        else:
            raise ValueError("The architecture specified is not provided choose : residual, direct or halfaveraged")

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

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(PersevalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.scale = 1/kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = x*self.scale
        return x
