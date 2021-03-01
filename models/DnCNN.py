import torch
import torch.nn.functional as F
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, depth=7, n_channels=64, image_channels=1, kernel_size=3, padding=1, architecture="residual", spectral_norm =True):
        super(DnCNN, self).__init__()
        layers = []
        self.architecture = architecture

        if spectral_norm:
            layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth-2):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding))
            self.dncnn = nn.Sequential(*layers)
        else:
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size,
                                    padding=padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(depth - 2):
                layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                        padding=padding)))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size,
                                    padding=padding)))
            self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        if self.architecture == "residual":
            return y-out
        elif self.architecture == "direct":
            return out


if __name__ == '__main__':
    x = torch.ones((5, 1, 50, 50))
    print(x.shape)
    model = DnCNN()
    out = model(x)
    print(out.shape)