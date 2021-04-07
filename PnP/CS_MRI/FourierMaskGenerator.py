import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import sigpy.mri
import sigpy.plot
from numpy.lib.stride_tricks import as_strided
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mask generation')
    parser.add_argument('--height', type=int, default=160, help='number of pixel in y')
    parser.add_argument('--width', type=int, default=160, help='number of pixel in y')
    parser.add_argument('--proportion', default=0.3, type=float, help='proportion of fourier coefficient kept')
    parser.add_argument('--radius', default=15, type=float, help='radius of lowest frequency kept 100 percent')
    args = parser.parse_args()
    return args


def FourierMaskRandom(width, height,proportion, R):
    s = np.zeros((width, height))
    center = [s.shape[0] // 2, s.shape[1] // 2]
    radius = np.sqrt(center[0] ** 2 + center[1] ** 2)
    print(center)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if (np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)) / radius < R:
                s[i, j] = 1
            else:
                s[i, j] = (1 / (1 - (np.sqrt(((i - center[0]) ** 2) + ((j - center[1]) ** 2)))) ** 2)
    mask = np.random.binomial(1, s)
    sum = sum = np.sum(mask)
    p = sum / (width * height)
    while p <proportion:
        draw = np.random.binomial(1, s)
        mask = mask+draw
        mask = np.where(mask > 0, 1, 0)
        sum = sum = np.sum(mask)
        p = sum / (width * height)
    return mask


if __name__ == '__main__':

    mat = sio.loadmat('Q_Random30.mat')
    mask = mat.get('Q1').astype(np.float64)
    mask = np.fft.fftshift(mask)
    plt.imshow(mask, cmap="gray")
    plt.show()
    mask = FourierMaskRandom(160, 160, 0.3, 0.05)
    plt.imshow(mask, cmap="gray")
    plt.show()
    sum = np.sum(mask)
    proportion = sum / (160 * 160)
    print(proportion)
    mask = np.fft.fftshift(mask)
    plt.imshow(mask, cmap="gray")
    plt.show()
    tensor_mask = torch.from_numpy(mask)
    torch.save(tensor_mask, f'Q_Random30.pt')
