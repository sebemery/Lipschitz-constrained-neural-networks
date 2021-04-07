import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import scipy.stats as stats
import scipy.ndimage.morphology
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
    """
    Create a random sampling pattern where each point is sampled from :
                            1 if r < R
                        1/(1-r)**2 if r > R
                            with  0<R<1
    Args:
        width (int):           Size of output mask in x direction (width)
        height (int):          Size of output mask in y direction (height)
        proportion (int):      proportion of the sample picked compare to the total number of pixel
        R (float):             Normed Radius (0-1) of ball around origin where all samples are included.
    Returns:
        np.ndarray: A numpy array (mask) depicting sampling pattern.
    """

    s = np.zeros((width, height))
    center = [s.shape[0] // 2, s.shape[1] // 2]
    radius = np.sqrt(center[0] ** 2 + center[1] ** 2)
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


"""
    The following sampling pattern function are borrowed from 
    https://github.com/mathialo/master_code
"""


def gaussian_sampling(len_x, len_y, num_samples, spread_factor=5, origin_ball=1):
    """
    Create a gaussian sampling pattern where each point is sampled from a
    bivariate, concatenated normal distribution.
    Args:
        len_x (int):            Size of output mask in x direction (width)
        len_y (int):            Size of output mask in y direction (height)
        num_samples (int):      Number of samples to pick
        spread_factor (float):  Concentration of samples (ie, the SD of the
                                probability distributions are len/spread_factor)
        origin_ball (int):      Radius of ball around origin where all samples
                                are included.
    Returns:
        np.ndarray: A boolean numpy array (mask) depicting sampling pattern.
    """
    # Create two truncated normal distributions for x and y dir
    lower = 0
    upper_x = len_x
    mu_x = len_x // 2
    sigma_x = len_x // spread_factor
    randgen_x = stats.truncnorm(
        (lower - mu_x) / sigma_x,
        (upper_x - mu_x) / sigma_x,
        loc=mu_x,
        scale=sigma_x
    )

    upper_y = len_y
    mu_y = len_y // 2
    sigma_y = len_y // spread_factor
    randgen_y = stats.truncnorm(
        (lower - mu_y) / sigma_y,
        (upper_y - mu_y) / sigma_y,
        loc=mu_y,
        scale=sigma_y
    )

    # Create mask
    mask = np.zeros([len_y, len_x], dtype=np.bool)

    # Add origin ball
    if origin_ball > 0:
        y_grid, x_grid = np.ogrid[:len_y, :len_x]
        dist_from_center = np.sqrt((y_grid - mu_y) ** 2 + (x_grid - mu_x) ** 2)

        mask = dist_from_center <= origin_ball

    # Subtract origin ball from number of samples
    num_samples -= np.sum(mask)

    # Sample points from distribution
    xs = randgen_x.rvs(num_samples).astype(np.uint32)
    ys = randgen_y.rvs(num_samples).astype(np.uint32)

    for i in range(num_samples):
        x, y = xs[i], ys[i]

        # Ensure unique samples
        while mask[y, x]:
            x = randgen_x.rvs(1).astype(np.uint32)
            y = randgen_y.rvs(1).astype(np.uint32)

        xs[i], ys[i] = x, y

        mask[y, x] = True

    return mask


def level_sampling(len_x, len_y, sampling_rates):
    """
    Create a level-based sampling where each level has its own sampling rate.
    The level sizes are log-based (ie, for a length of 16 and 3 levels, the
    level sizes will be 8, 4 and 2).
    Args:
        len_x (int):                Size of output mask in x direction (width)
        len_y (int):                Size of output mask in y direction (height)
        sampling_rates (iterable):  Iterable of floats between 0 and 1. The
                                    sampling rates for each level. The length of
                                    this list specifies the number of levels.
    Returns:
        np.ndarray: A boolean numpy array (mask) depicting sampling pattern.
    """
    levels = len(sampling_rates)

    mask = np.zeros([len_y, len_x], dtype=np.bool)


    # Local function for making each level (inplace)
    def level_gen(local_x, local_y, sampling_rate, placein):
        # Clean previous entries
        placein[:, :] = np.zeros_like(placein)

        # Add samples
        for y in range(local_y):
            picks = np.arange(local_x)
            np.random.shuffle(picks)
            for x in picks[0:int(local_x * sampling_rate)]:
                placein[y, x] = 1


    # Add all levels to sampling mask
    for level in range(levels):
        local_x = len_x // 2**level
        local_y = len_y // 2**level

        rest_x = len_x - local_x
        rest_y = len_y - local_y

        level_gen(
            local_x,
            local_y,
            sampling_rates[levels - level - 1],
            mask[rest_y // 2:rest_y // 2 + local_y, rest_x // 2:rest_x // 2 + local_x]
        )

    return mask


def uniform_sampling(len_x, len_y, sampling_rate):
    """
    Create a uniform sampling pattern.
    Args:
        len_x (int):                Size of output mask in x direction (width)
        len_y (int):                Size of output mask in y direction (height)
        sampling_rate (float):      Sampling rate
    Returns:
        np.ndarray: A boolean numpy array (mask) depicting sampling pattern.
    """
    mask = np.zeros([len_y, len_x], dtype=np.bool)

    for y in range(len_y):
        picks = np.arange(len_x)
        np.random.shuffle(picks)
        for x in picks[0:int(len_x * sampling_rate)]:
            mask[y, x] = 1

    return mask


def radial_sampling(len_x, len_y, line_num, dilations=1, close=True):
    """
    Creates a line sampling pattern where a specified number of lines is evenly
    distributed on angles between 0 and 2pi, with lines stroking from the center
    to the edges of the frame.
    Args:
        len_x (int):            Size of output mask in x direction (width)
        len_y (int):            Size of output mask in y direction (height)
        line_num (int):         Number of lines to add
        dilations (int):        Number of morphological dilations to perform
                                after the lines have been sampled. This affects
                                the line width.
        close (bool):           Whether to perform morphological closing or not
                                on the sampling pattern before applying dilations
    Returns:
        np.ndarray: A boolean numpy array (mask) depicting sampling pattern.
    """
    mask = np.zeros([len_y, len_x], dtype=np.bool)

    center = len_y // 2, len_x // 2

    thetas = np.arange(line_num) / line_num * 2 * np.pi

    # Sample a greater amount of points than what is strictly needed in order to
    # capture all the discretized points.
    point_num = int(np.floor(np.sqrt(len_x ** 2 + len_y ** 2)))

    # Points along radius
    points = np.linspace(0, 1, point_num)

    for line in range(line_num):
        theta = thetas[line]

        # Find length of the line from the center to the edge of the frame along
        # given angle.
        if np.pi / 4 < theta < 3 * np.pi / 4 or 5 * np.pi / 4 < theta < 7 * np.pi / 4:
            r = np.abs(len_y // 2 / np.sin(theta))
        else:
            r = np.abs(len_x // 2 / np.cos(theta))

        # Sample line
        for r_ in r * points:
            # Sample points along radius
            x = int(np.cos(theta) * r_) + center[1]
            y = int(np.sin(theta) * r_) + center[0]

            # Truncate x and y to avoid out-of-bounds errors at the edges
            if y >= len_y: y = len_y - 1
            if x >= len_x: x = len_x - 1
            if x < 0: x = 0
            if y < 0: y = 0

            mask[y, x] = 1

    # Perform morphological actions to better pattern
    if close:
        mask = scipy.ndimage.morphology.binary_closing(mask)

    for i in range(dilations):
        mask = scipy.ndimage.morphology.binary_dilation(mask)

    return mask


if __name__ == '__main__':

    mat = sio.loadmat('Q_Random30.mat')
    mask = mat.get('Q1').astype(np.float64)
    mask = np.fft.fftshift(mask)
    plt.imshow(mask, cmap="gray")
    plt.show()
    mask = FourierMaskRandom(160, 160, 0.3, 0.1)
    plt.imshow(mask, cmap="gray")
    plt.show()
    sum = np.sum(mask)
    proportion = sum / (160 * 160)
    print(proportion)
    mask = np.fft.fftshift(mask)
    plt.imshow(mask, cmap="gray")
    plt.show()
    tensor_mask = torch.from_numpy(mask)
    mask2 = gaussian_sampling(160, 160, int(0.25*160*160), spread_factor=5, origin_ball=10)
    plt.imshow(mask2, cmap="gray")
    plt.show()
    sum = np.sum(mask2)
    proportion = sum / (160 * 160)
    print(proportion)
    mask2 = np.fft.fftshift(mask2)
    plt.imshow(mask2, cmap="gray")
    plt.show()
    tensor_mask2 = torch.from_numpy(mask2)
    torch.save(tensor_mask2, f"Gaussian25.pt")
    mask3 = radial_sampling(160, 160, 30, dilations=0, close=True)
    plt.imshow(mask3, cmap="gray")
    plt.show()
    sum = np.sum(mask3)
    proportion = sum / (160 * 160)
    print(proportion)
    mask3 = np.fft.fftshift(mask3)
    plt.imshow(mask3, cmap="gray")
    plt.show()
    tensor_mask3 = torch.from_numpy(mask3)
    torch.save(tensor_mask3, f"Radial12.5.pt")
