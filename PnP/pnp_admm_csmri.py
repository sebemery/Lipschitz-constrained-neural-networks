"""
    Plug and Play ADMM for Compressive Sensing MRI
    Adapted from : Jialin Liu (danny19921123@gmail.com)
"""

import os
import numpy as np
import argparse
import json
import torch
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import models


def pnp_admm_csmri(model, im_orig, mask, device, **opts):

    alpha = opts.get('alpha', 2.0)
    maxitr = opts.get('maxitr', 100)
    verbose = opts.get('verbose', 1)
    sigma = opts.get('sigma', 5)
    device = device
    inc = []
    if verbose:
        snr = []

    """ Initialization. """

    m, n = im_orig.shape
    index = np.nonzero(mask)
    noise = np.random.normal(loc=0, scale=sigma/2.0, size=(m, n, 2)).view(np.complex128)
    noise = np.squeeze(noise)

    # observed value
    y = np.fft.fft2(im_orig) * mask + noise
    # zero fill
    x_init = np.fft.ifft2(y)

    zero_fill_snr = psnr(x_init, im_orig)
    print('zero-fill PSNR:', zero_fill_snr)
    if verbose:
        snr.append(zero_fill_snr)

    x = np.absolute(np.copy(x_init))
    v = np.copy(x)
    u = np.zeros((m, n), dtype=np.float64)

    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)
        vold = np.copy(v)
        uold = np.copy(u)
        """ Update variables. """

        vtilde = np.copy(x+u)
        vf = np.fft.fft2(vtilde)
        La2 = 1.0/2.0/alpha
        vf[index] = (La2 * vf[index] + y[index]) / (1.0 + La2)
        v = np.real(np.fft.ifft2(vf))

        """ Denoising step. """

        xtilde = np.copy(2*v - xold - uold)
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde = (xtilde - mintmp) / (maxtmp - mintmp)

        # the reason for the following scaling:
        # our denoisers are trained with "normalized images + noise"
        # so the scale should be 1 + O(sigma)
        scale_range = 1.0 + sigma/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift

        # pytorch denoising model
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).to(device, non_blocking=True)
        x = model(xtilde_torch).cpu().numpy()
        x = np.reshape(x, (m, n))

        # scale and shift the denoised image back
        x = (x - scale_shift) / scale_range
        x = x * (maxtmp - mintmp) + mintmp

        """ Update variables. """
        u = uold + xold - v

        """ Monitoring. """
        if verbose:
            snr_tmp = psnr(x, im_orig)
            print("i: {}, \t psnr: {}".format(i+1, snr_tmp))
            snr.append(snr_tmp)

        inc.append(np.sqrt(np.sum((np.absolute(x - xold)) ** 2)))

    x_init = np.real(x_init)
    if verbose:
        return x, inc, x_init, zero_fill_snr, snr
    else:
        return x, inc, x_init, zero_fill_snr


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json', type=str, help='Path to the config file')
    parser.add_argument('--model', default=None, type=str, help='Path to the trained .pth model')
    parser.add_argument('--device', default="cpu", type=str, help='device location')
    parser.add_argument('--experiment', default=None, type=str, help='name of the experiment')
    parser.add_argument("--sigma", type=int, default=0.05, help="Noise level for the denoising model")
    parser.add_argument("--alpha", type=float, default=2.0, help="Step size in Plug-and Play")
    parser.add_argument("--maxitr", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    args = parser.parse_args()
    return args


def check_directory(experiment):
    if not os.path.exists("Experiments"):
        os.makedirs("Experiments")
    path = os.path.join("Experiments", "admm")
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, experiment)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def scale(img):
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    image = 255 * img
    return image


def psnr(x, im_orig):
    xout = (x - np.min(x)) / (np.max(x) - np.min(x))
    norm1 = np.sum((np.absolute(im_orig)) ** 2)
    norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    psnr = 10 * np.log10(norm1 / norm2)
    return psnr


if __name__ == '__main__':

    # ---- input arguments ----
    args = parse_arguments()
    # CONFIG -> assert if config is here
    assert args.config
    config = json.load(open(args.config))

    # ---- load the model ----
    model = models.DnCNN(depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                         image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                         padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                         spectral_norm=config["model"]["spectral_norm"])
    device = args.device
    checkpoint = torch.load(args.model, device)
    if device == 'cpu':
        for key in list(checkpoint['state_dict'].keys()):
            if 'module.' in key:
                checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.float()
    model.eval()
    if args.device != 'cpu':
        model.to(device)

    # create the output directory and return the path to it
    path = check_directory(args.experiment)

    with torch.no_grad():

        # ---- load the ground truth ----
        im_orig = torch.load('CS_MRI/file1002252_2_bottomright.pt').numpy()
        cv2.imwrite(f'{path}/GroundTruth.png', 255*im_orig)

        # ---- load mask matrix ----
        mask = torch.load('CS_MRI/Q_Random30.pt').numpy()

        # ---- set options -----
        opts = dict(sigma=args.sigma, alpha=args.alpha, maxitr=args.maxitr, verbose=args.verbose)

        # ---- plug and play !!! -----
        if args.verbose:
            x_out, inc, x_init, zero_fill_snr, snr = pnp_admm_csmri(model, im_orig, mask, device, **opts)
        else:
            x_out, inc, x_init, zero_fill_snr = pnp_admm_csmri(model, im_orig, mask, device, **opts)

        # ---- print result -----
        out_snr = psnr(x_out, im_orig)
        print('Plug-and-Play PNSR: ', out_snr)
        metrics = {"PSNR": np.round(snr, 8), "Zero fill PSNR": np.round(zero_fill_snr, 8), }

        with open(f'{path}/snr.txt', 'w') as f:
            for k, v in list(metrics.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))

        # ---- save result -----
        fig, ax1 = plt.subplots()
        ax1.plot(inc, 'b-', linewidth=1)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Increment', color='b')
        ax1.set_title("Increment curve")
        fig.savefig(f'{path}/inc.png')
        plt.show()

        if args.verbose:
            fig, ax1 = plt.subplots()
            ax1.plot(snr, 'b-', linewidth=1)
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('PSNR', color='b')
            ax1.set_title("PSNR curve")
            fig.savefig(f'{path}/snr.png')
            plt.show()

        torch.save(torch.from_numpy(x_out), f'{path}/admm.pt')
        torch.save(torch.from_numpy(x_init), f'{path}/ifft.pt')
        x_out = scale(x_out)
        x_init = scale(x_init)
        cv2.imwrite(f'{path}/admm.png', x_out)
        cv2.imwrite(f'{path}/ifft.png', x_init)


