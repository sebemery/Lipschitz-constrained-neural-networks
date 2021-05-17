"""
    Plug and Play FBS for Compressive Sensing MRI
    Authors: Jialin Liu (danny19921123@gmail.com)
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


def pnp_fbs_csmri(model, im_orig, mask, noises, device, **opts):

    alpha = opts.get('alpha', 0.4)
    maxitr = opts.get('maxitr', 100)
    verbose = opts.get('verbose',1)
    sigma = opts.get('sigma', 5)
    device = device
    inc = []
    if verbose:
        snr = []

    """ Initialization. """

    m, n = im_orig.shape
    index = np.nonzero(mask)
    if noises is None:
        noise = np.random.normal(loc=0, scale=sigma / 2.0, size=(m, n, 2)).view(np.complex128)
        noise = np.squeeze(noise)
    else:
        noise = noises

    y = np.fft.fft2(im_orig) * mask + noise # observed value
    x_init = np.fft.ifft2(y) # zero fill

    zero_fill_snr = psnr(x_init, im_orig)
    print('zero-fill PSNR:', zero_fill_snr)
    if verbose:
        snr.append(zero_fill_snr)

    x = np.copy(x_init)

    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)

        """ Update variables. """

        res = np.fft.fft2(x) * mask
        index = np.nonzero(mask)
        res[index] = res[index] - y[index]
        x = x - alpha * np.fft.ifft2(res)
        # x = np.real( x )
        x = np.absolute(x)

        """ Denoising step. """

        xtilde = np.copy(x)
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde = (xtilde - mintmp) / (maxtmp - mintmp)
        
        # the reason for the following scaling:
        # our denoisers are trained with "normalized images + noise"
        # so the scale should be 1 + O(sigma)
        if noises is None:
            scale_range = 1.0 + sigma / 2.0
        else:
            scale_range = 1.0 + sigma / 255 / 2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift

        # pytorch denoising model
        xtilde_torch = np.reshape(xtilde, (1, 1, m, n))
        xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).to(device, non_blocking=True)
        x = model(xtilde_torch).cpu().numpy()
        x = np.reshape(x, (m, n))

        # scale and shift the denoised image back
        x = (x - scale_shift) / scale_range
        x = x * (maxtmp - mintmp) + mintmp

        """ Monitoring. """
        if verbose:
            snr_tmp = psnr(x, im_orig)
            print("i: {}, \t psnr: {}".format(i + 1, snr_tmp))
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
    parser.add_argument('--img', default='CS_MRI/file1002252_2_bottomright.pt', type=str, help='Path to the original image')
    parser.add_argument('--mask', default='CS_MRI/Q_Random30.pt', type=str, help='Path to the k-space mask file')
    parser.add_argument('--jpg', default=True, type=bool, help='file type either jpg or pt')
    parser.add_argument('--noise', default='CS_MRI/noises.mat', type=str, help='Path to the k-space noise file')
    parser.add_argument('--device', default="cpu", type=str, help='device location')
    parser.add_argument('--experiment', default=None, type=str, help='name of the experiment')
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise level for the denoising model")
    parser.add_argument("--alpha", type=float, default=2.0, help="Step size in Plug-and Play")
    parser.add_argument("--maxitr", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    args = parser.parse_args()
    return args


def check_directory(experiment):
    if not os.path.exists("Experiments"):
        os.makedirs("Experiments")
    path = os.path.join("Experiments", "fbs")
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
    model = models.DnCNN(config, depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                         image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                         padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                         spectral_norm=config["model"]["spectral_norm"],
                         shared_activation=config["model"]["shared_activation"],
                         shared_channels=config["model"]["shared_channels"], device=args.device)
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
        if args.jpg is True:
            im_orig = cv2.imread(f'{args.img}', 0) / 255.0
            cv2.imwrite(f'{path}/GroundTruth.png', 255 * im_orig)
        else:
            im_orig = torch.load(f'{args.img}').numpy()
            cv2.imwrite(f'{path}/GroundTruth.png', 255 * im_orig)

        # ---- load mask matrix ----
        if args.jpg is True:
            mat = sio.loadmat(f'{args.mask}')
            mask = mat.get('Q1').astype(np.float64)
        else:
            mask = torch.load(f'{args.mask}').numpy()

        # ---- load noises -----
        if args.jpg is True:
            noises = sio.loadmat(f'{args.noise}')
            noises = noises.get('noises').astype(np.complex128) * 3.0
        else:
            noises = None

        # ---- set options -----
        opts = dict(sigma=args.sigma, alpha=args.alpha, maxitr=args.maxitr, verbose=args.verbose)

        # ---- plug and play !!! -----
        if args.verbose:
            x_out, inc, x_init, zero_fill_snr, snr = pnp_fbs_csmri(model, im_orig, mask, noises, device, **opts)
        else:
            x_out, inc, x_init, zero_fill_snr = pnp_fbs_csmri(model, im_orig, mask, noises, device, **opts)

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

        torch.save(torch.from_numpy(x_out), f'{path}/fbs.pt')
        torch.save(torch.from_numpy(x_init), f'{path}/ifft.pt')
        x_out = scale(x_out)
        x_init = scale(x_init)
        cv2.imwrite(f'{path}/fbs.png', x_out)
        cv2.imwrite(f'{path}/ifft.png', x_init)



