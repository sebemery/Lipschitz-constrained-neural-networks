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
import PnP
import models


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
    parser.add_argument('--algo', default="admm", type=str, help='admm/fbs')
    parser.add_argument('--mu_upper', default=2.0, type=float, help='highest value of mu')
    parser.add_argument('--mu_lower', default=0.1, type=float, help='lowest value of mu')
    parser.add_argument('--mu_step', default=20, type=int, help='step')
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise level for the denoising model")
    parser.add_argument("--alpha", type=float, default=2.0, help="Step size in Plug-and Play")
    parser.add_argument("--maxitr", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    args = parser.parse_args()
    return args


def check_directory(experiment, algo):
    if not os.path.exists("Experiments"):
        os.makedirs("Experiments")
    path = os.path.join("Experiments", algo)
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
    path = check_directory(args.experiment, args.algo)

    with torch.no_grad():

        # ---- load the ground truth ----
        if args.jpg is True:
            im_orig = cv2.imread(f'{args.img}', 0) / 255.0
            cv2.imwrite(f'{path}/GroundTruth.png', 255 * im_orig)
        else:
            im_orig = torch.load(f'{args.img}').numpy()
            cv2.imwrite(f'{path}/GroundTruth.png', 255*im_orig)

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
        mu_snr = []
        mu_vec = np.linspace(args.mu_lower, args.mu_upper, args.mu_step)
        for mu in mu_vec:
            # ---- plug and play !!! -----
            if args.algo == "admm":
                if args.verbose:
                    x_out, inc, x_init, zero_fill_snr, snr = PnP.pnp_admm_csmri.pnp_admm_csmri_(model, im_orig, mask, noises, mu, device, **opts)
                else:
                    x_out, inc, x_init, zero_fill_snr = PnP.pnp_admm_csmri.pnp_admm_csmri_(model, im_orig, mask, noises, mu, device, **opts)
            elif args.algo == "fbs":
                if args.verbose:
                    x_out, inc, x_init, zero_fill_snr, snr = PnP.pnp_fbs_csmri.pnp_fbs_csmri_(model, im_orig, mask, noises, mu, device, **opts)
                else:
                    x_out, inc, x_init, zero_fill_snr = PnP.pnp_fbs_csmri.pnp_fbs_csmri_(model, im_orig, mask, noises, mu, device, **opts)

            # directory
            path_mu = os.path.join(path, f"{mu}")
            if not os.path.exists(path_mu):
                os.makedirs(path_mu)
            # ---- print result -----
            out_snr = psnr(x_out, im_orig)
            mu_snr.append(out_snr)
            print('Plug-and-Play PNSR: ', out_snr)
            metrics = {"PSNR": np.round(snr, 8), "Zero fill PSNR": np.round(zero_fill_snr, 8), }

            with open(f'{path_mu}/snr.txt', 'w') as f:
                for k, v in list(metrics.items()):
                    f.write("%s\n" % (k + ':' + f'{v}'))

            # ---- save result -----
            fig, ax1 = plt.subplots()
            ax1.plot(inc, 'b-', linewidth=1)
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('Increment', color='b')
            ax1.set_title("Increment curve")
            fig.savefig(f'{path_mu}/inc.png')
            plt.show()

            if args.verbose:
                fig, ax1 = plt.subplots()
                ax1.plot(snr, 'b-', linewidth=1)
                ax1.set_xlabel('iteration')
                ax1.set_ylabel('PSNR', color='b')
                ax1.set_title("PSNR curve")
                fig.savefig(f'{path_mu}/snr.png')
                plt.show()

            torch.save(torch.from_numpy(x_out), f'{path_mu}/{args.algo}.pt')
            torch.save(torch.from_numpy(x_init), f'{path_mu}/ifft.pt')
            x_out = scale(x_out)
            x_init = scale(x_init)
            cv2.imwrite(f'{path_mu}/{args.algo}.png', x_out)
            cv2.imwrite(f'{path_mu}/ifft.png', x_init)

        fig, ax1 = plt.subplots()
        ax1.plot(mu_vec, np.asarray(mu_snr), 'b-', linewidth=1)
        ax1.set_xlabel('mu')
        ax1.set_ylabel('SNR', color='b')
        ax1.set_title("SNR for different scaling mu")
        fig.savefig(f'{path}/mu.png')
        plt.show()
        idx_max = np.argmax(np.asarray(mu_snr))
        mu_max = mu_vec[idx_max]
        param = {"mu": mu_max}

        with open(f'{path}/mu.txt', 'w') as f:
            for k, v in list(param.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))
