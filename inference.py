import argparse
import numpy as np
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 as cv
import models
import dataloader
from utils.metrics import AverageMeter
from utils.ComputeSV import SingularValues
import math


def main():
    # get the argument from parser
    args = parse_arguments()

    # CONFIG -> assert if config is here
    assert args.config
    config = json.load(open(args.config))

    # DATA
    if config["dataset"] == "fastMRI":
        testdataset = dataloader.KneeMRI(config["test_loader"]["target_dir"], config["test_loader"]["noise_dirs"])
    elif config["dataset"] == "BSD500":
        testdataset = dataloader.BSD500(config["test_loader"]["target_dir"], config["sigma"])

    testloader = DataLoader(testdataset, batch_size=config["test_loader"]["batch_size"],
                           shuffle=config["test_loader"]["shuffle"], num_workers=config["test_loader"]["num_workers"])

    # MODEL
    model = models.DnCNN(config, depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                         image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                         padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                         spectral_norm=config["model"]["spectral_norm"],
                         shared_activation=config["model"]["shared_activation"],
                         shared_channels=config["model"]["shared_channels"], device=args.device)
    device = args.device
    checkpoint = torch.load(args.model, device)
    if config["dataset"] == "fastMRI":
        criterion = torch.nn.MSELoss(reduction="sum")
    elif config["dataset"] == "BSD500":
        criterion = torch.nn.MSELoss(size_average=False)

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

    check_directory(args.experiment)
    # LOOP OVER THE DATA
    tbar = tqdm(testloader, ncols=100)

    total_loss_val = AverageMeter()
    Signal = []
    Noise = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tbar):
            if config["dataset"] == "fastMRI":
                cropp1, cropp2, cropp3, cropp4, target1, target2, target3, target4, image_id = data
                cropp = torch.cat([cropp1, cropp2, cropp3, cropp4], dim=0)
                target = torch.cat([target1, target2, target3, target4], dim=0)
            elif config["dataset"] == "BSD500":
                cropp, target = data
            if args.device != 'cpu':
                cropp, target = cropp.to(non_blocking=True), target.cuda(non_blocking=True)
            batch_size = cropp.shape[0]
            output = model(cropp)

            # LOSS
            if config["dataset"] == "fastMRI":
                loss = criterion(output, target) / batch_size
            elif config["dataset"] == "BSD500":
                loss = criterion(output, target)/(output.size()[0]*2)
            total_loss_val.update(loss.cpu())

            # PRINT INFO
            tbar.set_description('EVAL | MSELoss: {:.5f} |'.format(total_loss_val.average))

            # save the images
            output = output.numpy()
            target = target.numpy()
            cropp = cropp.numpy()
            output = np.squeeze(output, axis=1)
            target = np.squeeze(target, axis=1)
            cropp = np.squeeze(cropp, axis=1)
            for idx in range(batch_size):
                signal = np.linalg.norm(target[idx].flatten())
                noise = np.linalg.norm(output[idx].flatten() - target[idx].flatten())
                Signal.append(signal)
                Noise.append(noise)
            output = batch_scale(output)
            target = batch_scale(target)
            cropp = batch_scale(cropp)
            for i in range(output.shape[0]):
                j = math.floor(i / 4)
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i%4}_prediction.png', output[i])
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i%4}_target.png', target[i])
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i % 4}_input.png', cropp[i])

        Signal = np.asarray(Signal)
        Noise = np.asarray(Noise)
        SNR = 20*np.log10(Signal.mean()/Noise.mean())
        print("The mean SNR over the test set is : {}".format(SNR))
        # save the metric
        metrics = {"MSE_Loss": np.round(total_loss_val.average, 8), "SNR": SNR}

        with open(f'{args.experiment}/test_result/test.txt', 'w') as f:
            for k, v in list(metrics.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))

        lipschitz_cte = SingularValues(model)
        lipschitz_cte.compute_layer_sv()
        merged_list = list(map(lambda x, y: (x, y), lipschitz_cte.names, lipschitz_cte.sigmas))
        metrics = {"Layer": merged_list}
        activation = config["model"]["activation_type"]
        QP = config["model"]["QP"]
        if (activation != "relu") and (activation != "leaky_relu") and (activation != "prelu"):
            C, slope = model.lipschtiz_exact()
            spline = {}
            splineAll = {}
            for i in range(len(C)):
                spline[f"activation_{i}"] = C[i]
                splineAll[f"activation_{i}"] = slope[i]

            with open(f'{args.experiment}/test_result/{QP}_ActivationSlopes.txt', 'w') as f:
                for k, v in list(spline.items()):
                    f.write("%s\n" % (k + ':' + f'{v}'))
                    torch.save(v, f"{args.experiment}/test_result/{QP}_{k}.pt")

            with open(f'{args.experiment}/test_result/{QP}_ActivationSlopesAll.txt', 'w') as f:
                for k, v in list(splineAll.items()):
                    f.write("%s\n" % (k + ':' + f'{v}'))
                    torch.save(v, f"{args.experiment}/test_result/{QP}_{k}_All.pt")

        with open(f'{args.experiment}/test_result/sv.txt', 'w') as f:
            for k, v in list(metrics.items()):
                for t in v:
                    f.write("%s\n" % (k + ':' + f'{t[0]}' + ':' + f'{t[1]}'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--model', default=None, type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--device', default="cpu", type=str,
                        help='device location')
    parser.add_argument('--experiment', default=None, type=str,
                        help='path to the folder experiment')
    args = parser.parse_args()
    return args


def check_directory(experiment):
    path = os.path.join(experiment, "test_result")
    if not os.path.exists(path):
        os.makedirs(path)


def batch_scale(image):
    for i, img in enumerate(image):
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        image[i, :, :] = 255 * img
    return image


if __name__ == '__main__':
    main()

