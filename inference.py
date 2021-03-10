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
import math


def main():
    # get the argument from parser
    args = parse_arguments()

    # CONFIG -> assert if config is here
    assert args.config
    config = json.load(open(args.config))

    # DATA
    testdataset = dataloader.KneeMRI(config["test_loader"]["target_dir"], config["test_loader"]["noise_dirs"])
    testloader = DataLoader(testdataset, batch_size=config["test_loader"]["batch_size"],
                           shuffle=config["test_loader"]["shuffle"], num_workers=config["test_loader"]["num_workers"])

    # MODEL
    model = models.DnCNN(depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                         image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                         padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                         spectral_norm=config["model"]["spectral_norm"])
    device = args.device
    checkpoint = torch.load(args.model, device)
    criterion = torch.nn.MSELoss(reduction="sum")

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

    with torch.no_grad():
        for batch_idx, data in enumerate(tbar):
            cropp1, cropp2, cropp3, cropp4, target1, target2, target3, target4, image_id = data
            cropp = torch.cat([cropp1, cropp2, cropp3, cropp4], dim=0)
            target = torch.cat([target1, target2, target3, target4], dim=0)
            if args.device != 'cpu':
                cropp, target = cropp.to(non_blocking=True), target.cuda(non_blocking=True)
            batch_size = cropp.shape[0]
            output = model(cropp)

            # LOSS
            loss = criterion(output, target)/batch_size
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
            output = batch_scale(output)
            target = batch_scale(target)
            cropp = batch_scale(cropp)
            for i in range(output.shape[0]):
                j = math.floor(i / 4)
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i%4}_prediction.png', output[i])
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i%4}_target.png', target[i])
                cv.imwrite(f'{args.experiment}/test_result/{image_id[j][:-4]}_{i % 4}_input.png', cropp[i])

        # save the metric
        metrics = {"MSE_Loss": np.round(total_loss_val.average, 8)}

        with open(f'{args.experiment}/test_result/test.txt', 'w') as f:
            for k, v in list(metrics.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))


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

