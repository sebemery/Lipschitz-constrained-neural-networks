import argparse
import numpy as np
import json
import os
import torch
import models
from utils.ComputeSV import SingularValues
import math
import time


def main():
    # get the argument from parser
    args = parse_arguments()

    # CONFIG -> assert if config is here
    assert args.config
    config = json.load(open(args.config))

    # MODEL
    model = models.DnCNN(config, depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                         image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                         padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                         spectral_norm=config["model"]["spectral_norm"], device=args.device)

    if args.activation:
        config["model"]["activation_type"] = args.activation
    if args.QP:
        config["model"]["QP"] = args.QP

    model_QP = models.DnCNN(config, depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                            image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                            padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                            spectral_norm=config["model"]["spectral_norm"], device=args.device)
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

    for name, param in model.dncnn.named_parameters():
        for name_QP, param_QP in model_QP.dncnn.named_parameters():
            if name == name_QP:
                param_QP.data = param.data

    model.float()
    model.eval()
    if args.device != 'cpu':
        model.to(device)

    model_QP.float()
    model_QP.eval()
    if args.device != 'cpu':
        model_QP.to(device)

    check_directory(args.experiment)

    if config["model"]["activation_type"] != "deepBspline":
        start_time = time.time()
        for i, module in enumerate(model_QP.modules_deepspline()):
            module.do_lipschitz_projection()
        t = time.time() - start_time
        print("--- %s seconds ---" % t)
        with open(f'{args.experiment}/QP_result/{args.QP}_time.txt', 'w') as f:
            f.write("%s\n" % ('Time :' + f'{t}'))

    lipschitz_cte = SingularValues(model_QP)
    lipschitz_cte.compute_layer_sv()
    merged_list = list(map(lambda x, y: (x, y), lipschitz_cte.names, lipschitz_cte.sigmas))
    metrics = {"Layer": merged_list}
    activation = config["model"]["activation_type"]
    if (activation != "relu") or (activation != "leaky_relu") or (activation != "prelu"):
        C = model_QP.lipschtiz_exact()
        spline = {}
        for i in range(len(C)):
            spline[f"activation_{i}"] = C[i]

        with open(f'{args.experiment}/QP_result/{args.QP}_ActivationSlopes.txt', 'w') as f:
            for k, v in list(spline.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))
                torch.save(v, f"{args.experiment}/QP_result/{args.QP}_{k}.pt")

    with open(f'{args.experiment}/QP_result/{args.QP}_sv.txt', 'w') as f:
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
    parser.add_argument('--activation', default="deepBspline_lipschitz_orthoprojection", type=str,
                        help='type of activation used')
    parser.add_argument('--QP', default="cvxpy", type=str,
                        help='quadratic proggraming library')
    args = parser.parse_args()
    return args


def check_directory(experiment):
    path = os.path.join(experiment, "QP_result")
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
