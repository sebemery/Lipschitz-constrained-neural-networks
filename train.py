import argparse
import json
import torch
from torch.utils.data import DataLoader
import dataloader
import models
from utils import Logger
from trainer import Trainer
import os


def main(config, resume, device):
    train_data = dataloader.KneeMRI(config["train_loader"]["target_dir"], config["train_loader"]["noise_dirs"])
    val_data = dataloader.KneeMRI(config["val_loader"]["target_dir"], config["val_loader"]["noise_dirs"])
    trainloader = DataLoader(train_data, batch_size=config["train_loader"]["batch_size"],
                             shuffle=config["train_loader"]["shuffle"], num_workers=config["train_loader"]["num_workers"])
    valloader = DataLoader(val_data, batch_size=config["val_loader"]["batch_size"],
                           shuffle=config["val_loader"]["shuffle"], num_workers=config["val_loader"]["num_workers"])
    experim_dir = os.path.join(config["trainer"]['save_dir'], config['experim_name'])
    if not os.path.exists(experim_dir):
        os.makedirs(experim_dir)
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = models.DnCNN(config, depth=config["model"]["depth"], n_channels=config["model"]["n_channels"],
                             image_channels=config["model"]["image_channels"], kernel_size=config["model"]["kernel_size"],
                             padding=config["model"]["padding"], architecture=config["model"]["architecture"],
                             spectral_norm=config["model"]["spectral_norm"],
                             shared_activation=config["model"]["shared_activation"],
                             shared_channels=config["model"]["shared_channels"], device=args.device)
        train_logger = Logger()
        trainer = Trainer(config, trainloader, valloader, model, train_logger, seed, resume, device)
        trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))
    # for performance on gpu when input size does not vary
    torch.backends.cudnn.benchmark = True
    main(config, args.resume,  args.device)
