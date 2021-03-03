import torch
import logging
import os
import datetime
import json
from utils import logger
from utils.htmlwriter import HTML
from torch.utils import tensorboard
from utils.metrics import AverageMeter
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config, trainloader, valloader, model, train_logger, seed, resume, device):
        self.config = config
        self.train_loader = trainloader
        self.val_loader = valloader
        self.model = model
        self.device = device
        if self.device != "cpu":
            self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["optimizer"]["args"]["lr"],
                                          weight_decay=config["optimizer"]["args"]["weight_decay"])
        self.criterion = torch.nn.MSELoss()
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_epoch = 1
        self.epochs = config["trainer"]['epochs']
        self.save_period = config["trainer"]['save_period']
        self.seed = seed
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(config["val_loader"]["batch_size"])))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / config["val_loader"]["batch_size"]) + 1

        # CHECKPOINTS & TENSOBOARD
        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        run_name = config['experim_name'] + '_' + str(seed)
        self.checkpoint_dir = os.path.join(config["trainer"]['save_dir'], config["experim_name"], run_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config["trainer"]['log_dir'], config["experim_name"], run_name)
        self.writer = tensorboard.SummaryWriter(writer_dir)
        self.html_results = HTML(web_dir=self.checkpoint_dir, exp_name=config['experim_name'],
                                 save_name=config['experim_name'], config=config, resume=resume)

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            results = self._train_epoch(epoch)
            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                self.logger.info('\n\n')
                for k, v in results.items():
                    self.logger.info(f'{str(k):15s}: {v}')
                    if epoch == self.epochs:
                        with open(f'{self.checkpoint_dir}/val.txt', 'w') as f:
                            f.write("%s\n" % (k + ':' + f'{v}'))

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
        self.html_results.save()
        self.writer.flush()
        self.writer.close()

    def _train_epoch(self, epoch):
        self.html_results.save()

        self.logger.info('\n')
        self.model.train()

        tbar = tqdm(self.train_loader, ncols=135)

        self._reset_metrics()
        for batch_idx, data in enumerate(tbar):
            cropp1, cropp2, cropp3, cropp4, cropp5, cropp6, cropp7, cropp8, target1, target2, target3, target4, _ = data
            cropp = torch.cat([cropp1, cropp2, cropp3, cropp4, cropp5, cropp6, cropp7, cropp8], dim=0)
            target = torch.cat([target1, target2, target3, target4, target1, target2, target3, target4], dim=0)
            if self.device != 'cpu':
                cropp, target = cropp.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            output = self.model(cropp)

            batch_loss = self.criterion(output, target)
            batch_loss.backward()
            self.optimizer.step()
            self._update_losses(batch_loss.detach().cpu().numpy())
            log = self._log_values()

            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self._write_scalars_tb(log)

            del batch_loss, output

            tbar.set_description('T ({}) | MSELoss {:.3f} |'.format(epoch, self.total_mse_loss.average))
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                cropp1, cropp2, cropp3, cropp4, target1, target2, target3, target4, _ = data
                cropp = torch.cat([cropp1, cropp2, cropp3, cropp4], dim=0)
                target = torch.cat([target1, target2, target3, target4], dim=0)
                if self.device == 'cpu':
                    cropp, target = cropp.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                output = self.model(cropp)

                # LOSS
                loss = self.criterion(output, target)
                total_loss_val.update(loss.cpu())

                # PRINT INFO
                tbar.set_description('EVAL ({}) | MSELoss: {:.3f} |'.format(epoch, total_loss_val.average))

            # METRICS TO TENSORBOARD
            self.wrt_step = epoch * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)

            log = {'val_loss': total_loss_val.average}
            self.html_results.add_results(epoch=epoch, results=log)
            self.html_results.save()
        return log

    def _reset_metrics(self):
        self.total_mse_loss = AverageMeter()

    def _update_losses(self, batch_loss):
        self.total_mse_loss.update(batch_loss.mean())

    def _log_values(self):
        logs = {}
        logs['mse_loss'] = self.total_mse_loss.average
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'config': self.config
        }

        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f'Error when loading: {e}')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if "logger" in checkpoint.keys():
            self.train_logger = checkpoint['logger']
        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

