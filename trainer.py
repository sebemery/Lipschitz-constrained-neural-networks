import torch
import logging
import os
import datetime
import json
from utils import logger
from utils.htmlwriter import HTML
from torch.utils import tensorboard
from utils.metrics import AverageMeter
from utils.utilities import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb


class Trainer:
    def __init__(self, config, trainloader, valloader, model, train_logger, seed, resume, device):
        self.config = config
        self.train_loader = trainloader
        self.val_loader = valloader
        self.model = model
        self.device = device
        if self.device != "cpu":
            self.model = self.model.to(device)
        print(self.model.get_num_params())
        self.set_optimization()
        if self.config["optimizer"]["args"]["stepscheduler"] is True:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.main_optimizer, step_size=self.config["optimizer"]["args"]["step"], gamma=self.config["optimizer"]["args"]["gamma"])
        if self.config["dataset"] == "fastMRI":
            self.criterion = torch.nn.MSELoss(reduction="sum")
        elif self.config["dataset"] == "BSD500":
            self.criterion = torch.nn.MSELoss(size_average=False)
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

    def set_optimization(self):
        """ """
        self.optim_names = self.config["optimizer"]["type"]

        # main optimizer/scheduler
        if len(self.optim_names) == 2:
            try:
                # main optimizer only for network parameters
                main_params_iter = self.model.parameters_no_deepspline_apl()
            except AttributeError:
                print('Cannot use aux optimizer.')
                raise
        else:
            # single optimizer for all parameters
            main_params_iter = self.model.parameters()

        self.main_optimizer = self.construct_optimizer(main_params_iter, self.optim_names[0], 'main')

        self.aux_optimizer = None

        if len(self.optim_names) == 2:
            # aux optimizer/scheduler for deepspline/apl parameters
            try:
                if self.model.deepspline is not None:
                    aux_params_iter = self.model.parameters_deepspline()
                # elif self.net.apl is not None:
                    # aux_params_iter = self.net.parameters_apl()
            except AttributeError:
                print('Cannot use aux optimizer.')
                raise

            self.aux_optimizer = self.construct_optimizer(aux_params_iter, self.optim_names[1], 'aux')

    def construct_optimizer(self, params_iter, optim_name, mode='main'):
        """ """
        lr = self.config["optimizer"]["args"]["lr"] if mode == 'main' else self.config["optimizer"]["args"]["lr"]

        # weight decay is added manually
        if optim_name == 'Adam':
            optimizer = torch.optim.Adam(params_iter, lr=lr)
        elif optim_name == 'SGD':
            optimizer = torch.optim.SGD(params_iter, lr=lr)
        else:
            raise ValueError('Need to provide a valid optimizer type')

        return optimizer

    def train(self):
        # losses
        Total_loss_train = []
        MSE_loss_val = []
        lr = []
        if self.config["dataset"] == "BSD500":
            psnr_val = []
            ssim_val = []
        self.model.init_hyperparams()

        # loop
        for epoch in range(self.start_epoch, self.epochs+1):

            if epoch == self.epochs and self.config["model"]["sparsify_activations"]:
                print('\nLast epoch: freezing network for sparsifying the activations and evaluating training accuracy.')
                self.model.eval()  # set network in evaluation mode
                self.model.sparsify_activations()
                self.model.freeze_parameters()

            results = self._train_epoch(epoch)
            Total_loss_train.append(results['mse_loss'])
            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                MSE_loss_val.append(results['val_loss'])
                if self.config["dataset"] == "BSD500":
                    psnr_val.append(results['val_psnr'])
                    ssim_val.append(results['val_ssim'])
                self.logger.info('\n\n')
                for k, v in results.items():
                    self.logger.info(f'{str(k):15s}: {v}')
            if epoch == self.epochs:
                with open(f'{self.checkpoint_dir}/val.txt', 'w') as f:
                    for k, v in results.items():
                        f.write("%s\n" % (k + ':' + f'{v}'))

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            for i, opt_group in enumerate(self.main_optimizer.param_groups):
                lr.append(opt_group['lr'])

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
        self.html_results.save()
        self.writer.flush()
        self.writer.close()
        Total_loss_train = np.array(Total_loss_train)
        MSE_loss_val = np.array(MSE_loss_val)
        if self.config["dataset"] == "BSD500":
            psnr_val = np.array(psnr_val)
            ssim_val = np.array(ssim_val)
        epochs = np.arange(self.config["trainer"]["val_per_epochs"], self.config["trainer"]["epochs"] + self.config["trainer"]["val_per_epochs"], self.config["trainer"]["val_per_epochs"])
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(Total_loss_train, 'g-',  linewidth=1)
        ax2.plot(epochs, MSE_loss_val, 'b-', linewidth=1)
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('Train loss', color='g')
        ax2.set_ylabel('Validation loss', color='b')
        ax1.set_title("Learning curves")
        fig.savefig(f'{self.checkpoint_dir}/curves.png')
        plt.show()
        if self.config["dataset"] == "BSD500":
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(epochs, psnr_val, 'g-', linewidth=1)
            ax2.plot(epochs, ssim_val, 'b-', linewidth=1)
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('val psnr', color='g')
            ax2.set_ylabel('Val ssim', color='b')
            ax1.set_title("Metric curves")
            fig.savefig(f'{self.checkpoint_dir}/metrics.png')
            plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(lr, 'k-', linewidth=1)
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('lr', color='b')
        ax1.set_title("Lr curves")
        fig.savefig(f'{self.checkpoint_dir}/lr.png')
        plt.show()

    def _train_epoch(self, epoch):
        self.html_results.save()

        self.logger.info('\n')
        self.model.train()

        tbar = tqdm(self.train_loader, ncols=135)

        self._reset_metrics()
        for batch_idx, data in enumerate(tbar):
            if self.config["dataset"] == "fastMRI":
                cropp1, cropp2, cropp3, cropp4, cropp5, cropp6, cropp7, cropp8, target1, target2, target3, target4, _ = data
                cropp = torch.cat([cropp1, cropp2, cropp3, cropp4, cropp5, cropp6, cropp7, cropp8], dim=0)
                target = torch.cat([target1, target2, target3, target4, target1, target2, target3, target4], dim=0)
            elif self.config["dataset"] == "BSD500":
                cropp, target = data
            if self.device != 'cpu':
                cropp, target = cropp.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.optimizer_zero_grad()
            batch_size = cropp.shape[0]
            output = self.model(cropp)

            # data fidelity
            if self.config["dataset"] == "fastMRI":
                data_fidelity = self.criterion(output, target) / batch_size
            elif self.config["dataset"] == "BSD500":
                data_fidelity = self.criterion(output, target)/(output.size()[0]*2)
            # data_fidelity.backward(retain_graph=True)

            # regularization
            regularization = torch.zeros_like(data_fidelity)
            if self.model.weight_decay_regularization is True:
                # the regularization weight is multiplied inside weight_decay()
                regularization = regularization + self.model.weight_decay()

            if self.model.tv_bv_regularization is True:
                # the regularization weight is multiplied inside TV_BV()
                tv_bv, tv_bv_unweighted = self.model.TV_BV()
                regularization = regularization + tv_bv
                # losses.append(tv_bv_unweighted)

            total_loss = data_fidelity + regularization
            total_loss.backward()
            self.optimizer_step()
            if self.config["model"]["spectral_norm"] == "Parseval":
                with torch.no_grad():
                    self.model.perseval_normalization(self.config["model"]["beta"])
            self._update_losses(total_loss.detach().cpu().numpy())
            log = self._log_values()

            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self._write_scalars_tb(log)

            del total_loss, output

            tbar.set_description('T ({}) | TotalLoss {:.3f} |'.format(epoch, self.total_mse_loss.average))

        if self.config["optimizer"]["args"]["stepscheduler"] is True:
            self.scheduler.step(epoch=epoch-1)
        return log

    def optimizer_zero_grad(self):
        """ """
        self.main_optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

    def optimizer_step(self):
        """ """
        self.main_optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()

        # Do the projection step to constrain the Lipschitz constant to 1
        if ((self.model.activation_type == 'deepBspline_lipschitz_orthoprojection') or (self.model.activation_type == 'deepBspline_lipschitz_maxprojection')):
            for module in self.model.modules_deepspline():
                module.do_lipschitz_projection()

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        psnr_val = 0
        ssim_val = 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                if self.config["dataset"] == "fastMRI":
                    cropp1, cropp2, cropp3, cropp4, target1, target2, target3, target4, _ = data
                    cropp = torch.cat([cropp1, cropp2, cropp3, cropp4], dim=0)
                    target = torch.cat([target1, target2, target3, target4], dim=0)
                elif self.config["dataset"] == "BSD500":
                    cropp, target = data
                if self.device != 'cpu':
                    cropp, target = cropp.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                batch_size = cropp.shape[0]
                output = self.model(cropp)

                # LOSS
                if self.config["dataset"] == "fastMRI":
                    loss = self.criterion(output, target) / batch_size
                elif self.config["dataset"] == "BSD500":
                    loss = self.criterion(output, target) / (output.size()[0] * 2)
                total_loss_val.update(loss.cpu())
                out_val = torch.clamp(output, 0., 1.)
                psnr_val += batch_PSNR(out_val, target, 1.)
                ssim_val += batch_SSIM(out_val, target, 1.)

                # PRINT INFO
                tbar.set_description('EVAL ({}) | MSELoss: {:.3f} |'.format(epoch, total_loss_val.average))

            # METRICS TO TENSORBOARD
            self.wrt_step = epoch * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
            psnr_val /= len(self.val_loader)
            ssim_val /= len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/Test PSNR', psnr_val, self.wrt_step)
            self.writer.add_scalar(f'{self.wrt_mode}/Test SSIM', ssim_val, self.wrt_step)

            log = {'val_loss': total_loss_val.average}
            self.html_results.add_results(epoch=epoch, results=log)
            self.html_results.save()
            log["val_psnr"] = psnr_val
            log["val_ssim"] = ssim_val
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

