import torch
import os
import shutil
import json

import torch.nn as nn

from tqdm import tqdm
from torchsummary import summary

from data import DataManger
from base import BaseTrainer
from losses import Softmax_Triplet_loss
from optimizers import WarmupMultiStepLR
from models import Baseline
from utils import MetricTracker

class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.datamanager = DataManger(config['data'])

        self.model = Baseline(
            num_classes=self.datamanager.datasource.get_num_classes('train'))

        summary(self.model, input_size=(3, 256, 128),
                batch_size=config['data']['batch_size'], device='cpu')

        cfg_losses = config['losses']
        self.criterion = Softmax_Triplet_loss(
            num_class=self.datamanager.datasource.get_num_classes('train'),
            margin=cfg_losses['margin'],
            epsilon=cfg_losses['epsilon'],
            use_gpu=self.use_gpu
        )

        cfg_optimizer = config['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg_optimizer['lr'],
            weight_decay=cfg_optimizer['weight_decay'])

        cfg_lr_scheduler = config['lr_scheduler']
        self.lr_scheduler = WarmupMultiStepLR(
            self.optimizer,
            milestones=cfg_lr_scheduler['steps'],
            gamma=cfg_lr_scheduler['gamma'],
            warmup_factor=cfg_lr_scheduler['factor'],
            warmup_iters=cfg_lr_scheduler['iters'],
            warmup_method=cfg_lr_scheduler['method'])

        self.train_metrics = MetricTracker('loss', 'accuracy')
        self.valid_metrics = MetricTracker('loss', 'accuracy')

        self.best_accuracy = None

        if config['resume'] != '':
            self._resume_checkpoint(config['resume'])

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            result = self._valid_epoch(epoch)

            # add scalars to tensorboard
            self.writer.add_scalars('Loss',
                {
                    'Train': self.train_metrics.avg('loss'),
                    'Val': self.valid_metrics.avg('loss')
                }, global_step=epoch)
            self.writer.add_scalars('Accuracy',
                {
                    'Train': self.train_metrics.avg('accuracy'),
                    'Val': self.valid_metrics.avg('accuracy')
                }, global_step=epoch)

            # logging result to console
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save model
            if self.best_accuracy == None or self.best_accuracy < self.valid_metrics.avg('accuracy'):
                self.best_accuracy = self.valid_metrics.avg('accuracy')
                self._save_checkpoint(epoch, save_best=True)
            else:
                self._save_checkpoint(epoch, save_best=False)

            # save logs
            if epoch % self.cfg_trainer['save_period'] == 0:
                self._save_logs(epoch)
        self._save_logs(epoch)

    def _train_epoch(self, epoch):
        """ Training step
        """
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=len(self.datamanager.get_dataloader('train'))) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            for batch_idx, (data, labels, _) in enumerate(self.datamanager.get_dataloader('train')):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                score, feat = self.model(data)

                loss = self.criterion(score, feat, labels)
                _, preds = torch.max(score.data, dim=1)
                loss.backward()
                self.optimizer.step()

                self.train_metrics.update('loss', loss.item())
                self.train_metrics.update('accuracy', torch.sum(
                    preds == labels.data).double().mean().item())

                epoch_pbar.set_postfix({
                    'train_loss': self.train_metrics.avg('loss'),
                    'train_acc': self.train_metrics.avg('accuracy')})
                epoch_pbar.update(1)
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        """ Validation step
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            with tqdm(total=len(self.datamanager.get_dataloader('val'))) as epoch_pbar:
                epoch_pbar.set_description(f'Epoch {epoch}')
                for batch_idx, (data, labels, _, _) in enumerate(self.datamanager.get_dataloader('val')):
                    data, labels = data.to(self.device), labels.to(self.device)

                    score, feat = self.model(data)

                    loss = self.criterion(score, feat, labels)
                    _, preds = torch.max(score.data, dim=1)

                    self.valid_metrics.update('loss', loss.item())
                    self.valid_metrics.update('accuracy', torch.sum(
                        preds == labels.data).double().mean().item())

                    epoch_pbar.set_postfix({
                        'val_loss': self.valid_metrics.avg('loss'),
                        'val_acc': self.valid_metrics.avg('accuracy')})
                    epoch_pbar.update(1)
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best=True):
        """ save model to file
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_accuracy': self.best_accuracy
        }
        filename = os.path.join(self.checkpoint_dir, 'model_last.pth')
        self.logger.info("Saving last model: model_last.pth ...")
        torch.save(state, filename)
        if save_best:
            filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
            self.logger.info("Saving current best: model_best.pth ...")
            torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        """ Load model from checkpoint
        """
        if not os.path.exists(resume_path):
            raise FileExistsError("Resume path not exist!")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.map_location)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_loss = checkpoint['best_accuracy']
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _save_logs(self, epoch):
        """ Save logs from google colab to google drive
        """
        if os.path.isdir(self.logs_dir_saved):
            shutil.rmtree(self.logs_dir_saved)
        destination = shutil.copytree(self.logs_dir, self.logs_dir_saved)
