import datetime
import os
from os import path
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks

from . import net
from .data import instance, loader

{%if cookiecutter.kaggle_competition %}
COMPETITION_NAME = {{cookiecutter.kaggle_competition}}{%if endif %}
class Experiment(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = net.model.get_model(**config['model'])

        loss_conf = config['loss']
        loss_cls = net.losses.__dict__[loss_conf['type']]
        self.loss = loss_cls(**loss_conf['kwargs'])

        self.batch_size = config['batch_size']
        if self.config['trainer']['distributed_backend'] == 'dp':
            self.batch_size *= self.config['trainer']['gpus']
        self.pretrain = True

    @property
    def batch_size(self):
        if self._batch_size is None:
            batch_size = self.config['data']['batch_size']
            if self.trainer.use_dp:
                batch_size *= self.config['trainer']['gpus']
            self._batch_size = batch_size
        return self._batch_size

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_val = self.loss(y_hat, y)
        return {'loss': loss_val}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss_val = self.loss(y_hat, y)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        val_loss = 0
        for out in outputs:
            loss = out['val_loss']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss = torch.mean(loss)
            val_loss += loss
        val_loss /= len(outputs)

        metrics = {'val_loss': val_loss}

        if self.pretrain:
            self.logger.log_hyperparams(self.config)
            self.pretrain = False
        else:
            self.logger.log_metrics(metrics)
        return metrics


    def configure_optimizers(self):
        optim_conf = self.config['optim']
        optim_cls = net.optim.__dict__[optim_conf['type']]
        optimizer = optim_cls(self.parameters(), **optim_conf['kwargs'])

        optim_scheduler_conf = self.config['optim_scheduler']
        optim_scheduler_cls = net.optim.lr_scheduler.__dict__[
            optim_scheduler_conf['type']]
        optim_scheduler = optim_scheduler_cls(optimizer,
                                              **optim_scheduler_conf['kwargs'])
        return [optimizer], [optim_scheduler]

    def prepare_data(self):{%if cookiecutter.kaggle_competition %}
        home_dir = pathlib.Path(__file__)
        data_dir = home_dir.parent.parent / 'data'
        dataset_dir = data_dir / COMPETITION_NAME
        if not dataset_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(COMPETITION_NAME, data_dir)
            with zipfile.ZipFile(data_dir / f"{COMPETITION_NAME}.zip",
                                 "r") as f:
                f.extractall(dataset_dir){% else %}
        dataset_dir = 'data/'{% endif %}

        self.train_instances, self.val_instances = instance.get_train_val_instances(
            dataset_dir, **self.config['instance'])

    def train_dataloader(self):
        dataset = loader.InstanceDataset(self.train_instances,
                                         **self.config['data'])

        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           **self.config['loader'])

    def val_dataloader(self):
        dataset = loader.InstanceDataset(self.val_instances,
                                         **self.config['data'])
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           **self.config['loader'])

    @staticmethod
    def run(config):
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = path.join("wandb", now)
        run_dir = path.abspath(run_dir)
        os.environ['WANDB_RUN_DIR'] = run_dir

        checkpoint_callback = callbacks.ModelCheckpoint(
            run_dir + "/{epoch}-{val_loss:.2f}",
            monitor=config['early_stopping']['monitor'])

        early_stopping_callback = callbacks.EarlyStopping(
            **config['early_stopping'])

        experiment = Experiment(config)
        trainer = pl.Trainer(logger=pl.loggers.WandbLogger(),
                             checkpoint_callback=checkpoint_callback,
                             early_stop_callback=early_stopping_callback,
                             **config['trainer'])

        trainer.fit(experiment)
