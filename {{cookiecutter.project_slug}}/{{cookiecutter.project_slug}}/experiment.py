import datetime
import os
from os import path
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import torch
from pytorch_lightning import callbacks

from . import net
from .data import instance, loader

{%if cookiecutter.kaggle_competition %}
COMPETITION_NAME = {{cookiecutter.kaggle_competition}}{%endif %}
class Experiment(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = net.model.get_model(**config['model'])

        loss_conf = config['loss']
        loss_cls = getattr(net.losses, loss_conf['type'])
        self.loss = loss_cls(**loss_conf['kwargs'])

        self.batch_size = config['batch_size']
        if self.config['trainer'].get('distributed_backend') == 'dp':
            self.batch_size *= self.config['trainer']['gpus']
        self.pretrain = True

    def forward(self, x):
        return self.model(x)

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
            self.log_hyperparams(self.config)
            self.pretrain = False
        else:
            self.log_metrics(metrics)
        return metrics

    @rank_zero_only
    def log_metrics(self, metrics):
        wandb.log(metrics)

    @rank_zero_only
    def log_hyperparams(self):
        wandb.init(config=self.config)



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

    def prepare_data(self):
        kwargs = self.config['instance']
        if 'dataset_dir' not in kwargs:
            home_dir = pathlib.Path(__file__)
            data_dir = home_dir.parent.parent / 'data'
            kwargs['dataset_dir'] = data_dir / COMPETITION_NAME

            
        self.train_instances, self.val_instances = instance.get_train_val_instances(
            **kwargs)

    def train_dataloader(self):
        dataset = loader.InstanceDataset(self.train_instances,
                                         **self.config['data'])

        return torch.utils.data.DataLoader(dataset, self.batch_size, 
                                            shuffle=True, pin_memory=True,
                                            **self.config['loader'])

    def val_dataloader(self):
        dataset = loader.InstanceDataset(self.val_instances,
                                         **self.config['data'])

        return torch.utils.data.DataLoader(dataset, self.batch_size, 
                                            shuffle=False, pin_memory=True,
                                            **self.config['loader'])

    @staticmethod
    def run(config):
        config = util.read_config(config)
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = path.join("wandb", now)
        run_dir = path.abspath(run_dir)
        os.environ['WANDB_RUN_DIR'] = run_dir

        checkpoint_callback = callbacks.ModelCheckpoint(
            run_dir + "/{epoch}-{val_loss:.2f}",
            **config['checkpoint'])

        os.environ['WANDB_PROJECT'] = COMPETITION_NAME
        os.environ['WANDB_RUN_DIR'] = run_dir

        if config['load_weights']:
            experiment = Experiment.load_from_checkpoint(config['load_weights'],
                                                         config=config)

        else:
            experiment = Experiment(config)

        trainer = pl.Trainer(logger=None,
                             checkpoint_callback=checkpoint_callback,
                             early_stop_callback=None,
                             **config['trainer'])

        trainer.fit(experiment)
