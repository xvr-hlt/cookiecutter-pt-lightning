import datetime
import os
import pathlib
from os import path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.utilities import rank_zero_only

from .data import loader

from . import factory, util

HERE = pathlib.Path(__file__)
BASE_CONFIG = HERE.parent.parent / "config" / "base.yml"

{%if cookiecutter.kaggle_competition %}
COMPETITION_NAME = {{cookiecutter.kaggle_competition}}{%endif %}
class Experiment(pl.LightningModule):

    def __init__(self, config=BASE_CONFIG):
        super().__init__()
        config = util.load_config(config)
        self.config = config

        pl.seed_everything(config['seed'])

        train_factory = factory.TrainConfig(config)
        self.tokenizer = util.load_tokenizer(config['tokenizer'])
        self.model = train_factory.model.load(vocab_size=self.tokenizer.get_vocab_size())
        self.loss = train_factory.loss.load()

        self.train_factory = train_factory
        self.is_ddp = self.config['trainer']['distributed_backend'] == "ddp"
        self.save_hyperparameters(self.config)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss_val = self.loss(y_hat, batch['tokens'])
        self.log('train_loss', loss_val, on_epoch=True)
        self.log('train_n', y_hat.shape[0], prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss_val = self.loss(y_hat, batch['tokens'])
        self.log('val_loss', loss_val, on_step=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_inst, self.val_inst = self.train_factory.instance.load()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = self.train_factory.dataset.load(instances=self.train_inst,
                                                        shuffle=True,
                                                        tokenizer=self.tokenizer)
        sampler = self.train_factory.sampler.load(dataset=train_dataset, shuffle=True, distributed=self.is_ddp)
        return torch.utils.data.DataLoader(train_dataset,
                                           batch_sampler=sampler,
                                           collate_fn=loader.dynamic_pad_collate,
                                           num_workers=64,
                                           pin_memory=True)
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = self.train_factory.dataset.load(instances=self.val_inst, shuffle=False, tokenizer=self.tokenizer)
        sampler = self.train_factory.sampler.load(dataset=val_dataset, shuffle=False, distributed=self.is_ddp)
        return torch.utils.data.DataLoader(val_dataset,
                                           batch_sampler=sampler,
                                           collate_fn=loader.dynamic_pad_collate,
                                           num_workers=64,
                                           pin_memory=True)

    @rank_zero_only
    def log_hyperparams(self):
        self.logger.experiment.config.update(self.config, allow_val_change=True)

    def configure_optimizers(self):
        optim = self.train_factory.optim.load(params=self.parameters())
        optim_scheduler = self.train_factory.optim_scheduler.load(optimizer=optim)
        scheduler = {'scheduler': optim_scheduler, 'interval': 'step'}
        return [optim], [scheduler]

    @staticmethod
    def run(config="config/base.yml"):
        config = util.load_config(config)
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = path.join("wandb", now)
        run_dir = path.abspath(run_dir)
        os.environ['WANDB_PROJECT'] = "linear_turing"
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss',
                                                        mode='min',
                                                        save_weights_only=True,
                                                        save_last=True,
                                                        filename='{epoch}_{val_loss:.2f}')

        other_callbacks = [
            pl.callbacks.LearningRateMonitor(),
            callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
        ]

        experiment = Experiment(config)

        trainer = pl.Trainer(logger=pl.loggers.WandbLogger(log_model=True),
                             checkpoint_callback=checkpoint_callback,
                             callbacks=other_callbacks,
                             **config['trainer'])

        trainer.fit(experiment)
