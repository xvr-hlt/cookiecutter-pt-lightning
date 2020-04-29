import torch
import yaml


def read_config(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)
    if 'wandb_version' in config:
        config.pop('wandb_version')
        config.pop('_wandb')
        config = {k: v['value'] for k, v in config.items()}
    return config


def load_weights(model, weights):
    weights = torch.load(weights)
    weights = {
        k.replace('model.', ''): v
        for k, v in weights['state_dict'].items()
        if k.startswith('model')
    }
    return model.load_state_dict(weights)
