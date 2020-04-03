import yaml


def load_config(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)
    return config
