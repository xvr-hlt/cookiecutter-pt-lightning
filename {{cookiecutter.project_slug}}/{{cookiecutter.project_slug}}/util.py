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

def download_kaggle_dataset(competition_name):
    home_dir = pathlib.Path(__file__)
    data_dir = home_dir.parent.parent / 'data'
    dataset_dir = data_dir / competition_name
    if not dataset_dir.exists():
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(competition_name, data_dir)
        zip_data = data_dir / f"{competition_name}.zip"
        with zipfile.ZipFile(zip_data, "r") as f:
            f.extractall(dataset_dir)

def get_model_from_dir(run_dir):
    run_dir = pathlib.Path(run_dir)
    model_config = read_config(str(run_dir / 'config.yaml'))
    net = model.get_model(load_weights=False, **model_config['model'])
    net = net.cuda()
    load_weights(net, run_dir / 'model.ckpt')
    net = net.eval()
    return net
