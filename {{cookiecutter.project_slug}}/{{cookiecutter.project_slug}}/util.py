import importlib
import pathlib
import typing
from typing import Any, Dict, Union

import yaml

def load_config(config: Union[Dict[str, Any], str, pathlib.PosixPath]) -> Dict[str, Any]:
    if isinstance(config, (str, pathlib.PosixPath)):
        with open(config) as f:
            config = yaml.safe_load(f)

    config = typing.cast(dict, config)  # Avoids mypy errors.

    if 'wandb_version' in config:
        config.pop('wandb_version')
        config.pop('_wandb')
        config = {k: v['value'] for k, v in config.items()}

    if 'from' in config:
        from_config = load_config(config['from'])

        def recursive_dict_merge(d1: Dict, d2: Dict) -> Dict:
            for key, val in d1.items():
                if isinstance(val, dict):
                    d2_node = d2.setdefault(key, {})
                    recursive_dict_merge(val, d2_node)
                else:
                    if key not in d2:
                        d2[key] = val
            return d2

        config = recursive_dict_merge(from_config, config)

    return config  # type: ignore


def load_class(module_name: str, name: str, kwargs) -> Any:
    """Takes a module name, class/fn name and kwargs and initialises/returns object.

    Args:
        module_name (str): Name of module (e.g. tensorflow.keras.applications.efficientnet)
        name (str): Name of class e.g. EfficientNetB0.
        kwargs (Optional[Dict[str, Any]]): Kwargs to initialise object with.

    Returns:
        Any: Instantiated class.
    """
    module = importlib.import_module(module_name)
    cls_fn = getattr(module, name)
    return cls_fn(**kwargs)
