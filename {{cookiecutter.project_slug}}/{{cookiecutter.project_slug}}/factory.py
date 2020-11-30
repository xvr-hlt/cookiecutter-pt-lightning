from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pydantic

from . import util


class ConfigItem(pydantic.BaseModel):
    """Dataclass representing generic config item"""

    module_name: str
    name: str
    kwargs: Optional[Dict] = None

    def load(self, **kwargs) -> Any:
        """Returns instantiated class represented by `self`.
        Returns:
            Any: Instantiated class.
        """
        self_kwargs = self.kwargs or {}
        kwargs = {**self_kwargs, **kwargs}
        return util.load_class(module_name=self.module_name, name=self.name, kwargs=kwargs)


class TrainConfig(pydantic.BaseModel):
    model: ConfigItem
    loss: ConfigItem
    optim: ConfigItem
    optim_scheduler: ConfigItem
    instance: ConfigItem
    dataset: ConfigItem
    sampler: ConfigItem

    def __init__(self, config: Union[Dict, str]):
        config = util.load_config(config)
        return super().__init__(**config)
