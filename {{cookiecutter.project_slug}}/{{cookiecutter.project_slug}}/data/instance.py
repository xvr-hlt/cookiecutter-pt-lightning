import dataclasses

import torch
import numpy as np


@dataclasses.dataclass
class Instance:
    uid: str
    label
    {%if cookiecutter.vision %}
    path: str{%end if %}
    {%if cookiecutter.segentation %}
    mask {%end if %}
    {%if cookiecutter.text %}
    text {%end if %}
