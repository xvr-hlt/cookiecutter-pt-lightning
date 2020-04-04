import torch
from torch import nn


{% if cookiecutter.vision %}
import albumentations
from PIL import Image
{% endif %}



class Dataset(torch.utils.data.Dataset):
    def __init__(self, instances, classes, {%if cookiecutter.text %} tokenizer,{% endif %}, {%if cookiecutter.vision %} size, augmentation=None,{% endif %}):
        self.instances = instances
        self.classes = classes
        {% if cookiecutter.text %}
        self.tokenizer = tokenizer
        {%end if %}
        {% if cookiecutter.vision %}
        pipeline = []
        if augmentation is not None:
            pipeline.append(augmentation)
        pipeline.append(albumentations.Resize(*size))
        pipeline.append(albumentations.pytorch.ToTensor())
        self.pipeline = albumentations.Compose(pipeline)
        {% endif %}

    def __len__(self):
        return len(self.instances)

    def get_target(self, instance):
        return None # @Todo

    def __getitem__(self, ix):
        instance = self.instances[ix]
        {%if cookiecutter.vision %}
        image = Image.open(instance.path)
        {%if cookiecutter.segmentation %}
        mask = self.get_target(mask)
        processed = self.pipeline(image=image, mask=mask)
        image, mask = processed['image'], processed['mask']
        return image, mask
        {% else %}
        processed = self.pipeline(image=image)
        image = processed['image']
        target = None
        return image, target
        {% end if %}
        {% end if %}
        return 
