import torch
from torch import nn

{% if cookiecutter.vision|int %}import albumentations
from PIL import Image
{% endif %}

class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self,
        instances,
        classes,{%if cookiecutter.text|int %}
        tokenizer,
        max_length,{% endif %}{%if cookiecutter.vision|int %}
        img_size,
        augmentation=None,{% endif %}
    ):
        self.instances = instances
        self.classes = classes{% if cookiecutter.text|int %}
        self.tokenizer = tokenizer
        self.max_length = max_length{%endif %}{% if cookiecutter.vision|int %}
        pipeline = []
        if augmentation is not None:
            pipeline.append(augmentation)
        pipeline.append(albumentations.Resize(*img_size))
        pipeline.append(albumentations.pytorch.ToTensor())
        self.pipeline = albumentations.Compose(pipeline){% endif %}

    def __len__(self):
        return len(self.instances)

    def get_target(self, instance):
        return 0 # @Todo

    def __getitem__(self, ix):
        instance = self.instances[ix]
        target = self.get_target(instance){%if cookiecutter.vision|int %}
        image = Image.open(instance.path){%if cookiecutter.segmentation|int %}
        processed = self.pipeline(image=image, mask=target)
        image, mask = processed['image'], processed['mask']
        return image, mask{% else %}
        processed = self.pipeline(image=image)
        image = processed['image']
        return image, target{% endif %}
        {% endif %}{%if cookiecutter.text|int %}
        text = instance.text
        text = self.tokenizer.encode_plus(text,
                                          max_length=self.max_length,
                                          pad_to_max_length=True,
                                          return_tensors='pt')
        text = {k: v.flatten() for k, v in text.items()}
        return text, target{% endif %}
