import dataclasses

@dataclasses.dataclass
class Instance:
    uid: str
    label: str{%if cookiecutter.vision|int %}
    path: str{%endif %}{%if cookiecutter.segmentation|int %}
    mask: str{%endif %}{%if cookiecutter.text %}
    text: str{%endif %}

def get_train_val_instances(val_split=0.3):
    return [], []