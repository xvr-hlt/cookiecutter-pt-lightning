import dataclasses

@dataclasses.dataclass
class Instance:
    uid: str
    label: str{%if cookiecutter.vision|int %}
    path: str{%endif %}{%if cookiecutter.segmentation|int %}
    mask: str{%endif %}{%if cookiecutter.text %}
    text: str{%endif %}

def get_train_val_instances(val_split=0.2, split_ix=0):
    instances = []

    def hash_str(s):
        return int(hashlib.sha512(s.encode()).hexdigest(), 16)

    instances = sorted(instances, key=lambda i: hash_str(i.uid))
    split_size = int(val_split * len(instances))
    val_start, val_end = split_size * split_ix, split_size * (split_ix + 1)
    val_instances = instances[val_start:val_end]
    train_instances = instances[:val_start] + instances[val_end:]
    return train_instances, val_instances
