import dataclasses

@dataclasses.dataclass
class Instance:
    uid: str
    label: str{%if cookiecutter.vision|int %}
    path: str{%endif %}{%if cookiecutter.segmentation|int %}
    mask: str{%endif %}{%if cookiecutter.text %}
    text: str{%endif %}

def get_train_val_instances(n_folds: int, val_fold: int = 0):
    instances = []

    def hash_str(s):
        return int(hashlib.sha512(s.encode()).hexdigest(), 16)

    train_instances, val_instances = [], []
    for i in instances:
        if hash_str(i.uid) % n_folds == val_fold:
            val_instances.append(i)
        else:
            train_instances.append(i)
    return train_instances, val_instances
