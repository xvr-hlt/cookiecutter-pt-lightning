{%if cookiecutter.text|int %}import transformers
{% endif %}{%if cookiecutter.segmentation|int %}import pytorch_segmentation_models
{% endif %}

def get_model(classes):
    return None
