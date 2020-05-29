{%if cookiecutter.text|int %}import transformers
{% endif %}{%if cookiecutter.segmentation|int %}import pytorch_segmentation_models
{% endif %}{%if cookiecutter.vision|int %}import efficientnet_pytorch as en
{% endif %}

def get_model(model_name, num_classes, load_weights=True, sync_bn=False, **model_kwargs):{%if cookiecutter.text|int %}
    config = transformers.AutoConfig.from_pretrained(model_name,
                                                    num_labels=num_classes)
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name, config=config){% endif %}{%if cookiecutter.vision|int %}
    if load_weights:
        backbone = en.EfficientNet.from_pretrained(model_name,
                                                    num_classes=num_classes,
                                                    **model_kwargs)
    else:
        model = en.EfficientNet.from_name(
            model_name, override_params={'num_classes': num_classes})
    {%endif%}
    return None
