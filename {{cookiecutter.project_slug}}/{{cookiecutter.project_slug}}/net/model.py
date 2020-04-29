{%if cookiecutter.text|int %}import transformers
{% endif %}{%if cookiecutter.segmentation|int %}import pytorch_segmentation_models
{% endif %}

def get_model(classes):{%if cookiecutter.text|int %}
    config = transformers.AutoConfig.from_pretrained(model_name,
                                                    num_labels=len(classes))
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name, config=config){% endif %}
    return None
