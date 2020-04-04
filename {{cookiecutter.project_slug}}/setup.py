from setuptools import find_packages, setup

REQUIRES = [
    'tqdm', 'pytest', 'torch', 'pytorch-lightning', 'pyyaml', 'pytorch_toolbelt',
    {% if cookiecutter.text %} 'transformers', {% endif %}
    {% if cookiecutter.vision %} * 'albumentations', 'segmentation-models-pytorch', {% endif %}
    ]

setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=[
        '{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*'
    ]),
    python_requires='>={{ cookiecutter.python_version }}',
    install_requires=REQUIRES,
    entry_points={
        'console_scripts': ['main={{ cookiecutter.project_slug }}.main:main'],
    })
