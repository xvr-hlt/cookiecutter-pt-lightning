from setuptools import find_packages, setup

REQUIRES = [
    'yapf',
    'pylint',
    'tqdm',
    'fire',
    'pytest',
    'torch',
    'pytorch-lightning',
    'pyyaml',
    'pytorch_toolbelt',{% if cookiecutter.text|int %}
    'transformers',{% endif %}{% if cookiecutter.vision|int %}
    'albumentations',
    'segmentation-models-pytorch',{% endif %}
]

setup(name='{{ cookiecutter.project_slug }}',
      version='{{ cookiecutter.version }}',
      packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
      python_requires='>={{ cookiecutter.python_version }}',
      install_requires=REQUIRES,
      entry_points={
          'console_scripts': ['main={{ cookiecutter.project_slug }}.main:main'],
      })
