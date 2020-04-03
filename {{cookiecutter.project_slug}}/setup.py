from setuptools import find_packages, setup

setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=[
        '{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*'
    ]),
    python_requires='>={{ cookiecutter.python_version }}',
    install_requires=['tqdm', 'pytest', 'torch', 'pytorch-lightning', 'pyyaml'],
    entry_points={
        'console_scripts': ['main={{ cookiecutter.project_slug }}.main:main'],
    })
