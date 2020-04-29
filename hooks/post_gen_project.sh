#!/bin/bash
direnv allow
git init
git remote add origin git@github.com:{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}
{% if not cookiecutter.kaggle_competition %} rm make_submission.py {% endif %}
