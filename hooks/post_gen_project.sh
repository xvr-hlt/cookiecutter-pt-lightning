#!/bin/bash
direnv allow
git init
git remote add origin git@github.com:{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}