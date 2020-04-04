#!/usr/bin/env python
import fire

from . import experiment

if __name__ == '__main__':
    fire.Fire(experiment.Experiment.run)
