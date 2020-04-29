#!/usr/bin/env python
import fire

from . import experiment

def main():
    fire.Fire(experiment.Experiment.run)

if __name__ == '__main__':
    main()
