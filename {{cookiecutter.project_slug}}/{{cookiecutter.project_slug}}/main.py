#!/usr/bin/env python
import fire

from . import experiment, submit


def main():
    fire.Fire({
        'train': experiment.Experiment.run,
        'submit': submit.make_submission
    })


if __name__ == '__main__':
    main()
