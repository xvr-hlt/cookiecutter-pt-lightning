import json
import os
import shutil
from datetime import datetime
from distutils import dir_util
from os import path

import fire
import kaggle

from {{cookiecutter.project_slug}} import util


def make_submission(inference_config="config/infer.yml",
                    run_base_dir="wandb",
                    save_local=False):
    submission_folder = pathlib.Path(datetime.now().strftime("%Y%m%d-%H%M"))
    submission_dir = "submissions" / submission_folder
    submission_dir.mkdir(exist_ok=True)

    dir_util.copy_tree('src', str(submission_dir / 'src'))
    dir_util.copy_tree('prodia', str(submission_dir / 'src' / 'prodia'))

    with open(submission_dir / 'dataset-metadata.json', 'w') as f:
        json.dump(
            {
                "title": str(submission_folder),
                "id": f"xvrhlt/{submission_folder}",
                "licenses": [{
                    "name": "CC0-1.0"
                }]
            }, f)

    shutil.copy(inference_config, submission_dir)
    config = util.read_config(inference_config)

    run_base_dir = pathlib.Path(run_base_dir)

    dst_run_dir = submission_dir / "runs"
    dst_run_dir.mkdir()

    for run in config['runs']:
        if isinstance(run, str):
            run_id = run
            run_kwargs = {}
        elif isinstance(run, dict):
            run_id = run['run_id']
            run_kwargs = run
        src_path = run_base_dir / run_id
        dst_run_dir = dst_run_dir / run_id
        dst_run_dir.mkdir()
        shutil.copy(str((src_path / "config.yaml")),
                    str(dst_run_dir / "config.yaml"))
        if 'model_pth' in run_kwargs:
            model = src_path / run_kwargs['model_pth']
        else:
            model = max(src_path.rglob('*.ckpt'))
        (dst_run_dir / 'model.ckpt').write_bytes(model.read_bytes())

    kaggle.api.dataset_create_new(submission_dir, dir_mode='zip')
    if save_local:
        shutil.make_archive(submission_folder, 'zip', root_dir='submissions')
    dir_util.remove_tree(submission_dir)
