import torch
import torch.nn as nn
import ml_collections
import yaml
import lib.utils.bookkeeping as bookkeeping
import lib.utils.utils as utils
from pathlib import Path
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm
import argparse


import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import lib.loggers.loggers as loggers
import lib.loggers.logger_utils as logger_utils


def main(eval_cfg, job_id=0, task_id=0):
    print("Evaluating with config", eval_cfg.eval_name)

    eval_folder, eval_named_folder, eval_named_folder_configs = \
        bookkeeping.setup_eval_folders(
            Path(eval_cfg.experiment_dir), eval_cfg.eval_name,
            job_id, task_id
    )
    bookkeeping.save_config_as_yaml(eval_cfg, eval_named_folder_configs)
    bookkeeping.save_git_hash(eval_named_folder)

    train_cfg = bookkeeping.load_ml_collections(
            bookkeeping.get_most_recent_config(
                Path(eval_cfg.experiment_dir).joinpath('config')
        )
    )
    for item in eval_cfg.train_config_overrides:
        utils.set_in_nested_dict(train_cfg, item[0], item[1])

    device = torch.device(eval_cfg.device)

    model = model_utils.create_model(train_cfg, device)

    dataset = dataset_utils.get_dataset(eval_cfg, device)

    model_state = torch.load(
        Path(eval_cfg.checkpoint_path), map_location=eval_cfg.device
    )['model']

    if utils.is_model_state_DDP(model_state):
        model_state = utils.remove_module_from_keys(model_state)

    model.load_state_dict(model_state)
    model.eval()

    writer = bookkeeping.NumpyWriter(eval_named_folder)

    for logger_name in eval_cfg.loggers:
        logging_func = logger_utils.get_logger(logger_name)
        logging_func(state={'model': model, 'n_iter': 1337}, cfg=eval_cfg, writer=writer,
            dataset=dataset)

    writer.save_to_disk()

    return eval_named_folder



if __name__ == "__main__":
    from config.eval.cifar10_elbo import get_config

    cfg = get_config()
    main(cfg)