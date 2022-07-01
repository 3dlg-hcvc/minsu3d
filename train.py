import warnings

warnings.filterwarnings('ignore')

import os
import argparse

from omegaconf import OmegaConf
from importlib import import_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor
from lib.dataset.scannet_data_module import ScanNetDataModule
from lib.callback import *


def load_conf(args):
    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    cfg.general.task = 'train'
    cfg.general.experiment = args.experiment

    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.general.experiment,
                        cfg.general.task)
    os.makedirs(root, exist_ok=True)
    cfg.general.root = root
    cfg_backup_path = os.path.join(cfg.general.root, "config.yaml")
    OmegaConf.save(cfg, cfg_backup_path)

    return cfg


def init_logger(cfg):
    if cfg.general.logger == "TensorBoard":
        logger = TensorBoardLogger(cfg.general.root, name="logs", default_hp_metric=False)
    elif cfg.general.logger == "Wandb":
        logger = WandbLogger(project=cfg.general.model, save_dir=cfg.general.root, name="logs")
    return logger


def init_callbacks(cfg):
    checkpoint_monitor = init_checkpoint_monitor(cfg)
    gpu_stats_monitor = DeviceStatsMonitor()
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    return [checkpoint_monitor, gpu_stats_monitor, gpu_cache_clean_monitor]


def init_trainer(cfg):
    trainer = pl.Trainer(
        gpus=-1,  # use all available GPUs
        strategy="ddp",
        num_nodes=args.num_nodes,
        max_epochs=cfg.train.epochs,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        profiler="simple"
    )

    return trainer


def init_model(cfg):
    model = getattr(import_module(cfg.model.module), cfg.model.classname)(**cfg)

    if cfg.model.pretrained_module:
        print("=> loading pretrained module from {} ...".format(cfg.model.pretrained_module_path))
        # for i, module_name in enumerate(cfg.model.pretrained_module):
        model_dict = model.state_dict()
        ckp = torch.load(cfg.model.pretrained_module_path)
        pretrained_module_dict = {k: v for k, v in ckp.items() if k.startswith(tuple(cfg.model.pretrained_module))}
        model_dict.update(pretrained_module_dict)
        model.load_state_dict(model_dict)

    if cfg.model.freeze_backbone:
        for param in model.unet.parameters():
            param.requires_grad = False

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="conf/pointgroup_scannet.yaml", type=str, help='path to config file')
    parser.add_argument('-e', '--experiment', type=str, default='', help='specify experiment')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='specify num of gpu nodes')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    # fix the seed
    pl.seed_everything(cfg.general.manual_seed, workers=True)

    print("=> initializing data...")
    data_module = ScanNetDataModule(cfg)

    print("=> initializing logger...")
    logger = init_logger(cfg)

    print("=> initializing monitor...")
    callbacks = init_callbacks(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start training...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.use_checkpoint)
