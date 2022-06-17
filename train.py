import warnings

warnings.filterwarnings('ignore')

import os
import argparse

from omegaconf import OmegaConf
from importlib import import_module

import torch
import pytorch_lightning as pl


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


def init_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    DATA_LOADER = getattr(DATA_MODULE, cfg.data.loader)

    if cfg.general.task == "train":
        print("=> loading the train and val datasets...")
    else:
        print("=> loading the {} dataset...".format(cfg.data.split))

    datasets, dataloaders = DATA_LOADER(cfg)
    print("=> loading dataset completed")

    return datasets, dataloaders


def init_logger(cfg):
    logger = pl.loggers.TensorBoardLogger(cfg.general.root, name="logs", default_hp_metric=False)
    return logger


def init_callbacks(cfg):
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="val/{}".format(cfg.general.monitor),
        mode="min",
        dirpath=cfg.general.root,
        filename="pointgroup-{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.train.save_checkpoint_every_n_epochs,
        save_last=True
    )

    return [monitor]


def init_trainer(cfg):
    if cfg.model.use_checkpoint:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.model.use_checkpoint,
                                  "last.ckpt")
    else:
        checkpoint = None

    trainer = pl.Trainer(
        gpus=-1,  # use all available GPUs
        strategy='ddp',  # use multiple GPUs on the same machine
        num_nodes=args.num_nodes,
        max_epochs=cfg.train.epochs,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,  # validate on all val data before training
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=callbacks,  # comment when debug
        logger=logger,
        profiler="simple",
        resume_from_checkpoint=checkpoint
    )

    return trainer


def init_model(cfg):
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)

    if cfg.model.use_checkpoint:
        print("=> loading pretrained checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.model.use_checkpoint,
                                  "last.ckpt")
        model.load_from_checkpoint(checkpoint)

    if cfg.model.pretrained_module:
        print("=> loading pretrained module from {} ...".format(cfg.model.pretrained_module_path))
        # for i, module_name in enumerate(cfg.model.pretrained_module):
        model_dict = model.state_dict()
        ckp = torch.load(cfg.model.pretrained_module_path)
        pretrained_module_dict = {k: v for k, v in ckp.items() if k.startswith(tuple(cfg.model.pretrained_module))}
        model_dict.update(pretrained_module_dict)
        model.load_state_dict(model_dict)

    if cfg.model.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/softgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-e', '--experiment', type=str, default='', help='specify experiment')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='specify num of gpu nodes')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    datasets, dataloaders = init_data(cfg)

    print("=> initializing logger...")
    logger = init_logger(cfg)

    print("=> initializing monitor...")
    callbacks = init_callbacks(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> initializing model...")
    pointgroup = init_model(cfg)

    print("=> start training...")
    trainer.fit(model=pointgroup, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"])
