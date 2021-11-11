import warnings
warnings.filterwarnings('ignore')

import os
import argparse

import pytorch_lightning as pl

from omegaconf import OmegaConf
from importlib import import_module


def load_conf(args):
    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.general.experiment.upper())
    os.makedirs(root, exist_ok=True)

    cfg.general.task = 'train'
    cfg.general.root = root
    # cfg.cluster.prepare_epochs = -1

    cfg_backup_path = os.path.join(cfg.general.root, "config.yaml")
    OmegaConf.save(cfg, cfg_backup_path)

    return cfg

def init_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    dataloader = getattr(DATA_MODULE, cfg.data.loader)

    if cfg.general.task == "train":
        print("=> loading the train and val datasets...")
    else:
        print("=> loading the {} dataset...".format(cfg.data.split))
        
    dataset, dataloader = dataloader(cfg)
    print("=> loading dataset completed")

    return dataset, dataloader

def init_logger(cfg):
    logger = pl.loggers.TensorBoardLogger(cfg.general.root, name="logs")

    return logger

def init_monitor(cfg):
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="val/{}".format(cfg.general.monitor),
        mode="min",
        # save_weights_only=True,
        dirpath=cfg.general.root,
        filename="model",
        save_last=True
    )

    return monitor

def init_trainer(cfg):
    if cfg.model.use_checkpoint:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.model.use_checkpoint, "last.ckpt")
    else:
        checkpoint = None

    trainer = pl.Trainer(
        gpus=-1, # use all available GPUs 
        accelerator='ddp', # use multiple GPUs on the same machine
        max_epochs=cfg.train.epochs, 
        num_sanity_val_steps=cfg.train.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=[monitor], # comment when debug
        logger=logger,
        profiler="simple",
        resume_from_checkpoint=checkpoint
    )

    return trainer

def init_model(cfg):
    PointGroup = getattr(import_module("model.pointgroup"), "PointGroup")
    model = PointGroup(cfg)

    if cfg.model.use_checkpoint:
        print("=> loading pretrained checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.model.use_checkpoint, "last.ckpt")
        # import torch
        # ckpt = torch.load(checkpoint)
        # ckpt["hyper_parameters"]["cfg"]["ROOT_PATH"] = "/local-scratch/qiruiw/research/pointgroup-minkowski"
        # ckpt["hyper_parameters"]["cfg"]["DATA_PATH"] = "${ROOT_PATH}/data"
        # torch.save(ckpt, checkpoint)
        model.load_from_checkpoint(checkpoint)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-e', '--experiment', type=str, default='', help='specify experiment')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing logger...")
    logger = init_logger(cfg)
    
    print("=> initializing monitor...")
    monitor = init_monitor(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)
    
    print("=> initializing model...")
    pointgroup = init_model(cfg)

    print("=> start training...")
    trainer.fit(model=pointgroup, train_dataloader=dataloader["train"], val_dataloaders=dataloader["val"])
