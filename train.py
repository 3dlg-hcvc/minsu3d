import warnings
warnings.filterwarnings('ignore')

import argparse

import pytorch_lightning as pl

from omegaconf import OmegaConf
from importlib import import_module


def init_data(cfg):
    print("=> initialize data...")
    DATA_MODULE = import_module(cfg.data.module)
    dataloader = getattr(DATA_MODULE, cfg.data.loader)

    if cfg.general.task == "train":
        print("=> loading the train and val datasets...")
    else:
        print("=> loading the {} dataset...".format(cfg.data.split))
        
    dataset, dataloader = dataloader(cfg)
    print("=> loading dataset completed")

    return dataset, dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    args = parser.parse_args()

    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    cfg.general.task = 'train'

    PointGroup = getattr(import_module("model.pointgroup"), "PointGroup")
    pointgroup = PointGroup(cfg)

    logger = pl.loggers.TensorBoardLogger(pointgroup.root, name="logs")
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="val/{}".format(cfg.general.monitor),
        mode="min",
        # save_weights_only=True,
        dirpath=pointgroup.root,
        filename="model",
        save_last=True
    )

    trainer = pl.Trainer(
        gpus=-1, # use all available GPUs 
        accelerator='ddp', # use multiple GPUs on the same machine
        max_epochs=cfg.train.epochs, 
        num_sanity_val_steps=cfg.train.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=[monitor], # comment when debug
        logger=logger,
        profiler="simple"
    )

    dataset, dataloader = init_data(cfg)
    trainer.fit(model=pointgroup, train_dataloader=dataloader["train"], val_dataloaders=dataloader["val"])
