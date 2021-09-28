import warnings
warnings.filterwarnings('ignore')

import argparse

import pytorch_lightning as pl

from omegaconf import OmegaConf
from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    args = parser.parse_args()

    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    cfg.general.task = 'train'

    # Solver = getattr(import_module('lib.solver'), cfg.general.solver)
    # solver = Solver(cfg)

    # ##### train and val
    # for epoch in range(solver.start_epoch, cfg.train.epochs + 1):
    #     solver.train(epoch)

    #     if solver.check_save_condition(epoch):
    #         solver.eval(epoch)

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
        num_sanity_val_steps=-1, # validate on all val data before training 
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[monitor], # comment when debug
        logger=logger
    )

    trainer.fit(model=pointgroup)
