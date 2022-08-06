import os
import hydra
from importlib import import_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from minsu3d.callback import *
from minsu3d.data.data_module import DataModule


def init_callbacks(cfg, output_path):
    checkpoint_monitor = ModelCheckpoint(dirpath=output_path,
                                         filename=f"{cfg.model.model.module}-{cfg.data.dataset}" + "-{epoch}",
                                         **cfg.model.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, gpu_cache_clean_monitor, lr_monitor]


def init_model(cfg):
    model = getattr(import_module("minsu3d.model"), cfg.model.model.module) \
        (cfg.model.model, cfg.data, cfg.model.optimizer, cfg.model.lr_decay, None)
    if cfg.model.model.pretrained_module:
        print("=> loading pretrained module from {} ...".format(cfg.model.pretrained_module_path))
        model_dict = model.state_dict()
        ckp = torch.load(cfg.model.pretrained_module_path)
        pretrained_module_dict = {k: v for k, v in ckp.items() if k.startswith(tuple(cfg.model.pretrained_module))}
        model_dict.update(pretrained_module_dict)
        model.load_state_dict(model_dict)
    if cfg.model.model.freeze_backbone:
        for param in model.unet.parameters():
            param.requires_grad = False
    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_seed, workers=True)

    output_path = os.path.join(cfg.exp_output_root_path, cfg.data.dataset,
                               cfg.model.model.module, cfg.model.model.experiment_name, "training")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing logger ...")
    logger = getattr(import_module("pytorch_lightning.loggers"), cfg.model.log.module) \
        (save_dir=output_path, **cfg.model.log[cfg.model.log.module])

    print("==> initializing monitor ...")
    callbacks = init_callbacks(cfg, output_path)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer)

    print("==> initializing model ...")
    model = init_model(cfg)

    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()
