import os
import hydra
import torch
from lib.callback import *
import pytorch_lightning as pl
from importlib import import_module
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from lib.data.data_module import DataModule


def init_callbacks(cfg):
    checkpoint_monitor = ModelCheckpoint(dirpath=cfg.general.output_root,
                                         filename=f"{cfg.model.model.module}-{cfg.data.dataset}" + "-{epoch}",
                                         **cfg.model.checkpoint_monitor)
    gpu_stats_monitor = DeviceStatsMonitor()
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, gpu_stats_monitor, gpu_cache_clean_monitor, lr_monitor]


def init_model(cfg):
    model = getattr(import_module("model"), cfg.model.model.module)(cfg.model.model, cfg.data, cfg.model.optimizer)
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
    pl.seed_everything(cfg.general.global_seed, workers=True)

    cfg.general.output_root = os.path.join(cfg.ROOT_PATH, cfg.general.output_root, cfg.data.dataset, cfg.model.model.module, "train")
    os.makedirs(cfg.general.output_root, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing logger ...")
    logger = getattr(import_module("pytorch_lightning.loggers"), cfg.model.log.module) \
        (save_dir=cfg.general.output_root, **cfg.model.log[cfg.model.log.module])

    print("==> initializing monitor ...")
    callbacks = init_callbacks(cfg)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer)

    print("==> initializing model ...")
    model = init_model(cfg)

    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()
