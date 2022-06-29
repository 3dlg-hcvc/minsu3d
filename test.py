import os
import argparse
from omegaconf import OmegaConf
from importlib import import_module
from lib.dataset.scannet_data_module import ScanNetDataModule
import pytorch_lightning as pl


def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    cfg.general.experiment = args.experiment
    cfg.general.task = "predict"
    if args.pretrain is not None:
        cfg.model.use_checkpoint = args.pretrain

    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.general.experiment,
                        cfg.general.task)
    root = os.path.join(root, "predict")
    # os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.data.batch_size = 1

    cfg.general.root = root
    cfg.model.prepare_epochs = -1

    return cfg


def init_trainer():
    trainer = pl.Trainer(
        gpus=1,  # use all available GPUs
        num_nodes=1,
        profiler="simple",
    )
    return trainer


def init_model(cfg):
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/softgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('-e', '--experiment', type=str, default='', help='specify experiment')
    parser.add_argument('-p', '--pretrain', type=str, default=None, help='specify pretrained model')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing trainer...")
    trainer = init_trainer()

    print("=> initializing data...")
    data_module = ScanNetDataModule(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start inferencing...")
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.use_checkpoint)
