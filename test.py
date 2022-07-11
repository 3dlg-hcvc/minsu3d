import hydra
from importlib import import_module
from lib.data.data_module import DataModule
import pytorch_lightning as pl
import os


def init_model(cfg):
    return getattr(import_module("model"), cfg.model.model.module)(cfg.model.model, cfg.data, cfg.model.optimizer, cfg.model.lr_decay, cfg.model.inference)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    print("=> initializing trainer...")
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=1, logger=False)

    cfg.general.output_root = os.path.join(cfg.ROOT_PATH, cfg.general.output_root, cfg.data.dataset,
                                           cfg.model.model.module, "test")
    cfg.model.inference.output_dir = os.path.join(cfg.general.output_root, "predictions")
    os.makedirs(cfg.general.output_root, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start inference...")
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()
