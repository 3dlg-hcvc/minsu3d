import hydra
from importlib import import_module
from lib.dataset.scannet_data_module import ScanNetDataModule
import pytorch_lightning as pl
import os


def init_model(cfg):
    return getattr(import_module("model"), cfg.model.model.module)(cfg.model.model, cfg.data, cfg.model.optimizer)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    print("=> initializing trainer...")
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=1)

    cfg.general.output_root = os.path.join(cfg.ROOT_PATH, cfg.general.output_root, cfg.data.dataset,
                                           cfg.model.model.module, "test")
    os.makedirs(cfg.general.output_root, exist_ok=True)

    print("==> initializing data ...")
    data_module = ScanNetDataModule(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start inferencing...")
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.model.use_checkpoint)


if __name__ == '__main__':
    main()
