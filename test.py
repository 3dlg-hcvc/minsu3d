import hydra
from importlib import import_module
from minpg.lib.data.data_module import DataModule
import pytorch_lightning as pl
import os


def init_model(cfg):
    return getattr(import_module("minpg.model"), cfg.model.model.module) \
        (cfg.model.model, cfg.data, cfg.model.optimizer, cfg.model.lr_decay, cfg.model.inference)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # fix the seed
    pl.seed_everything(cfg.global_seed, workers=True)

    print("=> initializing trainer...")
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=1, logger=False)

    output_path = os.path.join(cfg.exp_output_root_path, cfg.data.dataset, cfg.model.model.module,
                               cfg.model.model.experiment_name, "inference", cfg.model.inference.split)
    cfg.model.inference.output_dir = os.path.join(output_path, "predictions")
    os.makedirs(cfg.model.inference.output_dir, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start inference...")
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()
