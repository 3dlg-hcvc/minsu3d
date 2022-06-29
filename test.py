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
    
    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.test.use_exp)
    assert os.path.exists(root), "wrong experiment path"
    root = os.path.join(root, f"{args.task}")
    os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.data.batch_size = 1
    cfg.general.task = args.task
    cfg.general.root = root
    cfg.cluster.prepare_epochs = -1

    return cfg

# def init_model(cfg):
#     MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
#     model = MODEL(cfg)

#     # checkpoint_path = "/project/3dlg-hcvc/pointgroup-minkowski/pointgroup.tar"
#     checkpoint_path = "/local-scratch/qiruiw/research/pointgroup-minkowski/output/scannet/pointgroup/DETECTOR_F/detector.pth"
#     # checkpoint_path = os.path.join(cfg.general.root, checkpoint_name)
#     # model.load_from_checkpoint(checkpoint_path, cfg)

#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint)

#     model.cuda()
#     model.eval()

#     return model


# TODO: refactor
def init_trainer(cfg):
    trainer = pl.Trainer(
        gpus=-1,  # use all available GPUs
        strategy='ddp',  # use multiple GPUs on the same machine
        num_nodes=args.num_nodes,
        profiler="simple",
    )
    return trainer


def init_model(cfg):
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('-t', '--task', type=str, default='test', help='specify task')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> initializing data...")
    data_module = ScanNetDataModule(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    print("=> start inferencing...")
    trainer.predict(model=model, datamodule=data_module, ckpt_path=cfg.model.use_checkpoint)
