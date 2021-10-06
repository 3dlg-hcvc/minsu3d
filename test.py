import argparse
from omegaconf import OmegaConf
from importlib import import_module


def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.data.batch_size = 1
    cfg.general.task = "test"
    cfg.cluster.prepare_epochs = -1

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

def init_model(cfg):
    PointGroup = getattr(import_module("model.pointgroup"), "PointGroup")
    model = PointGroup(cfg)

    # "/project/3dlg-hcvc/pointgroup-minkowski/pointgroup.tar"
    checkpoint_path = cfg.model.pretrained_path
    # model.load_from_checkpoint(checkpoint_path, cfg)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.cuda()
    model.eval()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    ##### test
    # solver.test(cfg.data.split)