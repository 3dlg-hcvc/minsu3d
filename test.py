import argparse
from omegaconf import OmegaConf
from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    args = parser.parse_args()

    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    cfg.general.task = f'test'
    cfg.data.split = args.split
    cfg.data.batch_size = 1

    Solver = getattr(import_module('lib.solver'), cfg.general.solver)
    solver = Solver(cfg)

    ##### test
    solver.test(cfg.data.split)