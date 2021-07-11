import argparse
from omegaconf import OmegaConf
from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml.yaml', help='path to config file')
    args = parser.parse_args()

    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    cfg.general.task = 'train'

    Solver = getattr(import_module('lib.solver'), cfg.general.solver)
    solver = Solver(cfg)

    ##### train and val
    for epoch in range(solver.start_epoch, cfg.train.epochs + 1):
        solver.train(epoch)

        if solver.check_save_condition(epoch):
            solver.eval(epoch)