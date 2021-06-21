import argparse
from omegaconf import OmegaConf
from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('-t', '--task', type=str, default='', help='specify task: semantic | instance', required=True)
    args = parser.parse_args()

    base_cfg = OmegaConf.load('conf/path.yaml')
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    assert args.task != ''
    cfg.evaluation.task = f'{args.task}.{args.split}'
    cfg.general.task = 'eval'
    cfg.data.split = args.split

    if args.task == 'semantic':
        evaluate_func = getattr(import_module('lib.evaluation'), 'evaluate_semantic')
    elif args.task == 'instance':
        evaluate_func = getattr(import_module('lib.evaluation'), 'evaluate_instance')
        
    evaluate_func(cfg)
        

