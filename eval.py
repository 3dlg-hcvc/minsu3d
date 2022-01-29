import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from omegaconf import OmegaConf
from importlib import import_module


def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    # root = os.path.join(cfg.general.output_root, args.folder)
    # os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.evaluation.task = f"{args.task}.{args.split}"
    cfg.general.task = "eval"
    # cfg.general.root = root
    cfg.cluster.prepare_epochs = -1

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_scannet.yaml", help="path to config file")
    parser.add_argument("-s", "--split", type=str, default="val", help="specify data split")
    parser.add_argument("-t", "--task", type=str, choices=["semantic", "instance", "detection"], \
        help="specify task: semantic | instance | detection", required=True)
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)
    
    print("=> start evaluating {}...".format(args.task))
    if args.task == "semantic":
        from lib.evaluation import evaluate_semantic
        evaluate_semantic(cfg)
    elif args.task == "instance":
        from lib.evaluation import evaluate_instance
        evaluate_instance(cfg)
    elif args.task == "detection":
        from lib.evaluation import evaluate_detection
        evaluate_detection(cfg)
    
