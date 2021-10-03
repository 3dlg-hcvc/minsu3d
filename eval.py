import warnings
warnings.filterwarnings("ignore")


import os
import torch
import argparse

import pytorch_lightning as pl

from omegaconf import OmegaConf
from importlib import import_module
from tqdm import tqdm

from data.scannet.model_util_scannet import ScannetDatasetConfig

from lib.utils.eval import APCalculator, parse_predictions, parse_groundtruths

def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.general.output_root, args.folder)
    os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.general.task = "eval"
    cfg.general.root = root
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

    # checkpoint_name = "model.ckpt"
    checkpoint_name = "last.ckpt"
    checkpoint_path = os.path.join(cfg.general.root, checkpoint_name)
    # model.load_from_checkpoint(checkpoint_path, cfg)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    model.cuda()
    model.eval()

    return model

def eval_detection(cfg, dataloader, model):
    DC = ScannetDatasetConfig(cfg)

    # config
    POST_DICT = {
        "remove_empty_box": False, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.09,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            for key in batch.keys():
                batch[key] = batch[key].cuda()

            torch.cuda.empty_cache()

            ##### prepare input and forward
            ret = model._feed(batch, 1)
            _, loss_input = model._parse_feed_ret(batch, ret)
            meter_dict = model._loss(loss_input, 1)
            model.get_bbox_iou(loss_input, batch, meter_dict)
            
            batch_pred_map_cls = parse_predictions(ret, batch, POST_DICT) 
            batch_gt_map_cls = parse_groundtruths(batch, POST_DICT) 
            for ap_calculator in AP_CALCULATOR_LIST:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

def eval_semantic(cfg, dataloader, model):
    raise NotImplementedError

def eval_instance(cfg, dataloader, model):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_scannet.yaml", help="path to config file")
    parser.add_argument("-s", "--split", type=str, default="val", help="specify data split")
    parser.add_argument("-t", "--task", type=str, choices=["semantic", "instance", "detection"], \
        help="specify task: semantic | instance | detection", required=True)
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg)
    
    print("=> start evaluating {}...".format(args.task))
    if args.task == "semantic":
        eval_semantic(cfg, dataloader[args.split], model)
    elif args.task == "instance":
        eval_instance(cfg, dataloader[args.split], model)
    elif args.task == "detection":
        eval_detection(cfg, dataloader[args.split], model)
        

