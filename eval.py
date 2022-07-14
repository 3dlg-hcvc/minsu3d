from lib.evaluation.instance_segmentation import ScanNetEval, get_gt_instances
from lib.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
from lib.evaluation.semantic_segmentation import *
import torch
import hydra
import os
import numpy as np


def read_gt_files_from_disk(data_path):
    pth_file = torch.load(data_path)
    return pth_file["xyz"], pth_file["sem_labels"], pth_file["instance_ids"]


def read_pred_files_from_disk(data_path, gt_xyz):
    pred_instances = []
    with open(data_path, "r") as f:
        for line in f:
            mask_relative_path, sem_label, confidence = line.strip().split()
            mask_path = os.path.join(os.path.dirname(data_path), mask_relative_path)
            pred = {"scan_id": os.path.basename(data_path), "label_id": int(sem_label), "conf": float(confidence),
                    "pred_mask": np.loadtxt(mask_path, dtype=bool)}
            pred_xyz = gt_xyz[pred["pred_mask"]]
            pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
    return pred_instances


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    split = cfg.model.model.inference.split
    cfg.general.output_root = os.path.join(cfg.project_root_path, cfg.general.output_root,
                                           cfg.data.dataset, cfg.model.model.module,
                                           cfg.model.model.experiment_name, "inference", split)

    cfg.model.inference.output_dir = os.path.join(cfg.general.output_root, "predictions")

    if not os.path.exists(cfg.model.inference.output_dir):
        print("Error: prediction files do not exist.")
        exit(-1)

    print(f"==> start evaluating {split} set ...")

    print("==> Evaluating instance segmentation ...")
    inst_seg_evaluator = ScanNetEval(cfg.data.class_names)

    all_pred_insts = []
    all_gt_insts = []

    data_map = {
        "train": cfg.SCANNETV2_PATH.train_list,
        "val": cfg.SCANNETV2_PATH.val_list,
        "test": cfg.SCANNETV2_PATH.test_list
    }

    for scan_id in data_map[split]:
        scan_path = os.path.join(cfg.SCANNETV2_PATH.splited_data, split, scan_id + cfg.data.file_suffix)
        # read ground truth files
        gt_xyz, gt_sem_labels, gt_instance_ids = read_gt_files_from_disk(scan_path)
        gt_instances = get_gt_instances(gt_sem_labels, gt_instance_ids, cfg.data.ignore_classes)
        all_gt_insts.append(gt_instances)

        # read prediction files
        pred_instances = read_pred_files_from_disk(scan_path, gt_xyz)
        all_pred_insts.append(pred_instances)

    inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)


if __name__ == "__main__":
    main()
