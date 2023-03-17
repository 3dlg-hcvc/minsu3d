import os
import hydra
from tqdm import tqdm
from minsu3d.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
from minsu3d.evaluation.instance_segmentation import GeneralDatasetEvaluator, get_gt_instances
from minsu3d.util.io import read_gt_files_from_disk, read_pred_files_from_disk


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    split = cfg.model.inference.split
    pred_file_path = os.path.join(cfg.exp_output_root_path, "inference", cfg.model.inference.split, "predictions", "instance")

    if not os.path.exists(pred_file_path):
        print("Error: prediction files do not exist.")
        exit(-1)

    print(f"==> start evaluating {split} set ...")

    print("==> Evaluating instance segmentation ...")
    inst_seg_evaluator = GeneralDatasetEvaluator(cfg.data.class_names, -1, cfg.data.ignore_classes)

    all_pred_insts = []
    all_gt_insts = []
    all_gt_insts_bbox = []

    data_map = {
        "train": cfg.data.metadata.train_list,
        "val": cfg.data.metadata.val_list,
        "test": cfg.data.metadata.test_list
    }

    with open(data_map[split]) as f:
        scene_names = [line.strip() for line in f]

    for scan_id in tqdm(scene_names):
        scan_path = os.path.join(cfg.data.dataset_path, split, f"{scan_id}.pth")
        pred_path = os.path.join(pred_file_path, scan_id + ".txt")

        # read ground truth files
        gt_xyz, gt_sem_labels, gt_instance_ids = read_gt_files_from_disk(scan_path)
        gt_instances = get_gt_instances(gt_sem_labels, gt_instance_ids, cfg.data.ignore_classes)
        all_gt_insts.append(gt_instances)

        # read prediction files
        pred_instances = read_pred_files_from_disk(pred_path, gt_xyz, cfg.data.mapping_classes_ids, cfg.data.ignore_classes)
        all_pred_insts.append(pred_instances)

        # parse gt bounding boxes
        gt_instances_bbox = get_gt_bbox(gt_xyz, gt_instance_ids, gt_sem_labels,
                                        -1, cfg.data.ignore_classes)
        all_gt_insts_bbox.append(gt_instances_bbox)


    inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
    obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox, cfg.data.class_names,
                                               cfg.data.ignore_classes, print_result=True)


if __name__ == "__main__":
    main()
