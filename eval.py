import os
import hydra
from minsu3d.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
from minsu3d.evaluation.instance_segmentation import GeneralDatasetEvaluator, get_gt_instances
from minsu3d.util.io import read_gt_files_from_disk, read_pred_files_from_disk


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    split = cfg.model.model.inference.split
    pred_file_path = os.path.join(cfg.exp_output_root_path, cfg.data.dataset,
                               cfg.model.model.module, cfg.model.model.experiment_name,
                               "inference", cfg.model.model.inference.split, "predictions")

    if not os.path.exists(pred_file_path):
        print("Error: prediction files do not exist.")
        exit(-1)

    print(f"==> start evaluating {split} set ...")

    print("==> Evaluating instance segmentation ...")
    inst_seg_evaluator = GeneralDatasetEvaluator(cfg.data.class_names, cfg.data.ignore_label)

    all_pred_insts = []
    all_gt_insts = []

    data_map = {
        "train": cfg.data.metadata.train_list,
        "val": cfg.data.metadata.val_list,
        "test": cfg.data.metadata.test_list
    }

    for scan_id in data_map[split]:
        scan_path = os.path.join(cfg.SCANNETV2_PATH.splited_data, split, scan_id + cfg.data.file_suffix)
        # read ground truth files
        gt_xyz, gt_sem_labels, gt_instance_ids = read_gt_files_from_disk(scan_path)
        gt_instances = get_gt_instances(gt_sem_labels, gt_instance_ids, cfg.data.ignore_classes)
        all_gt_insts.append(gt_instances)

        # read prediction files
        pred_instances = read_pred_files_from_disk(scan_path, gt_xyz, cfg.data.mapping_classes_ids)
        all_pred_insts.append(pred_instances)

    inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
    obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox,
                      self.hparams.data.class_names, print_result=True)


if __name__ == "__main__":
    main()
