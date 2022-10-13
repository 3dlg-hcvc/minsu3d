import os
import torch
from tqdm import tqdm
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.evaluation.instance_segmentation import rle_decode, rle_encode


def save_prediction(save_path, all_pred_insts, mapping_ids, ignored_classes_indices):
    inst_pred_path = os.path.join(save_path, "instance")
    inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
    os.makedirs(inst_pred_masks_path, exist_ok=True)
    scan_instance_count = {}
    filtered_mapping_ids = [elem for i, elem in enumerate(mapping_ids) if i not in ignored_classes_indices]
    id_mappings = {}
    for i, label in enumerate(filtered_mapping_ids):
        id_mappings[i] = label
    for preds in tqdm(all_pred_insts, desc="==> Saving predictions ..."):
        tmp_info = []
        scan_id = preds[0]["scan_id"]
        for pred in preds:
            if scan_id not in scan_instance_count:

                scan_instance_count[scan_id] = 0
            mapped_label_id = id_mappings[pred['label_id'] - 1]
            tmp_info.append(
                f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {mapped_label_id} {pred['conf']:.4f}\n")
            np.savetxt(
                os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                rle_decode(pred["pred_mask"]), fmt="%d")
            scan_instance_count[scan_id] += 1
        with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "w") as f:
            for mask_info in tmp_info:
                f.write(mask_info)


def read_gt_files_from_disk(data_path):
    pth_file = torch.load(data_path)
    pth_file["xyz"] -= pth_file["xyz"].mean(axis=0)
    return pth_file["xyz"], pth_file["sem_labels"], pth_file["instance_ids"], pth_file


def read_pred_files_from_disk(data_path, gt_xyz, mapping_ids, ignored_classes_indices, erase_pred):

    sem_label_mapping = {}

    filtered_mapping_ids = [elem for i, elem in enumerate(mapping_ids) if i not in ignored_classes_indices]

    for i, item in enumerate(filtered_mapping_ids, 1):
        sem_label_mapping[item] = i
    pred_instances = []

    with open(data_path, "r") as f:
        for line in f:
            mask_relative_path, sem_label, confidence = line.strip().split()
            mask_path = os.path.join(os.path.dirname(data_path), mask_relative_path)
            pred_mask = np.loadtxt(mask_path, dtype=bool)
            pred = {"scan_id": os.path.basename(data_path), "label_id": sem_label_mapping[int(sem_label)],
                    "conf": float(confidence), "pred_mask": rle_encode(pred_mask)}

            # for multiscan part seg using pred obj
            if pred_mask.shape[0] < gt_xyz.shape[0]:
                pred_mask = np.concatenate((pred_mask, np.zeros(shape=gt_xyz.shape[0] - pred_mask.shape[0], dtype=bool)), axis=0)
            if erase_pred:
                pred_mask = np.zeros_like(pred_mask, dtype=bool)
            pred["pred_mask"] = rle_encode(pred_mask)

            assert len(pred_mask) == len(gt_xyz)

            # pred_xyz = gt_xyz[pred_mask]
            # pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
    return pred_instances
