import os
import numpy as np
from tqdm import tqdm
from minsu3d.evaluation.instance_segmentation import rle_decode


def save_prediction(save_path, all_pred_insts, mapping_ids):
    inst_pred_path = os.path.join(save_path, "instance")
    inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
    os.makedirs(inst_pred_masks_path, exist_ok=True)
    scan_instance_count = {}

    for preds in tqdm(all_pred_insts, desc="==> Saving predictions ..."):
        tmp_info = []
        scan_id = preds[0]["scan_id"]
        for pred in preds:
            if scan_id not in scan_instance_count:
                scan_instance_count[scan_id] = 0
            mapped_label_id = mapping_ids[pred['label_id'] - 1]
            tmp_info.append(
                f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {mapped_label_id} {pred['conf']:.4f}\n")
            np.savetxt(
                os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                rle_decode(pred["pred_mask"]), fmt="%d")
            scan_instance_count[scan_id] += 1
        with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "w") as f:
            for mask_info in tmp_info:
                f.write(mask_info)
