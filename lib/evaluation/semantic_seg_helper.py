import numpy as np


def evaluate_semantic_accuracy(pred, gt, ignore_label):
    assert gt.shape == pred.shape
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    correct = np.count_nonzero(valid_gt == valid_pred)
    whole = len(valid_gt)
    acc = correct / whole * 100
    return acc


def evaluate_semantic_miou(pred, gt, ignore_label):
    assert gt.shape == pred.shape
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    iou_list = []
    for i in np.unique(valid_gt):
        intersection = np.count_nonzero(((valid_gt == i) & (valid_pred == i)))
        union = np.count_nonzero(((valid_gt == i) | (valid_pred == i)))
        iou = intersection / union * 100
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    return mean_iou


def evaluate_offset_mae(pred, gt, gt_instance_list, ignore_label):
    assert gt.shape == pred.shape
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    mae = np.abs(gt - pred).sum() / np.count_nonzero(pos_inds)
    return mae
