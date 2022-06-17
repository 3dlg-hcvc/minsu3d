import numpy as np


def evaluate_semantic_accuracy(pred, gt, ignore_label):

    assert gt.shape == pred.shape
    valid_idx = pred != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]

    correct = np.count_nonzero(valid_gt == valid_pred)
    whole = len(valid_gt)
    acc = correct / whole * 100
    return acc


def evaluate_semantic_miou(pred, gt, ignore_label):

    assert gt.shape == pred.shape
    valid_idx = pred != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]

    iou_list = []
    for i in np.unique(valid_gt):
        intersection = ((valid_gt == i) & (valid_pred == i)).sum()
        union = ((valid_gt == i) | (valid_pred == i)).sum()
        iou = intersection / union * 100
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    return mean_iou
