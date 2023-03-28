import torch


def evaluate_semantic_accuracy(pred, gt, ignore_label):
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    return torch.count_nonzero(valid_gt == valid_pred) / len(valid_gt) * 100


def evaluate_semantic_miou(pred, gt, ignore_label):
    valid_idx = gt != ignore_label
    valid_pred = pred[valid_idx]
    valid_gt = gt[valid_idx]
    unique_valid_gt = torch.unique(valid_gt)
    ious = torch.empty(shape=unique_valid_gt.shape[0], dtype=torch.float32, device=gt.device)
    for i, gt_id in enumerate(unique_valid_gt):
        intersection = torch.count_nonzero(((valid_gt == gt_id) & (valid_pred == gt_id)))
        union = torch.count_nonzero(((valid_gt == gt_id) | (valid_pred == gt_id)))
        ious[i] = intersection / union
    return ious.mean() * 100
