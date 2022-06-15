import torch.nn as nn
from lib.softgroup_ops.functions import softgroup_ops


class ClassificationLoss(nn.Module):

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset, instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = softgroup_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # overlap > thr on fg instances are positive samples
        max_iou, gt_inds = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        pos_gt_inds = gt_inds[pos_inds]

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0),), self.instance_classes)
        labels[pos_inds] = fg_instance_cls[pos_gt_inds]
        classification_loss = self.criterion(cls_scores, labels)
        return classification_loss
