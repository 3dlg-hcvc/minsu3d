import torch
import time
import torch.nn as nn
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.common_ops.functions import hais_ops
from minsu3d.common_ops.functions import common_ops
from minsu3d.loss import MaskScoringLoss, ScoreLoss
from minsu3d.model.helper import clusters_voxelization, get_batch_offsets
from minsu3d.loss.utils import get_segmented_scores
from minsu3d.model.module import TinyUnet
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.model.general_model import GeneralModel


class HAIS(GeneralModel):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__(model, data, optimizer, lr_decay, inference)
        output_channel = model.m

        """
            Intra-instance Block
        """
        self.tiny_unet = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)
        self.mask_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, 1)
        )

    def forward(self, data_dict):
        output_dict = super().forward(data_dict)
        if self.current_epoch > self.hparams.model.prepare_epochs:
            # get proposal clusters
            batch_idxs = data_dict["vert_batch_ids"]
            semantic_preds = output_dict["semantic_scores"].max(1)[1]
            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in self.hparams.data.ignore_classes:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)  # exclude predicted wall and floor

            batch_idxs_ = batch_idxs[object_idxs].int()
            batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
            coords_ = data_dict["locs"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].cpu().int()

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                      batch_offsets_,
                                                                      self.hparams.model.point_aggr_radius,
                                                                      self.hparams.model.cluster_shift_meanActive)

            using_set_aggr = self.hparams.model.using_set_aggr_in_training if self.training else self.hparams.model.using_set_aggr_in_testing
            proposals_idx, proposals_offset = hais_ops.hierarchical_aggregation(
                semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx_shift.cpu(), start_len_shift.cpu(),
                batch_idxs_.cpu(), using_set_aggr, self.hparams.data.point_num_avg, self.hparams.data.radius_avg,
                self.hparams.data.ignore_label)

            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = clusters_voxelization(
                clusters_idx=proposals_idx,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["locs"],
                scale=self.hparams.model.score_scale,
                spatial_shape=self.hparams.model.score_fullscale,
                mode=4,
                device=self.device
            )

            # predict instance scores
            inst_score = self.tiny_unet(proposals_voxel_feats)
            score_feats = inst_score.features[proposals_p2v_map.long()]

            # predict mask scores
            # first linear than voxel to point,  more efficient  (because voxel num < point num)
            mask_scores = self.mask_branch(inst_score.features)[proposals_p2v_map.long()]

            # predict instance scores
            if self.current_epoch > self.hparams.model.use_mask_filter_score_feature_start_epoch:
                mask_index_select = torch.ones_like(mask_scores)
                mask_index_select[torch.sigmoid(mask_scores) < self.hparams.model.mask_filter_score_feature_thre] = 0.
                score_feats = score_feats * mask_index_select
            score_feats = common_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_branch(score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset, mask_scores)
        return output_dict

    def _loss(self, data_dict, output_dict):
        losses, total_loss = super()._loss(data_dict, output_dict)

        total_loss += self.hparams.model.loss_weight[0] * losses["semantic_loss"] + \
                      self.hparams.model.loss_weight[1] * losses["offset_norm_loss"]

        if self.current_epoch > self.hparams.model.prepare_epochs:
            """score and mask loss"""
            scores, proposals_idx, proposals_offset, mask_scores = output_dict['proposal_scores']

            # get iou and calculate mask label and mask loss
            mask_scores_sigmoid = torch.sigmoid(mask_scores)

            proposals_idx = proposals_idx[:, 1].cuda()
            proposals_offset = proposals_offset.cuda()

            if self.current_epoch > self.hparams.model.cal_iou_based_on_mask_start_epoch:
                ious = common_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                       data_dict["instance_num_point"],
                                                       mask_scores_sigmoid.detach())
            else:
                ious = common_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                          data_dict["instance_ids"],
                                                          data_dict["instance_num_point"])

            mask_label, mask_label_mask = common_ops.get_mask_label(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_semantic_cls"],
                                                                    data_dict["instance_num_point"], ious,
                                                                    self.hparams.data.ignore_label, 0.5)
            mask_label = mask_label.unsqueeze(1)
            mask_label_mask = mask_label_mask.unsqueeze(1)
            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_mask, reduction='mean')
            mask_loss = mask_scoring_criterion(mask_scores_sigmoid, mask_label.float())
            losses["mask_loss"] = mask_loss

            gt_ious, _ = ious.max(1)  # gt_ious: (nProposal) float, long

            gt_scores = get_segmented_scores(gt_ious, self.hparams.model.fg_thresh, self.hparams.model.bg_thresh)
            score_criterion = ScoreLoss()
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            losses["score_loss"] = score_loss
            total_loss += self.hparams.model.loss_weight[2] * score_loss + self.hparams.model.loss_weight[3] * mask_loss
        return losses, total_loss

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)
        self.log("val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["proposal_scores"][3].cpu(),
                                                      output_dict["semantic_scores"].cpu(), len(self.hparams.data.ignore_classes))
            gt_instances = get_gt_instances(data_dict["sem_labels"].cpu(), data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["locs"].cpu().numpy(),
                                            data_dict["instance_ids"].cpu().numpy(),
                                            data_dict["sem_labels"].cpu().numpy(),
                                            self.hparams.data.ignore_label,
                                            self.hparams.data.ignore_classes)

            return pred_instances, gt_instances, gt_instances_bbox

    def test_step(self, data_dict, idx):
        # prepare input and forward
        start_time = time.time()
        output_dict = self._feed(data_dict)
        end_time = time.time() - start_time
        sem_labels_cpu = data_dict["sem_labels"].cpu()
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions,
                                                       sem_labels_cpu.numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, sem_labels_cpu.numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["proposal_scores"][3].cpu(),
                                                      output_dict["semantic_scores"].cpu(), len(self.hparams.data.ignore_classes))
            gt_instances = get_gt_instances(sem_labels_cpu, data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["locs"].cpu().numpy(),
                                            data_dict["instance_ids"].cpu().numpy(),
                                            data_dict["sem_labels"].cpu().numpy(),
                                            self.hparams.data.ignore_label,
                                            self.hparams.data.ignore_classes)

            return semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox, end_time

    def _get_pred_instances(self, scan_id, gt_xyz, scores, proposals_idx, num_proposals, mask_scores, semantic_scores, num_ignored_classes):
        semantic_pred_labels = semantic_scores.max(1)[1]
        scores_pred = torch.sigmoid(scores.view(-1))

        N = semantic_scores.shape[0]
        # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu
        proposals_pred = torch.zeros((num_proposals, N), dtype=torch.bool, device="cpu")
        # (nProposal, N), int, cuda

        # outlier filtering
        _mask = mask_scores.squeeze(1) > self.hparams.model.test.test_mask_score_thre
        proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = True

        # score threshold
        score_mask = (scores_pred > self.hparams.model.test.TEST_SCORE_THRESH)
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        # semantic_id = semantic_id[score_mask]

        # npoint threshold
        proposals_pointnum = torch.count_nonzero(proposals_pred, dim=1)
        npoint_mask = (proposals_pointnum >= self.hparams.model.test.TEST_NPOINT_THRESH)
        scores_pred = scores_pred[npoint_mask]
        proposals_pred = proposals_pred[npoint_mask]

        clusters = proposals_pred.numpy()
        cluster_scores = scores_pred.numpy()

        nclusters = clusters.shape[0]

        pred_instances = []
        for i in range(nclusters):
            cluster_i = clusters[i]
            pred = {'scan_id': scan_id, 'label_id': semantic_pred_labels[cluster_i][0].item() - num_ignored_classes + 1,
                    'conf': cluster_scores[i], 'pred_mask': rle_encode(cluster_i)}
            pred_xyz = gt_xyz[cluster_i]
            pred['pred_bbox'] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
        return pred_instances
