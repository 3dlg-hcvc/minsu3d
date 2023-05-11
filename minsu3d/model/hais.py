import numpy as np
import torch.nn as nn
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.common_ops.functions import hais_ops, common_ops
from minsu3d.model.general_model import get_segmented_scores
from minsu3d.model.module import TinyUnet
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.model.general_model import GeneralModel, clusters_voxelization


class HAIS(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        output_channel = cfg.model.network.m

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
        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:

            # get proposal clusters
            semantic_preds = output_dict["semantic_scores"].argmax(1).to(torch.int16)
            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in self.hparams.cfg.data.ignore_classes:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != (class_label - 1))
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)

            batch_idxs_ = data_dict["vert_batch_ids"][object_idxs]
            batch_offsets_ = torch.cumsum(torch.bincount(batch_idxs_ + 1), dim=0, dtype=torch.int32)

            offset_coords_ = data_dict["point_xyz"][object_idxs] + output_dict["point_offsets"][object_idxs]

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(
                offset_coords_, batch_idxs_, batch_offsets_,
                self.hparams.cfg.model.network.point_aggr_radius,
                self.hparams.cfg.model.network.cluster_shift_meanActive
            )

            using_set_aggr = self.hparams.cfg.model.network.using_set_aggr_in_training if self.training else self.hparams.cfg.model.network.using_set_aggr_in_testing
            cluster_idxs_kept, cluster_points_idxs_kept, proposals_offset = hais_ops.hierarchical_aggregation(
                semantic_preds[object_idxs].cpu(), offset_coords_.cpu(), idx_shift.cpu(), start_len_shift.cpu(),
                batch_idxs_.cpu(), using_set_aggr, self.hparams.cfg.data.point_num_avg,
                self.hparams.cfg.data.radius_avg, -1
            )

            cluster_idxs_kept = cluster_idxs_kept.to(self.device)
            cluster_points_idxs_kept = cluster_points_idxs_kept.to(self.device)

            proposals_offset = proposals_offset.to(self.device)
            cluster_points_idxs_kept = object_idxs[cluster_points_idxs_kept]


            # proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = clusters_voxelization(
                cluster_obj_idxs=cluster_idxs_kept,
                cluster_point_idxs=cluster_points_idxs_kept,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["point_xyz"],
                scale=self.hparams.cfg.model.network.score_scale,
                spatial_shape=self.hparams.cfg.model.network.score_fullscale,
                device=self.device
            )

            # predict instance scores
            inst_score = self.tiny_unet(proposals_voxel_feats)
            score_feats = inst_score.features[proposals_p2v_map]

            # predict mask scores
            # first linear than voxel to point, more efficient (because voxel num < point num)
            mask_scores = self.mask_branch(inst_score.features)[proposals_p2v_map]

            # predict instance scores
            if self.current_epoch > self.hparams.cfg.model.network.use_mask_filter_score_feature_start_epoch:
                mask_index_select = torch.ones_like(mask_scores)
                mask_index_select[torch.sigmoid(mask_scores) < self.hparams.cfg.model.network.mask_filter_score_feature_thre] = 0.
                score_feats = score_feats * mask_index_select
            score_feats = common_ops.roipool(score_feats, proposals_offset)  # (nProposal, C)
            scores = self.score_branch(score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, cluster_idxs_kept, cluster_points_idxs_kept, proposals_offset, mask_scores)
        return output_dict

    def _loss(self, data_dict, output_dict):
        losses = super()._loss(data_dict, output_dict)

        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            """score and mask loss"""
            scores, cluster_idxs_kept, cluster_points_idxs_kept, proposals_offset, mask_scores = output_dict['proposal_scores']

            # get iou and calculate mask label and mask loss
            mask_scores_sigmoid = torch.sigmoid(mask_scores)

            cluster_points_idxs_kept = cluster_points_idxs_kept.contiguous()

            if self.current_epoch > self.hparams.cfg.model.network.cal_iou_based_on_mask_start_epoch:
                ious = common_ops.get_mask_iou_on_pred(
                    cluster_points_idxs_kept, proposals_offset, data_dict["instance_ids"],
                    data_dict["instance_num_point"],
                    mask_scores_sigmoid.detach()
                )
            else:
                ious = common_ops.get_mask_iou_on_cluster(
                    cluster_points_idxs_kept, proposals_offset, data_dict["instance_ids"], data_dict["instance_num_point"]
                )

            mask_label, mask_label_mask = common_ops.get_mask_label(
                cluster_points_idxs_kept, proposals_offset, data_dict["instance_ids"],
                data_dict["instance_semantic_cls"], data_dict["instance_num_point"], ious, -1, 0.5
            )
            mask_label = mask_label.unsqueeze(1)
            mask_label_mask = mask_label_mask.unsqueeze(1)
            losses["mask_loss"] = nn.functional.binary_cross_entropy(
                mask_scores_sigmoid, mask_label.float(), weight=mask_label_mask, reduction="mean"
            )
            gt_scores = get_segmented_scores(
                ious.max(1)[0], self.hparams.cfg.model.network.fg_thresh, self.hparams.cfg.model.network.bg_thresh
            )
            losses["score_loss"] = nn.functional.binary_cross_entropy_with_logits(scores.view(-1), gt_scores)
        return losses

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
        losses = self._loss(data_dict, output_dict)

        # log losses
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1]
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        self.log(
            "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        
        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            point_xyz_cpu = data_dict["point_xyz"].cpu().numpy()
            instance_ids_cpu = data_dict["instance_ids"].cpu()
            sem_labels = data_dict["sem_labels"].cpu()
            pred_instances = self._get_pred_instances(
                data_dict["scan_ids"][0], point_xyz_cpu, output_dict["proposal_scores"][0].cpu(),
                output_dict["proposal_scores"][1].cpu(), output_dict["proposal_scores"][1].cpu(),
                output_dict["proposal_scores"][3].size(0) - 1,
                output_dict["proposal_scores"][4].cpu(), output_dict["semantic_scores"].cpu(),
                len(self.hparams.cfg.data.ignore_classes)
            )
            gt_instances = get_gt_instances(
                sem_labels, instance_ids_cpu, self.hparams.cfg.data.ignore_classes
            )
            gt_instances_bbox = get_gt_bbox(
                point_xyz_cpu, instance_ids_cpu.numpy(), sem_labels.numpy(), -1, self.hparams.cfg.data.ignore_classes
            )

            self.val_test_step_outputs.append((pred_instances, gt_instances, gt_instances_bbox))

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)

        semantic_accuracy = None
        semantic_mean_iou = None
        if self.hparams.cfg.model.inference.evaluate:
            semantic_predictions = output_dict["semantic_scores"].max(1)[1]
            semantic_accuracy = evaluate_semantic_accuracy(
                semantic_predictions, data_dict["sem_labels"], ignore_label=-1
            )
            semantic_mean_iou = evaluate_semantic_miou(
                semantic_predictions, data_dict["sem_labels"], ignore_label=-1
            )

        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["point_xyz"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].cpu(),
                                                      output_dict["proposal_scores"][3].size(0) - 1,
                                                      output_dict["proposal_scores"][4].cpu(),
                                                      output_dict["semantic_scores"].cpu(), len(self.hparams.cfg.data.ignore_classes))
            gt_instances = None
            gt_instances_bbox = None
            if self.hparams.cfg.model.inference.evaluate:
                gt_instances = get_gt_instances(
                    data_dict["sem_labels"].cpu(), data_dict["instance_ids"].cpu(),
                    self.hparams.cfg.data.ignore_classes
                )
                gt_instances_bbox = get_gt_bbox(data_dict["point_xyz"].cpu().numpy(),
                                                data_dict["instance_ids"].cpu().numpy(),
                                                data_dict["sem_labels"].cpu().numpy(),
                                                -1,
                                                self.hparams.cfg.data.ignore_classes)
            self.val_test_step_outputs.append(
                (semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox)
            )

    def _get_pred_instances(self, scan_id, gt_xyz, scores, proposals_idx, proposals_points_idx, num_proposals, mask_scores, semantic_scores, num_ignored_classes):
        semantic_pred_labels = semantic_scores.max(1)[1]
        scores_pred = torch.sigmoid(scores.view(-1))

        N = semantic_scores.shape[0]
        # proposals_idx: (sumNPoint, 2), [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
        # proposals_offset: (nProposal + 1)
        proposals_pred = torch.zeros((num_proposals, N), dtype=torch.bool, device="cpu")

        # outlier filtering
        _mask = mask_scores.squeeze(1) > self.hparams.cfg.model.network.test.test_mask_score_thre


        proposals_pred[proposals_idx[_mask].long(), proposals_points_idx[_mask]] = True

        # score threshold
        score_mask = (scores_pred > self.hparams.cfg.model.network.test.TEST_SCORE_THRESH)
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]

        # npoint threshold
        proposals_pointnum = torch.count_nonzero(proposals_pred, dim=1)
        npoint_mask = (proposals_pointnum >= self.hparams.cfg.model.network.test.TEST_NPOINT_THRESH)
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
