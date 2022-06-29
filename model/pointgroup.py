import os
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from lib.evaluation.instance_seg_helper import ScanNetEval, rle_encode
from lib.common_ops.functions import pointgroup_ops
from lib.common_ops.functions import common_ops
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.loss import *
from lib.loss.utils import get_segmented_scores
from lib.util.eval import get_nms_instances
from model.module import Backbone, TinyUnet
from lib.optimizer import init_optimizer
from lib.evaluation.semantic_seg_helper import *


class PointGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.DC = ScannetDatasetConfig(cfg)

        self.task = cfg.general.task

        input_channel = cfg.model.use_color * 3 + cfg.model.use_normal * 3 + cfg.model.use_coords * 3
        output_channel = cfg.model.m
        semantic_classes = cfg.data.classes
        self.instance_classes = semantic_classes - len(cfg.data.ignore_classes)


        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.cluster_meanActive = cfg.cluster.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster.cluster_npoint_thre

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        """
            Backbone Block
        """
        self.backbone = Backbone(input_channel=input_channel,
                                 output_channel=cfg.model.m,
                                 block_channels=cfg.model.blocks,
                                 block_reps=cfg.model.block_reps,
                                 sem_classes=semantic_classes)

        """
            ScoreNet Block
        """
        self.score_net = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)


    def get_batch_offsets(self, batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + torch.count_nonzero(batch_idxs == i)
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = common_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean_all

        clusters_coords_min = common_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = common_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        #### make sure the the range of scaled clusters are at most fullscale
        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        clusters_voxel_coords, clusters_p2v_map, clusters_v2p_map = common_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # clusters_voxel_coords: M * (1 + 3) long
        # clusters_p2v_map: sumNPoint int, in M
        # clusters_v2p_map: M * (maxActive + 1) int, in N

        clusters_voxel_feats = common_ops.voxelization(clusters_feats, clusters_v2p_map.cuda(), mode)  # (M, C), float, cuda

        clusters_voxel_feats = ME.SparseTensor(features=clusters_voxel_feats, coordinates=clusters_voxel_coords.int().cuda())

        return clusters_voxel_feats, clusters_p2v_map
    

    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        output_dict = {}

        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        output_dict.update(backbone_output_dict)


        if self.current_epoch > self.hparams.cfg.model.prepare_epochs or self.hparams.cfg.model.freeze_backbone:
            # get prooposal clusters
            batch_idxs = data_dict["locs_scaled"][:, 0].int()
            semantic_preds = output_dict["semantic_scores"].max(1)[1]
            if not self.requires_gt_mask:
                # set mask
                semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
                for class_label in self.hparams.cfg.data.ignore_classes:
                    semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
                object_idxs = torch.nonzero(semantic_preds_mask).view(-1)  # exclude predicted wall and floor

                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = output_dict["point_offsets"][object_idxs]

                semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

                idx_shift, start_len_shift = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.hparams.cfg.cluster.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                proposals_batchId_shift_all = batch_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int
                # proposals_batchId_shift_all: (sumNPoint,) batch id

                idx, start_len = common_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.hparams.cfg.cluster.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
                # proposals_idx = proposals_idx_shift
                # proposals_offset = proposals_offset_shift
            else:
                proposals_idx = data_dict["gt_proposals_idx"].cpu()
                proposals_offset = data_dict["gt_proposals_offset"].cpu()

            # proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_dict["point_features"],
                data_dict["locs"],
                self.score_fullscale,
                self.score_scale,
                self.mode
            )
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)

            ## score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long()] # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_branch(proposals_score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)
            
        return output_dict

    def configure_optimizers(self):
        return init_optimizer(parameters=self.parameters(), **self.cfg.train.optim)

    def _loss(self, data_dict, output_dict):
        losses = {}

        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.cfg.data.ignore_label)
        semantic_loss = sem_seg_criterion(output_dict["semantic_scores"], data_dict["sem_labels"])
        losses["semantic_loss"] = semantic_loss

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"][:, 0:3] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.cfg.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(output_dict["point_offsets"], gt_offsets, valid_mask=valid)
        losses["offset_norm_loss"] = offset_norm_loss
        losses["offset_dir_loss"] = offset_dir_loss

        total_loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + \
                     self.cfg.train.loss_weight[2] * offset_dir_loss

        if self.current_epoch > self.hparams.cfg.model.prepare_epochs:
            """score loss"""
            scores, proposals_idx, proposals_offset = output_dict["proposal_scores"]
            instance_pointnum = data_dict["instance_num_point"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), data_dict["instance_ids"], instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.cfg.train.fg_thresh, self.cfg.train.bg_thresh)
            score_criterion = ScoreLoss()
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            losses["score_loss"] = score_loss

            total_loss += self.cfg.train.loss_weight[3] * score_loss

        return losses, total_loss
        
    def _feed(self, data_dict):
        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"], data_dict["p2v_map"], self.cfg.data.mode)  # (M, C), float, cuda
        output_dict = self.forward(data_dict)
        return output_dict

    def training_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.hparams.cfg.train.clear_cache_every_n_epochs == 0:
            torch.cuda.empty_cache()

    def validation_step(self, data_dict, idx):
        torch.cuda.empty_cache()
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                       ignore_label=self.hparams.cfg.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                   ignore_label=self.hparams.cfg.data.ignore_label)
        self.log("val_accuracy/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True)

        return data_dict, output_dict

    def test_step(self, data_dict, idx):
        torch.cuda.empty_cache()
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return output_dict

    def validation_epoch_end(self, outputs):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.cfg.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            for batch, output in outputs:
                pred_instances = self._get_pred_instances(output["proposal_scores"][0].cpu(), output["proposal_scores"][1].cpu(),
                                                     output["proposal_scores"][2].cpu(), output["semantic_scores"].cpu())
                gt_instances = self._get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu())
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            evaluator = ScanNetEval(self.hparams.cfg.data.class_names)
            evaluation_result = evaluator.evaluate(all_pred_insts, all_gt_insts)
            self.log("val_accuracy/AP", evaluation_result["all_ap"], sync_dist=True)
            self.log("val_accuracy/AP_50", evaluation_result['all_ap_50%'], sync_dist=True)
            self.log("val_accuracy/AP_25", evaluation_result["all_ap_25%"], sync_dist=True)

    def test_step(self, data_dict, idx):
        torch.cuda.empty_cache()
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return output_dict

    def predict_step(self, data_dict, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return data_dict, output_dict


    def _get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = len(self.hparams.cfg.data.ignore_classes)
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins
        return gt_ins

    def _get_pred_instances(self, proposals_scores, proposals_idx, proposals_offset, semantic_scores):

        semantic_pred_labels = semantic_scores.max(1)[1]

        proposals_score = torch.sigmoid(proposals_scores.view(-1))  # (nProposal,) float
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        num_proposals = proposals_offset.shape[0] - 1
        N = semantic_scores.shape[0]

        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int, device="cpu")  # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        # score threshold & min_npoint mask
        proposals_npoint = proposals_mask.sum(1)
        proposals_thres_mask = torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH,
                                                 proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)

        proposals_score = proposals_score[proposals_thres_mask]
        proposals_mask = proposals_mask[proposals_thres_mask]

        # instance masks non_max_suppression
        if proposals_score.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_mask_f = proposals_mask.float()  # (nProposal, N), float
            intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float
            proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
            proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
            proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
            cross_ious = intersection / (
                    proposals_np_repeat_h + proposals_np_repeat_v - intersection)  # (nProposal, nProposal), float, cuda
            pick_idxs = get_nms_instances(cross_ious.numpy(), proposals_score.numpy(),
                                          self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].numpy()  # int, (nCluster, N)
        score_pred = proposals_score[pick_idxs].numpy()  # float, (nCluster,)
        nclusters = clusters_mask.shape[0]
        instances = []
        for i in range(nclusters):
            cluster_i = clusters_mask[i]  # (N)
            pred = {}
            pred['label_id'] = semantic_pred_labels[cluster_i == 1][0]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(cluster_i)
            instances.append(pred)
        return instances
