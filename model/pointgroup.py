import os
import torch
import functools

import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.softgroup_ops.functions import softgroup_ops
from lib.loss import SemSegLoss, PTOffsetLoss
from lib.loss.utils import get_segmented_scores
from lib.utils.eval import get_nms_instances
from model.common import ResidualBlock, VGGBlock, UBlock


class PointGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.DC = ScannetDatasetConfig(cfg)

        self.task = cfg.general.task

        in_channel = cfg.model.use_color * 3 + cfg.model.use_normal * 3 + cfg.model.use_coords * 3 + cfg.model.use_multiview * 128
        m = cfg.model.m
        D = 3
        classes = cfg.data.classes
        blocks = cfg.model.blocks
        cluster_blocks = cfg.model.cluster_blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual
        
        self.freeze_backbone = cfg.model.freeze_backbone
        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.cluster_radius = cfg.cluster.cluster_radius
        self.cluster_meanActive = cfg.cluster.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster.cluster_npoint_thre
        self.prepare_epochs = cfg.cluster.prepare_epochs

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock
        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        #### backbone
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m*c for c in blocks], sp_norm, block_reps, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        #### semantic segmentation
        self.sem_seg = nn.Linear(m, classes) # bias(default): True

        #### offset
        self.offset_net = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, 3)
        )

        #### score
        self.score_net = nn.Sequential(
            UBlock([m*c for c in cluster_blocks], sp_norm, 2, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )
        
        self.score_linear = nn.Linear(m, 1)

        self._init_criterion()
        
    
    @staticmethod
    def get_batch_offsets(batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1).int().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
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

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean_all

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

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

        clusters_voxel_coords, clusters_p2v_map, clusters_v2p_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # clusters_voxel_coords: M * (1 + 3) long
        # clusters_p2v_map: sumNPoint int, in M
        # clusters_v2p_map: M * (maxActive + 1) int, in N

        clusters_voxel_feats = pointgroup_ops.voxelization(clusters_feats, clusters_v2p_map.cuda(), mode)  # (M, C), float, cuda

        clusters_voxel_feats = ME.SparseTensor(features=clusters_voxel_feats, coordinates=clusters_voxel_coords.int().cuda())

        return clusters_voxel_feats, clusters_p2v_map
    

    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        #### backbone
        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()] # (N, m)

        #### semantic segmentation
        semantic_scores = self.sem_seg(pt_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long, {0, 1, ..., classes}
        data_dict["semantic_scores"] = semantic_scores

        #### offsets
        pt_offsets = self.offset_net(pt_feats) # (N, 3), float32
        data_dict["pt_offsets"] = pt_offsets

        if self.current_epoch > self.prepare_epochs or self.freeze_backbone:
            #### get prooposal clusters
            batch_idxs = data_dict["locs_scaled"][:, 0].int()
            
            if not self.requires_gt_mask:
                object_idxs = torch.nonzero(semantic_preds > 1, as_tuple=False).view(-1) # exclude predicted wall and floor ???
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]

                semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                proposals_batchId_shift_all = batch_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int
                # proposals_batchId_shift_all: (sumNPoint,) batch id

                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
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

            #### proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = self.clusters_voxelization(proposals_idx, proposals_offset, pt_feats, data_dict["locs"], self.score_fullscale, self.score_scale, self.mode)
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)

            #### score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long()] # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(proposals_score_feats)  # (nProposal, 1)
            data_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)
            
        return data_dict
    
    def _init_criterion(self):
        self.sem_seg_criterion = SemSegLoss(self.cfg.data.ignore_label)
        self.pt_offset_criterion = PTOffsetLoss()
        self.score_criterion = nn.BCELoss()

    
    def configure_optimizers(self):
        print("=> configure optimizer...")

        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == "Adam":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr)
        elif optim_class_name == "SGD":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr, momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented

        return [optimizer]
        
        
    def _loss(self, data_dict):
        N = data_dict["sem_labels"].shape[0]
        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        semantic_loss = self.sem_seg_criterion(data_dict["semantic_scores"], data_dict["sem_labels"])
        data_dict["semantic_loss"] = (semantic_loss, N)

        """offset loss"""
        pt_offsets, coords, instance_info, instance_ids = data_dict["pt_offsets"], data_dict["locs"], data_dict["instance_info"], data_dict["instance_ids"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        valid = (instance_ids != self.cfg.data.ignore_label).float()
        offset_norm_loss, offset_dir_loss = self.pt_offset_criterion(pt_offsets, gt_offsets, valid_mask=valid)
        data_dict["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        data_dict["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            """score loss"""
            scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
            instance_pointnum = data_dict["instance_num_point"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_ids, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.cfg.train.fg_thresh, self.cfg.train.bg_thresh)
            score_loss = self.score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            data_dict["score_loss"] = (score_loss, gt_ious.shape[0])

        """total loss"""
        loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + self.cfg.train.loss_weight[2] * offset_dir_loss
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            loss += (self.cfg.train.loss_weight[3] * score_loss)
        data_dict["total_loss"] = (loss, N)

        return data_dict
        
        
    def _feed(self, data_dict):
        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"], self.cfg.data.mode)  # (M, C), float, cuda

        data_dict = self.forward(data_dict)
        
        return data_dict

        
    def training_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self._feed(data_dict)
        data_dict = self._loss(data_dict)
        loss = data_dict["total_loss"][0]

        in_prog_bar = ["total_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("train/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    def validation_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self._feed(data_dict)
        data_dict = self._loss(data_dict)

        in_prog_bar = ["total_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("val/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=False, on_epoch=True, sync_dist=True)
                
        return data_dict

    
    def test_step(self, data_dict, idx):
        torch.cuda.empty_cache()
        
        data_dict = self._feed(data_dict)
        
        return data_dict
    

    def predict_step(self, data_dict, idx):
        torch.cuda.empty_cache()
        data_dict = self._feed(data_dict)
        self.parse_semantic_predictions(data_dict)
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            self.parse_instance_predictions(data_dict)


    # TODO: move to somewhere else
    def parse_semantic_predictions(self, data_dict):
        # from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        # NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:]  # for scannet

        ##### (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
        semantic_scores = data_dict["semantic_scores"]  # (N, nClass) float32, cuda
        semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
        semantic_class_idx = torch.tensor(self.cfg.evaluation.semantic_gt_class_idx, dtype=torch.int).cuda()  # (nClass)
        semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
        data_dict["semantic_pred_class_idx"] = semantic_pred_class_idx

        ##### save predictions
        scene_id = data_dict["scene_id"][0]
        pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
        sem_pred_path = os.path.join(pred_path, "semantic")
        os.makedirs(sem_pred_path, exist_ok=True)
        sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_id}.txt")
        np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")

    def parse_instance_predictions(self, data_dict):
        scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
        proposals_score = torch.sigmoid(scores.view(-1))  # (nProposal,) float, cuda
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        num_proposals = proposals_offset.shape[0] - 1
        N = data_dict["semantic_scores"].shape[0]

        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int).cuda()  # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        ##### score threshold & min_npoint mask
        proposals_npoint = proposals_mask.sum(1)
        proposals_thres_mask = torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH,
                                                 proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)

        proposals_score = proposals_score[proposals_thres_mask]
        proposals_mask = proposals_mask[proposals_thres_mask]

        ##### instance masks non_max_suppression
        if proposals_score.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_mask_f = proposals_mask.float()  # (nProposal, N), float, cuda
            intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float, cuda
            proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
            proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
            proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
            cross_ious = intersection / (
                        proposals_np_repeat_h + proposals_np_repeat_v - intersection)  # (nProposal, nProposal), float, cuda
            pick_idxs = get_nms_instances(cross_ious.cpu().numpy(), proposals_score.cpu().numpy(),
                                          self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].cpu().numpy()  # int, (nCluster, N)
        clusters_score = proposals_score[pick_idxs].cpu().numpy()  # float, (nCluster,)
        nclusters = clusters_mask.shape[0]

        assert "semantic_pred_class_idx" in data_dict, "make sure you parse semantic predictions at first"
        scene_id = data_dict["scene_id"][0]
        semantic_pred_class_idx = data_dict["semantic_pred_class_idx"]
        pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
        inst_pred_path = os.path.join(pred_path, "instance")
        inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
        # os.makedirs(inst_pred_path, exist_ok=True)
        os.makedirs(inst_pred_masks_path, exist_ok=True)
        cluster_ids = np.ones(shape=(N)) * -1  # id starts from 0
        with open(os.path.join(inst_pred_path, f"{scene_id}.txt"), "w") as f:
            for c_id in range(nclusters):
                cluster_i = clusters_mask[c_id]  # (N)
                cluster_ids[cluster_i == 1] = c_id
                assert np.unique(semantic_pred_class_idx[cluster_i == 1]).size == 1
                cluster_i_class_idx = semantic_pred_class_idx[cluster_i == 1][0]
                score = clusters_score[c_id]
                f.write(f"predicted_masks/{scene_id}_{c_id:03d}.txt {cluster_i_class_idx} {score:.4f}\n")
                np.savetxt(os.path.join(inst_pred_masks_path, f"{scene_id}_{c_id:03d}.txt"), cluster_i, fmt="%d")
        np.savetxt(os.path.join(inst_pred_path, f"{scene_id}.cluster_ids.txt"), cluster_ids, fmt="%d")
