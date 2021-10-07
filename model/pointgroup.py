import os
import re
import torch
import functools
import random

import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from importlib import import_module

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.eval import get_nms_instances
from lib.utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from lib.utils.bbox import get_3d_box_batch, get_aabb3d_iou_batch, get_3d_box

from model.common import ResidualBlock, VGGBlock, UBlock


class PointGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.DC = ScannetDatasetConfig(cfg)

        self.task = cfg.general.task
        self.total_epoch = cfg.train.epochs

        in_channel = cfg.model.use_color * 3 + cfg.model.use_normal * 3 + cfg.model.use_coords * 3 + cfg.model.use_multiview * 128
        m = cfg.model.m
        D = 3
        classes = cfg.data.classes
        blocks = cfg.model.blocks
        cluster_blocks = cfg.model.cluster_blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual
        
        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.cluster_radius = cfg.cluster.cluster_radius
        self.cluster_meanActive = cfg.cluster.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster.cluster_npoint_thre
        self.freeze_backbone = cfg.cluster.freeze_backbone

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        self.prepare_epochs = cfg.cluster.prepare_epochs

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
        
        # self.proposal_mlp = nn.Sequential(
        #     nn.Linear(m, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 128)
        # )
        
        if cfg.model.pred_bbox:
            num_class = cfg.model.num_bbox_class
            num_heading_bin = cfg.model.num_heading_bin
            num_size_cluster = cfg.model.num_size_cluster
            self.bbox_regressor = nn.Sequential(
                nn.Linear(m, m, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, m, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, 3+num_heading_bin*2+num_size_cluster*4+num_class)
            )
        
        self.score_linear = nn.Linear(m, 1)
        # self.score_linear = nn.Linear(128, 1)

        self._init_random_seed()

        # NOTE do NOT manually load the pretrained weights during training!!!
        # if cfg.general.task == "test":
        #     self._load_pretrained_model()
        
    
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

        #### extract bbox info for each cluster
        clusters_size = clusters_coords_max - clusters_coords_min # (nCluster, 3), float
        clusters_center = (clusters_coords_max + clusters_coords_min) / 2 + clusters_coords_mean # (nCluster, 3), float
        ####

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

        return clusters_voxel_feats, clusters_p2v_map, (clusters_center, clusters_size)


    def decode_bbox_prediction(self, encoded_bbox, data_dict):
        """
        decode the predicted parameters for the bounding boxes
        """
        num_heading_bin = self.cfg.model.num_heading_bin
        num_size_cluster = self.cfg.model.num_size_cluster
        # encoded_bbox = encoded_bbox.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        num_proposal = encoded_bbox.shape[0]

        # objectness_scores = encoded_bbox[:,:,0:2]

        base_xyz = data_dict["proposal_info"][0] # (num_proposal, 3)
        center = base_xyz + encoded_bbox[:, :3] # (num_proposal, 3)

        heading_scores = encoded_bbox[:, 3:3+num_heading_bin] # (num_proposal, 1)
        heading_residuals_normalized = encoded_bbox[:, 3+num_heading_bin:3+num_heading_bin*2] # (num_proposal, 1)
        
        size_scores = encoded_bbox[:, 3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = encoded_bbox[:, 3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([num_proposal, num_size_cluster, 3]) # (num_proposal, num_size_cluster, 3)
        
        sem_cls_scores = encoded_bbox[:, 3+num_heading_bin*2+num_size_cluster*4:] # num_proposalx18

        # store
        # data_dict["objectness_scores"] = objectness_scores
        data_dict["center"] = center
        data_dict["heading_scores"] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict["heading_residuals_normalized"] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict["heading_residuals"] = heading_residuals_normalized * (np.pi/num_heading_bin) # (num_proposal, num_heading_bin)
        data_dict["size_scores"] = size_scores
        data_dict["size_residuals_normalized"] = size_residuals_normalized
        data_dict["size_residuals"] = size_residuals_normalized * torch.from_numpy(self.DC.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0)
        data_dict["sem_cls_scores"] = sem_cls_scores

        return data_dict
    
    
    def convert_stack_to_batch(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        max_num_proposal = self.cfg.model.max_num_proposal
        data_dict["proposal_feats_batched"] = torch.zeros(batch_size, max_num_proposal, self.cfg.model.m).type_as(data_dict["proposal_feats"])
        data_dict["proposal_bbox_batched"] = torch.zeros(batch_size, max_num_proposal, 8, 3).type_as(data_dict["proposal_feats"])
        data_dict["proposal_center_batched"] = torch.zeros(batch_size, max_num_proposal, 3).type_as(data_dict["proposal_feats"])
        data_dict["proposal_sem_cls_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])
        data_dict["proposal_scores_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])
        data_dict["proposal_batch_mask"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])

        proposal_bbox = data_dict["proposal_crop_bboxes"].detach().cpu().numpy()
        proposal_bbox = get_3d_box_batch(proposal_bbox[:, :3], proposal_bbox[:, 3:6], proposal_bbox[:, 6]) # (nProposals, 8, 3)
        proposal_bbox_tensor = torch.tensor(proposal_bbox).type_as(data_dict["proposal_feats"])

        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(data_dict["proposals_batchId"] == b).squeeze(-1)
            pred_num = len(proposal_batch_idx)
            pred_num = pred_num if pred_num < max_num_proposal else max_num_proposal
            
            # NOTE proposals should be truncated if more than max_num_proposal proposals are predicted
            data_dict["proposal_feats_batched"][b, :pred_num, :] = data_dict["proposal_feats"][proposal_batch_idx][:pred_num]
            data_dict["proposal_bbox_batched"][b, :pred_num, :, :] = proposal_bbox_tensor[proposal_batch_idx][:pred_num]
            data_dict["proposal_center_batched"][b, :pred_num, :] = data_dict["proposal_crop_bboxes"][proposal_batch_idx, :3][:pred_num]
            data_dict["proposal_sem_cls_batched"][b, :pred_num] = data_dict["proposal_crop_bboxes"][proposal_batch_idx, 7][:pred_num]
            data_dict["proposal_scores_batched"][b, :pred_num] = data_dict["proposal_objectness_scores"][proposal_batch_idx][:pred_num]
            data_dict["proposal_batch_mask"][b, :pred_num] = 1

        return data_dict


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
                object_idxs = torch.nonzero(semantic_preds > 0, as_tuple=False).view(-1)
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
                proposals_batchId_all = batch_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int
                # proposals_batchId_all: (sumNPoint,) batch id

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
                proposals_batchId_all = torch.cat((proposals_batchId_all, proposals_batchId_shift_all[1:])) # (sumNPoint,)
                # proposals_idx = proposals_idx_shift
                # proposals_offset = proposals_offset_shift
                # proposals_batchId_all = proposals_batchId_shift_all
            else:
                proposals_idx = data_dict["gt_proposals_idx"]
                proposals_offset = data_dict["gt_proposals_offset"]
                proposals_batchId_all = batch_idxs[proposals_idx[:, 1].long()].int() # (sumNPoint,)

            #### proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map, (proposals_center, proposals_size) = self.clusters_voxelization(proposals_idx, proposals_offset, pt_feats, data_dict["locs"], self.score_fullscale, self.score_scale, self.mode)
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)
            # proposals_center / proposals_size: (nProposals, 3)

            #### score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long()] # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset.cuda())  # (nProposal, C)
            # proposals_score_feats = self.proposal_mlp(proposals_score_feats) # (nProposal, 128)
            scores = self.score_linear(proposals_score_feats)  # (nProposal, 1)
            data_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)

            ############ extract batch related features and bbox #############
            num_proposals = proposals_offset.shape[0] - 1
            
            proposals_npoint = torch.zeros(num_proposals).cuda()
            for i in range(num_proposals):
                proposals_npoint[i] = (proposals_idx[:, 0] == i).sum()
            thres_mask = torch.logical_and(torch.sigmoid(scores.view(-1)) > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH) # (nProposal,)
            data_dict["proposals_npoint"] = proposals_npoint
            data_dict["proposal_thres_mask"] = thres_mask
            
            proposals_batchId = proposals_batchId_all[proposals_offset[:-1].long()] # (nProposal,)
            proposals_batchId = proposals_batchId[thres_mask]
            data_dict["proposals_batchId"] = proposals_batchId # (nProposal,)
            data_dict["proposal_feats"] = proposals_score_feats[thres_mask]
            data_dict["proposal_objectness_scores"] = torch.sigmoid(scores.view(-1))[thres_mask]
            
            if self.cfg.model.crop_bbox:
                proposal_crop_bboxes = torch.zeros(num_proposals, 9).cuda() # (nProposals, center+size+heading+label+score)
                proposal_crop_bboxes[:, :3] = proposals_center
                proposal_crop_bboxes[:, 3:6] = proposals_size
                proposal_crop_bboxes[:, 7] = semantic_preds[proposals_idx[proposals_offset[:-1].long(), 1].long()]
                proposal_crop_bboxes[:, 8] = torch.sigmoid(scores.view(-1))
                proposal_crop_bboxes = proposal_crop_bboxes[thres_mask]
                data_dict["proposal_crop_bboxes"] = proposal_crop_bboxes

            if self.cfg.model.pred_bbox:
                encoded_pred_bbox = self.bbox_regressor(proposals_score_feats) # (nProposal, 3+num_heading_bin*2+num_size_cluster*4+num_class)
                encoded_pred_bbox = encoded_pred_bbox[thres_mask]
                data_dict["proposal_info"] = (proposals_center[thres_mask], proposals_size[thres_mask])
                data_dict = self.decode_bbox_prediction(encoded_pred_bbox)
            
        return data_dict
                

    def _init_random_seed(self):
        print("=> setting random seed...")
        if self.cfg.general.manual_seed:
            random.seed(self.cfg.general.manual_seed)
            np.random.seed(self.cfg.general.manual_seed)
            torch.manual_seed(self.cfg.general.manual_seed)
            torch.cuda.manual_seed_all(self.cfg.general.manual_seed)

    
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
    
    
    # NOTE deprecated - will be removed soon
    def _resume_from_checkpoint(self):
        if self.cfg.model.use_checkpoint:
            print("=> restoring checkpoint from {} ...".format(self.cfg.model.use_checkpoint))
            self.start_epoch = self.restore_checkpoint()      # resume from the latest epoch, or specify the epoch to restore
        else: 
            self.start_epoch = 1

            if self.cfg.model.pretrained_module:
                self._load_pretrained_module()
    
    
    # NOTE deprecated - will be removed soon
    def _load_pretrained_module(self):
        self.logger.info("=> loading pretrained {}...".format(self.cfg.model.pretrained_module))
        for i, module_name in enumerate(self.cfg.model.pretrained_module):
            module = getattr(self, module_name)
            ckp = torch.load(self.cfg.model.pretrained_module_path[i])
            module.load_state_dict(ckp)
            
        if self.cfg.cluster.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    # NOTE deprecated - will be removed soon
    def _load_pretrained_model(self):
        pretrained_path = self.cfg.model.pretrained_path
        self.logger.info("=> load pretrained model from {} ...".format(pretrained_path))
        
        model_state = torch.load(pretrained_path)
        self.load_state_dict(model_state["state_dict"])
        
        self.start_epoch = self.cfg.model.resume_epoch
    
    
    # NOTE deprecated - will be removed soon
    def restore_checkpoint(self):
        ckp_filename = os.path.join(self.cfg.general.output_root, self.cfg.model.use_checkpoint, "last.ckpt")
        assert os.path.isfile(ckp_filename), "Invalid checkpoint file: {}".format(ckp_filename)

        checkpoint = torch.load(ckp_filename)
        epoch = checkpoint["epoch"]
        
        print("=> relocating epoch at {} ...".format(epoch))

        self.load_state_dict(checkpoint["state_dict"])
        
        return checkpoint["epoch"]
        
        
    def _loss(self, loss_input):

        def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
            """
            :param scores: (N), float, 0~1
            :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
            """
            fg_mask = scores > fg_thresh
            bg_mask = scores < bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            segmented_scores = (fg_mask > 0).float()
            k = 1 / (fg_thresh - bg_thresh)
            b = bg_thresh / (bg_thresh - fg_thresh)
            segmented_scores[interval_mask] = scores[interval_mask] * k + b

            return segmented_scores

        loss_dict = {}

        """semantic loss"""
        semantic_scores, semantic_labels = loss_input["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.data.ignore_label)
        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_dict["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """offset loss"""
        pt_offsets, coords, instance_info, instance_ids = loss_input["pt_offsets"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_ids != self.cfg.data.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_dict["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        loss_dict["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            """score loss"""
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_input["proposal_scores"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_ids, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.cfg.train.fg_thresh, self.cfg.train.bg_thresh)

            score_criterion = nn.BCELoss(reduction="none")
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_dict["score_loss"] = (score_loss, gt_ious.shape[0])

        """total loss"""
        loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + self.cfg.train.loss_weight[2] * offset_dir_loss
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            loss += (self.cfg.train.loss_weight[3] * score_loss)
        loss_dict["total_loss"] = (loss, semantic_labels.shape[0])

        return loss_dict
        
        
    def _feed(self, data_dict):
        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"], self.cfg.data.mode)  # (M, C), float, cuda

        data_dict = self.forward(data_dict)
        data_dict = self.convert_stack_to_batch(data_dict)
        
        return data_dict


    def _parse_feed_ret(self, data_dict):
        # semantic_scores = data_dict["semantic_scores"] # (N, nClass) float32, cuda
        # pt_offsets = data_dict["pt_offsets"]           # (N, 3), float32, cuda
        
        # preds = {}
        loss_input = {}
        
        # preds["semantic"] = semantic_scores
        # preds["pt_offsets"] = pt_offsets
        if self.task != "test":
            loss_input["semantic_scores"] = (semantic_scores, data_dict["sem_labels"])
            loss_input["pt_offsets"] = (pt_offsets, data_dict["locs"], data_dict["instance_info"], data_dict["instance_ids"])
        
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
            # preds["score"] = scores
            # preds["proposals"] = (proposals_idx, proposals_offset)
            # preds["proposals_npoint"] = data_dict["proposals_npoint"]
            # preds["proposal_thres_mask"] = data_dict["proposal_thres_mask"]
            # if self.cfg.model.crop_bbox:
            #     preds["proposal_crop_bboxes"] = data_dict["proposal_crop_bboxes"] # (nProposals, center+size+heading+label+score)
            
            if self.task != "test":
                loss_input["proposal_scores"] = (scores, proposals_idx, proposals_offset, data_dict["instance_num_point"])
                # scores: (nProposal, 1) float, cuda
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                loss_input["proposal_thres_mask"] = data_dict["proposal_thres_mask"]
                loss_input["proposals_batchId"] = data_dict["proposals_batchId"]
                if self.cfg.model.crop_bbox:
                    loss_input["proposal_crop_bboxes"] = data_dict["proposal_crop_bboxes"]
                if self.cfg.model.pred_bbox:
                    loss_input["proposal_pred_bboxes"] = (data_dict["center"], data_dict["heading_scores"], data_dict["heading_residuals_normalized"], data_dict["heading_residuals"], data_dict["size_scores"], data_dict["size_residuals_normalized"], data_dict["size_residuals"], data_dict["sem_cls_scores"])
        
        # return preds, loss_input
        return loss_input
        

    def get_bbox_iou(self, loss_input, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        proposal_thres_mask = loss_input["proposal_thres_mask"]
        instance_offsets = data_dict["instance_offsets"].detach().cpu().numpy()
        proposals_batchId = loss_input["proposals_batchId"]
        
        # parse gt_bbox
        gt_bboxes = data_dict["gt_bbox"]
        
        if self.cfg.model.crop_bbox:
            # parse crop_bbox
            proposal_crop_bboxes = loss_input["proposal_crop_bboxes"]
            proposal_crop_bboxes = proposal_crop_bboxes.detach().cpu().numpy() # (nProposals, center+size+heading+label)
            proposal_crop_bbox_corners = get_3d_box_batch(proposal_crop_bboxes[:, 0:3], proposal_crop_bboxes[:, 3:6], proposal_crop_bboxes[:, 6]) # (nProposals, 8, 3)
            num_proposal = proposal_crop_bboxes.shape[0] 
            
            crop_bbox_ious = np.zeros(num_proposal)
            for b in range(batch_size):
                proposal_batch_idx = torch.nonzero(proposals_batchId == b)
                pred_num = len(proposal_batch_idx) #pred_batch_end - pred_batch_start # N
                gt_batch_start, gt_batch_end = instance_offsets[b], instance_offsets[b+1]
                gt_num = gt_batch_end - gt_batch_start # M
                for i in range(pred_num):
                    crop_bbox_iou = get_aabb3d_iou_batch(np.tile(proposal_crop_bbox_corners[proposal_batch_idx[i]], (gt_num, 1, 1)), gt_bboxes[gt_batch_start:gt_batch_end])
                    crop_bbox_ious[proposal_batch_idx[i]] = np.max(crop_bbox_iou)
            data_dict["crop_bbox_iou"] = (crop_bbox_ious.mean(), num_proposal)
            data_dict["proposal_crop_bbox_corners"] = proposal_crop_bbox_corners
            
        if self.cfg.model.pred_bbox:
            # parse pred_bbox
            pred_center, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals, sem_cls_scores = loss_input["proposal_pred_bboxes"]
            pred_center = pred_center.detach().cpu().numpy()
            pred_heading_class = torch.argmax(heading_scores, -1).detach().cpu().numpy() # num_proposal
            pred_heading_residual = torch.gather(heading_scores, 1, pred_heading_class.unsqueeze(-1)).squeeze(1).detach().cpu().numpy() # num_proposal
            pred_size_class = torch.argmax(size_scores, -1).detach().cpu().numpy() # num_proposal
            pred_size_residual = torch.gather(size_residuals, 1, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1).detach().cpu().numpy() # num_proposal,3

            num_proposal = pred_center.shape[0] 
            proposal_pred_bbox_corners = np.zeros((num_proposal, 8, 3))
            for j in range(num_proposal):
                heading_angle = self.DC.class2angle(pred_heading_class[j], pred_heading_residual[j])
                box_size = self.DC.class2size(int(pred_size_class[j]), pred_size_residual[j])
                proposal_pred_bbox_corners[j] = get_3d_box(pred_center[j,:], box_size, heading_angle)
            
            pred_bbox_ious = np.zeros(num_proposal)
            for b in range(batch_size):
                proposal_batch_idx = torch.nonzero(proposals_batchId == b)
                pred_num = len(proposal_batch_idx) #pred_batch_end - pred_batch_start # N
                gt_batch_start, gt_batch_end = instance_offsets[b], instance_offsets[b+1]
                gt_num = gt_batch_end - gt_batch_start # M
                for i in range(pred_num):
                    pred_bbox_iou = get_aabb3d_iou_batch(np.tile(proposal_pred_bbox_corners[proposal_batch_idx[i]], (gt_num, 1, 1)), gt_bboxes[gt_batch_start:gt_batch_end])
                    pred_bbox_ious[proposal_batch_idx[i]] = np.max(pred_bbox_iou)
            data_dict["pred_bbox_iou"] = (pred_bbox_ious.mean(), num_proposal)
            data_dict["proposal_pred_bbox_corners"] = proposal_pred_bbox_corners
            
        if self.cfg.model.crop_bbox and self.cfg.model.pred_bbox:
            assert proposal_crop_bbox_corners.shape == proposal_pred_bbox_corners.shape
            ious = get_aabb3d_iou_batch(proposal_crop_bbox_corners, proposal_pred_bbox_corners)
            data_dict["pred_crop_bbox_iou"] = (ious.mean(), num_proposal)

        
    def training_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self._feed(data_dict)
        loss_input = self._parse_feed_ret(data_dict)
        loss_dict = self._loss(loss_input)
        loss = loss_dict["total_loss"][0]

        in_prog_bar = ["total_loss"]
        for key, value in loss_dict.items():
            if "loss" in key:
                self.log("train/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    def validation_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self._feed(data_dict)
        loss_input = self._parse_feed_ret(data_dict)
        loss_dict = self._loss(loss_input)

        in_prog_bar = ["total_loss"]
        for key, value in loss_dict.items():
            if "loss" in key:
                self.log("val/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=False, on_epoch=True, sync_dist=True)


    #### inference only works with batch_size = 1 for now
    def inference(self, dataloader, split="val"):
        POST_DICT = {
            "remove_empty_box": False, 
            "use_3d_nms": True, 
            "nms_iou": 0.25,
            "use_old_type_nms": False, 
            "cls_nms": True, 
            "per_class_proposal": True,
            "conf_thresh": 0.09,
        }
        self.eval()
        
        print(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(dataloader)):
                # import pdb; pdb.set_trace()
                for key in data_dict.keys():
                    if isinstance(data_dict[key], tuple): continue
                    if isinstance(data_dict[key], dict): continue
                    if isinstance(data_dict[key], list): continue
                    data_dict[key] = data_dict[key].cuda()
                
                # N = data_dict["feats"].shape[0]
                # scene_id = data_dict["scene_id"][0] #self.dataset[split].scene_names[i]
                
                data_dict = self._feed(data_dict)
                # preds, _ = self._parse_feed_ret(data_dict)
                # preds["scene_id"] = scene_id
                
                ##### parse semantic predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
                self.parse_semantic_predictions(data_dict, save_preds=True)
                # semantic_scores = preds["semantic"]  # (N, nClass) float32, cuda, 0: unannotated
                # semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
                # semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
                # semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
                
                ##### parse instance predictions
                if self.current_epoch > self.cfg.cluster.prepare_epochs:
                    self.parse_instance_predictions(data_dict, save_preds=True)
                    self.parse_bbox_predictions(data_dict, POST_DICT, save_preds=True)
                    
                    # proposals_score = torch.sigmoid(preds["score"].view(-1)) # (nProposal,) float, cuda
                    # proposals_idx, proposals_offset = preds["proposals"]
                    # # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                    # # proposals_offset: (nProposal + 1), int, cpu

                    # num_proposals = proposals_offset.shape[0] - 1
                    # # N = semantic_scores.shape[0]
                    
                    # proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int).cuda() # (nProposal, N), int, cuda
                    # proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                    
                    # ##### score threshold & min_npoint mask
                    # proposals_npoint = preds["proposals_npoint"] #proposals_mask.sum(1)
                    # proposals_thres_mask = preds["proposal_thres_mask"] #torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)
                    
                    # proposals_score = proposals_score[proposals_thres_mask]
                    # proposals_mask = proposals_mask[proposals_thres_mask]
                    
                    # ##### instance masks non_max_suppression
                    # if proposals_score.shape[0] == 0:
                    #     pick_idxs = np.empty(0)
                    # else:
                    #     proposals_mask_f = proposals_mask.float()  # (nProposal, N), float, cuda
                    #     intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float, cuda
                    #     proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
                    #     proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
                    #     proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
                    #     cross_ious = intersection / (proposals_np_repeat_h + proposals_np_repeat_v - intersection) # (nProposal, nProposal), float, cuda
                    #     pick_idxs = get_nms_instances(cross_ious.cpu().numpy(), proposals_score.cpu().numpy(), self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

                    # clusters_mask = proposals_mask[pick_idxs].cpu().numpy() # int, (nCluster, N)
                    # clusters_score = proposals_score[pick_idxs].cpu().numpy() # float, (nCluster,)
                    # nclusters = clusters_mask.shape[0]
                
                ##### save predictions
                # pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
                
                # sem_pred_path = os.path.join(pred_path, "semantic")
                # os.makedirs(sem_pred_path, exist_ok=True)
                # sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_id}.txt")
                # np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")
                
                # if self.current_epoch > self.cfg.cluster.prepare_epochs:
                #     inst_pred_path = os.path.join(pred_path, "instance")
                #     inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
                #     os.makedirs(inst_pred_path, exist_ok=True)
                #     os.makedirs(inst_pred_masks_path, exist_ok=True)
                #     cluster_ids = np.ones(shape=(N)) * -1 # id starts from 0
                #     with open(os.path.join(inst_pred_path, f"{scene_id}.txt"), "w") as f:
                #         for c_id in range(nclusters):
                #             cluster_i = clusters_mask[c_id]  # (N)
                #             cluster_ids[cluster_i == 1] = c_id
                #             assert np.unique(semantic_pred_class_idx[cluster_i == 1]).size == 1
                #             cluster_i_class_idx = semantic_pred_class_idx[cluster_i == 1][0]
                #             score = clusters_score[c_id]
                #             f.write(f"predicted_masks/{scene_id}_{c_id:03d}.txt {cluster_i_class_idx} {score:.4f}\n")
                #             np.savetxt(os.path.join(inst_pred_masks_path, f"{scene_id}_{c_id:03d}.txt"), cluster_i, fmt="%d")
                #     np.savetxt(os.path.join(inst_pred_path, f"{scene_id}.cluster_ids.txt"), cluster_ids, fmt="%d")


    def parse_semantic_predictions(self, data_dict, save_preds=True):
        from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:] # for scannet
        
        ##### parse semantic predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
        semantic_scores = data_dict["semantic_scores"]  # (N, nClass) float32, cuda, 0: unannotated
        semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
        semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
        semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
        data_dict["semantic_pred_class_idx"] = semantic_pred_class_idx
        
        ##### save predictions
        if save_preds:
            scene_id = data_dict["scene_id"][0]
            pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
            sem_pred_path = os.path.join(pred_path, "semantic")
            os.makedirs(sem_pred_path, exist_ok=True)
            sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_id}.txt")
            np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")
            
            
    def parse_instance_predictions(self, data_dict, save_preds=True):
        scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
        proposals_score = torch.sigmoid(scores.view(-1)) # (nProposal,) float, cuda
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        num_proposals = proposals_offset.shape[0] - 1
        N = data_dict["semantic_scores"].shape[0]
        
        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int).cuda() # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        
        ##### score threshold & min_npoint mask
        proposals_npoint = data_dict["proposals_npoint"] #proposals_mask.sum(1)
        proposals_thres_mask = data_dict["proposal_thres_mask"] #torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)
        
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
            cross_ious = intersection / (proposals_np_repeat_h + proposals_np_repeat_v - intersection) # (nProposal, nProposal), float, cuda
            pick_idxs = get_nms_instances(cross_ious.cpu().numpy(), proposals_score.cpu().numpy(), self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].cpu().numpy() # int, (nCluster, N)
        clusters_score = proposals_score[pick_idxs].cpu().numpy() # float, (nCluster,)
        nclusters = clusters_mask.shape[0]
        
        if save_preds:
            assert "semantic_pred_class_idx" in data_dict, "make sure you parse semantic predictions at first"
            scene_id = data_dict["scene_id"][0]
            semantic_pred_class_idx = data_dict["semantic_pred_class_idx"]
            pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
            inst_pred_path = os.path.join(pred_path, "instance")
            inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
            # os.makedirs(inst_pred_path, exist_ok=True)
            os.makedirs(inst_pred_masks_path, exist_ok=True)
            cluster_ids = np.ones(shape=(N)) * -1 # id starts from 0
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
            
    
    def parse_bbox_predictions(self, data_dict, config_dict, save_preds=True):
        """ Parse predictions to OBB parameters and suppress overlapping boxes
        Args:
            data_dict: dict
            config_dict: dict
                {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
                use_old_type_nms, conf_thresh, per_class_proposal}
        """
        batch_size = len(data_dict["batch_offsets"]) - 1
        
        num_proposal = data_dict["proposal_crop_bboxes"].shape[0]
        pred_bboxes = data_dict["proposal_crop_bboxes"].detach().cpu().numpy() # (nProposals, 9)
        bbox_corners = get_3d_box_batch(pred_bboxes[:, :3], pred_bboxes[:, 3:6], pred_bboxes[:, 6]) # (nProposals, 8, 3) numpy
        pred_sem_cls = pred_bboxes[:, 7] - 2
        pred_sem_cls[pred_sem_cls < 0] = 17
        obj_prob = pred_bboxes[:, 8]

        proposals_batchId = data_dict['proposals_batchId']
        # batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)

        nonempty_box_mask = np.ones((num_proposal,))
        if config_dict['remove_empty_box']:
            # -------------------------------------
            # Remove predicted boxes without any point within them..
            batch_pc = data_dict['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
            for i in range(batch_size):
                pc = batch_pc[i,:,:] # (N,3)
                for j in range(K):
                    box3d = bbox_corners[i,j,:,:] # (8,3)
                    # box3d = flip_axis_to_depth(box3d)
                    pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i,j] = 0
            # -------------------------------------

        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(proposals_batchId == b).view(-1).detach().cpu().numpy()
            num_proposal_batch = len(proposal_batch_idx)
            bbox_corners_batch = bbox_corners[proposal_batch_idx]
            obj_prob_batch = obj_prob[proposal_batch_idx]
            pred_sem_cls_batch = pred_sem_cls[proposal_batch_idx]
            pred_mask = np.zeros((num_proposal_batch,))
            if not config_dict['use_3d_nms']:
                # ---------- NMS input: pred_with_prob in (B,K,7) -----------
                boxes_2d_with_prob = np.zeros((num_proposal_batch, 5))
                for j in range(num_proposal_batch):
                    boxes_2d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_2d_with_prob[j,2] = np.max(bbox_corners_batch[j,:,0])
                    boxes_2d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,2])
                    boxes_2d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,2])
                    boxes_2d_with_prob[j,4] = obj_prob_batch[j]
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
            elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
                # ---------- NMS input: pred_with_prob in (B,K,7) -----------
                boxes_3d_with_prob = np.zeros((num_proposal_batch, 7))
                for j in range(num_proposal_batch):
                    boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,6] = obj_prob_batch[j]
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
            elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
                # ---------- NMS input: pred_with_prob in (B,K,8) -----------
                boxes_3d_with_prob = np.zeros((num_proposal_batch, 8))
                for j in range(num_proposal_batch):
                    boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,6] = obj_prob_batch[j]
                    boxes_3d_with_prob[j,7] = pred_sem_cls_batch[j] # only suppress if the two boxes are of the same class!!
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
                
            bbox_corners_batch = bbox_corners_batch[pred_mask == 1]
            pred_sem_cls_batch = pred_sem_cls_batch[pred_mask == 1]
            obj_prob_batch = obj_prob_batch[pred_mask == 1]
            
            if save_preds:
                scene_id = data_dict["scene_id"][0]
                pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
                bbox_path = os.path.join(pred_path, "detection")
                os.makedirs(bbox_path, exist_ok=True)
                torch.save({"pred_bbox": bbox_corners_batch, "pred_sem_cls": pred_sem_cls_batch, "pred_obj_prob": obj_prob_batch, "gt_bbox": data_dict['gt_bbox'][b].detach().cpu().numpy(), "gt_bbox_label": data_dict['gt_bbox_label'].detach().cpu().numpy(), "gt_sem_cls": data_dict['sem_cls_label'].detach().cpu().numpy()}, os.path.join(bbox_path, f"{scene_id}.pth"))
            
            # if config_dict['per_class_proposal']:
            #     cur_list = []
            #     for ii in range(self.DC.num_class):
            #         cur_list += [(ii, bbox_corners_batch[j], obj_prob_batch[j]) \
            #             for j in range(num_proposal_batch) if pred_mask[j]==1 and pred_sem_cls_batch[j]==ii and obj_prob_batch[j]>config_dict['conf_thresh']]
            #     batch_pred_map_cls.append(cur_list)
            # else:
            #     batch_pred_map_cls.append([(pred_sem_cls_batch[j], bbox_corners_batch[j], obj_prob_batch[j]) \
            #         for j in range(num_proposal_batch) if pred_mask[j]==1 and obj_prob_batch[j]>config_dict['conf_thresh']])
        
        # data_dict['batch_pred_map_cls'] = batch_pred_map_cls

        # return batch_pred_map_cls
    
    # def parse_bbox_groundtruths(self, data_dict, config_dict):
    #     """ Parse groundtruth labels to OBB parameters.
        
    #     Args:
    #         data_dict: dict
    #         config_dict: dict

    #     Returns:
    #         batch_gt_map_cls: a list  of len == batch_size (BS)
    #             [gt_list_i], i = 0, 1, ..., BS-1
    #             where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
    #             where j = 0, ..., num of objects - 1 at sample input i
    #     """
        
    #     bbox_corner_labels = data_dict['gt_bbox'].detach().cpu().numpy()
    #     box_mask_labels = data_dict['gt_bbox_label'].detach().cpu().numpy()
    #     sem_cls_labels = data_dict['sem_cls_label'].detach().cpu().numpy()
    #     bsize = bbox_corner_labels.shape[0]
    #     max_num_obj = bbox_corner_labels.shape[1] # K2==MAX_NUM_OBJ

    #     batch_gt_map_cls = []
    #     for i in range(bsize):
    #         batch_gt_map_cls.append([(sem_cls_labels[i,j], bbox_corner_labels[i,j]) for j in range(max_num_obj) if box_mask_labels[i,j]==1])
    #     data_dict['batch_gt_map_cls'] = batch_gt_map_cls

    #     return batch_gt_map_cls