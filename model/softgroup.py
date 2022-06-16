import os
import torch
import functools
import random
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.softgroup_ops.functions import softgroup_ops
from lib.loss import *
from lib.utils.eval import get_nms_instances
from model.common import ResidualBlock, VGGBlock, UBlock


class SoftGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.DC = ScannetDatasetConfig(cfg)

        input_channel = cfg.model.use_coords * 3 + cfg.model.use_color * 3 + cfg.model.use_normal * 3
        m = cfg.model.m
        D = 3
        semantic_classes = cfg.data.classes
        blocks = cfg.model.blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual

        self.freeze_backbone = cfg.model.freeze_backbone
        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.grouping_radius = cfg.model.grouping.grouping_radius
        self.grouping_meanActive = cfg.model.grouping.mean_active
        self.grouping_npoint_threshold = cfg.model.grouping.npoint_threshold

        self.prepare_epochs = cfg.model.prepare_epochs

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock
        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        """
        Bottom-up Grouping Block
        """
        # 1. backbone U-Net
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(input_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m * c for c in blocks], sp_norm, block_reps, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, semantic_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, 3)
        )

        """
        Top-down Refinement Block
        """
        # 3 tiny U-Net
        self.tiny_unet = nn.Sequential(
            UBlock([m, 2 * m], sp_norm, 2, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        # 4.1 classification branch
        self.classification_branch = nn.Linear(m, semantic_classes + 1)

        # 4.2 mask scoring branch
        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(inplace=True),
            nn.Linear(m, semantic_classes + 1)
        )

        # 5
        self.iou_score = nn.Linear(m, semantic_classes + 1)

        self._init_random_seed()
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

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, scale, spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = softgroup_ops.sec_min(coords, clusters_offset.cuda())
        coords_max = softgroup_ops.sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = softgroup_ops.voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = softgroup_ops.voxelization(feats, out_map.cuda())

        voxelization_feats = ME.SparseTensor(features=out_feats, coordinates=out_coords.int().cuda())
        return voxelization_feats, inp_map

    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1

        """
            Bottom-up Grouping Block
        """
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()]  # (N, m) TODO: the naming p2v is wrong! should be v2p
        semantic_scores = self.semantic_branch(pt_feats)  # (N, nClass), float
        data_dict["semantic_scores"] = semantic_scores
        pt_offsets = self.offset_branch(pt_feats)  # (N, 3), float32
        data_dict["pt_offsets"] = pt_offsets

        if self.current_epoch > self.prepare_epochs or self.freeze_backbone:
            """
                Top-down Refinement Block
            """
            semantic_scores = semantic_scores.softmax(dim=-1)
            batch_idxs = data_dict["locs_scaled"][:, 0].int()

            # hyperparameters from config
            grouping_radius = self.hparams.cfg.model.grouping_cfg.radius
            grouping_mean_active = self.hparams.cfg.model.grouping_cfg.mean_active
            grouping_num_point_threshold = self.hparams.cfg.model.grouping_cfg.npoint_thr

            class_num_point_mean = torch.tensor(self.hparams.cfg.model.grouping_cfg.class_numpoint_mean,
                                                dtype=torch.float32)
            proposals_offset_list = []
            proposals_idx_list = []

            for class_id in range(self.hparams.cfg.data.classes):
                if class_id in self.hparams.cfg.data.ignore_classes:
                    continue
                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > self.hparams.cfg.model.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.hparams.cfg.model.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]
                idx, start_len = softgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                                                 grouping_radius, grouping_mean_active)
                proposals_idx, proposals_offset = softgroup_ops.bfs_cluster(class_num_point_mean, idx.cpu(),
                                                                            start_len.cpu(),
                                                                            grouping_num_point_threshold, class_id)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)

            if proposals_offset.shape[0] > self.hparams.cfg.model.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.hparams.cfg.model.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            data_dict["proposals_idx"] = proposals_idx
            data_dict["proposals_offset"] = proposals_offset

            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                pt_feats,
                data_dict["locs"],
                rand_quantize=True,
                **self.instance_voxel_cfg)

            feats = self.tiny_unet(inst_feats)

            # predict mask scores
            mask_scores = self.mask_scoring_branch(feats.features)
            data_dict["mask_scores"] = mask_scores[inst_map.long()]
            data_dict["instance_batch_idxs"] = feats.indices[:, 0][inst_map.long()]

            # predict instance cls and iou scores
            feats = self.global_pool(feats)
            data_dict["cls_scores"] = self.classification_branch(feats)
            data_dict["iou_scores"] = self.iou_score(feats)

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
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr,
                              momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented

        return [optimizer]

    def _loss(self, data_dict):
        N = data_dict["sem_labels"].shape[0]
        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.cfg.data.ignore_label)
        semantic_loss = sem_seg_criterion(data_dict["semantic_scores"], data_dict["sem_labels"])
        data_dict["semantic_loss"] = (semantic_loss, N)

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"][:, 0:3] - data_dict["locs"]  # (N, 3)
        valid = (data_dict["instance_ids"] != self.cfg.data.ignore_label).float()
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(data_dict["pt_offsets"], gt_offsets, valid_mask=valid)
        data_dict["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        data_dict["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + \
               self.cfg.train.loss_weight[2] * offset_dir_loss

        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            proposals_idx = data_dict["proposals_idx"][:, 1].cuda()
            proposals_offset = data_dict["proposals_offset"].cuda()

            # calculate iou of clustered instance
            ious_on_cluster = softgroup_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_num_point"])

            # filter out background instances
            fg_inds = (data_dict["instance_semantic_cls"] != self.ignore_label)
            fg_instance_cls = data_dict["instance_semantic_cls"][fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # overlap > thr on fg instances are positive samples
            max_iou, gt_inds = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= self.train_cfg.pos_iou_thr
            pos_gt_inds = gt_inds[pos_inds]

            """classification loss"""
            # follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0),), self.instance_classes)
            labels[pos_inds] = fg_instance_cls[pos_gt_inds]
            classification_criterion = ClassificationLoss()
            classification_loss = classification_criterion(data_dict["cls_scores"], labels)
            data_dict["classification_loss"] = (classification_loss, )

            """mask scoring loss"""
            mask_cls_label = labels[data_dict["instance_batch_idxs"].long()]
            slice_inds = torch.arange(0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
            mask_scores_sigmoid_slice = data_dict["mask_scores"].sigmoid()[slice_inds, mask_cls_label]
            mask_label = softgroup_ops.get_mask_label(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_semantic_cls"],
                                                      data_dict["instance_num_point"], ious_on_cluster,
                                                      self.train_cfg.pos_iou_thr)
            mask_label_weight = (mask_label != -1).float()
            mask_label[mask_label == -1.] = 0.5  # any value is ok
            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_weight, reduction='sum')
            mask_scoring_loss = mask_scoring_criterion(mask_scores_sigmoid_slice, mask_label)
            mask_scoring_loss /= (mask_label_weight.sum() + 1)

            """iou scoring loss"""
            ious = softgroup_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_num_point"],
                                                      mask_scores_sigmoid_slice.detach())
            fg_ious = ious[:, fg_inds]
            gt_ious, _ = fg_ious.max(1)
            slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            iou_score_weight = (labels < self.instance_classes).float()
            iou_score_slice = data_dict["iou_scores"][slice_inds, labels]
            iou_scoring_criterion = IouScoringLoss(reduction="none")
            iou_scoring_loss = iou_scoring_criterion(iou_score_slice, gt_ious)
            iou_scoring_loss = (iou_scoring_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)

            loss += + self.cfg.train.loss_weight[3] * classification_loss + self.cfg.train.loss_weight[
                4] * mask_scoring_loss + self.cfg.train.loss_weight[5] * iou_scoring_loss

        """total loss"""
        data_dict["total_loss"] = (loss, N)
        return data_dict

    def _feed(self, data_dict):
        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = softgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"],
                                                              self.cfg.data.mode)  # (M, C), float, cuda
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
                self.log("train/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=True, on_epoch=True,
                         sync_dist=True)

        return loss

    def validation_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        # prepare input and forward
        data_dict = self._feed(data_dict)
        data_dict = self._loss(data_dict)

        in_prog_bar = ["total_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("val/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=False, on_epoch=True,
                         sync_dist=True)

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
