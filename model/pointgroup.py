import torch
import torch.nn as nn
import MinkowskiEngine as ME
from lib.pointgroup_ops.functions import pointgroup_ops
from model.common import ResidualBlock, VGGBlock, UBlock

import functools


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_channel = cfg.model.input_channel + cfg.model.use_coords * 3
        m = cfg.model.m
        D = 3
        classes = cfg.data.classes
        blocks = cfg.model.blocks
        cluster_blocks = cfg.model.cluster_blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual

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
        
        self.score_linear = nn.Linear(m, 1)
        # self.score_linear = nn.Linear(128, 1)
        
    
    @staticmethod
    def get_batch_offsets(batch_idxs, batch_size):
        '''
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        '''
        batch_offsets = torch.zeros(batch_size + 1).int().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
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


    def forward(self, data):
        ret = {}
        batch_size = self.cfg.data.batch_size #len(data["batch_offsets"]) - 1
        x = ME.SparseTensor(features=data['voxel_feats'], coordinates=data['voxel_locs'].int())

        #### backbone
        out = self.backbone(x)
        pt_feats = out.features[data['p2v_map'].long()] # (N, m)

        #### semantic segmentation
        semantic_scores = self.sem_seg(pt_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long, {0, 1, ..., classes}
        ret['semantic_scores'] = semantic_scores

        #### offsets
        pt_offsets = self.offset_net(pt_feats) # (N, 3), float32
        ret['pt_offsets'] = pt_offsets

        if data['epoch'] > self.prepare_epochs or self.freeze_backbone:
            #### get prooposal clusters
            batch_idxs = data['locs_scaled'][:, 0].int()
            object_idxs = torch.nonzero(semantic_preds > 0, as_tuple=False).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = data['locs'][object_idxs]
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

            #### proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map, (proposals_center, proposals_size) = self.clusters_voxelization(proposals_idx, proposals_offset, pt_feats, data['locs'], self.score_fullscale, self.score_scale, self.mode)
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)
            # proposals_center / proposals_size: (nProposals, 3)

            #### score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long()] # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset.cuda())  # (nProposal, C)
            # proposals_score_feats = self.proposal_mlp(proposals_score_feats) # (nProposal, 128)
            scores = self.score_linear(proposals_score_feats)  # (nProposal, 1)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
        
        return ret
