import torch
from torch.autograd import Function
import COMMON_OPS


class PGBFSCluster(Function):
    @staticmethod
    def forward(ctx, semantic_label, ball_query_idxs, start_len, threshold):
        '''
        :param ctx:
        :param semantic_label: (N), int
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        '''

        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = semantic_label.new()
        cluster_offsets = semantic_label.new()

        COMMON_OPS.pg_bfs_cluster(semantic_label, ball_query_idxs, start_len, cluster_idxs, cluster_offsets, N, threshold)

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


pg_bfs_cluster = PGBFSCluster.apply


class RoiPool(Function):
    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")
        output_maxidx = torch.zeros((nProposal, C), dtype=torch.int32, device="cuda")

        COMMON_OPS.roipool_fp(feats, proposals_offset, output_feats, output_maxidx, nProposal, C)

        ctx.for_backwards = (output_maxidx, proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        output_maxidx, proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.zeros((sumNPoint, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.roipool_bp(d_feats, proposals_offset, output_maxidx, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


roipool = RoiPool.apply


class GetIoU(Function):
    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -1
        :param instance_pointnum: (total_nInst), int
        :return: proposals_iou: (nProposal, total_nInst), float
        '''
        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        proposals_iou = torch.zeros((nProposal, nInstance), dtype=torch.float32, device="cuda")

        COMMON_OPS.get_iou(proposals_idx, proposals_offset, instance_labels, instance_pointnum, proposals_iou, nInstance,
                      nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_iou = GetIoU.apply


class PointRecover(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, nPoint):
        '''
        :param ctx:
        :param feats: cuda float M * C
        :param map_rule: cuda int M * (maxActive + 1)
        :param nPoint: int
        :return: output_feats: cuda float N * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        M, C = feats.size()
        maxActive = map_rule.size(1) - 1

        output_feats = torch.zeros((nPoint, C), dtype=torch.float32, device="cuda")

        ctx.for_backwards = (map_rule, maxActive, M)

        COMMON_OPS.point_recover_fp(feats, output_feats, map_rule, M, maxActive, C)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, maxActive, M = ctx.for_backwards
        N, C = d_output_feats.size()

        d_feats = torch.zeros((M, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.point_recover_bp(d_output_feats.contiguous(), d_feats, map_rule, M, maxActive, C)

        return d_feats, None, None


point_recover = PointRecover.apply
