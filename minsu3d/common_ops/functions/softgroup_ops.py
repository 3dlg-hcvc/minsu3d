import torch
from torch.autograd import Function
import COMMON_OPS



class SGBFSCluster(Function):

    @staticmethod
    def forward(ctx, class_numpoint_mean, ball_query_idxs, start_len, threshold, class_id):

        N = start_len.size(0)
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_numpoint_mean = torch.tensor(class_numpoint_mean, dtype=torch.float32)

        cluster_obj_idxs, cluster_point_idxs, cluster_offsets = COMMON_OPS.sg_bfs_cluster(
            cluster_numpoint_mean, ball_query_idxs, start_len, N, threshold, class_id
        )

        return cluster_obj_idxs, cluster_point_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


sg_bfs_cluster = SGBFSCluster.apply


class GlobalAvgPool(Function):

    @staticmethod
    def forward(ctx, feats, proposals_offset):
        """
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        """
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.global_avg_pool_fp(feats, proposals_offset, output_feats, nProposal, C)

        ctx.for_backwards = (proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.zeros((sumNPoint, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.global_avg_pool_bp(d_feats, proposals_offset, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


global_avg_pool = GlobalAvgPool.apply
