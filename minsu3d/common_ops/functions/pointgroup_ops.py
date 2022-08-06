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
