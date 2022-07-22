import torch
from torch.autograd import Function
import COMMON_OPS


class GetMaskIoUOnCluster(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.zeros((nProposal, nInstance), dtype=torch.float32, device="cuda")

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        COMMON_OPS.get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, proposals_iou, nInstance, nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_cluster = GetMaskIoUOnCluster.apply


class GetMaskIoUOnPred(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum,
                mask_scores_sigmoid):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.zeros((nProposal, nInstance), dtype=torch.float32, device="cuda")

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda
        assert mask_scores_sigmoid.is_contiguous() and mask_scores_sigmoid.is_cuda

        COMMON_OPS.get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                 instance_pointnum, proposals_iou, nInstance, nProposal,
                                 mask_scores_sigmoid)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_pred = GetMaskIoUOnPred.apply


class GetMaskLabel(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_cls,
                instance_pointnum, proposals_iou, ignored_label, iou_thr):
        """
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        """

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        mask_label = torch.full(proposals_idx.shape, fill_value=-1, dtype=torch.float32, device="cuda")

        assert proposals_iou.is_contiguous() and proposals_iou.is_cuda
        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_cls.is_contiguous() and instance_cls.is_cuda

        COMMON_OPS.get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                           proposals_iou, nInstance, nProposal, ignored_label, iou_thr, mask_label)

        return mask_label

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_label = GetMaskLabel.apply

class SGBFSCluster(Function):

    @staticmethod
    def forward(ctx, class_numpoint_mean, ball_query_idxs, start_len, threshold, class_id):
        """
        :param ctx:
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        """

        N = start_len.size(0)
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = ball_query_idxs.new()
        cluster_offsets = ball_query_idxs.new()
        cluster_numpoint_mean = torch.tensor(class_numpoint_mean, dtype=torch.float32)

        COMMON_OPS.sg_bfs_cluster(cluster_numpoint_mean, ball_query_idxs, start_len, cluster_idxs,
                        cluster_offsets, N, threshold, class_id)

        return cluster_idxs, cluster_offsets

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
