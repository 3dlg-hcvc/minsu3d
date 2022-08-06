'''
PointGroup operations
Written by Li Jiang
'''

import torch
from torch.autograd import Function
import COMMON_OPS


class Voxelization_Idx(Function):
    @staticmethod
    def forward(ctx, coords, vert_batch_idxs, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int (M, (maxActive + 1))
        :return: input_map: int (N,)
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()
        input_map = torch.zeros(N, dtype=torch.int32, device="cpu")
        output_map = input_map.new()
        COMMON_OPS.voxelize_idx(coords, output_coords, vert_batch_idxs, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None


voxelization_idx = Voxelization_Idx.apply


class Voxelization(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int (M, (maxActive + 1))
        :param feats: cuda float (N, C)
        :return: output_feats: cuda float (M, C)
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1
        output_feats = torch.zeros((M, C), dtype=torch.float32, device="cuda")
        ctx.for_backwards = (map_rule, mode, maxActive, N)
        COMMON_OPS.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()
        d_feats = torch.zeros((N, C), dtype=torch.float32, device="cuda")
        COMMON_OPS.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None


voxelization = Voxelization.apply


class BallQueryBatchP(Function):
    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        '''
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) int
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        '''

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.zeros(n * meanActive, dtype=torch.int32, device="cuda")
            start_len = torch.zeros((n, 2), dtype=torch.int32, device="cuda")
            nActive = COMMON_OPS.ballquery_batch_p(coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius)
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class SecMean(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device=inp.device)

        COMMON_OPS.sec_mean(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_mean = SecMean.apply


class SecMin(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.sec_min(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_min = SecMin.apply


class SecMax(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply

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
        mask_label = torch.zeros(proposals_idx.shape, dtype=torch.bool, device="cuda")
        mask_label_mask = torch.zeros(proposals_idx.shape, dtype=torch.bool, device="cuda")

        assert proposals_iou.is_contiguous() and proposals_iou.is_cuda
        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_cls.is_contiguous() and instance_cls.is_cuda

        COMMON_OPS.get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                           proposals_iou, nInstance, nProposal, ignored_label, iou_thr, mask_label, mask_label_mask)

        return mask_label, mask_label_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_label = GetMaskLabel.apply
