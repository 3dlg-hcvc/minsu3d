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

        input_map = torch.IntTensor(N).zero_()
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
        output_feats = torch.cuda.FloatTensor(M, C).zero_()
        ctx.for_backwards = (map_rule, mode, maxActive, N)
        COMMON_OPS.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()
        d_feats = torch.cuda.FloatTensor(N, C).zero_()
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
            idx = torch.cuda.IntTensor(n * meanActive).zero_()
            start_len = torch.cuda.IntTensor(n, 2).zero_()
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

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

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

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

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

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        COMMON_OPS.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply
