import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.utils.bbox import get_3d_box_batch, get_aabb3d_iou_batch


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


def nn_distance_batch(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2


def nn_distance_stack(pc1, pc2, batch_offsets_pc1, batch_offsets_pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (N,C) torch tensor
        pc2: (M,C) torch tensor
        batch_offsets_pc1: (B+1,)
        batch_offsets_pc2: (B+1,)
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    dist1, idx1 = [], []
    dist2, idx2 = [], []
    batch_size = len(batch_offsets_pc1) - 1
    for b in range(batch_size):
        N = batch_offsets_pc1[b+1] - batch_offsets_pc1[b]
        M = batch_offsets_pc2[b+1] - batch_offsets_pc2[b]
        pc1_b = pc1[batch_offsets_pc1[b+1]:batch_offsets_pc1[b]]
        pc2_b = pc2[batch_offsets_pc2[b+1]:batch_offsets_pc2[b]]
        pc1_expand_tile = pc1_b.unsqueeze(1).repeat(1,M,1) # (N, M, C)
        pc2_expand_tile = pc2_b.unsqueeze(0).repeat(N,1,1) # (N, M, C)
        pc_diff_b = pc1_expand_tile - pc2_expand_tile
        
        if l1smooth:
            pc_dist_b = torch.sum(huber_loss(pc_diff_b, delta), dim=-1) # (N,M)
        elif l1:
            pc_dist_b = torch.sum(torch.abs(pc_diff_b), dim=-1) # (N,M)
        else:
            pc_dist_b = torch.sum(pc_diff_b**2, dim=-1) # (N,M)
        
        dist1_b, idx1_b = torch.min(pc_dist_b, dim=1) # (N)
        dist2_b, idx2_b = torch.min(pc_dist_b, dim=0) # (M)
        dist1.append(dist1_b)
        idx1.append(idx1_b)
        dist2.append(dist2_b)
        idx2.append(idx2_b)
        
    return dist1, idx1, dist2, idx2