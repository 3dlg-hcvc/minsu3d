import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.utils.bbox import get_3d_box_batch, get_aabb3d_iou_batch, get_3d_box


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


def nn_distance_stack(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (N,C) torch tensor
        pc2: (M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (N,) torch float32 tensor
        idx1: (N,) torch int64 tensor
        dist2: (M,) torch float32 tensor
        idx2: (M,) torch int64 tensor
    """
    N = pc1.shape[0]
    M = pc2.shape[0]
    pc1_expand_tile = pc1.unsqueeze(1).repeat(1,M,1) # (N, M, C)
    pc2_expand_tile = pc2.unsqueeze(0).repeat(N,1,1) # (N, M, C)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (N,M)
    
    dist1, idx1 = torch.min(pc_dist, dim=1) # (N)
    dist2, idx2 = torch.min(pc_dist, dim=0) # (M)
        
    return dist1, idx1, dist2, idx2


def compute_box_and_sem_cls_loss(loss_input, data_dict, loss_dict, mean_size_arr, DC):
    """ Compute 3D bounding box and semantic classification loss.
    Args:
        data_dict: dict (read-only)
    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    batch_size = len(data_dict["batch_offsets"]) - 1
    num_heading_bin = 1
    num_size_cluster = 18
    num_class = 18
    
    pred_center, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals, sem_cls_scores, proposal_offsets = loss_input['proposal_pred_bboxes']
    instance_offsets = data_dict['instance_offsets']
    
    ####### eval
    # pred_center = end_points['center'] # num_proposal,3
    pred_heading_class = torch.argmax(heading_scores, -1) # num_proposal
    pred_heading_residual = torch.gather(heading_scores, 1, pred_heading_class.unsqueeze(-1)).squeeze(1) # num_proposal
    pred_size_class = torch.argmax(size_scores, -1) # num_proposal
    pred_size_residual = torch.gather(size_residuals, 1, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1) # num_proposal,3
    # pred_sem_cls = torch.argmax(sem_cls_scores, -1) # num_proposal
    # sem_cls_probs = softmax(sem_cls_scores.detach().cpu().numpy()) # num_proposal,18
    # pred_sem_cls_prob = np.max(sem_cls_probs,-1) # B,num_proposal

    num_proposal = pred_center.shape[0] 
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    pred_corners_3d_upright_camera = np.zeros((num_proposal, 8, 3))
    # pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    pred_center_upright_camera = pred_center.detach().cpu().numpy()
    for j in range(num_proposal):
        heading_angle = DC.class2angle(pred_heading_class[j].detach().cpu().numpy(), pred_heading_residual[j].detach().cpu().numpy())
        box_size = DC.class2size(int(pred_size_class[j].detach().cpu().numpy()), pred_size_residual[j].detach().cpu().numpy())
        corners_3d_upright_camera = get_3d_box(pred_center_upright_camera[j,:], box_size, heading_angle)
        pred_corners_3d_upright_camera[j] = corners_3d_upright_camera
    
    # iou between crop_bbox and pred_bbox
    proposal_crop_bboxes, proposals_batchId = loss_input['proposal_crop_bboxes']
    # dist1, ind1, dist2, ind2 = nn_distance_stack(proposal_crop_bboxes[:, 0:3], data_dict['center_label'])
    proposal_crop_bboxes = proposal_crop_bboxes.detach().cpu().numpy() # (nProposals, center+size+heading+label)
    proposal_crop_bboxes = get_3d_box_batch(proposal_crop_bboxes[:, 0:3], proposal_crop_bboxes[:, 3:6], proposal_crop_bboxes[:, 6]) # (nProposals, 8, 3)
    assert proposal_crop_bboxes.shape == pred_corners_3d_upright_camera.shape
    ious = get_aabb3d_iou_batch(proposal_crop_bboxes, pred_corners_3d_upright_camera)
    loss_dict['pred_crop_bbox_ious@25'] = ((ious > 0.25).mean(), num_proposal)
    loss_dict['pred_crop_bbox_ious@50'] = ((ious > 0.5).mean(), num_proposal)
    
    # gt bbox
    center_label = data_dict['center_label']
    heading_class_label = data_dict['heading_class_label']
    heading_residual_label = data_dict['heading_residual_label']
    size_class_label = data_dict['size_class_label']
    size_residual_label = data_dict['size_residual_label']
    num_instance = center_label.shape[0] 
    
    gt_corners_3d_upright_camera = np.zeros((num_proposal, 8, 3))
    gt_center_upright_camera = center_label.detach().cpu().numpy()
    for j in range(num_instance):
        heading_angle = DC.class2angle(heading_class_label[j].detach().cpu().numpy(), heading_residual_label[j].detach().cpu().numpy())
        box_size = DC.class2size(int(size_class_label[j].detach().cpu().numpy()), size_residual_label[j].detach().cpu().numpy())
        corners_3d_upright_camera = get_3d_box(gt_center_upright_camera[j,:], box_size, heading_angle)
        gt_corners_3d_upright_camera[j] = corners_3d_upright_camera
    
    # gt_ious = get_aabb3d_iou_batch(proposal_bbox, gt_corners_3d_upright_camera[ind1.cpu().numpy(),:,:])
    # iou between crop_bbox and gt_bbox
    crop_bbox_ious = np.zeros(num_proposal)
    for b in range(batch_size):
        pred_batch_start, pred_batch_end = proposal_offsets[b].cpu().numpy(), proposal_offsets[b+1].cpu().numpy()
        pred_num = pred_batch_end - pred_batch_start # N
        gt_batch_start, gt_batch_end = instance_offsets[b].cpu().numpy(), instance_offsets[b+1].cpu().numpy()
        gt_num = gt_batch_end - gt_batch_start # M
        
        for i in range(pred_num):
            crop_bbox_iou = get_aabb3d_iou_batch(np.tile(proposal_crop_bboxes[pred_batch_start+i], (gt_num, 1, 1)), gt_corners_3d_upright_camera[gt_batch_start:gt_batch_end])
            crop_bbox_ious[pred_batch_start+i] = np.max(crop_bbox_iou)
    loss_dict['crop_bbox_ious@25'] = ((crop_bbox_ious > 0.25).mean(), num_proposal)
    loss_dict['crop_bbox_ious@50'] = ((crop_bbox_ious > 0.5).mean(), num_proposal)
    
    pred_bbox_ious = np.zeros(num_proposal)
    for b in range(batch_size):
        pred_batch_start, pred_batch_end = proposal_offsets[b].cpu().numpy(), proposal_offsets[b+1].cpu().numpy()
        pred_num = pred_batch_end - pred_batch_start # N
        gt_batch_start, gt_batch_end = instance_offsets[b].cpu().numpy(), instance_offsets[b+1].cpu().numpy()
        gt_num = gt_batch_end - gt_batch_start # M
        
        for i in range(pred_num):
            pred_bbox_iou = get_aabb3d_iou_batch(np.tile(pred_corners_3d_upright_camera[pred_batch_start+i], (gt_num, 1, 1)), gt_corners_3d_upright_camera[gt_batch_start:gt_batch_end])
            pred_bbox_ious[pred_batch_start+i] = np.max(pred_bbox_iou)
    loss_dict['pred_bbox_ious@25'] = ((pred_bbox_ious > 0.25).mean(), num_proposal)
    loss_dict['pred_bbox_ious@50'] = ((pred_bbox_ious > 0.5).mean(), num_proposal)
    ######## end eval
    
    center_loss = torch.tensor(0.).cuda()
    heading_class_loss = torch.tensor(0.).cuda()
    heading_reg_loss = torch.tensor(0.).cuda()
    size_class_loss = torch.tensor(0.).cuda()
    size_reg_loss = torch.tensor(0.).cuda()
    sem_cls_loss = torch.tensor(0.).cuda()

    # object_assignment = data_dict['object_assignment']
    # batch_size = object_assignment.shape[0]
    batch_size = len(proposal_offsets) - 1
    for b in range(batch_size):
        pred_batch_start, pred_batch_end = proposal_offsets[b], proposal_offsets[b+1]
        pred_num = pred_batch_end - pred_batch_start # N
        gt_batch_start, gt_batch_end = instance_offsets[b], instance_offsets[b+1]
        gt_num = gt_batch_end - gt_batch_start # M
        # Compute center loss
        # pred_center = data_dict['center']
        pred_center_batch = pred_center[pred_batch_start:pred_batch_end]
        gt_center_batch = data_dict['center_label'][gt_batch_start:gt_batch_end]
        dist1, ind1, dist2, ind2 = nn_distance_stack(pred_center_batch, gt_center_batch) # dist1: (N,), dist2: (M,)
        object_assignment = ind1 # N
        # box_label_mask = data_dict['box_label_mask']
        # objectness_label = data_dict['objectness_label'].float()
        centroid_reg_loss1 = torch.sum(dist1)/(pred_num+1e-6)
        centroid_reg_loss2 = torch.sum(dist2)/(gt_num+1e-6)
        center_loss += (centroid_reg_loss1 + centroid_reg_loss2)

        # Compute heading loss
        heading_scores_batch = heading_scores[pred_batch_start:pred_batch_end] # (N, 1)
        heading_class_label_batch = data_dict['heading_class_label'][gt_batch_start:gt_batch_end] # (M,)
        heading_class_label_batch = torch.gather(heading_class_label_batch, 0, object_assignment) # select (N,) from (M,)
        criterion_heading_class = nn.CrossEntropyLoss()
        heading_class_loss += criterion_heading_class(heading_scores_batch, heading_class_label_batch) # (N,)
        # heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

        heading_residuals_normalized_batch = heading_residuals_normalized[pred_batch_start:pred_batch_end] # (N, 1)
        heading_residual_label_batch = data_dict['heading_residual_label'][gt_batch_start:gt_batch_end] # (M,)
        heading_residual_label_batch = torch.gather(heading_residual_label_batch, 0, object_assignment) # select (N,) from (M,)
        heading_residual_normalized_label_batch = heading_residual_label_batch / (np.pi/num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(heading_class_label_batch.shape[0], num_heading_bin).zero_() # (N, 1)
        heading_label_one_hot.scatter_(1, heading_class_label_batch.unsqueeze(-1), 1) # src==1 so it's *one-hot* (N,1)
        heading_residual_normalized_loss = huber_loss(torch.sum(heading_residuals_normalized_batch*heading_label_one_hot, -1) - heading_residual_normalized_label_batch, delta=1.0) # (N,)
        heading_reg_loss += torch.sum(heading_residual_normalized_loss)/(pred_num+1e-6)

        # Compute size loss
        size_scores_batch = size_scores[pred_batch_start:pred_batch_end] # (N, 18)
        size_class_label_batch = data_dict['size_class_label'][gt_batch_start:gt_batch_end] # (M,)
        size_class_label_batch = torch.gather(size_class_label_batch, 0, object_assignment) # select (N,) from (M,)
        criterion_size_class = nn.CrossEntropyLoss()
        size_class_loss += criterion_size_class(size_scores_batch, size_class_label_batch) # (N,)
        # size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

        size_residuals_normalized_batch = size_residuals_normalized[pred_batch_start:pred_batch_end] # (N, num_size_cluster, 3)
        size_residual_label_batch = data_dict['size_residual_label'][gt_batch_start:gt_batch_end] # (M, 3)
        size_residual_label_batch = torch.gather(size_residual_label_batch, 0, object_assignment.unsqueeze(-1).repeat(1,3)) # select (N,3) from (M,3)
        size_label_one_hot = torch.cuda.FloatTensor(size_class_label_batch.shape[0], num_size_cluster).zero_() # (N, num_size_cluster)
        size_label_one_hot.scatter_(1, size_class_label_batch.unsqueeze(-1), 1) # src==1 so it's *one-hot* (N,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,3) # (N,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(size_residuals_normalized_batch*size_label_one_hot_tiled, 1) # (N,3)

        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0) # (1,num_size_cluster,3) 
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 1) # (N,3)
        size_residual_label_normalized_batch = size_residual_label_batch / mean_size_label # (N,3)
        size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized_batch, delta=1.0), -1) # (N,3) -> (N,)
        size_reg_loss += torch.sum(size_residual_normalized_loss)/(pred_num+1e-6)

        # 3.4 Semantic cls loss
        sem_cls_scores_batch = sem_cls_scores[pred_batch_start:pred_batch_end] # (N, 18)
        sem_cls_label_batch = data_dict['sem_cls_label'][gt_batch_start:gt_batch_end] # (M,)
        sem_cls_label_batch = torch.gather(sem_cls_label_batch, 0, object_assignment) # select (N,) from (M,)
        criterion_sem_cls = nn.CrossEntropyLoss()
        sem_cls_loss += criterion_sem_cls(sem_cls_scores_batch, sem_cls_label_batch) # (N,)
        # sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
        
        # center_loss_total += center_loss
        # heading_class_loss_total += heading_class_loss
        # heading_residual_normalized_loss_total += heading_residual_normalized_loss
        # size_class_loss_total += size_class_loss
        # size_residual_normalized_loss_total += size_residual_normalized_loss
        # sem_cls_loss_total += sem_cls_loss
        
    loss_dict['center_loss'] = (center_loss/batch_size, batch_size)
    loss_dict['heading_cls_loss'] = (heading_class_loss/batch_size, batch_size)
    loss_dict['heading_reg_loss'] = (heading_reg_loss/batch_size, batch_size)
    loss_dict['size_cls_loss'] = (size_class_loss/batch_size, batch_size)
    loss_dict['size_reg_loss'] = (size_reg_loss/batch_size, batch_size)
    loss_dict['sem_cls_loss'] = (sem_cls_loss/batch_size, batch_size)
    
    bbox_loss = center_loss + 0.1*heading_class_loss + heading_reg_loss + 0.1*size_class_loss + size_reg_loss
    loss_dict['bbox_loss'] = (bbox_loss/batch_size, batch_size)
    # import pdb; pdb.set_trace()

    return loss_dict['bbox_loss'][0]
