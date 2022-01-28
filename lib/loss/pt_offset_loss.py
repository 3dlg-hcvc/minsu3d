import torch
import torch.nn as nn


class PTOffsetLoss(nn.Module):
    
    def __init__(self):
        super(PTOffsetLoss, self).__init__()
    
    def forward(self, preds, gts, valid_mask=None):
        """Pointwise offset prediction losses in norm and direction

        Args:
            preds (torch.Tensor): predicted point offsets, (B, 3), float32, cuda
            gts (torch.Tensor): GT point offsets, (B, 3), float32, cuda
            valid_mask (torch.Tensor): indicate valid points involving in loss, (B,), bool, cuda

        Returns:
            torch.Tensor: [description]
        """
        pt_diff = preds - gts   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        
        if valid_mask is not None:
            offset_norm_loss = torch.sum(pt_dist * valid_mask) / (torch.sum(valid_mask) + 1e-6)
            offset_dir_loss = torch.sum(direction_diff * valid_mask) / (torch.sum(valid_mask) + 1e-6)
        else:
            offset_norm_loss = torch.mean(pt_dist)
            offset_dir_loss = torch.mean(direction_diff)
        
        return offset_norm_loss, offset_dir_loss