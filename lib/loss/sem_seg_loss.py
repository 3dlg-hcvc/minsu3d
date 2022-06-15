import torch.nn as nn


class SemSegLoss(nn.Module):
    
    def __init__(self, ignore_label=None):
        super(SemSegLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    
    def forward(self, predictions, gts):
        """Semantic segmentation loss using CrossEntropyLoss

        Args:
            predictions (torch.Tensor): predicted scores, (B, nClass), float32, cuda
            gts (torch.Tensor): ground truth label, (B,), long, cuda

        Returns:
            torch.Tensor: [description]
        """
        return self.criterion(predictions, gts)
