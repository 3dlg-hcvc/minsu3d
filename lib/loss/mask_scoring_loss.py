import torch.nn as nn


class MaskScoringLoss(nn.Module):

    def __init__(self, weight, reduction):
        super(MaskScoringLoss, self).__init__()
        self.criterion = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)
