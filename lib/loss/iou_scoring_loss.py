import torch.nn as nn


class IouScoringLoss(nn.Module):

    def __init__(self, reduction):
        super(IouScoringLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)