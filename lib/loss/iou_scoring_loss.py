import pytorch_lightning as pl
import torch.nn as nn


class IouScoringLoss(pl.LightningModule):

    def __init__(self, reduction):
        super(IouScoringLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)