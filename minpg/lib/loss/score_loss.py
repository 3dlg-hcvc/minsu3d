import pytorch_lightning as pl
import torch.nn as nn


class ScoreLoss(pl.LightningModule):

    def __init__(self):
        super(ScoreLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)
